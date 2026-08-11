"""A small, fully deterministic transformer for injection experiments.

Deliberately minimal. This is a substrate for measuring corruption, not a model
anyone should train for its own sake — so it has no dropout (a needless source of
RNG coupling), no fused kernels, and no attention implementation that dispatches
differently by shape.

Every ``nn.Linear`` is wrapped so the injector can reach GEMM outputs, which is
where an arithmetic defect in real hardware would actually surface.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .inject import Injector


@dataclass(frozen=True)
class ModelConfig:
    """Sizes are small by default so gates run in seconds on CPU."""

    vocab_size: int = 128
    n_layer: int = 2
    n_head: int = 4
    n_embd: int = 128
    block_size: int = 64
    dtype: torch.dtype = torch.float32

    @property
    def head_dim(self) -> int:
        if self.n_embd % self.n_head:
            raise ValueError("n_embd must be divisible by n_head")
        return self.n_embd // self.n_head


class InjectedLinear(nn.Linear):
    """``nn.Linear`` whose output passes through the injector's ``gemm_out`` site."""

    injector: Injector | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = super().forward(x)
        if self.injector is not None:
            out = self.injector.corrupt(out, "gemm_out")
        return out


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.qkv = InjectedLinear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = InjectedLinear(cfg.n_embd, cfg.n_embd, bias=False)
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size)).view(
            1, 1, cfg.block_size, cfg.block_size
        )
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        # (B, n_head, T, head_dim)
        shape = (B, T, self.cfg.n_head, self.cfg.head_dim)
        q = q.view(shape).transpose(1, 2)
        k = k.view(shape).transpose(1, 2)
        v = v.view(shape).transpose(1, 2)

        # Written out rather than using scaled_dot_product_attention, whose
        # backend selection varies by shape and platform and would put a
        # nondeterminism risk directly under the thing we are measuring.
        att = (q @ k.transpose(-2, -1)) * (self.cfg.head_dim ** -0.5)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.fc = InjectedLinear(cfg.n_embd, 4 * cfg.n_embd, bias=False)
        self.out = InjectedLinear(4 * cfg.n_embd, cfg.n_embd, bias=False)
        self._injector: Injector | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        h = self.out(F.gelu(self.fc(self.ln2(x))))
        if self._injector is not None:
            h = self._injector.corrupt(h, "activation")
        return x + h


class TinyTransformer(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = InjectedLinear(cfg.n_embd, cfg.vocab_size, bias=False)

    def attach_injector(self, injector: Injector | None) -> None:
        """Wire (or unwire) the injector into every site this model exposes."""
        for module in self.modules():
            if isinstance(module, InjectedLinear):
                module.injector = injector
            if isinstance(module, Block):
                module._injector = injector

    def forward(self, idx: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok(idx) + self.pos(pos)
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.ln_f(x))
        return F.cross_entropy(
            logits.reshape(-1, self.cfg.vocab_size), targets.reshape(-1)
        )

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
