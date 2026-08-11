"""Deterministic training loop with clean and injected modes.

Determinism here is a configuration, not a hope. Gate 0.2 — two clean runs at the
same seed producing bit-identical weights — is the control that makes every later
comparison meaningful, so it is established before any injection happens.

Clean and injected runs take the *identical* code path and consume the *identical*
data stream. The only difference is whether an injector is attached. Batches are
materialised up front from a dedicated generator so that data ordering cannot
drift between the two, which is the failure mode that would quietly invalidate
every divergence number this project produces.
"""

from __future__ import annotations

import os
import random
from dataclasses import asdict, dataclass

import torch

from .inject import Injector
from .model import ModelConfig, TinyTransformer
from .record import (
    RunRecord,
    StepRecord,
    delta_norm,
    fingerprint,
    grad_norm,
    snapshot,
)


@dataclass(frozen=True)
class TrainConfig:
    steps: int = 40
    batch_size: int = 16
    lr: float = 3e-3
    seed: int = 1234
    data_seed: int = 99
    signal: float = 0.9  # P(next token follows the learnable permutation)
    model: ModelConfig = ModelConfig()


def set_determinism(seed: int) -> None:
    """Pin every source of run-to-run variation we can reach from Python.

    CUDA note — UNVALIDATED IN THIS CONTAINER (no GPU present): cuBLAS reads
    ``CUBLAS_WORKSPACE_CONFIG`` when its handle is created, which for most
    programs is before this function runs. Setting it here is best-effort only;
    export it in the shell before launching python:

        CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m sdc.runner
    """
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("PYTHONHASHSEED", str(seed))

    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)

    # CPU reduction order depends on the thread count, so bit-identical results
    # require pinning it. This costs wall-clock and is worth it.
    torch.set_num_threads(1)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():  # pragma: no cover - no GPU in this container
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


def make_batches(cfg: TrainConfig) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Materialise the whole data stream up front, from its own generator.

    Separate from both the model's RNG and the injector's. Any of the three
    sharing a stream would couple them, and the coupling would look exactly like
    a corruption effect.

    The sequence is a noisy first-order Markov chain: the successor of a token is
    a fixed permutation of it with probability ``cfg.signal``, and uniform noise
    otherwise. Uniform random tokens would be unlearnable, and a loop that cannot
    learn passes every determinism gate trivially while proving nothing — the
    corruption has to have real optimisation dynamics to perturb.
    """
    gen = torch.Generator(device="cpu")
    gen.manual_seed(cfg.data_seed)
    m = cfg.model
    V, T = m.vocab_size, m.block_size + 1

    successor = torch.randperm(V, generator=gen)

    batches = []
    for _ in range(cfg.steps):
        tokens = torch.empty((cfg.batch_size, T), dtype=torch.int64)
        tokens[:, 0] = torch.randint(
            0, V, (cfg.batch_size,), generator=gen, dtype=torch.int64
        )
        for t in range(1, T):
            signal = successor[tokens[:, t - 1]]
            noise = torch.randint(
                0, V, (cfg.batch_size,), generator=gen, dtype=torch.int64
            )
            take_signal = torch.rand(cfg.batch_size, generator=gen) < cfg.signal
            tokens[:, t] = torch.where(take_signal, signal, noise)
        batches.append((tokens[:, :-1].contiguous(), tokens[:, 1:].contiguous()))
    return batches


def train(cfg: TrainConfig, injector: Injector | None = None) -> RunRecord:
    """Run training. Attaching an injector is the only difference between modes."""
    set_determinism(cfg.seed)

    model = TinyTransformer(cfg.model)
    model.attach_injector(injector)
    if injector is not None:
        injector.reset()

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    batches = make_batches(cfg)

    record = RunRecord(
        config={
            **{k: v for k, v in asdict(cfg).items() if k != "model"},
            "model": {
                k: str(v) if isinstance(v, torch.dtype) else v
                for k, v in asdict(cfg.model).items()
            },
            "injector": repr(injector) if injector else None,
        }
    )

    for step, (x, y) in enumerate(batches):
        if injector is not None:
            injector.set_step(step)
        before = snapshot(model)
        seen = len(injector.events) if injector is not None else 0

        opt.zero_grad(set_to_none=True)
        loss = model(x, y)
        loss.backward()

        if injector is not None:
            # Stands in for corruption on the wire during gradient all-reduce.
            # Single-device here, so the payload is corrupted directly; the site
            # is the same one a real NCCL path would expose.
            for p in model.parameters():
                if p.grad is not None:
                    injector.corrupt_(p.grad, "allreduce")

        opt.step()

        if injector is not None:
            for group in opt.param_groups:
                for p in group["params"]:
                    state = opt.state.get(p, {})
                    for key in ("exp_avg", "exp_avg_sq"):
                        if key in state:
                            injector.corrupt_(state[key], "optim_state")

        loss_value = float(loss.detach())
        record.steps.append(
            StepRecord(
                step=step,
                loss=loss_value,
                grad_norm=grad_norm(model),
                weight_delta=delta_norm(model, before),
                n_injections=(len(injector.events) - seen) if injector else 0,
            )
        )
        if loss_value != loss_value or loss_value in (float("inf"), float("-inf")):
            record.diverged = True
            break

    record.final_fingerprint = fingerprint(model)
    if injector is not None:
        record.injection_manifest = injector.manifest()
    return record


if __name__ == "__main__":  # pragma: no cover - manual smoke check
    cfg = TrainConfig()
    clean = train(cfg)
    print(f"params      : {TinyTransformer(cfg.model).n_params:,}")
    print(f"clean loss  : {clean.losses[0]:.4f} -> {clean.final_loss:.4f}")
    print(f"fingerprint : {clean.final_fingerprint[:16]}")
