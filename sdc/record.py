"""Per-step telemetry, and bit-exact fingerprints of model state.

The fingerprint is what Gate 0.2 compares. It hashes raw bytes rather than
comparing with a tolerance, because the whole point of the clean-determinism
control is that two clean runs agree *exactly* — a tolerance would hide the
nondeterminism this gate exists to catch.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import torch
import torch.nn as nn


def _raw_bytes(t: torch.Tensor) -> bytes:
    """Raw little-endian bytes of a tensor, via an integer view where needed."""
    t = t.detach().cpu().contiguous().flatten()
    if t.dtype == torch.bfloat16:
        # numpy has no bfloat16; reinterpret the same bits as int16.
        t = t.view(torch.int16)
    return t.numpy().tobytes()


def fingerprint(model: nn.Module) -> str:
    """SHA-256 over every parameter and buffer, in sorted name order."""
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(_raw_bytes(tensor))
    return h.hexdigest()


def tensor_digest(t: torch.Tensor) -> str:
    return hashlib.sha256(_raw_bytes(t)).hexdigest()


@dataclass
class StepRecord:
    step: int
    loss: float
    grad_norm: float
    weight_delta: float
    n_injections: int


@dataclass
class RunRecord:
    """Everything one training run produced.

    ``final_fingerprint`` decides Gate 0.2; ``losses`` and ``weight_delta``
    feed the divergence measurements in Gate 0.3.
    """

    config: dict
    steps: list[StepRecord] = field(default_factory=list)
    final_fingerprint: str = ""
    injection_manifest: list[tuple] = field(default_factory=list)
    diverged: bool = False
    alarms: list = field(default_factory=list)
    checks: int = 0

    def fired(self, detector: str | None = None, site: str | None = None) -> bool:
        return any(
            (detector is None or a.detector == detector)
            and (site is None or a.site == site)
            for a in self.alarms
        )

    @property
    def losses(self) -> list[float]:
        return [s.loss for s in self.steps]

    @property
    def final_loss(self) -> float:
        return self.steps[-1].loss if self.steps else float("nan")

    @property
    def n_injections(self) -> int:
        return len(self.injection_manifest)

    def loss_divergence(self, other: "RunRecord") -> float:
        """Max absolute per-step loss gap against a reference run.

        NaN in either run counts as infinite divergence — a run that blew up has
        certainly diverged, and propagating NaN through a max() would silently
        report 0.0 instead.
        """
        gaps = []
        for a, b in zip(self.losses, other.losses):
            if a != a or b != b:  # NaN check without importing math
                return float("inf")
            gaps.append(abs(a - b))
        return max(gaps) if gaps else 0.0


@torch.no_grad()
def grad_norm(model: nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += float(p.grad.detach().float().pow(2).sum())
    return total ** 0.5


@torch.no_grad()
def snapshot(model: nn.Module) -> list[torch.Tensor]:
    return [p.detach().clone() for p in model.parameters()]


@torch.no_grad()
def delta_norm(model: nn.Module, previous: list[torch.Tensor]) -> float:
    total = 0.0
    for p, prev in zip(model.parameters(), previous):
        total += float((p.detach() - prev).float().pow(2).sum())
    return total ** 0.5
