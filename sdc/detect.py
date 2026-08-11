"""L0 detectors — the always-on layer.

Phase 0 demoted the signal this layer was originally built around. The loss
cannot resolve low-mantissa corruption at all: the gap sits at one ULP of the
loss value, which is a representation floor, not a statistical one. No threshold
recovers a signal below the resolution of the quantity being thresholded. So loss
and norm watching become weak corroborators, and the load moves onto checksums.

The central design point is that a checksum must live in the **integer domain**.
A float sum fails for exactly the reason the loss fails — floating-point addition
is lossy, and a mantissa-LSB flip moves the sum by less than the sum's own ULP.
Reinterpreting the same bits as integers makes a single flip a change of exactly
±2^k, which no accumulator can round away. Detection becomes exact rather than
statistical at essentially the same cost.

What this layer structurally *cannot* do is catch corruption at a site where no
uncorrupted reference exists. A GEMM output is the only copy of itself; there is
nothing to compare it against. That partition — referenced vs unreferenced sites —
is the coverage map this phase exists to establish, and it is what justifies the
more expensive layers above.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .inject import _INT_VIEW

# Mersenne prime. Values stay under 2^31 so an int64 sum over realistic tensor
# sizes cannot overflow: 2^31 * 2^20 = 2^51, well inside int64.
_MOD = (1 << 31) - 1

# Weight modulus for the position-weighted accumulator. Small enough that
# value * weight stays under 2^47 before the reduction back into _MOD.
_WMOD = 65521

# Sites where an uncorrupted reference copy exists, so an exact checksum applies.
REFERENCED_SITES = ("allreduce", "optim_state")
UNREFERENCED_SITES = ("gemm_out", "activation")


@dataclass(frozen=True)
class Alarm:
    step: int
    detector: str
    site: str
    detail: str


@dataclass(frozen=True)
class Stamp:
    """Two accumulators over the integer view of a tensor.

    ``plain`` alone detects any *single* bit flip, but has a blind spot: two
    flips of the same bit index in different elements, one 0→1 and one 1→0,
    cancel exactly. ``weighted`` closes it — the contributions are 2^k·w_i and
    −2^k·w_j, which cannot cancel for i ≠ j because the weights differ.

    Weights are ``(i mod _WMOD) + 1``, never 0: a zero weight at index 0 would
    make a flip there invisible to the weighted accumulator.
    """

    plain: int
    weighted: int
    numel: int

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Stamp):
            return NotImplemented
        return (
            self.plain == other.plain
            and self.weighted == other.weighted
            and self.numel == other.numel
        )


_WEIGHT_CACHE: dict[tuple[int, str], torch.Tensor] = {}


def _weights(n: int, device: torch.device) -> torch.Tensor:
    """Position weights, cached by (size, device).

    Rebuilding these per call would dominate the checksum's own cost and make the
    overhead gate measure the allocator rather than the checksum.
    """
    key = (n, str(device))
    cached = _WEIGHT_CACHE.get(key)
    if cached is None:
        cached = torch.arange(n, device=device, dtype=torch.int64) % _WMOD + 1
        _WEIGHT_CACHE[key] = cached
    return cached


def stamp(tensor: torch.Tensor) -> Stamp:
    """Fold a tensor into a Stamp. O(n), vectorised, no hashing."""
    flat = tensor.detach().contiguous().reshape(-1)
    if flat.dtype not in _INT_VIEW:
        raise TypeError(f"unsupported dtype {flat.dtype}")
    values = flat.view(_INT_VIEW[flat.dtype]).to(torch.int64) % _MOD
    weighted = (values * _weights(values.numel(), values.device)) % _MOD
    return Stamp(
        plain=int(values.sum() % _MOD),
        weighted=int(weighted.sum() % _MOD),
        numel=flat.numel(),
    )


def verify(tensor: torch.Tensor, expected: Stamp) -> bool:
    """True if ``tensor`` still folds to ``expected``."""
    return stamp(tensor) == expected


class ChecksumGuard:
    """Stamp at the sender, verify at the receiver.

    Mirrors how this deploys for real: the checksum is taken where the data is
    known good and checked where it arrives. Corruption in between is caught with
    certainty, at any bit position, because the comparison is bit-exact rather
    than tolerance-based.
    """

    name = "checksum"

    def __init__(self) -> None:
        self._stamps: dict[str, Stamp] = {}
        self.alarms: list[Alarm] = []
        self.checks = 0

    def stamp_(self, key: str, tensor: torch.Tensor) -> None:
        self._stamps[key] = stamp(tensor)

    def verify_(self, key: str, tensor: torch.Tensor, step: int, site: str) -> bool:
        expected = self._stamps.get(key)
        if expected is None:
            return True
        self.checks += 1
        if verify(tensor, expected):
            return True
        self.alarms.append(
            Alarm(step=step, detector=self.name, site=site, detail=f"stamp mismatch on {key}")
        )
        return False

    def reset(self) -> None:
        self._stamps.clear()
        self.alarms.clear()
        self.checks = 0


class NormMonitor:
    """Rolling z-score on gradient norm and weight delta.

    The statistical detector, and the weak one. Phase 0 predicts it will fail in
    the low-mantissa regime — the whole point of measuring it is to establish how
    far that failure extends, not to hope it does not happen.

    ``z`` is the only tunable here, and it is what the false-positive gate
    actually constrains.
    """

    name = "norm"

    def __init__(self, z: float = 6.0, warmup: int = 8, window: int = 24) -> None:
        self.z = z
        self.warmup = warmup
        self.window = window
        self.alarms: list[Alarm] = []
        self.checks = 0
        self._history: dict[str, list[float]] = {"grad_norm": [], "weight_delta": []}

    def observe(self, step: int, **metrics: float) -> None:
        for key, value in metrics.items():
            history = self._history.setdefault(key, [])

            # Non-finite values must be tested for explicitly. Every comparison
            # against NaN is False, so a z-score test silently ignores the single
            # loudest signal available — found by the coverage sweep, not review.
            if value != value or value in (float("inf"), float("-inf")):
                self.checks += 1
                self.alarms.append(
                    Alarm(
                        step=step,
                        detector=self.name,
                        site="?",
                        detail=f"{key} is non-finite ({value})",
                    )
                )
                continue

            if len(history) >= self.warmup:
                self.checks += 1
                recent = history[-self.window :]
                mean = sum(recent) / len(recent)
                var = sum((v - mean) ** 2 for v in recent) / max(len(recent) - 1, 1)
                sd = var ** 0.5
                # A flat history gives sd == 0; only a literal change counts then.
                if (sd > 0 and abs(value - mean) > self.z * sd) or (
                    sd == 0 and value != mean
                ):
                    self.alarms.append(
                        Alarm(
                            step=step,
                            detector=self.name,
                            site="?",  # statistical: cannot attribute to a site
                            detail=f"{key}={value:.6g} vs mean {mean:.6g} sd {sd:.3g}",
                        )
                    )
            history.append(value)

    def reset(self) -> None:
        self.alarms.clear()
        self.checks = 0
        for history in self._history.values():
            history.clear()


@dataclass
class Guards:
    """The L0 bundle attached to a run.

    ``optim_stride`` trades coverage for cost. Optimizer state is two moments per
    parameter, and stamping plus verifying both accounts for four of the six
    passes this layer makes over parameter-sized memory each step — the dominant
    term, since the checksum is memory-bound. Checking every Nth step cuts that
    to 4/N.

    The cost is real and must not be glossed: corruption of optimizer state
    between checked steps is missed outright. Gradients stay checked every step,
    because a gradient exists for one step only and a missed check is a permanently
    missed corruption.
    """

    checksum: ChecksumGuard = field(default_factory=ChecksumGuard)
    norm: NormMonitor = field(default_factory=NormMonitor)
    optim_stride: int = 1

    def checks_optim(self, step: int) -> bool:
        return step % self.optim_stride == 0

    @property
    def alarms(self) -> list[Alarm]:
        return sorted(
            self.checksum.alarms + self.norm.alarms, key=lambda a: (a.step, a.detector)
        )

    def fired(self, detector: str | None = None) -> bool:
        if detector is None:
            return bool(self.alarms)
        return any(a.detector == detector for a in self.alarms)

    def reset(self) -> None:
        self.checksum.reset()
        self.norm.reset()
