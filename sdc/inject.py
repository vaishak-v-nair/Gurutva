"""Seeded, rate-controlled, bit-position-parameterised fault injection.

Real silent data corruption is far too rare to develop a detector against, so
everything in this project starts with manufacturing corruption on demand.

Two invariants this module exists to protect:

1.  **Independent RNG stream.** The injector draws from its own
    ``torch.Generator``, never the global one. Sharing a stream would make an
    injected run diverge from its clean twin through perturbed data ordering and
    dropout rather than through injection, silently invalidating every
    comparison downstream.

2.  **Bit position is first-class.** Flipping the sign or a high exponent bit
    produces a catastrophic, trivially detectable error; flipping the mantissa
    LSB of an fp32 value moves it by ~1.2e-7 and is invisible. A harness that
    only flips exponent bits makes any detector look excellent and be worthless.
    Every injection records the bit it flipped, and every result is reported
    against that axis rather than marginalised into one "detection rate".
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable, Sequence

import torch

# Sites are named rather than enumerated so a caller can add one without
# touching this module. These four are the ones Phase 0 commits to.
SITES = ("gemm_out", "allreduce", "optim_state", "activation")

_INT_VIEW = {
    torch.float32: torch.int32,
    torch.bfloat16: torch.int16,
    torch.float16: torch.int16,
}

_WIDTH = {
    torch.float32: 32,
    torch.bfloat16: 16,
    torch.float16: 16,
}

# Field layout per dtype, as (sign_bit, exponent_bits, mantissa_bits). Used only
# for human-readable labelling of a bit index; the flip itself is layout-blind.
_LAYOUT = {
    torch.float32: (31, range(23, 31), range(0, 23)),
    torch.bfloat16: (15, range(7, 15), range(0, 7)),
    torch.float16: (15, range(10, 15), range(0, 10)),
}


def field_of(bit: int, dtype: torch.dtype) -> str:
    """Name the IEEE-754 field a bit index falls in. Labelling only."""
    sign, exponent, mantissa = _LAYOUT[dtype]
    if bit == sign:
        return "sign"
    if bit in exponent:
        return "exponent"
    if bit in mantissa:
        return "mantissa"
    raise ValueError(f"bit {bit} out of range for {dtype}")


def max_bit(dtype: torch.dtype) -> int:
    """Highest valid bit index for a dtype."""
    return _WIDTH[dtype] - 1


def _xor_operand(bit: int, dtype: torch.dtype) -> int:
    """``1 << bit`` expressed in the signed integer view's range.

    Torch integer views are signed, so the top bit of an int16 view cannot be
    written as 32768. Wrap it into the negative half of the range instead.
    """
    width = _WIDTH[dtype]
    if not 0 <= bit < width:
        raise ValueError(f"bit {bit} out of range for {dtype} (width {width})")
    value = 1 << bit
    if value >= 1 << (width - 1):
        value -= 1 << width
    return value


@dataclass(frozen=True)
class InjectionEvent:
    """One flipped bit. The manifest of these is what Gate 0.1 compares."""

    step: int
    site: str
    flat_index: int
    bit: int
    field: str
    before: float
    after: float

    def key(self) -> tuple:
        """Identity tuple — excludes float payloads so it compares exactly."""
        return (self.step, self.site, self.flat_index, self.bit)


class Injector:
    """Corrupts tensors at a controlled rate and bit position.

    Args:
        seed: seeds this injector's private generator.
        bits: bit positions to draw from, uniformly. A single-element sequence
            pins every flip to one position, which is what the difficulty sweep
            uses to isolate the bit-position axis.
        rate: per-element flip probability. Mutually exclusive with ``count``.
        count: exact number of flips per corrupted call. Mutually exclusive with
            ``rate``. Preferred for controlled experiments — one flip per step
            isolates the effect of position from the effect of volume.
        sites: which injection sites are live. Calls from other sites pass
            through untouched.
        every: only act on steps where ``step % every == 0``.
    """

    def __init__(
        self,
        seed: int,
        bits: Sequence[int],
        rate: float | None = None,
        count: int | None = None,
        sites: Iterable[str] = SITES,
        every: int = 1,
    ) -> None:
        if (rate is None) == (count is None):
            raise ValueError("supply exactly one of rate= or count=")
        if rate is not None and not 0.0 <= rate <= 1.0:
            raise ValueError(f"rate must be in [0, 1], got {rate}")
        if count is not None and count < 1:
            raise ValueError(f"count must be >= 1, got {count}")
        if not bits:
            raise ValueError("bits must be non-empty")
        if every < 1:
            raise ValueError(f"every must be >= 1, got {every}")

        self.seed = seed
        self.bits = tuple(bits)
        self.rate = rate
        self.count = count
        self.sites = frozenset(sites)
        self.every = every

        unknown = self.sites - set(SITES)
        if unknown:
            raise ValueError(f"unknown sites: {sorted(unknown)}")

        # Deliberately private and separate from the global torch RNG.
        self._gen = torch.Generator(device="cpu")
        self._gen.manual_seed(seed)

        self.events: list[InjectionEvent] = []
        self._step = 0

    # -- lifecycle -------------------------------------------------------

    def set_step(self, step: int) -> None:
        self._step = step

    def reset(self) -> None:
        """Rewind to the constructed state. Same seed, empty manifest."""
        self._gen.manual_seed(self.seed)
        self.events.clear()
        self._step = 0

    @property
    def active(self) -> bool:
        return self._step % self.every == 0

    # -- the flip --------------------------------------------------------

    def _select(self, numel: int) -> torch.Tensor:
        """Flat indices to corrupt, drawn from the private stream.

        Count mode samples *without replacement*. Sampling with replacement would
        let the same (index, bit) pair be drawn twice in one call, and two XORs of
        the same bit cancel — the manifest would then record two corruptions whose
        net effect on the tensor is nil. ``randperm`` costs O(numel) per call,
        which is acceptable at Phase 0 model sizes and would need revisiting
        before this runs against production-scale tensors.
        """
        if self.count is not None:
            n = min(self.count, numel)
            return torch.randperm(numel, generator=self._gen)[:n]
        draws = torch.rand(numel, generator=self._gen)
        return (draws < self.rate).nonzero(as_tuple=True)[0]

    def _pick_bits(self, n: int) -> torch.Tensor:
        if len(self.bits) == 1:
            return torch.full((n,), self.bits[0], dtype=torch.int64)
        choice = torch.randint(
            low=0, high=len(self.bits), size=(n,), generator=self._gen, dtype=torch.int64
        )
        return torch.tensor(self.bits, dtype=torch.int64)[choice]

    def corrupt_(self, tensor: torch.Tensor, site: str) -> torch.Tensor:
        """Corrupt ``tensor`` in place. Returns it for convenience.

        Used for gradients and optimizer state, where there is no autograd graph
        to preserve. For activations use :meth:`corrupt` instead.
        """
        if site not in self.sites or not self.active:
            return tensor
        if tensor.dtype not in _INT_VIEW:
            raise TypeError(f"unsupported dtype {tensor.dtype}")

        flat = tensor.detach().reshape(-1)
        if not flat.is_contiguous():
            raise ValueError("corrupt_ requires a contiguous tensor")

        idx = self._select(flat.numel())
        if idx.numel() == 0:
            return tensor

        bits = self._pick_bits(idx.numel())
        int_view = flat.view(_INT_VIEW[tensor.dtype])

        for i, bit in zip(idx.tolist(), bits.tolist()):
            before = float(flat[i])
            int_view[i] ^= _xor_operand(bit, tensor.dtype)
            self.events.append(
                InjectionEvent(
                    step=self._step,
                    site=site,
                    flat_index=int(i),
                    bit=int(bit),
                    field=field_of(bit, tensor.dtype),
                    before=before,
                    after=float(flat[i]),
                )
            )
        return tensor

    def corrupt(self, tensor: torch.Tensor, site: str) -> torch.Tensor:
        """Corrupt ``tensor`` while preserving its autograd graph.

        Straight-through: the forward value is corrupted, and the backward pass
        sees an identity. That is exactly what a hardware fault does — downstream
        consumes a wrong number and differentiates it as if it were right.
        """
        if site not in self.sites or not self.active:
            return tensor

        corrupted = tensor.detach().clone()
        self.corrupt_(corrupted, site)
        if not tensor.requires_grad:
            return corrupted
        return tensor + (corrupted - tensor.detach()).detach()

    # -- manifest --------------------------------------------------------

    def manifest(self) -> list[tuple]:
        """Identity tuples of every event, for exact comparison in Gate 0.1."""
        return [e.key() for e in self.events]

    def as_records(self) -> list[dict]:
        return [asdict(e) for e in self.events]

    def __repr__(self) -> str:
        spec = f"rate={self.rate}" if self.rate is not None else f"count={self.count}"
        return (
            f"Injector(seed={self.seed}, bits={self.bits}, {spec}, "
            f"sites={sorted(self.sites)}, every={self.every}, "
            f"events={len(self.events)})"
        )
