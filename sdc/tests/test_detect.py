"""Phase 1 gates — L0 detectors.

The claim this phase rests on is that a checksum in the *integer domain* escapes
the floor that defeats loss watching. Phase 0 showed a mantissa-LSB flip moves the
loss by less than one ULP of the loss, so no threshold on a float quantity can
recover it. Reinterpreting the same bits as integers turns that flip into a change
of exactly ±2^k, which no accumulator rounds away.

The gates below test that claim at its hardest point — bit 0 — and test the
structural limit that goes with it: at a site with no uncorrupted reference copy,
there is nothing to compare against and this layer cannot help at any price.
"""

import pytest
import torch

from sdc.detect import (
    REFERENCED_SITES,
    UNREFERENCED_SITES,
    Guards,
    NormMonitor,
    stamp,
    verify,
)
from sdc.inject import Injector
from sdc.runner import TrainConfig, train

STEPS = 12
ALL_BITS = [0, 1, 6, 12, 18, 22, 23, 27, 30, 31]


def run(site, bit, count=2, steps=STEPS, seed=5):
    guards = Guards()
    record = train(
        TrainConfig(steps=steps),
        Injector(seed=seed, bits=(bit,), count=count, sites=(site,)),
        guards=guards,
    )
    return record


# -- Gate 1.1: exact detection where a reference exists ------------------


@pytest.mark.parametrize("site", REFERENCED_SITES)
@pytest.mark.parametrize("bit", ALL_BITS)
def test_checksum_catches_every_bit_on_referenced_sites(site, bit):
    """100%, at every bit position including 0. Anything less means the
    checksum is not integer-domain and the design is wrong."""
    record = run(site, bit)
    assert record.n_injections > 0, f"nothing injected at {site}/bit {bit}"
    assert record.fired("checksum", site), (
        f"checksum missed {record.n_injections} flips at {site}, bit {bit}"
    )


def test_checksum_catches_a_single_lsb_flip_in_isolation():
    """The whole thesis, reduced to one flip in one tensor."""
    t = torch.randn(4096)
    before = stamp(t)
    assert verify(t, before)
    t.view(torch.int32)[1234] ^= 1  # mantissa LSB — smallest change representable
    assert not verify(t, before), "integer checksum missed a single LSB flip"


def test_float_sum_would_have_missed_it():
    """Why the checksum is integer-domain, demonstrated rather than asserted."""
    t = torch.randn(4096)
    float_before = float(t.sum())
    t.view(torch.int32)[1234] ^= 1
    assert float(t.sum()) == float_before, (
        "a float sum caught this; the motivating premise needs re-checking"
    )


# -- Gate 1.2: the accumulator blind spot --------------------------------


def test_same_bit_flips_in_two_elements_are_caught():
    """A plain sum has a blind spot the weighted accumulator exists to close.

    Two flips of the same bit index, one 0->1 and one 1->0, contribute +2^k and
    -2^k and cancel exactly in an unweighted sum.
    """
    t = torch.zeros(256, dtype=torch.float32)
    iv = t.view(torch.int32)
    iv[10] = 0
    iv[20] = 1 << 15
    before = stamp(t)

    iv[10] ^= 1 << 15  # 0 -> 1
    iv[20] ^= 1 << 15  # 1 -> 0

    after = stamp(t)
    assert after.plain == before.plain, "the blind spot this gate exists for is gone"
    assert after.weighted != before.weighted, "weighted accumulator missed the pair"
    assert not verify(t, before)


def test_index_zero_is_weighted_nonzero():
    """A weight of 0 at index 0 would make a flip there invisible."""
    t = torch.zeros(64, dtype=torch.float32)
    before = stamp(t)
    t.view(torch.int32)[0] ^= 1
    after = stamp(t)
    assert after.weighted != before.weighted
    assert not verify(t, before)


# -- Gate 1.3: false positives -------------------------------------------

CLEAN_SEEDS = [1, 2, 3, 4, 5, 6, 7, 8]


@pytest.mark.parametrize("seed", CLEAN_SEEDS)
def test_no_false_positives_on_clean_runs(seed):
    """Zero alarms across the clean corpus. K is disclosed, not extrapolated."""
    guards = Guards()
    record = train(TrainConfig(steps=STEPS, seed=seed), guards=guards)
    assert record.checks > 0, "no checks ran; the gate would be vacuous"
    assert record.alarms == [], f"false positive on clean seed {seed}: {record.alarms}"


def test_norm_monitor_flags_non_finite():
    """NaN fails every comparison, so it needs an explicit test.

    Regression guard: the z-score path silently ignored non-finite values, which
    meant the loudest possible signal produced no alarm.
    """
    monitor = NormMonitor(warmup=2)
    for step in range(4):
        monitor.observe(step, grad_norm=1.0 + 0.01 * step)
    assert monitor.alarms == []
    monitor.observe(99, grad_norm=float("nan"))
    assert monitor.alarms, "non-finite value produced no alarm"


# -- Gate 1.5: the structural limit, documented not hidden ---------------


@pytest.mark.parametrize("site", UNREFERENCED_SITES)
def test_checksum_cannot_see_unreferenced_sites(site):
    """Passes by documenting a failure.

    A GEMM output is the only copy of itself. There is no uncorrupted reference
    to compare against, so this layer cannot catch corruption there at any price.
    That is the argument for L1 and L2 existing, and it must not be papered over
    by a detector that appears to work.
    """
    record = run(site, bit=0, count=4)
    assert record.n_injections > 0
    assert not record.fired("checksum"), (
        f"checksum reported a catch at {site}; it has no reference to compare "
        "against, so this is a bug in the harness, not a capability"
    )


def test_catastrophic_arithmetic_is_still_caught_statistically():
    """The consolation at unreferenced sites: loud corruption is loud."""
    record = run("gemm_out", bit=30, count=4, steps=30)
    assert record.diverged
    assert record.fired("norm"), "NaN gradients produced no alarm"


def test_subtle_arithmetic_is_caught_by_nothing():
    """The gap that justifies the whole layered stack.

    Low-mantissa corruption at an unreferenced site: real, landed, and invisible
    to every L0 detector. If this ever starts passing, L0 got stronger and the
    plan's cost model should be revisited.
    """
    record = run("gemm_out", bit=0, count=16, steps=30)
    assert record.n_injections > 0
    assert not record.fired(), (
        "L0 caught subtle arithmetic corruption; the coverage map is wrong"
    )
