"""Gate 0.3 — divergence, in both directions.

The positive half is nearly uninformative on its own: flip the exponent MSB and
training falls over. The negative half is the one that matters. If every
injection we can manufacture were trivially visible, Phase 1 would have nothing
to detect and any detector would score perfectly. This gate proves the opposite —
that corruption exists which is real and which the loss curve does not notice.

CALIBRATION DISCLOSURE. The *structure* of these gates was declared before any
code was written (visible / invisible / separated). The *constants* below were
set after running ``sdc/figures/difficulty.py``, and are recorded here rather
than tuned silently inside the assertions. They are measurements, not
predictions, and one of them contradicted the prediction — see
``test_sign_flip_is_mild_not_catastrophic``.

Measured landscape (fp32, gemm_out, 30 steps — see figures/difficulty.json):

    bit 30  exponent MSB   inf/NaN at every volume tested
    bit 27  exponent       4e-5 at x1, 9e-3 at x4, 2e+2 at x16  (non-monotonic)
    bit 31  sign           8e-5 at x1, 8e-3 at x4, 1e-2 at x16  (mild)
    bit 22  mantissa MSB   ~1e-3
    bit <=12 mantissa      2.4e-7 flat — one ULP of the loss itself
"""

import pytest

from sdc.inject import Injector
from sdc.runner import TrainConfig, train

STEPS = 30

# Loss gap against the clean run, in nats. Calibrated — see disclosure above.
INVISIBLE = 1e-6  # low mantissa flips sit at the loss ULP, ~2.4e-7
MILD_FLOOR = 1e-3  # sign flips at volume land here...
MILD_CEIL = 1.0  # ...and conspicuously do not blow up
SEPARATION = 100.0


def run_with(bit, count=4, site="gemm_out", steps=STEPS, seed=11):
    return train(
        TrainConfig(steps=steps),
        Injector(seed=seed, bits=(bit,), count=count, sites=(site,)),
    )


@pytest.fixture(scope="module")
def clean():
    return train(TrainConfig(steps=STEPS))


# -- the visible end -----------------------------------------------------


def test_exponent_msb_is_catastrophic(clean):
    """Bit 30 scales a value by ~2^128. Nothing survives that."""
    injected = run_with(bit=30)
    assert injected.n_injections > 0
    assert injected.diverged, "exponent-MSB corruption did not blow the run up"
    assert injected.loss_divergence(clean) == float("inf")


def test_sign_flip_is_mild_not_catastrophic(clean):
    """A prediction this harness falsified.

    The plan's difficulty table called a sign flip "trivial" to detect. Measured,
    it is not: 16 sign flips per step move the loss by ~1e-2 and the run trains
    through it. Sign corruption is visible in the *tensor* but nearly invisible in
    the *loss*, which means a loss-watching detector would miss it — and loss
    watching is the cheapest thing in the L0 layer. Phase 1 is harder than the
    plan assumed, and this gate pins that finding so it cannot quietly regress.
    """
    injected = run_with(bit=31, count=16)
    gap = injected.loss_divergence(clean)
    assert not injected.diverged, "sign flip blew up; the finding has changed"
    assert MILD_FLOOR < gap < MILD_CEIL, f"sign-flip gap {gap:.3e} left the mild band"


# -- the invisible end, which is the point -------------------------------


def test_low_mantissa_flip_is_invisible(clean):
    """The negative control, and the reason Phase 1 is not trivial."""
    injected = run_with(bit=0, count=16)
    assert injected.n_injections > 0, "nothing was injected; the control is vacuous"
    gap = injected.loss_divergence(clean)
    assert gap < INVISIBLE, (
        f"mantissa-LSB corruption was visible (gap {gap:.3e}); if this is real, "
        "the detection problem is easier than the plan assumes"
    )


def test_invisible_corruption_still_lands(clean):
    """Invisible in the loss is not the same as absent.

    Without this, the negative control above would pass just as happily if the
    injector had silently done nothing at all.
    """
    injected = run_with(bit=0, count=16)
    assert injected.final_fingerprint != clean.final_fingerprint


def test_bit_position_separates_the_regimes(clean):
    """Same field, same volume — only the position differs."""
    high = run_with(bit=22, count=16).loss_divergence(clean)
    low = run_with(bit=0, count=16).loss_divergence(clean)
    assert high > low * SEPARATION, (
        f"bit position barely mattered (bit 22 {high:.3e} vs bit 0 {low:.3e}); "
        "the difficulty axis this project reports against would be meaningless"
    )


# -- coverage ------------------------------------------------------------


@pytest.mark.parametrize("site", ["gemm_out", "allreduce", "optim_state", "activation"])
def test_every_site_can_corrupt(site, clean):
    injected = run_with(bit=30, site=site)
    assert injected.n_injections > 0, f"site {site} never fired"
    assert injected.final_fingerprint != clean.final_fingerprint
