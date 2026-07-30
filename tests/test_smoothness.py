"""Gates for the smoothness-precision opening block."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import smoothness


@pytest.fixture(scope="module")
def qreg():
    Q, reg, tree, active = smoothness.build_precision()
    return Q, reg, int(active.sum())


def test_energy_identity_pins_the_convention(qreg):
    """phi(m) == m^T Q m for random m — pins SimPEG's factor-of-2 empirically."""
    Q, reg, n = qreg
    rng = np.random.default_rng(20260801)
    for _ in range(3):
        m = rng.standard_normal(n)
        phi = float(reg(m))
        quad = float(m @ (Q @ m))
        assert quad == pytest.approx(phi, rel=1e-10)


def test_precision_is_symmetric(qreg):
    Q, _, _ = qreg
    d = abs(Q - Q.T)
    assert d.max() < 1e-10 * abs(Q).max()


def test_smoothness_annihilates_constants(qreg):
    """With alpha_s = 0, a constant model has zero smoothness energy —
    the null space a pure-gradient prior must have."""
    Q0, reg0, *_ = (None, None)
    Qs, regs, tree, active = smoothness.build_precision(alpha_s=0.0)
    n = int(active.sum())
    c = np.full(n, 3.7)
    assert float(c @ (Qs @ c)) == pytest.approx(0.0, abs=1e-8 * n)


def test_positive_definite_with_smallness(qreg):
    Q, _, n = qreg
    rng = np.random.default_rng(7)
    for _ in range(3):
        x = rng.standard_normal(n)
        assert float(x @ (Q @ x)) > 0
