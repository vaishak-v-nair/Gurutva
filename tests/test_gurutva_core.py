"""Gates for the domain-agnostic core and the astro adapter."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.gurutva_core import astro, gates


# ------------------------------------------------------------- gate suite
def test_licensing_flags_an_outsider_and_passes_a_native():
    rng = np.random.default_rng(0)
    sim = rng.standard_normal((500, 8))
    assert gates.licensing(rng.standard_normal(8), sim).passed
    assert not gates.licensing(np.full(8, 9.0), sim).passed


def test_adequacy_is_the_only_gate_that_sees_reality():
    """The S3 lesson, executable: a model can pass licensing while being
    unable to reproduce the data — which is why adequacy exists."""
    rng = np.random.default_rng(1)
    wide = rng.standard_normal((500, 8)) * 50.0     # over-wide prior
    x_obs = rng.standard_normal(8) * 3.0
    assert gates.licensing(x_obs, wide).passed, "over-wide prior licenses anything"
    assert not gates.adequacy(120.7, 19.3).passed, "…but adequacy still fails"
    assert gates.adequacy(21.0, 19.3).passed


def test_suite_refuses_to_claim_without_all_four():
    s = gates.GateSuite()
    s.add(gates.licensing(np.zeros(4), np.random.default_rng(2).standard_normal((100, 4))))
    assert not s.claimable and "INCOMPLETE" in s.verdict()


# ---------------------------------------------------------- astro adapter
def test_cr3bp_l1_is_an_equilibrium():
    """Physics check: at L1 with zero velocity the acceleration vanishes."""
    s = [astro.L1_X, 0.0, 0.0, 0.0, 0.0, 0.0]
    a = np.array(astro.accel(0.0, s))[3:]
    assert np.linalg.norm(a) < 1e-6, f"L1 not an equilibrium: {a}"


def test_jacobi_constant_is_conserved():
    """The CR3BP's conserved quantity — the integrator's honesty check."""
    s0 = [0.72, 0.0, 0.02, 0.05, 0.42, 0.0]
    t, y = astro.propagate(s0, (0.0, 2.0), n_out=50)

    def jacobi(st):
        x, yy, z, vx, vy, vz = st
        r1 = np.sqrt((x + astro.MU) ** 2 + yy**2 + z**2)
        r2 = np.sqrt((x - 1 + astro.MU) ** 2 + yy**2 + z**2)
        om = 0.5 * (x**2 + yy**2) + (1 - astro.MU) / r1 + astro.MU / r2
        return 2 * om - (vx**2 + vy**2 + vz**2)

    j = np.array([jacobi(y[:, k]) for k in range(y.shape[1])])
    drift = float(np.max(np.abs(j - j[0])) / abs(j[0]))
    assert drift < 1e-9, f"Jacobi constant drifted {drift:.2e}"


def test_linearisation_is_adequate_in_benign_cislunar_regimes():
    """The measured finding, pinned: Gurutva does NOT claim standard
    astrodynamics tooling is broken. If this ever fails, the claim map
    changed and the README must be updated."""
    s0 = [0.72, 0.0, 0.02, 0.05, 0.42, 0.0]
    span = (0.0, 12 * 86_400.0 / astro.TU_S)
    t, ens = astro.ensemble(s0, span, n=200, n_out=60)
    _, _, vol_mc = astro.covariance_tube(ens)
    P0 = np.diag([(1.0 / astro.LU_KM) ** 2] * 3
                 + [((10 * 1e-6) / (astro.LU_KM / astro.TU_S)) ** 2] * 3)
    _, _, _, vol_lin = astro.linear_covariance(s0, span, P0, n_out=60)
    ratio = vol_mc[-1] / vol_lin[-1]
    assert 0.5 < ratio < 1.6, f"benign-regime ratio moved to {ratio:.2f}"
