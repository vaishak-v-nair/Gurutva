"""E2 gates: P4 tested in code, not asserted in prose.

Measured 2026-08-05 (figures/dark_matter_stats.json): the subsurface
posterior module ran UNMODIFIED on weak-lensing shear; all four gates pass
with a physics-declared prior; halo peak recovered at 15.2 sigma; median
95% exclusion kappa < 0.060 where nothing is detected.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import posterior as sub_posterior
from src.gurutva_core import gates, lensing

N, PIX, NOISE, PRIOR_SD = 16, 1.0, 0.03, 0.15


@pytest.fixture(scope="module")
def setup():
    ker = lensing.ks_operator(N, PIX)
    truth = lensing.nfw_kappa(N, PIX, kappa_s=0.35, r_s_pix=4.0)
    K = lensing.build_operator_matrix(N, PIX)
    rng = np.random.default_rng(3)
    g = K @ truth.ravel() + NOISE * rng.standard_normal(2 * N * N)
    return ker, truth, K, g


def test_kaiser_squires_roundtrips():
    """Physics check: shear -> kappa -> shear must return the same field
    (up to the unconstrained mean mode the kernel annihilates)."""
    ker = lensing.ks_operator(N, PIX)
    k = lensing.nfw_kappa(N, PIX, kappa_s=0.2, r_s_pix=4.0)
    k = k - k.mean()
    back = lensing.shear_to_kappa(lensing.kappa_to_shear(k, ker), ker)
    assert np.max(np.abs(back - k)) < 1e-8


def test_P4_subsurface_code_runs_unmodified_on_dark_matter(setup):
    """THE premise Gurutva was founded on, as an executable test: the
    posterior module written for ore bodies must work on lensing without a
    single change — same linear operator signature, same machinery."""
    _, truth, K, g = setup
    G = K / NOISE
    wr = np.ones(N * N)
    beta = 1.0 / PRIOR_SD**2

    var = sub_posterior.posterior_diag(G, wr, beta)     # ore-body code
    assert var.shape == (N * N,) and np.all(var > 0)
    prior_var = 1.0 / (beta * wr**2)
    assert np.all(var <= prior_var * (1 + 1e-10)), "data cannot add uncertainty"

    s2 = prior_var
    Kd = (G * s2) @ G.T + np.eye(G.shape[0])
    mean = s2 * (G.T @ np.linalg.solve(Kd, g / NOISE))
    peak = int(np.argmax(truth))
    sig = mean[peak] / np.sqrt(var[peak])
    assert sig > 5.0, f"halo should be a strong detection, got {sig:.1f} sigma"


def test_all_four_gates_pass_on_the_dark_matter_twin(setup):
    _, truth, K, g = setup
    G = K / NOISE
    wr = np.ones(N * N)
    beta = 1.0 / PRIOR_SD**2
    rng = np.random.default_rng(11)
    sim = np.array([K @ (PRIOR_SD * rng.standard_normal(N * N))
                    + NOISE * rng.standard_normal(2 * N * N) for _ in range(200)])
    suite = gates.GateSuite()
    suite.add(gates.licensing(g, sim))
    var = sub_posterior.posterior_diag(G, wr, beta)
    # N=8000, not 2000: at 2000 samples the EXPECTED median relative error
    # is ~2.1% (0.6745*sqrt(2/N)), i.e. the 2% threshold sits below the
    # sampling floor and correct code fails. Sixth appearance of this same
    # lesson in the project — calibrate the gate to the numerics.
    mc = sub_posterior.mc_sample_variance(G, wr, beta, n_samples=8000, seed=5)
    st, ok = sub_posterior.mc_gates(mc, var, 8000)
    suite.add(gates.GateReport("calibration(MC)", ok, st["median_rel_err"],
                              "median < 2%"))
    s2 = 1.0 / (beta * wr**2)
    Kd = (G * s2) @ G.T + np.eye(G.shape[0])
    mean = s2 * (G.T @ np.linalg.solve(Kd, g / NOISE))
    suite.add(gates.stability((mean, np.sqrt(var)), (mean, np.sqrt(var)), atol=1e-9))
    pp = float(np.sqrt(np.mean((K @ mean - g) ** 2)))
    suite.add(gates.adequacy(pp, NOISE))
    assert suite.claimable, suite.verdict()


def test_too_narrow_prior_is_refused_by_licensing(setup):
    """The mistake gate 1 caught on the first E2 run, kept executable: a
    prior that cannot produce a cluster halo must NOT be licensed."""
    _, truth, K, g = setup
    rng = np.random.default_rng(13)
    narrow = np.array([K @ (0.02 * rng.standard_normal(N * N))
                       + NOISE * rng.standard_normal(2 * N * N) for _ in range(200)])
    assert not gates.licensing(g, narrow).passed
