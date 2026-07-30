"""Posterior gates — the plan's calibrated criteria, executable.

Test-speed note: the pytest runs N=6,000 samples (noise floor sqrt(2/N)
= 1.8%); the notebook runs the full N=30,000 (floor 0.8%). The criteria
below are the plan's numbers and pass correct code with margin at both N.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import inversion, posterior


@pytest.fixture(scope="module")
def setup():
    a = inversion.assemble()
    wr = inversion.depth_weights(a["G"])
    r, _ = inversion.residual_data(a["G"], a["d_obs"], a["mref"])
    beta = inversion.tune_beta(a["G"], r, wr)
    return a, wr, beta


def test_posterior_variance_positive_and_bounded_by_prior(setup):
    a, wr, beta = setup
    var = posterior.posterior_diag(a["G"], wr, beta)
    prior = 1.0 / (beta * wr**2)
    assert np.all(var > 0)
    assert np.all(var <= prior * (1 + 1e-10)), "data cannot ADD uncertainty"


def test_mc_gate(setup):
    a, wr, beta = setup
    n = 6_000
    var_mc = posterior.mc_sample_variance(a["G"], wr, beta, n_samples=n)
    var_cf = posterior.posterior_diag(a["G"], wr, beta)
    stats, ok = posterior.mc_gates(var_mc, var_cf, n)
    assert ok, f"MC gate failed: {stats}"


def test_coverage_gate(setup):
    """Pooled 95%-interval coverage over prior-drawn truths. Band is modest
    (cells within a rep are correlated, widening the spread beyond
    binomial) — a sane gate, not a max-over-anything."""
    a, wr, beta = setup
    cov, per_rep = posterior.coverage(a["G"], wr, beta, a["mref"], reps=10)
    assert 0.93 < cov < 0.97, f"coverage {cov:.4f}, per-rep {per_rep}"
