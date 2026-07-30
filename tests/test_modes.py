"""S1 mode-machinery gates: the mode path must reproduce validated Phase-1
results to numerical precision, and the design note's measured numbers."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import inversion, modes, posterior


@pytest.fixture(scope="module")
def M():
    return modes.build(bridged=True)


def test_mode_posterior_variance_reproduces_woodbury(M):
    var_modes = modes.cell_variance_from_modes(M["s2"], M["S"], M["Vt"])
    var_wood = posterior.posterior_diag(M["a"]["G"], M["wr"], M["beta"])
    rel = np.max(np.abs(var_modes - var_wood) / var_wood)
    assert rel < 1e-8, f"mode path diverges from Woodbury: rel {rel:.2e}"


def test_mode_map_reproduces_woodbury_solve(M):
    dm_modes = modes.map_from_modes(M["s2"], M["S"], M["Vt"], M["t_obs"])
    dm_wood = inversion.solve_map(M["a"]["G"], M["r"], M["wr"], M["beta"])
    rel = (np.linalg.norm(dm_modes - dm_wood)
           / np.linalg.norm(dm_wood))
    assert rel < 1e-8, f"mode MAP diverges: rel {rel:.2e}"


def test_trace_R_matches_design_note(M):
    """Pins the design note's measured 32.4 effective parameters."""
    assert float(modes.gains(M["S"]).sum()) == pytest.approx(32.4, abs=0.2)


def test_k128_covers_99_percent(M):
    g = modes.gains(M["S"])
    cum = np.cumsum(g) / g.sum()
    assert cum[modes.K_DEFAULT - 1] >= 0.99


def test_simulator_is_calibrated(M):
    """The simulator's own statistics: with c ~ N(0,1), t_i has variance
    s_i^2 + 1 — checked at MC tolerance (never max-over-modes)."""
    rng = np.random.default_rng(20260802)
    k = 64
    c, t = modes.simulate(M["S"], 20_000, k, rng)
    expected = M["S"][:k] ** 2 + 1.0
    rel = np.abs(t.var(axis=0) - expected) / expected
    assert np.median(rel) < 0.03
    assert np.percentile(rel, 95) < 0.08
