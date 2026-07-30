"""Inversion gates: mean-match (D6), discrepancy convergence, sanity."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import inversion


@pytest.fixture(scope="module")
def setup():
    a = inversion.assemble()
    wr = inversion.depth_weights(a["G"])
    r, dc = inversion.residual_data(a["G"], a["d_obs"], a["mref"])
    return a, wr, r, dc


def test_mean_match_woodbury_vs_cg(setup):
    """D6 gate: two independent routes to the same quadratic minimum.
    Tolerance stated relative to CG convergence (rtol 1e-12 -> ~1e-8 in the
    solution is comfortable; solver slop, not physics, sets this bound)."""
    a, wr, r, _ = setup
    beta = 1.0
    dm_w = inversion.solve_map(a["G"], r, wr, beta)
    dm_cg = inversion.solve_map_cg(a["G"], r, wr, beta)
    rel = np.linalg.norm(dm_w - dm_cg) / np.linalg.norm(dm_w)
    assert rel < 1e-7, f"mean-match broken: rel {rel:.2e}"


def test_discrepancy_hits_measured_floor(setup):
    a, wr, r, _ = setup
    beta = inversion.tune_beta(a["G"], r, wr)
    dm = inversion.solve_map(a["G"], r, wr, beta)
    rms = float(np.sqrt(np.mean((a["G"] @ dm - r) ** 2)))
    assert rms == pytest.approx(inversion.TARGET_RMS, abs=0.02), (
        f"beta tuning missed the floor target: {rms}")


def test_recovered_contrast_is_geologically_sane(setup):
    """No bounds are imposed (Gaussian math stays exact) — so sanity is a
    CHECK, not a constraint: recovered contrast must stay within a broad
    physical range, else the regularization/DC handling is broken."""
    a, wr, r, _ = setup
    beta = inversion.tune_beta(a["G"], r, wr)
    dm = inversion.solve_map(a["G"], r, wr, beta)
    model = a["mref"] + dm
    assert model.min() > -0.8 and model.max() < 0.6, (
        f"contrast range implausible: [{model.min():.2f}, {model.max():.2f}]")


def test_g_rows_match_prism_forward(setup):
    """Spot-check the assembled G against the independent prism code:
    unit density in one coarse cell -> one column of G."""
    from src import prism_forward
    a, *_ = setup
    tree, active = a["tree"], a["active"]
    cc = tree.cell_centers[active]
    hh = tree.h_gridded[active]
    rng = np.random.default_rng(20260731)
    cols = rng.choice(a["G"].shape[1], size=3, replace=False)
    for j in cols:
        gj = prism_forward.prism_gz(a["stations"][:5],
                                    cc[j:j + 1], hh[j:j + 1], np.array([1.0]))
        rel = np.max(np.abs(gj - a["G"][:5, j]) /
                     np.maximum(np.abs(gj), 1e-12))
        assert rel < 1e-4, f"G column {j} mismatch: rel {rel:.2e}"
