"""Gates on the product layer. The customer-facing number must be exact.

The one thing this company may never do is report a tighter interval than
the data supports, so the headline test is functional_sd against a
brute-force dense covariance.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import posterior as P
from src.product import verdict as V


def _toy(seed=0, n_cells=40, n_data=12):
    rng = np.random.default_rng(seed)
    G = rng.standard_normal((n_data, n_cells))
    wr = 1.0 + 0.5 * rng.random(n_cells)
    return G, wr, 3.0, rng


def test_functional_sd_matches_dense_covariance():
    """w^T Sigma w via the data-space identity == the explicit n x n form."""
    G, wr, beta, rng = _toy()
    s2 = 1.0 / (beta * wr**2)
    S = np.diag(s2)
    K = G @ S @ G.T + np.eye(G.shape[0])
    Sigma = S - S @ G.T @ np.linalg.solve(K, G @ S)     # dense, O(n^2)
    for _ in range(5):
        w = rng.standard_normal(G.shape[1])
        assert np.isclose(V.functional_sd(G, wr, beta, w),
                          np.sqrt(w @ Sigma @ w), rtol=1e-10)


def test_diagonal_only_would_overstate_confidence():
    """Why functional_sd exists: summing per-cell variances is NOT the
    variance of the sum. This test pins that the shortcut is wrong, so
    nobody 'optimises' it back in later."""
    G, wr, beta, _ = _toy()
    w = np.ones(G.shape[1])
    exact = V.functional_sd(G, wr, beta, w)
    naive = float(np.sqrt(np.sum(w**2 * P.posterior_diag(G, wr, beta))))
    assert not np.isclose(exact, naive, rtol=1e-3)


def test_informed_fraction_is_bounded_and_data_only_informs():
    G, wr, beta, _ = _toy()
    f = V.informed_fraction(G, wr, beta)
    assert np.all(f >= -1e-12) and np.all(f <= 1.0 + 1e-12)
    assert (f > 0.05).any(), "12 data should inform something"


def test_mass_in_region_recovers_a_known_mass():
    """End-to-end: plant a mass, invert, and the reported interval must
    contain the truth."""
    G, wr, beta, rng = _toy(seed=4, n_cells=30, n_data=25)
    truth = np.zeros(30)
    truth[10:16] = 0.4
    d = G @ truth + 0.05 * rng.standard_normal(25)
    mask = np.zeros(30, bool); mask[8:18] = True
    vol = np.ones(30)
    m, sd = V.mass_in_region(G, wr, beta, d, mask, vol)
    assert sd > 0
    assert abs(m - truth[mask].sum()) < 3 * sd


def test_verdict_refuses_when_a_gate_fails():
    """The product's whole differentiator: it must say NO out loud."""
    from src.gurutva_core import gates
    suite = gates.GateSuite()
    suite.add(gates.GateReport("licensing", True, 10.0, "<95 pct"))
    suite.add(gates.GateReport("calibration", True, 0.90, "~0.90"))
    suite.add(gates.GateReport("stability", True, 0.001, "<0.01"))
    suite.add(gates.GateReport("adequacy", False, 120.7, "<3x19.3"))
    v = V.assess(suite, {"tonnes": "1.2e6"}, subject="the ore body")
    assert not v.claimable
    assert "NOT CLAIMED" in v.headline and "DIAGNOSTIC ONLY" in v.headline
    assert "adequacy" in v.headline


def test_recovery_gate_catches_what_the_other_four_cannot():
    """The dark-matter failure, pinned: four core gates green, and an
    interval that misses a known truth by 11 sigma."""
    from src.gurutva_core import gates
    r = gates.recovery(11.33, 0.74, 19.86)
    assert not r.passed
    suite = gates.GateSuite()
    for n in ("licensing", "calibration(MC)", "stability", "adequacy"):
        suite.reports.append(gates.GateReport(n, True, 1.0, "x"))
    assert suite.claimable, "four core gates alone must still be claimable"
    suite.reports.append(r)
    assert not suite.claimable, "a failing recovery gate must block the claim"
    assert "recovery" in suite.verdict()


def test_untested_recovery_is_a_distinct_verdict_state():
    """Three genuinely different situations get three different words, so the
    headline can never say CLAIMABLE and 'nothing was checked' at once."""
    from src.gurutva_core import gates
    suite = gates.GateSuite()
    for n in ("licensing", "calibration", "stability", "adequacy"):
        suite.reports.append(gates.GateReport(n, True, 1.0, "x"))
    assert suite.claimable
    assert suite.status == "PROVISIONAL"
    assert "PROVISIONAL" in suite.verdict()
    suite.reports.append(gates.recovery(0.0, 1.0, 0.1))
    assert suite.status == "CLAIMABLE"
    suite.reports.append(gates.GateReport("adequacy2", False, 9.0, "x"))
    assert suite.status == "NOT CLAIMED"


def test_core_gate_missing_is_incomplete_not_claimable():
    from src.gurutva_core import gates
    suite = gates.GateSuite()
    for n in ("licensing", "calibration", "stability"):
        suite.reports.append(gates.GateReport(n, True, 1.0, "x"))
    assert not suite.claimable
    assert "INCOMPLETE" in suite.verdict() and "adequacy" in suite.verdict()


def test_recovery_reports_worst_miss_not_just_average():
    """A field can cover 99% of cells and be badly wrong where it matters,
    so the gate fails on the worst miss even when mean coverage is fine."""
    import numpy as np
    from src.gurutva_core import gates
    est = np.zeros(200); sd = np.ones(200); truth = np.zeros(200)
    truth[7] = 9.0                      # one cell catastrophically wrong
    r = gates.recovery(est, sd, truth)
    assert not r.passed and "9.0 sigma" in r.note
