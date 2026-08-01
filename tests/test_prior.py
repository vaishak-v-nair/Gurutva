"""Gates on the correlated prior — the piece licensing demanded.

A wrong sampler here is the most dangerous bug this repo could have: every
downstream gate would still report green while the error bars came from the
wrong distribution. So the sampler is checked against a dense inverse, and
the tolerances are calibrated to the SAMPLE COUNT rather than picked to
pass (seventh appearance of that lesson).
"""

import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.product import prior as PR


def _spd(n=120, seed=3):
    B = sp.random(n, n, density=0.05, random_state=seed)
    return (B @ B.T + sp.eye(n) * 1.5).tocsc()


def test_factor_identity_is_checked_not_assumed():
    """SmoothPrior must verify U == D L^T; a silent permutation bug would
    produce plausible-looking samples from the wrong distribution."""
    p = PR.SmoothPrior(_spd())
    gap = abs(p._lu.U - sp.diags(p._lu.U.diagonal()) @ p._lu.L.T.tocsc()).max()
    assert gap < 1e-9 * abs(p.Q).max()
    assert np.array_equal(p._lu.perm_r, p._lu.perm_c)


def test_sampler_reproduces_the_dense_inverse():
    Q = _spd()
    p = PR.SmoothPrior(Q)
    S = np.linalg.inv(Q.toarray())
    n = 60_000
    x = p.sample(np.random.default_rng(0), n)
    # relative error of a sampled variance is ~sqrt(2/n); the gate must sit
    # above that floor or correct code fails.
    floor = np.sqrt(2.0 / n)
    rel = np.abs(x.var(axis=1) - np.diag(S)) / np.diag(S)
    assert np.median(rel) < 4 * floor, f"{np.median(rel):.4f} vs {4*floor:.4f}"


def test_apply_inv_is_the_inverse():
    Q = _spd()
    p = PR.SmoothPrior(Q)
    rng = np.random.default_rng(1)
    b = rng.standard_normal(Q.shape[0])
    assert np.allclose(Q @ p.apply_inv(b), b, atol=1e-9)
    B = rng.standard_normal((Q.shape[0], 4))
    assert np.allclose(Q @ p.apply_inv(B), B, atol=1e-9)


def test_functional_sd_matches_dense_posterior():
    """The customer's number must be exact, not sampled."""
    Q = _spd(n=90, seed=5)
    p = PR.SmoothPrior(Q)
    rng = np.random.default_rng(2)
    G = rng.standard_normal((14, 90))
    S = np.linalg.inv(Q.toarray())
    K = G @ S @ G.T + np.eye(14)
    Sigma = S - S @ G.T @ np.linalg.solve(K, G @ S)
    for _ in range(4):
        w = rng.standard_normal(90)
        assert np.isclose(PR.functional_sd(G, p, w),
                          np.sqrt(w @ Sigma @ w), rtol=1e-9)


def test_posterior_mean_matches_dense():
    Q = _spd(n=90, seed=6)
    p = PR.SmoothPrior(Q)
    rng = np.random.default_rng(4)
    G = rng.standard_normal((14, 90))
    d = rng.standard_normal(14)
    S = np.linalg.inv(Q.toarray())
    ref = S @ G.T @ np.linalg.solve(G @ S @ G.T + np.eye(14), d)
    assert np.allclose(PR.posterior_mean(G, p, d), ref, atol=1e-9)


def test_scaling_moves_marginal_sd_the_declared_way():
    """build() calibrates amplitude by scaling Q; sd must go as 1/sqrt(scale)."""
    Q = _spd()
    rng = np.random.default_rng(9)
    a = np.median(PR.SmoothPrior(Q, 1.0).marginal_sd(rng, 4000))
    b = np.median(PR.SmoothPrior(Q, 4.0).marginal_sd(rng, 4000))
    assert np.isclose(a / b, 2.0, rtol=0.06)
