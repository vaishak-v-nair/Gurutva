"""The posterior — the product. Closed-form Gaussian uncertainty, guarded.

Sigma_post = Sigma_m - Sigma_m G^T K^-1 G Sigma_m,  K = G Sigma_m G^T + I
(diagonal prior Sigma_m = (beta W^2)^-1 as declared; Sigma_d = I).

Guards (the reason this repo is trustworthy):
- Monte-Carlo gate: Matheron's-rule samples vs the closed-form diagonal,
  with noise-floor-calibrated criteria (median rel err < 2%, 95th pct < 5%,
  <= 1% of cells beyond 3x their MC standard error — never max-over-cells,
  which correct code fails on MC noise).
- Coverage gate (SBC-style): truths drawn FROM THE PRIOR; the 95% credible
  interval must cover at nominal rate. Fixed realistic truths under-cover
  by construction and are diagnostic only.
"""

import numpy as np
from scipy.linalg import cho_factor, cho_solve


def _factor(G, wr, beta):
    s2 = 1.0 / (beta * wr**2)
    K = (G * s2) @ G.T + np.eye(G.shape[0])
    return s2, cho_factor(K)


def posterior_diag(G, wr, beta):
    """diag(Sigma_post) via the data-space identity — n_data-sized solve."""
    s2, cf = _factor(G, wr, beta)
    B = cho_solve(cf, G)                       # K^-1 G
    return s2 - s2**2 * np.einsum("ij,ij->j", G, B)


def mc_sample_variance(G, wr, beta, n_samples=30_000, seed=20260731,
                       chunk=2_000):
    """Per-cell sample variance of posterior fluctuations (Matheron's rule).

    Zero-mean form: fluct = mp - S G^T K^-1 (G mp + eps), mp ~ N(0, S),
    eps ~ N(0, I). The data vector shifts the mean only; variance validation
    does not need it.
    """
    s2, cf = _factor(G, wr, beta)
    sd = np.sqrt(s2)
    rng = np.random.default_rng(seed)
    n = len(s2)
    acc = np.zeros(n)
    acc2 = np.zeros(n)
    done = 0
    while done < n_samples:
        c = min(chunk, n_samples - done)
        mp = sd[:, None] * rng.standard_normal((n, c))
        eps = rng.standard_normal((G.shape[0], c))
        alpha = cho_solve(cf, G @ mp + eps)
        fl = mp - s2[:, None] * (G.T @ alpha)
        acc += fl.sum(axis=1)
        acc2 += (fl**2).sum(axis=1)
        done += c
    mean = acc / n_samples
    return acc2 / n_samples - mean**2


def mc_gates(var_mc, var_cf, n_samples):
    """The plan's calibrated criteria; returns (stats dict, all_pass)."""
    rel = np.abs(var_mc - var_cf) / var_cf
    se = np.sqrt(2.0 / n_samples)
    frac3 = float(np.mean(rel > 3 * se))
    stats = {
        "n_samples": n_samples,
        "median_rel_err": float(np.median(rel)),
        "p95_rel_err": float(np.percentile(rel, 95)),
        "frac_beyond_3se": frac3,
        "noise_floor_se": se,
    }
    ok = (stats["median_rel_err"] < 0.02 and stats["p95_rel_err"] < 0.05
          and frac3 <= 0.01)
    return stats, ok


def coverage(G, wr, beta, mref, reps=20, seed=20260801):
    """SBC-style: truths from the prior; fraction inside the 95% interval."""
    s2, cf = _factor(G, wr, beta)
    sd = np.sqrt(s2)
    sig_post = np.sqrt(posterior_diag(G, wr, beta))
    rng = np.random.default_rng(seed)
    n = len(s2)
    hits, per_rep = 0, []
    for _ in range(reps):
        m_true = mref + sd * rng.standard_normal(n)
        d = G @ m_true + rng.standard_normal(G.shape[0])
        mu = mref + s2 * (G.T @ cho_solve(cf, d - G @ mref))
        inside = np.abs(m_true - mu) < 1.96 * sig_post
        per_rep.append(float(inside.mean()))
        hits += int(inside.sum())
    return hits / (reps * n), per_rep
