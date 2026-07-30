"""S1 mode machinery: the KL basis where the survey's knowledge lives.

Whitened model u = (m - mref)/sqrt(s2) has prior N(0, I). The data operator
A = G * sqrt(s2) factors as A = U S V^T (economy SVD). Then:

- mode coefficients c = V^T u have prior N(0, I_k), and are INDEPENDENT in
  the posterior: c_i | d ~ N( s_i t_i/(1+s_i^2), 1/(1+s_i^2) ), t = U^T d'.
- per-mode resolution gain = s_i^2/(1+s_i^2); their sum is trace(R) — the
  32.4 of the design note.
- the linear-limit simulator in mode space is exact and free:
  t_i = s_i c_i + eta_i, eta ~ N(0, I).

Every claim above is pinned by tests/test_modes.py against the validated
Phase-1 Woodbury results — the mode path must reproduce them to numerical
precision or S1 does not proceed.
"""

import numpy as np

from . import inversion

K_DEFAULT = 128       # design note: covers > 99% of measured resolution


def build(bridged: bool = True, k: int = K_DEFAULT):
    """Returns everything S1 needs, plus full-rank pieces for cross-gates."""
    a = inversion.assemble(bridged=bridged)
    G, mref, stations = a["G"], a["mref"], a["stations"]
    wr = inversion.depth_weights(G)
    r, coef = inversion.residual_data(G, a["d_obs"], mref,
                                      stations=stations, order=2)
    beta = inversion.tune_beta(G, r, wr)
    s2 = 1.0 / (beta * wr**2)

    A = G * np.sqrt(s2)
    U, S, Vt = np.linalg.svd(A, full_matrices=False)   # 323 modes max
    t_obs = U.T @ r

    return dict(a=a, wr=wr, beta=beta, s2=s2, r=r,
                U=U, S=S, Vt=Vt, t_obs=t_obs, k=k)


def gains(S: np.ndarray) -> np.ndarray:
    return S**2 / (1.0 + S**2)


def analytic_posterior(S: np.ndarray, t: np.ndarray):
    """Exact per-mode posterior (mean, variance) given projected data t."""
    mean = S * t / (1.0 + S**2)
    var = 1.0 / (1.0 + S**2)
    return mean, var


def simulate(S: np.ndarray, n_draws: int, k: int, rng) -> tuple:
    """Linear-limit simulator: (c, t) pairs for the top-k modes."""
    c = rng.standard_normal((n_draws, k))
    t = S[:k] * c + rng.standard_normal((n_draws, k))
    return c, t


def cell_variance_from_modes(s2, S, Vt) -> np.ndarray:
    """Reconstruct diag(Sigma_post) from ALL modes — must equal the
    validated Woodbury diagonal (cross-gate)."""
    g = gains(S)
    return s2 * (1.0 - np.einsum("ij,i->j", Vt**2, g))


def map_from_modes(s2, S, Vt, t_obs) -> np.ndarray:
    """Reconstruct the full-space MAP increment from mode means (all
    modes) — must equal the validated Woodbury solve (cross-gate)."""
    mean, _ = analytic_posterior(S, t_obs)
    return np.sqrt(s2) * (Vt.T @ mean)
