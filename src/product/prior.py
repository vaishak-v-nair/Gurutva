"""A prior that is actually a statement about rock.

This module exists because a gate refused to let us ship without it.

The product's headline number is "how much mass is in this box, plus or
minus what". Its honesty rests entirely on the prior being a believable
description of geology, and the two priors in common use are not:

  1. beta from the discrepancy principle. Tuning the regularization weight
     until the misfit hits a target implies, at Utah FORGE, that rock
     density varies by 0.021 g/cc. Measured: licensing fails at the 100th
     percentile — that prior could not have produced the anomaly that was
     actually measured — and against the licensed prior it still under-states
     the error bar on box mass (2.6x on the Utah demo's box).

  2. a flat, independent-per-cell prior at a physical amplitude (0.25 g/cc,
     the published report's own basin contrast). Defensible per cell, and
     still refused by licensing at the 99th percentile. Independent draws
     are white noise: neighbouring cells cancel, so the long-wavelength
     gravity that a real basin produces never appears. Rock is CORRELATED
     over hundreds of metres; a diagonal prior cannot say so.

So the prior here is Gaussian with a sparse PRECISION Q = A^T A assembled
from SimPEG's own smallness + gradient operators (src/smoothness.py), which
makes draws look like geology instead of like static. Two declared numbers:

    PRIOR_SD_GCC   marginal density sd            (amplitude — what rock is)
    CORR_LEN_M     smoothness/smallness ratio      (shape — how rock varies)

Both are read off the site, never tuned until a gate turns green.

What stays exact: the customer's number. For a linear functional w^T m,

    var = w^T S w - (G S w)^T K^-1 (G S w),   S = Q^-1,  K = G S G^T + I

needs only sparse solves against Q, so the mass interval is computed, not
sampled. Per-cell maps are sampled (Matheron) and carry their MC error.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


class SmoothPrior:
    """N(0, Q^-1) for sparse SPD Q, factorised once.

    SciPy has no sparse Cholesky, so the factor for sampling is recovered
    from SuperLU run in symmetric mode with pivoting disabled: there
    U == D L^T, giving Q[ip][:,ip] = L D L^T with ip = argsort(perm_c).
    That identity is CHECKED in __init__, not assumed — an unnoticed
    permutation convention would silently produce samples from the wrong
    distribution, and every gate downstream would still look green.
    """

    def __init__(self, Q, scale=1.0):
        self.Q = (Q * scale).tocsc()
        self._lu = spla.splu(self.Q, diag_pivot_thresh=0.0,
                             permc_spec="MMD_AT_PLUS_A",
                             options=dict(SymmetricMode=True))
        self._ip = np.argsort(self._lu.perm_c)
        self._Lt = self._lu.L.T.tocsr()
        self._d = self._lu.U.diagonal()
        if not np.array_equal(self._lu.perm_r, self._lu.perm_c):
            raise RuntimeError("SuperLU pivoted despite symmetric mode — "
                               "the sampling factor is not valid")
        if self._d.min() <= 0:
            raise RuntimeError("non-positive pivot — Q is not SPD")
        gap = abs(self._lu.U - sp.diags(self._d) @ self._lu.L.T.tocsc()).max()
        if gap > 1e-9 * abs(self.Q).max():
            raise RuntimeError(f"U != D L^T (gap {gap:.3g}); refusing to "
                               "sample from an unverified factor")

    def apply_inv(self, b):
        """S b = Q^-1 b, for one vector or a stack of columns."""
        b = np.asarray(b, dtype=float)
        if b.ndim == 1:
            return self._lu.solve(b)
        return np.column_stack([self._lu.solve(c) for c in b.T])

    def sample(self, rng, n=1):
        """Exact draws from N(0, Q^-1): solve L^T y = D^-1/2 xi, unpermute."""
        z = rng.standard_normal((self.Q.shape[0], n)) / np.sqrt(self._d)[:, None]
        y = spla.spsolve_triangular(self._Lt, z, lower=False)
        x = np.empty_like(y)
        x[self._ip] = y
        return x

    def marginal_sd(self, rng, n=400):
        """Sampled marginal sd per cell (diag(Q^-1) has no cheap exact form)."""
        return self.sample(rng, n).std(axis=1)


def build(tree, active, prior_sd, corr_len_m, rng, n_calib=400,
          alpha_s=1.0):
    """Assemble the declared geological prior and calibrate its amplitude.

    corr_len_m sets alpha_x/y/z = corr_len^2 * alpha_s, the standard
    smallness-to-smoothness ratio: the prior's correlation length. The
    overall scale is then fixed so the SAMPLED marginal sd equals prior_sd
    — the amplitude is declared, and the assembly is made to honour it.
    """
    from ..smoothness import build_precision

    Q0, _, _, _ = build_precision(tree, active, alpha_s=alpha_s,
                                  alpha_x=corr_len_m**2 * alpha_s,
                                  alpha_y=corr_len_m**2 * alpha_s,
                                  alpha_z=corr_len_m**2 * alpha_s)

    p0 = SmoothPrior(Q0)
    sd0 = float(np.median(p0.marginal_sd(rng, n_calib)))
    scale = (sd0 / prior_sd) ** 2          # Q -> Q*scale shrinks sd by sqrt
    return SmoothPrior(Q0, scale), dict(
        raw_median_sd=sd0, scale=scale, corr_len_m=corr_len_m,
        declared_sd=prior_sd, n_calib=n_calib)


# --------------------------------------------------------------- posterior
def _kfactor(G, prior):
    SGt = prior.apply_inv(G.T)                    # n x m
    K = G @ SGt + np.eye(G.shape[0])
    return SGt, K


def posterior_mean(G, prior, d):
    SGt, K = _kfactor(G, prior)
    return SGt @ np.linalg.solve(K, d)


def functional_sd(G, prior, w):
    """Exact posterior sd of w^T m under the correlated prior."""
    Sw = prior.apply_inv(np.asarray(w, float))
    v = float(np.asarray(w) @ Sw)
    GSw = G @ Sw
    _, K = _kfactor(G, prior)
    return np.sqrt(max(v - float(GSw @ np.linalg.solve(K, GSw)), 0.0))


def posterior_sd(G, prior, rng, n=600):
    """Per-cell sd by Matheron's rule. Returns (sd, mc_standard_error).

    Sampled, not closed form: diag(Q^-1) has no cheap exact expression.
    The MC error is returned rather than hidden, because a map whose error
    bars are themselves uncertain must say so.
    """
    SGt, K = _kfactor(G, prior)
    m = prior.sample(rng, n)                              # n_cells x n
    eps = rng.standard_normal((G.shape[0], n))
    fl = m - SGt @ np.linalg.solve(K, G @ m + eps)
    sd = fl.std(axis=1)
    return sd, sd / np.sqrt(2.0 * (n - 1))


def build_regular(shape, spacing, prior_sd, corr_len_m, rng, n_calib=400,
                  active=None):
    """Same declared prior on a regular 3D grid (nx, ny, nz).

    The TreeMesh path above borrows SimPEG's regularization operators; a
    regular grid does not need them, so the smallness + first-difference
    precision is assembled directly:

        Q0 = I + L^2 * sum_axis D_axis^T D_axis

    with D the forward difference along each axis in CELL units, scaled by
    (corr_len / spacing) so the correlation length is a physical length and
    not an accident of the mesh. Amplitude is then calibrated exactly as in
    build(), so both paths mean the same thing by "0.25 g/cc at 500 m".

    `active`: optional boolean mask over the flattened grid. Cells that are
    False are dropped entirely and no difference row is built across them.
    This matters once the mesh follows topography: with a flat-topped block
    the smoothness term happily correlates rock on one side of a valley with
    rock on the other THROUGH THE AIR, which is not a statement any geologist
    would sign. Returns a prior over the ACTIVE cells only, so callers must
    index their own arrays with the same mask.
    """
    nx, ny, nz = shape
    n_full = nx * ny * nz
    if active is None:
        active = np.ones(n_full, dtype=bool)
    active = np.asarray(active, dtype=bool).ravel()
    if active.size != n_full:
        raise ValueError(f"active mask has {active.size} entries, mesh has "
                         f"{n_full}")
    n = int(active.sum())
    if n < 8:
        raise ValueError(f"only {n} active cells left after masking")
    # map full-grid index -> compact index, -1 where inactive
    remap = np.full(n_full, -1, dtype=np.int64)
    remap[active] = np.arange(n)
    idx = np.arange(n_full).reshape(shape)

    rows = []
    for axis, h in zip(range(3), spacing):
        a = np.moveaxis(idx, axis, 0)
        if a.shape[0] < 2:
            continue
        i0, i1 = a[:-1].ravel(), a[1:].ravel()
        keep = active[i0] & active[i1]        # never difference through air
        i0, i1 = remap[i0[keep]], remap[i1[keep]]
        m = len(i0)
        if m == 0:
            continue
        w = corr_len_m / h                    # dimensionless smoothing weight
        rows.append(sp.csr_matrix(
            (np.concatenate([w * np.ones(m), -w * np.ones(m)]),
             (np.concatenate([np.arange(m), np.arange(m)]),
              np.concatenate([i0, i1]))), shape=(m, n)))
    D = sp.vstack(rows).tocsr() if rows else sp.csr_matrix((0, n))
    Q0 = (sp.eye(n, format="csr") + D.T @ D).tocsc()

    p0 = SmoothPrior(Q0)
    sd0 = float(np.median(p0.marginal_sd(rng, n_calib)))
    scale = (sd0 / prior_sd) ** 2
    return SmoothPrior(Q0, scale), dict(raw_median_sd=sd0, scale=scale,
                                        corr_len_m=corr_len_m,
                                        declared_sd=prior_sd, n_calib=n_calib,
                                        n_active=n, n_full=n_full)


def expected_residual(G, prior):
    """RMS the posterior mean SHOULD leave behind, in whitened units.

    Added 2026-08-05 after the CLI's first real run exposed a bug in the
    adequacy gate. That gate was made two-sided so it would catch a model
    that had "absorbed the noise floor" — but it compared the residual to the
    NOISE, and that reference is wrong whenever the model has more freedom
    than the data has points.

    With whitened data (Sigma_d = I) and K = G S G^T + I:

        prediction = G mu = (I - K^-1) d
        residual   = d - G mu = K^-1 d
        E||r||^2   = tr(K^-1 E[d d^T] K^-1) = tr(K^-1)   under d ~ N(0, K)

    so the expected RMS is sqrt(tr(K^-1)/n), which falls below 1 exactly to
    the degree the model is flexible. Comparing against 1 instead flags a
    correctly-behaving posterior as overfitting, and the only way to satisfy
    it is to shrink the prior back to the tuned-regularizer value this whole
    project exists to reject. Measured on a 120-station demo: a prior of
    0.25, 0.10 and 0.05 g/cc all "failed", and only 0.02 g/cc passed.

    Against this reference the gate asks the right question instead: is the
    misfit consistent with what THIS model, with THIS much freedom, predicts
    of itself?
    """
    SGt = prior.apply_inv(G.T)
    K = G @ SGt + np.eye(G.shape[0])
    return float(np.sqrt(np.trace(np.linalg.inv(K)) / G.shape[0]))
