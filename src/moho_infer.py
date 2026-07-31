"""S3 full: a Bayesian Moho for South America — the continent's first error bars.

The target map (Uieda & Barbosa 2017) is published with no uncertainties of
any kind. S3's spike proved the compute is feasible (27 ms/forward via the
layered-sensitivity route). This module defines the generative model and the
gates that decide whether a posterior may be claimed at all.

Declared model (every constant stated, provenance-style):
  depth(x) = MEAN_DEPTH + Phi(x) @ w      Phi = 48 RBFs (6 lon x 8 lat),
                                          length scale 800 km
  w   ~ N(0, SIGMA_W^2)                   SIGMA_W = 12 km
  drho ~ N(350, 50) kg/m^3                crust-mantle contrast, unknown
  theta = [w (48), drho]                  -> 49 parameters
  x     = top-K principal components of the simulated gravity + unit noise

Resolution is declared, not hidden: the working grid is 2.0 deg (~220 km),
coarse against the published 0.2 deg map. This is a continental-scale
uncertainty study, not a re-mapping.

ORDER OF OPERATIONS (the S2.5 lesson, encoded):
  1. prior-predictive check FIRST — is the real observation in-distribution?
  2. only then train;
  3. SBC calibration gate;
  4. run-to-run stability at the real observation (the gate that caught S2);
  5. POSTERIOR-PREDICTIVE ADEQUACY — added 2026-08-04 after all of gates
     1-3 passed for a model that could not reproduce the observed gravity
     (120.7 mGal residual vs 19.3 for the published model).
No posterior is claimed at an out-of-distribution observation, ever — and
none is claimed by a model that cannot reproduce the data, either.

WHY GATE 5 EXISTS (the finding that cost this stage its headline):
  - OOD/licensing can PASS on an over-wide prior: if the prior predictive
    spread is enormous, every observation looks typical. S2.5's failure was
    a prior too NARROW; this was the mirror image.
  - SBC only tests self-consistency INSIDE the model's own world.
  - Run-to-run stability only says two networks agree, not that either is
    right.
  None of the three touch whether the model can reproduce reality.
"""

from pathlib import Path

import numpy as np

from . import moho, moho_fast

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "moho" / "sensitivity_step10.npz"

STEP = 10                 # 2.0 deg working grid
N_LON, N_LAT = 6, 8       # RBF centres
RBF_LEN_DEG = 7.0         # ~800 km
MEAN_DEPTH = 35_000.0     # m, declared reference
SIGMA_W = 12_000.0        # m, relief prior
DRHO_PRIOR = (350.0, 50.0)
N_LAYERS = 24
K_DATA = 64


def build(rebuild: bool = False):
    """Grid, basis, and the cached layered-sensitivity operator."""
    g = moho.load_gravity(step=STEP)
    glat, glon = g["lat"], g["lon"]
    deg = 0.2 * STEP
    edges = moho_fast.layer_edges(5_000.0, 80_000.0, N_LAYERS)

    if CACHE.exists() and not rebuild:
        A = np.load(CACHE)["A"]
    else:
        A = moho_fast.build_sensitivity(glat, glon, deg, deg, edges,
                                        verbose=True)
        np.savez_compressed(CACHE, A=A)

    lons = np.linspace(glon.min(), glon.max(), N_LON)
    lats = np.linspace(glat.min(), glat.max(), N_LAT)
    C = np.array([(lo, la) for lo in lons for la in lats])
    d2 = ((glon[:, None] - C[None, :, 0]) ** 2
          + (glat[:, None] - C[None, :, 1]) ** 2)
    Phi = np.exp(-0.5 * d2 / RBF_LEN_DEG**2)

    return dict(g=g, glat=glat, glon=glon, deg=deg, edges=edges, A=A,
                Phi=Phi, centers=C, n_basis=Phi.shape[1],
                obs=g["sedfree_bouguer"])


def sample_theta(S, n, rng):
    w = SIGMA_W * rng.standard_normal((n, S["n_basis"]))
    drho = DRHO_PRIOR[0] + DRHO_PRIOR[1] * rng.standard_normal((n, 1))
    return np.hstack([w, drho])


def depth_from_theta(S, theta):
    theta = np.atleast_2d(theta)
    d = MEAN_DEPTH + theta[:, :S["n_basis"]] @ S["Phi"].T
    return np.clip(d, 6_000.0, 79_000.0)      # keep inside the layer stack


def forward_theta(S, theta):
    """Gravity (mGal) for each theta row — the matvec route."""
    theta = np.atleast_2d(theta)
    depth = depth_from_theta(S, theta)
    out = np.empty((len(theta), len(S["glat"])))
    for i in range(len(theta)):
        out[i] = moho_fast.forward(S["A"], depth[i], S["edges"],
                                   z_ref=moho.Z_REF, drho=theta[i, -1])
    return out


def fit_projection(S, n=600, seed=7):
    """Data-summary basis: principal components of the prior predictive.

    Declared, and fitted on SIMULATED data only — never on the observation."""
    rng = np.random.default_rng(seed)
    d = forward_theta(S, sample_theta(S, n, rng))
    mu = d.mean(axis=0)
    U, s, _ = np.linalg.svd(d - mu, full_matrices=False)
    V = np.linalg.lstsq(d - mu, U[:, :K_DATA] * s[:K_DATA], rcond=None)[0]
    S["proj_mu"], S["proj_V"] = mu, V
    scale = float(np.std((d - mu) @ V))
    S["proj_scale"] = scale
    return S


def project(S, d):
    return ((d - S["proj_mu"]) @ S["proj_V"]) / S["proj_scale"]


def simulate(S, n, rng, noise=1.0):
    theta = sample_theta(S, n, rng)
    d = forward_theta(S, theta)
    x = project(S, d) + noise * rng.standard_normal((n, K_DATA))
    return theta, x


def x_observed(S, noise=1.0):
    return project(S, S["obs"])


def ood_percentile(S, n_ref=800, seed=99):
    """The licensing gate: where the real observation sits among prior draws."""
    _, xs = simulate(S, n_ref, np.random.default_rng(seed))
    xo = x_observed(S)
    mu, sd = xs.mean(0), xs.std(0)
    d_sim = np.linalg.norm((xs - mu) / sd, axis=1)
    d_obs = float(np.linalg.norm((xo - mu) / sd))
    return float((d_sim < d_obs).mean() * 100), d_obs, float(np.median(d_sim))
