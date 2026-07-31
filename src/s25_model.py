"""S2.5: the geology the data actually demands — measured, then declared.

Why this module exists: S2's two-unit story left the real observation
OUTSIDE the simulator's world (OOD percentile 100), and neural posteriors
evaluated out-of-distribution proved unstable across identical trainings
(0.49 vs 1.07). That instability is why the S2 interface claim was
retracted. S2.5 fixes the cause, not the symptom.

How the model class was chosen (diagnostic, 2026-08-03, recorded because
it is the whole argument):

  model class                        explains   required |w| median
  interface 12 RBF @2km + rho        65.4%      4443 m   (unphysical)
  interface 24 RBF @1.2km + rho      83.1%       982 m
  interface 40 RBF @0.9km + rho      88.4%       488 m   (physical for
                                                 Basin-and-Range relief)
  + within-basin compaction gradient  +0.0%      (adds nothing — dropped)
  8 broad heterogeneity RBFs @4km    83.8% but requires 1.14 g/cc — larger
                                     than the whole basin/basement contrast,
                                     so REJECTED as unphysical.

Conclusion the data forced: the misfit is not exotic rock, it is BASEMENT
RELIEF the smooth delivered surface does not carry. So S2.5 gives the
interface shorter wavelengths and a wider (declared) amplitude, and keeps
only a small heterogeneity term for the remainder.

    theta = [ w (40 interface RBF coeffs @0.9 km, sigma 300 m),
              rho_basin, rho_basement,
              h (8 heterogeneity RBF coeffs @4 km, small) ]  -> 50 params
"""

import numpy as np

from . import forge_data, s2_model

N_INT, INT_NX, INT_NY, INT_LEN = 40, 8, 5, 900.0
SIGMA_Z = 300.0          # m — declared; Basin-and-Range basement relief
N_HET, HET_LEN = 8, 4000.0
SIGMA_HET = 0.02         # g/cc — deliberately SMALL: heterogeneity is the
                         # remainder term, never the explanation (see above)
N_THETA = N_INT + 2 + N_HET


def _grid(cc, nx, ny):
    gx = np.linspace(cc[:, 0].min(), cc[:, 0].max(), nx)
    gy = np.linspace(cc[:, 1].min(), cc[:, 1].max(), ny)
    return np.array([(x, y) for x in gx for y in gy])


def _rbf(cc, centers, ell):
    d2 = ((cc[:, None, 0] - centers[None, :, 0]) ** 2
          + (cc[:, None, 1] - centers[None, :, 1]) ** 2)
    return np.exp(-0.5 * d2 / ell**2)


def build(M=None):
    S = s2_model.build(M)
    cc = S["cc"]
    S["int_centers"] = _grid(cc, INT_NX, INT_NY)
    S["Phi"] = _rbf(cc, S["int_centers"], INT_LEN)      # replaces S2's 12
    S["het_centers"] = _grid(cc, 4, 2)
    S["Psi"] = _rbf(cc, S["het_centers"], HET_LEN)
    return S


def sample_theta(S, n, rng):
    w = SIGMA_Z * rng.standard_normal((n, N_INT))
    rb = s2_model.RHO_BASIN[0] + s2_model.RHO_BASIN[1] * rng.standard_normal((n, 1))
    rg = s2_model.RHO_BASEMENT[0] + s2_model.RHO_BASEMENT[1] * rng.standard_normal((n, 1))
    h = SIGMA_HET * rng.standard_normal((n, N_HET))
    return np.hstack([w, rb, rg, h])


def model_from_theta(S, theta):
    theta = np.atleast_2d(theta)
    z = S["z_base"][None, :] + theta[:, :N_INT] @ S["Phi"].T
    basin = S["cc"][None, :, 2] > z
    m = np.where(basin, theta[:, N_INT:N_INT + 1], theta[:, N_INT + 1:N_INT + 2])
    return m + theta[:, N_INT + 2:] @ S["Psi"].T


def simulate(S, n, rng, chunk=1_500):
    theta = sample_theta(S, n, rng)
    Uk = S["U"][:, :s2_model.K_DATA]
    xs = np.empty((n, s2_model.K_DATA))
    for a in range(0, n, chunk):
        m = model_from_theta(S, theta[a:a + chunk])
        d = (m - S["mref"][None, :]) @ S["G"].T
        xs[a:a + chunk] = d @ Uk + rng.standard_normal((len(m), s2_model.K_DATA))
    return theta, xs


def ood_percentile(S, x_obs, n_ref=1_500, seed=99):
    """Where the real observation sits among simulated draws (50 = typical,
    100 = outside the simulator's world). The precondition for trusting any
    posterior at x_obs."""
    _, xs = simulate(S, n_ref, np.random.default_rng(seed))
    mu, sd = xs.mean(axis=0), xs.std(axis=0)
    d_sim = np.linalg.norm((xs - mu) / sd, axis=1)
    d_obs = float(np.linalg.norm((x_obs - mu) / sd))
    return float((d_sim < d_obs).mean() * 100.0), d_obs, float(np.median(d_sim))
