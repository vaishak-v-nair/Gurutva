"""S2 generative model: geology as it is — bimodal, with an uncertain
interface. The first beyond-Gaussian engine.

Declared story (all hyperparameters stated, provenance-style):
- interface: z(x,y) = delivered Top-of-Granite + smooth perturbation
  dz(x,y) = Phi(x,y) @ w, with Phi 12 RBF features (4x3 grid over the
  footprint, length scale 2 km) and w ~ N(0, SIGMA_Z^2), SIGMA_Z = 150 m.
- densities: contrast-vs-2.67 of the two units are themselves unknowns:
  rho_basin ~ N(-0.25, 0.03), rho_basement ~ N(-0.02, 0.02) (per draw).
- theta = [w (12), rho_basin, rho_basement] — 14 dims, all interpretable.
- data summary x = top-K_DATA components of U^T (G (m - mref)) + N(0, I)
  (the S1-validated projection; sufficient in the linear-Gaussian case,
  declared summary here).

The product this enables: the granite surface WITH ERROR BARS, and
per-cell posteriors that are allowed to be bimodal.
"""

import numpy as np

from . import forge_data, modes

SIGMA_Z = 150.0          # m, interface perturbation amplitude (declared)
RBF_LEN = 2000.0         # m, correlation length (declared)
RHO_BASIN = (-0.25, 0.03)
RHO_BASEMENT = (-0.02, 0.02)
K_DATA = 64              # data-summary dimension (covers >90% resolution)
N_THETA = 14


def _rbf_centers(cc):
    x0, x1 = cc[:, 0].min(), cc[:, 0].max()
    y0, y1 = cc[:, 1].min(), cc[:, 1].max()
    gx = np.linspace(x0, x1, 4)
    gy = np.linspace(y0, y1, 3)
    return np.array([(x, y) for x in gx for y in gy])


def build(M=None):
    """Precompute geometry: returns dict with Phi (n_act x 12), base z, etc."""
    if M is None:
        M = modes.build(bridged=True)
    tree, active = M["a"]["tree"], M["a"]["active"]
    cc = tree.cell_centers[active]

    basement = forge_data.basement_surface("Original")
    from scipy.spatial import cKDTree
    kd = cKDTree(basement[:, :2])
    _, j = kd.query(cc[:, :2])
    z_base = basement[j, 2]

    centers = _rbf_centers(cc)
    d2 = ((cc[:, None, 0] - centers[None, :, 0]) ** 2
          + (cc[:, None, 1] - centers[None, :, 1]) ** 2)
    Phi = np.exp(-0.5 * d2 / RBF_LEN**2)

    return dict(M=M, cc=cc, z_base=z_base, Phi=Phi, centers=centers,
                G=M["a"]["G"], mref=M["a"]["mref"], U=M["U"], r=M["r"])


def sample_theta(n, rng):
    w = SIGMA_Z * rng.standard_normal((n, 12))
    rb = RHO_BASIN[0] + RHO_BASIN[1] * rng.standard_normal((n, 1))
    rg = RHO_BASEMENT[0] + RHO_BASEMENT[1] * rng.standard_normal((n, 1))
    return np.hstack([w, rb, rg])


def model_from_theta(S2, theta):
    """Density-contrast field(s) from theta; theta (n, 14) -> m (n, n_act)."""
    theta = np.atleast_2d(theta)
    z_pert = S2["z_base"][None, :] + theta[:, :12] @ S2["Phi"].T
    basin = S2["cc"][None, :, 2] > z_pert
    rb = theta[:, 12:13]
    rg = theta[:, 13:14]
    return np.where(basin, rb, rg)


def simulate(S2, n, rng, chunk=2_000):
    """(theta, x) training pairs; x = top-K_DATA summary with unit noise."""
    theta = sample_theta(n, rng)
    Uk = S2["U"][:, :K_DATA]
    xs = np.empty((n, K_DATA))
    for a in range(0, n, chunk):
        m = model_from_theta(S2, theta[a:a + chunk])
        d = (m - S2["mref"][None, :]) @ S2["G"].T
        xs[a:a + chunk] = d @ Uk + rng.standard_normal((len(m), K_DATA))
    return theta, xs


def x_observed(S2):
    return S2["r"] @ S2["U"][:, :K_DATA]
