"""S3: the Moho spike — satellite gravity, the space debut, with a kill-criterion.

Target problem (Phase-2 design note): the South American Moho from GOCO5S
satellite gravity (Uieda & Barbosa 2017, GJI; data CC-BY). Their published
map carries NO uncertainty estimates of any kind — which is exactly the gap
Gurutva exists to fill. The obstacle is compute: their nonlinear tesseroid
inversion is a multi-day, Python-2.7-era pipeline, and simulation-based
inference needs 10^4-10^5 forward evaluations.

So S3 is a SPIKE, not a commitment: measure whether a modern tesseroid
forward (harmonica) is fast and accurate enough on a laptop.

KILL-CRITERION, declared before measuring (design note, S3):
  ACCURACY — forward error at working resolution < 20% of the data noise
             scale, judged against the same forward at reference tolerance.
  SPEED    — <= 0.5 s per forward evaluation, so 10^5 simulations fit in
             ~14 h of free-tier compute.
  If either fails, the Moho DEFERS — recorded, not fudged — and Phase 2
  completes on S2/S2.5 + E2.

Physics: the Moho is modelled as a tesseroid layer between a reference
depth z_ref and the Moho surface, with density contrast drho (crust-mantle).
Both are hyperparameters the original paper chose by cross-validation
against seismic points; the spike uses their published values as declared
constants — the spike measures COMPUTE, not geology.
"""

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MOHO_DIR = ROOT / "data" / "moho"

# declared constants (Uieda & Barbosa 2017 best-fit hyperparameters)
Z_REF = 30_000.0        # m, reference depth
DRHO = 350.0            # kg/m^3, crust-mantle density contrast
OBS_HEIGHT = 50_000.0   # m, the data's computation height


def load_gravity(step: int = 1):
    """Processed GOCO5S grid. Columns per the file header (13 fields)."""
    cols = ["lat", "lon", "height", "topo", "gravity", "disturbance",
            "topo_effect", "bouguer", "upper_sed", "middle_sed", "lower_sed",
            "total_sed", "sedfree_bouguer"]
    a = np.loadtxt(MOHO_DIR / "goco5s.txt")
    d = {c: a[:, i] for i, c in enumerate(cols)}
    nlat, nlon = 401, 301
    if step > 1:
        for k in d:
            d[k] = d[k].reshape(nlat, nlon)[::step, ::step].ravel()
        d["shape"] = (len(range(0, nlat, step)), len(range(0, nlon, step)))
    else:
        d["shape"] = (nlat, nlon)
    return d


def load_published_moho():
    """The published Moho depth map (CC-BY). Returns lon, lat, depth[m].

    Column order verified empirically, not assumed: column 0 spans
    [-60, 20] (latitudes of South America) and column 1 spans [270, 330]
    (its longitudes east of Greenwich), so the file is lat-first.
    """
    a = np.loadtxt(MOHO_DIR / "south-american-moho.txt")
    return a[:, 1], a[:, 0], a[:, 2]


def moho_on_grid(glat, glon):
    """Nearest-neighbour interpolation of the published Moho onto a grid."""
    from scipy.spatial import cKDTree
    lon, lat, depth = load_published_moho()
    kd = cKDTree(np.column_stack([lon % 360, lat]))
    _, j = kd.query(np.column_stack([glon % 360, glat]))
    return depth[j]


def build_tesseroids(glat, glon, depth, dlat, dlon):
    """Tesseroid layer from z_ref down to the Moho, on the given grid.

    Returns (tesseroids, densities) in harmonica's convention:
    each tesseroid is [w, e, s, n, bottom_radius, top_radius].
    """
    import boule

    R = boule.WGS84.mean_radius
    w = glon - dlon / 2
    e = glon + dlon / 2
    s = glat - dlat / 2
    n = glat + dlat / 2
    # relief relative to the reference depth; sign of the contrast follows
    # whether the Moho is deeper (more crust -> negative anomaly) or shallower
    top = R - np.minimum(depth, Z_REF)
    bottom = R - np.maximum(depth, Z_REF)
    # Sign, established empirically (first run gave corr = -0.998 — right
    # physics, wrong sign): a Moho DEEPER than the reference means extra
    # light crust where the reference assumed dense mantle -> mass deficit
    # -> negative anomaly. So the contrast is -DRHO below the reference.
    dens = np.where(depth > Z_REF, -DRHO, DRHO)
    tess = np.column_stack([w, e, s, n, bottom, top])
    keep = (top - bottom) > 1.0            # drop degenerate cells
    return tess[keep], dens[keep]


def forward(glat, glon, depth, dlat, dlon, height=OBS_HEIGHT, **kw):
    """Gravity disturbance (mGal) of the Moho layer at observation height."""
    import boule
    import harmonica as hm

    tess, dens = build_tesseroids(glat, glon, depth, dlat, dlon)
    R = boule.WGS84.mean_radius
    coords = (glon, glat, np.full(len(glat), R + height))
    return hm.tesseroid_gravity(coords, tess, dens, field="g_z", **kw)
