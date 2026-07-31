"""S3: the layered-sensitivity route — turning a 37-second forward into a matvec.

The spike's first measurement was decisive: a direct tesseroid forward costs
~37 s at a heavily decimated 1.6-deg grid, against a declared 0.5 s budget.
Naive simulation-based inference is therefore impossible here (10^5 sims
would take ~6 weeks of laptop time).

The structural fix is the same one Phase 1 already relies on: the forward is
LINEAR in mass, and only the GEOMETRY is nonlinear. Slice the depth axis into
thin layers once; precompute the gravity response of each (cell, layer)
element; then ANY Moho relief is a fill pattern over those elements and the
forward becomes A @ fill — microseconds.

    precompute:  n_obs x (n_cells * n_layers) sensitivity, ONE time
    forward:     matvec, with the ONLY approximation being the vertical
                 discretization (measured against the exact tesseroid).

This trades one-time cost for per-evaluation cost. Whether that trade clears
the S3 kill-criterion is measured in notebooks/10_s3_moho_spike.py.
"""

import numpy as np

from . import moho


def layer_edges(z_min=5_000.0, z_max=75_000.0, n_layers=24):
    return np.linspace(z_min, z_max, n_layers + 1)


def build_sensitivity(glat, glon, dlat, dlon, edges, height=moho.OBS_HEIGHT,
                      chunk=None, verbose=True):
    """A: (n_obs, n_cells*n_layers) response of each unit-density element.

    Element (c, l) is the tesseroid under cell c spanning depth layer l with
    density 1 kg/m^3. Built layer by layer so memory stays flat.
    """
    import boule
    import harmonica as hm
    import time

    R = boule.WGS84.mean_radius
    n_obs = len(glat)
    n_lay = len(edges) - 1
    coords = (glon, glat, np.full(n_obs, R + height))
    w, e = glon - dlon / 2, glon + dlon / 2
    s, n = glat - dlat / 2, glat + dlat / 2

    A = np.empty((n_obs, n_obs * n_lay), dtype=np.float64)
    t0 = time.time()
    for l in range(n_lay):
        top = R - edges[l]
        bot = R - edges[l + 1]
        for c in range(n_obs):
            tess = np.array([[w[c], e[c], s[c], n[c], bot, top]])
            A[:, l * n_obs + c] = hm.tesseroid_gravity(
                coords, tess, np.array([1.0]), field="g_z")
        if verbose:
            print(f"  layer {l+1}/{n_lay}  ({time.time()-t0:.0f}s)", flush=True)
    return A


def fill_from_depth(depth, edges, z_ref=moho.Z_REF, drho=moho.DRHO):
    """Fill vector: signed density of each (cell, layer) element.

    Fractional at the boundary layer — the vertical discretization is
    linearised there rather than snapped, which is what keeps the
    approximation error small (measured in the spike).
    """
    n_obs = len(depth)
    n_lay = len(edges) - 1
    f = np.zeros(n_obs * n_lay)
    for l in range(n_lay):
        lo, hi = edges[l], edges[l + 1]
        # overlap of [z_ref, depth] (or [depth, z_ref]) with this layer
        a = np.minimum(depth, z_ref)
        b = np.maximum(depth, z_ref)
        frac = (np.minimum(b, hi) - np.maximum(a, lo)) / (hi - lo)
        frac = np.clip(frac, 0.0, 1.0)
        sign = np.where(depth > z_ref, -1.0, 1.0)
        f[l * n_obs:(l + 1) * n_obs] = sign * drho * frac
    return f


def forward(A, depth, edges, z_ref=moho.Z_REF, drho=moho.DRHO):
    return A @ fill_from_depth(depth, edges, z_ref, drho)
