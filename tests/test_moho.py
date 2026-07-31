"""S3 gates: the conventions that were traps, and the route that cleared
the kill-criterion.

Full-run measurements (2026-08-03, figures/s3_stats.json): direct forward
0.16/0.36/1.51/3.96 s at n=494/884/1938/3417; layered sensitivity at
n=1938 x 24 layers = 3023 s precompute, 721 MB, 27.2 ms/forward,
discretisation error 0.02 mGal against a 3.6 mGal budget. Tests below use
a small configuration so the suite stays fast.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import moho, moho_fast

pytestmark = pytest.mark.skipif(
    not (moho.MOHO_DIR / "goco5s.txt").exists(),
    reason="Moho data not downloaded (see src/moho.py)")


def test_moho_file_is_latitude_first():
    """The trap: the published file is (lat, lon, depth). Reading it as
    (lon, lat) silently relocates South America."""
    lon, lat, depth = moho.load_published_moho()
    assert 270 <= lon.min() and lon.max() <= 330, "longitudes must be 270-330E"
    assert -60 <= lat.min() and lat.max() <= 20, "latitudes must be -60..20"
    assert 5_000 < depth.min() and depth.max() < 80_000


def test_density_sign_reproduces_observed_gravity():
    """The second trap: a deeper Moho is a MASS DEFICIT. The first run gave
    corr = -0.998 (right physics, wrong sign); this pins the fix."""
    g = moho.load_gravity(step=16)
    dep = moho.moho_on_grid(g["lat"], g["lon"])
    d = moho.forward(g["lat"], g["lon"], dep, 3.2, 3.2)
    corr = float(np.corrcoef(d, g["sedfree_bouguer"])[0, 1])
    assert corr > 0.95, f"sign/physics broken: corr {corr:+.3f}"


def test_layered_route_matches_exact_and_is_fast():
    """The kill-criterion, in miniature: the matvec route must reproduce the
    exact tesseroid forward far inside the accuracy budget, and run far
    inside the speed budget."""
    g = moho.load_gravity(step=20)
    glat, glon, sp = g["lat"], g["lon"], 4.0
    dep = moho.moho_on_grid(glat, glon)
    edges = moho_fast.layer_edges(n_layers=6)

    A = moho_fast.build_sensitivity(glat, glon, sp, sp, edges, verbose=False)
    d_fast = moho_fast.forward(A, dep, edges)
    d_exact = moho.forward(glat, glon, dep, sp, sp)
    err = float(np.sqrt(np.mean((d_fast - d_exact) ** 2)))
    scale = float(np.std(d_exact))
    assert err < 0.05 * scale, (
        f"layered discretisation error {err:.2f} mGal too large vs signal "
        f"{scale:.0f} mGal")

    moho_fast.forward(A, dep, edges)                     # warm
    t0 = time.time()
    for _ in range(20):
        moho_fast.forward(A, dep, edges)
    per = (time.time() - t0) / 20
    assert per < 0.5, f"matvec forward {per*1e3:.0f} ms exceeds the 0.5 s budget"
