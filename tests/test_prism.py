"""Gates for the prism forward and the published-model geometry inference."""

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.constants import G

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import prism_forward, published_model


def test_prism_matches_point_mass_far_field():
    """A 10 m cube at 50 cell-widths is a point mass to O((a/r)^4)."""
    centers = np.array([[0.0, 0.0, 0.0]])
    dims = np.array([[10.0, 10.0, 10.0]])
    drho = np.array([1.0])                       # g/cc
    st = np.array([[0.0, 0.0, 500.0], [300.0, 0.0, 400.0]])
    got = prism_forward.prism_gz(st, centers, dims, drho)
    mass = 1000.0 * 10.0**3
    r = np.linalg.norm(st, axis=1)
    expected = -(G * mass / r**2) * (st[:, 2] / r) / 1e-5   # z-up, mGal
    rel = np.abs(got - expected) / np.abs(expected)
    assert rel.max() < 1e-6, f"far-field mismatch: {rel}"


def test_prism_matches_simpeg_near_field():
    """Same single cell, station only 100 m above — pins sign, units, and
    the exact-prism math against SimPEG's integral kernel."""
    import discretize
    from simpeg import maps
    from simpeg.potential_fields import gravity

    h = [(50.0, 5)]
    mesh = discretize.TensorMesh([h, h, h], origin="CCC")
    i_c = int(np.argmin(np.linalg.norm(mesh.cell_centers, axis=1)))
    rho = np.zeros(mesh.n_cells)
    rho[i_c] = 0.3
    st = np.array([[30.0, -20.0, 100.0], [0.0, 0.0, 60.0]])
    rx = gravity.receivers.Point(st, components="gz")
    survey = gravity.survey.Survey(gravity.sources.SourceField(receiver_list=[rx]))
    sim = gravity.simulation.Simulation3DIntegral(
        mesh=mesh, survey=survey, rhoMap=maps.IdentityMap(nP=mesh.n_cells),
        store_sensitivities="ram")
    d_simpeg = sim.dpred(rho)

    d_prism = prism_forward.prism_gz(
        st, np.array([[0.0, 0.0, 0.0]]), np.array([[50.0, 50.0, 50.0]]),
        np.array([0.3]))
    rel = np.abs(d_prism - d_simpeg) / np.abs(d_simpeg)
    assert rel.max() < 1e-5, f"prism vs SimPEG mismatch: {d_prism} vs {d_simpeg}"


def test_published_model_geometry_has_zero_overlaps():
    """Cell-size inference audit: exact sub-voxel occupancy. Oversized or
    misplaced cells double-claim sub-voxels. Holes are legitimate (the model
    is topo-cropped: air above ground is absent — a naive volume/box audit
    reads ~0.89 and proves nothing)."""
    assert published_model.overlap_count() == 0


def test_published_model_all_cells_classified():
    centers, dims, drho = published_model.load_geometry()
    assert len(centers) == 268_773
    assert np.all(dims > 0)
    # measured level structure: (50,50,30), (100,100,60), (200,200,120)
    assert set(np.unique(dims[:, 0])) == {50.0, 100.0, 200.0}
    assert set(np.unique(dims[:, 2])) == {30.0, 60.0, 120.0}
    levels = {(50.0, 30.0), (100.0, 60.0), (200.0, 120.0)}
    assert set(map(tuple, np.unique(dims[:, [0, 2]], axis=0))) == levels
