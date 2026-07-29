"""First validation gate: G-vs-analytic (plan finding E-4).

A single dense cell at 50 cell-widths is a point mass to O((a/r)^4) ~ 1e-8
(uniform cube: quadrupole vanishes by symmetry), so SimPEG's integral
sensitivity must reproduce Newton to ~6 significant digits or the unit
wiring is wrong somewhere (mGal vs SI, g/cc vs kg/m^3, cell volumes, sign).

Conventions established empirically by src/units_probe.py on 2026-07-29
(SimPEG 0.25.2): density in g/cc; output gz in mGal (1 mGal = 1e-5 m/s^2);
z-UP sign convention — attraction toward a mass below the station is
NEGATIVE gz. Measured ratio vs down-positive analytic: -1.000000006.
"""

import numpy as np
import pytest
from scipy.constants import G

import discretize
from simpeg import maps
from simpeg.potential_fields import gravity

DENSITY_G_CC = 1.0
CELL = 10.0                                   # m
STATIONS = np.array([
    [0.0, 0.0, 500.0],
    [0.0, 0.0, -500.0],
    [300.0, 0.0, 400.0],
])
RTOL = 1e-6                                   # ~6 significant digits


@pytest.fixture(scope="module")
def simulation():
    h = [(CELL, 21)]
    mesh = discretize.TensorMesh([h, h, h], origin="CCC")
    i_center = int(np.argmin(np.linalg.norm(mesh.cell_centers, axis=1)))
    assert np.allclose(mesh.cell_centers[i_center], 0.0)

    rho = np.zeros(mesh.n_cells)
    rho[i_center] = DENSITY_G_CC

    rx = gravity.receivers.Point(STATIONS, components="gz")
    survey = gravity.survey.Survey(gravity.sources.SourceField(receiver_list=[rx]))
    sim = gravity.simulation.Simulation3DIntegral(
        mesh=mesh, survey=survey, rhoMap=maps.IdentityMap(nP=mesh.n_cells),
        store_sensitivities="ram",
    )
    return sim, rho


def analytic_gz_mgal(stations):
    """Point-mass gz in SimPEG's convention: mGal, z-up (mass below -> negative)."""
    mass_kg = DENSITY_G_CC * 1000.0 * CELL**3
    r = np.linalg.norm(stations, axis=1)
    g_si = G * mass_kg / r**2                  # magnitude, m/s^2
    gz_zup_si = -g_si * (stations[:, 2] / r)   # z-up: station above mass -> negative
    return gz_zup_si / 1e-5                    # -> mGal


def test_gz_matches_point_mass(simulation):
    sim, rho = simulation
    d = sim.dpred(rho)
    expected = analytic_gz_mgal(STATIONS)
    rel_err = np.abs(d - expected) / np.abs(expected)
    assert rel_err.max() < RTOL, (
        f"unit/sign wiring broken: max rel err {rel_err.max():.3e}\n"
        f"simpeg:   {d}\nanalytic: {expected}"
    )


def test_forward_is_linear_in_density(simulation):
    sim, rho = simulation
    np.testing.assert_allclose(sim.dpred(2.0 * rho), 2.0 * sim.dpred(rho),
                               rtol=1e-12)
