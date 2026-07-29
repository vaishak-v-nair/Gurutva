"""Units probe — establish SimPEG's gravity conventions empirically.

A single dense cell far from the stations is indistinguishable from a point
mass (uniform cube: quadrupole vanishes by symmetry; leading correction is
O((a/r)^4) ~ 1e-8 at r = 50 cell widths). We compare SimPEG's gz against the
analytic point-mass field and print the ratio. Whatever convention this
reveals gets encoded — with this probe as the receipt — in
tests/test_g_analytic.py.

Run: py -3 src/units_probe.py
"""

import numpy as np
from scipy.constants import G  # 6.674e-11 m^3 kg^-1 s^-2

import discretize
from simpeg import maps
from simpeg.potential_fields import gravity

# --- mesh: 21^3 cells of 10 m, centered so ONE cell sits exactly at origin
h = [(10.0, 21)]
mesh = discretize.TensorMesh([h, h, h], origin="CCC")
i_center = int(np.argmin(np.linalg.norm(mesh.cell_centers, axis=1)))
assert np.allclose(mesh.cell_centers[i_center], 0.0), "no cell exactly at origin"

DENSITY_G_CC = 1.0                      # SimPEG's documented density unit: g/cc
rho = np.zeros(mesh.n_cells)
rho[i_center] = DENSITY_G_CC

cell_volume = 10.0**3                    # m^3
mass_kg = DENSITY_G_CC * 1000.0 * cell_volume  # 1 g/cc = 1000 kg/m^3

# --- stations: on-axis above, on-axis below, and one oblique point
stations = np.array([
    [0.0, 0.0, 500.0],
    [0.0, 0.0, -500.0],
    [300.0, 0.0, 400.0],
])

rx = gravity.receivers.Point(stations, components="gz")
src = gravity.sources.SourceField(receiver_list=[rx])
survey = gravity.survey.Survey(src)
sim = gravity.simulation.Simulation3DIntegral(
    mesh=mesh, survey=survey, rhoMap=maps.IdentityMap(nP=mesh.n_cells),
    store_sensitivities="ram",
)
d_simpeg = sim.dpred(rho)

# --- analytic point mass: attraction vector points from station toward mass
print(f"{'station':>22s} {'SimPEG gz':>14s} {'analytic |g| m/s2':>18s} "
      f"{'as mGal':>12s} {'ratio simpeg/mGal_downpos':>26s}")
for loc, d in zip(stations, d_simpeg):
    r = np.linalg.norm(loc)
    g_si = G * mass_kg / r**2                      # magnitude, m/s^2
    # z-component with "positive down toward mass" geophysics convention:
    # station above mass (z>0): attraction is downward -> +; below: -> -
    gz_downpos_si = g_si * (loc[2] / r)            # projection with down-positive sign
    gz_downpos_mgal = gz_downpos_si / 1e-5         # 1 mGal = 1e-5 m/s^2
    ratio = d / gz_downpos_mgal if gz_downpos_mgal != 0 else float("nan")
    print(f"{str(loc):>22s} {d:14.6e} {g_si:18.6e} {gz_downpos_mgal:12.6e} {ratio:26.9f}")
