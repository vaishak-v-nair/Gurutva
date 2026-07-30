"""Measure the coarse-mesh error floor.

d_fine  : exact prism forward of the published 268,773-cell model (validated
          geometry, validated prism code) at the 323 stations.
d_coarse: SimPEG forward of the SAME source, mass-preservingly projected onto
          our 10,093-cell mesh.
floor   = d_coarse - d_fine. This is the misfit our mesh coarseness alone
          injects — measured, not assumed, per the plan.

Run: py -3 notebooks/02_error_floor.py
"""

import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import forge_data, mesh as mesh_mod, prism_forward, published_model

ROOT = Path(__file__).resolve().parents[1]

inv = forge_data.inversion_stations()
stations = np.column_stack([inv.Easting, inv.Northing, inv.z_sensor]).astype(float)

centers, dims, drho = published_model.load_geometry()
tree, active, meta = mesh_mod.build_mesh()

# --- project fine cells onto coarse mesh (mass-preserving) -------------------
try:
    idx = tree.get_containing_cells(centers)
except AttributeError:
    idx = tree._get_containing_cell_indexes(centers)
idx = np.asarray(idx)

act_index = -np.ones(tree.n_cells, dtype=int)
act_index[active] = np.arange(int(active.sum()))
fine_in_active = act_index[idx] >= 0

n_dropped = int((~fine_in_active).sum())
mass_dropped = float(np.sum(np.abs(drho[~fine_in_active])
                            * dims[~fine_in_active].prod(axis=1)))
mass_total = float(np.sum(np.abs(drho) * dims.prod(axis=1)))
print(f"fine cells outside active coarse cells: {n_dropped} "
      f"({100 * mass_dropped / mass_total:.2f}% of |mass|) — excluded from BOTH sides")

fc = fine_in_active
vol_fine = dims[fc].prod(axis=1)
tgt = act_index[idx[fc]]
n_act = int(active.sum())
mass = np.bincount(tgt, weights=drho[fc] * vol_fine, minlength=n_act)
vol_coarse = tree.cell_volumes[active]
model_coarse = mass / vol_coarse            # mass-preserving smear

# --- d_fine: exact prisms ----------------------------------------------------
t0 = time.time()
d_fine = prism_forward.prism_gz(stations, centers[fc], dims[fc], drho[fc])
print(f"d_fine done in {time.time() - t0:.0f}s")

# --- d_coarse: SimPEG on our mesh -------------------------------------------
from simpeg import maps
from simpeg.potential_fields import gravity

rx = gravity.receivers.Point(stations, components="gz")
survey = gravity.survey.Survey(gravity.sources.SourceField(receiver_list=[rx]))
kw = dict(mesh=tree, survey=survey, rhoMap=maps.IdentityMap(nP=n_act),
          store_sensitivities="ram")
try:
    sim = gravity.simulation.Simulation3DIntegral(active_cells=active, **kw)
except TypeError:
    sim = gravity.simulation.Simulation3DIntegral(ind_active=active, **kw)
t0 = time.time()
d_coarse = sim.dpred(model_coarse)
print(f"d_coarse done in {time.time() - t0:.0f}s")

# --- the floor ---------------------------------------------------------------
floor = d_coarse - d_fine
stats = {
    "n_stations": len(stations),
    "rms_floor_mgal": float(np.sqrt(np.mean(floor**2))),
    "mean_abs_floor_mgal": float(np.mean(np.abs(floor))),
    "p95_abs_floor_mgal": float(np.percentile(np.abs(floor), 95)),
    "max_abs_floor_mgal": float(np.max(np.abs(floor))),
    "published_rms_mgal": 0.0298,
    "signal_span_mgal": float(d_fine.max() - d_fine.min()),
    "fine_cells_excluded": n_dropped,
    "excluded_abs_mass_fraction": mass_dropped / mass_total,
    "n_active_cells": n_act,
}
print(json.dumps(stats, indent=1))
(ROOT / "figures" / "floor_stats.json").write_text(json.dumps(stats, indent=1))

fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6), dpi=150)
ax = axes[0]
sc = ax.scatter(inv.Easting / 1000, inv.Northing / 1000, c=floor, cmap="RdBu_r",
                vmin=-np.percentile(np.abs(floor), 98),
                vmax=np.percentile(np.abs(floor), 98), s=16, linewidths=0)
fig.colorbar(sc, ax=ax, label="floor error [mGal]")
ax.set_title("where the mesh coarseness bites")
ax.set_xlabel("Easting [km]"); ax.set_ylabel("Northing [km]"); ax.set_aspect("equal")

ax = axes[1]
ax.scatter(d_fine, d_coarse, s=8, alpha=0.6, linewidths=0, color="#6b4fd8")
lim = [min(d_fine.min(), d_coarse.min()), max(d_fine.max(), d_coarse.max())]
ax.plot(lim, lim, "k--", lw=1)
ax.set_xlabel("d_fine: exact prisms, published model [mGal]")
ax.set_ylabel("d_coarse: SimPEG, our 10k mesh [mGal]")
ax.set_title("same source, two discretizations")

ax = axes[2]
ax.hist(floor, bins=40, color="#3a86a8", edgecolor="white", linewidth=0.4)
ax.axvline(0.0298, color="r", ls="--", lw=1.2, label="published RMS 0.0298")
ax.axvline(-0.0298, color="r", ls="--", lw=1.2)
ax.set_xlabel("floor error [mGal]"); ax.set_ylabel("stations")
ax.set_title(f"floor RMS = {stats['rms_floor_mgal']:.3f} mGal")
ax.legend(fontsize=8)

fig.tight_layout()
fig.savefig(ROOT / "figures" / "error_floor.png")
print("figure -> figures/error_floor.png")
