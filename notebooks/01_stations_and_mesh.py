"""Session artifact: load the 323 stations, raise the coarse mesh, render both.

Run: py -3 notebooks/01_stations_and_mesh.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import forge_data, mesh as mesh_mod

ROOT = Path(__file__).resolve().parents[1]

inv = forge_data.inversion_stations()
basement = forge_data.basement_surface("Original")
tree, active, meta = mesh_mod.build_mesh()

print(f"stations: {len(inv)} (published inversion set)")
print(f"mesh: {meta['n_total']} total cells | {meta['n_active']} active (below topo)")
print(f"finest cell: {meta['finest_cell']} m | model extent: {meta['model_extent']}")

fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6), dpi=150)

# left: the 323 stations, colored by the delivered anomaly
ax = axes[0]
sc = ax.scatter(inv.Easting / 1000, inv.Northing / 1000, c=inv.gCBGA,
                cmap="viridis", s=14, linewidths=0)
fig.colorbar(sc, ax=ax, label="complete Bouguer anomaly, 2.67 g/cc [mGal]")
ax.set_xlabel("Easting [km, NAD83 z12]"); ax.set_ylabel("Northing [km]")
ax.set_title(f"{len(inv)} stations of the published inversion")
ax.set_aspect("equal")

# right: E-W section of the mesh through the station-cloud median northing,
# with the basement surface trace
ax = axes[1]
y_sec = float(inv.Northing.median())
sl = np.abs(basement[:, 1] - y_sec) < 300
order = np.argsort(basement[sl, 0])
cc = tree.cell_centers[active]
band = np.abs(cc[:, 1] - y_sec) < 400
h = tree.h_gridded[active]
for (x, _, z), (hx, _, hz) in zip(cc[band], h[band]):
    ax.add_patch(plt.Rectangle(((x - hx / 2) / 1000, z - hz / 2), hx / 1000, hz,
                               fill=False, edgecolor="#8890a8", linewidth=0.3))
ax.plot(basement[sl, 0][order] / 1000, basement[sl, 2][order],
        "r-", lw=1.4, label="top of granite (from geoh5)")
zs = inv[np.abs(inv.Northing - y_sec) < 400]
ax.plot(zs.Easting / 1000, zs.z_sensor, "kv", ms=4, label="stations (sensor height)")
ax.set_xlim((meta["model_extent"][0][0] - 500) / 1000,
            (meta["model_extent"][0][1] + 500) / 1000)
ax.set_ylim(meta["model_extent"][2][0] - 200, 2300)
ax.set_xlabel("Easting [km]"); ax.set_ylabel("elevation [m, NAVD88]")
ax.set_title(f"E-W mesh section at N={y_sec/1000:.1f} km — "
             f"{meta['n_active']} active cells")
ax.legend(loc="lower left", fontsize=8)

fig.tight_layout()
out = ROOT / "figures" / "stations_and_mesh.png"
fig.savefig(out)
print(f"figure -> {out}")
