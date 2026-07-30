"""Pass-one inversion: run, compare to the published model, render.

Run: py -3 notebooks/03_inversion.py
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import forge_data, inversion, published_model

ROOT = Path(__file__).resolve().parents[1]

res = inversion.run()
tree, active = res["tree"], res["active"]
print(f"beta = {res['beta']:.4g} | DC = {res['dc']:.3f} mGal | "
      f"RMS = {res['rms']:.3f} mGal (target {inversion.TARGET_RMS})")

# published model projected onto the same mesh (mass-preserving, as in 02)
centers, dims, drho = published_model.load_geometry()
try:
    idx = tree.get_containing_cells(centers)
except AttributeError:
    idx = tree._get_containing_cell_indexes(centers)
idx = np.asarray(idx)
act_index = -np.ones(tree.n_cells, dtype=int)
act_index[active] = np.arange(int(active.sum()))
fc = act_index[idx] >= 0
vol = dims[fc].prod(axis=1)
pub = np.bincount(act_index[idx[fc]], weights=drho[fc] * vol,
                  minlength=int(active.sum())) / tree.cell_volumes[active]

np.save(ROOT / "figures" / "model_recovered.npy", res["model"])
np.save(ROOT / "figures" / "model_published_projected.npy", pub)

# ---------------------------------------------------------------- figures
inv_st = forge_data.inversion_stations()
y_sec = float(inv_st.Northing.median())
cc = tree.cell_centers[active]
h = tree.h_gridded[active]
band = np.abs(cc[:, 1] - y_sec) < 400

fig, axes = plt.subplots(2, 2, figsize=(13.5, 9), dpi=150)

vlim = 0.28
for ax, m, title in [
    (axes[0, 0], res["model"], "OURS: pass-one recovered contrast"),
    (axes[0, 1], pub, "THEIRS: published model (projected, minus 2.67)"),
]:
    sc = ax.scatter(cc[band, 0] / 1000, cc[band, 2], c=m[band], cmap="RdBu_r",
                    vmin=-vlim, vmax=vlim, s=np.sqrt(h[band, 0]) * 1.7,
                    marker="s", linewidths=0)
    fig.colorbar(sc, ax=ax, label="density contrast [g/cc]")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Easting [km]"); ax.set_ylabel("elevation [m]")
    ax.set_xlim(331.4, 340.6); ax.set_ylim(-1400, 2300)

ax = axes[1, 0]
sc = ax.scatter(inv_st.Easting / 1000, inv_st.Northing / 1000,
                c=res["misfit"], cmap="RdBu_r",
                vmin=-2.2, vmax=2.2, s=16, linewidths=0)
fig.colorbar(sc, ax=ax, label="misfit d_obs - d_pred [mGal]")
ax.set_title("misfit map", fontsize=10)
ax.set_xlabel("Easting [km]"); ax.set_ylabel("Northing [km]")
ax.set_aspect("equal")

ax = axes[1, 1]
ax.hist(res["misfit"], bins=40, color="#3a86a8", edgecolor="white",
        linewidth=0.4)
ax.axvline(inversion.TARGET_RMS, color="r", ls="--", lw=1.2,
           label=f"floor target ±{inversion.TARGET_RMS}")
ax.axvline(-inversion.TARGET_RMS, color="r", ls="--", lw=1.2)
ax.set_title(f"misfit histogram — RMS {res['rms']:.3f} mGal", fontsize=10)
ax.set_xlabel("misfit [mGal]"); ax.set_ylabel("stations")
ax.legend(fontsize=8)

fig.tight_layout()
fig.savefig(ROOT / "figures" / "inversion_pass_one.png")

stats = {"beta": res["beta"], "dc_mgal": res["dc"], "rms_mgal": res["rms"],
         "target_rms_mgal": inversion.TARGET_RMS,
         "model_min": float(res["model"].min()),
         "model_max": float(res["model"].max()),
         "corr_ours_vs_published": float(np.corrcoef(res["model"], pub)[0, 1])}
print(json.dumps(stats, indent=1))
(ROOT / "figures" / "inversion_stats.json").write_text(json.dumps(stats, indent=1))
print("figure -> figures/inversion_pass_one.png")
