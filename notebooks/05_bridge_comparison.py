"""The 2.55 bridge + the limited-fidelity model comparison.

Run: py -3 notebooks/05_bridge_comparison.py
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

res = inversion.run(bridged=True)
tree, active, G = res["tree"], res["active"], res["G"]
print(f"bridged inversion: beta={res['beta']:.4g} RMS={res['rms']:.3f} mGal")

centers, dims, drho = published_model.load_geometry()
try:
    idx = tree.get_containing_cells(centers)
except AttributeError:
    idx = tree._get_containing_cell_indexes(centers)
idx = np.asarray(idx)
ai = -np.ones(tree.n_cells, dtype=int)
ai[active] = np.arange(int(active.sum()))
fc = ai[idx] >= 0
vol = dims[fc].prod(axis=1)
pub = (np.bincount(ai[idx[fc]], weights=drho[fc] * vol,
                   minlength=int(active.sum())) / tree.cell_volumes[active])

# their-model misfit ladder (the unblocking story)
r = res["d_obs"] - G @ pub
ladder = {}
for name, order in [("DC only", 0), ("+ plane", 1), ("+ quadratic", 2)]:
    A = inversion.regional_basis(res["stations"], order)
    coef, *_ = np.linalg.lstsq(A, r, rcond=None)
    ladder[name] = float(np.sqrt(np.mean((r - A @ coef) ** 2)))
print("their-model misfit ladder:", ladder)

# correlations, all + constrained (baselines from the 2.67 run: 0.125 / 0.287)
cc = tree.cell_centers[active]
from scipy.spatial import cKDTree
st_all = forge_data.load_stations()
kd = cKDTree(np.column_stack([st_all.Easting, st_all.Northing]))
d2, j = kd.query(cc[:, :2])
depth = st_all.NAVD88.to_numpy()[j] - cc[:, 2]
shallow = (depth < 700) & (d2 < 1000)
corr_all = float(np.corrcoef(res["model"], pub)[0, 1])
corr_con = float(np.corrcoef(res["model"][shallow], pub[shallow])[0, 1])
print(f"corr all: {corr_all:.3f} (was 0.125) | constrained: {corr_con:.3f} (was 0.287)")

# ---------------------------------------------------------------- figures
inv_st = forge_data.inversion_stations()
y_sec = float(inv_st.Northing.median())
h = tree.h_gridded[active]
band = np.abs(cc[:, 1] - y_sec) < 400

fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), dpi=150)
vlim = 0.28
for ax, m, title in [
    (axes[0], res["model"], "OURS (bridged 2.55 data)"),
    (axes[1], pub, "THEIRS (projected)"),
]:
    sc = ax.scatter(cc[band, 0] / 1000, cc[band, 2], c=m[band], cmap="RdBu_r",
                    vmin=-vlim, vmax=vlim, marker="s",
                    s=np.sqrt(h[band, 0]) * 1.6, linewidths=0)
    fig.colorbar(sc, ax=ax, label="Δρ [g/cc]")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Easting [km]"); ax.set_ylabel("elevation [m]")
    ax.set_xlim(331.4, 340.6); ax.set_ylim(-1400, 2300)

ax = axes[2]
names = list(ladder) + ["floor ⊕ their fit"]
vals = list(ladder.values()) + [np.sqrt(0.755**2 + 0.03**2)]
colors = ["#b3543f", "#c98a3d", "#3a86a8", "#4aa86b"]
ax.bar(names, vals, color=colors)
ax.axhline(np.sqrt(0.755**2 + 0.03**2), color="k", ls="--", lw=1)
ax.set_ylabel("their-model RMS vs our bridged data [mGal]")
ax.set_title("the unblocking ladder — and the honest limit", fontsize=10)
ax.tick_params(axis="x", labelsize=8)

fig.tight_layout()
fig.savefig(ROOT / "figures" / "bridge_comparison.png")

stats = {"ladder": ladder, "corr_all": corr_all, "corr_constrained": corr_con,
         "baseline_corr_all": 0.125, "baseline_corr_constrained": 0.287,
         "bridged_rms": res["rms"], "beta": res["beta"]}
(ROOT / "figures" / "bridge_stats.json").write_text(json.dumps(stats, indent=1))
print(json.dumps(stats, indent=1))
print("figure -> figures/bridge_comparison.png")
