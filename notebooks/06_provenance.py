"""Provenance artifact (deliverable #5): Sigma_d, Sigma_m, and what survives.

Run: py -3 notebooks/06_provenance.py
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import forge_data, inversion, posterior

ROOT = Path(__file__).resolve().parents[1]

a = inversion.assemble(bridged=True)
G, mref, stations = a["G"], a["mref"], a["stations"]
tree, active = a["tree"], a["active"]
wr = inversion.depth_weights(G)
r, coef = inversion.residual_data(G, a["d_obs"], mref, stations=stations, order=2)
beta_star = inversion.tune_beta(G, r, wr)
print(f"beta* = {beta_star:.4g}")

def solve_all(wr_use, beta):
    dm = inversion.solve_map(G, r, wr_use, beta)
    var = posterior.posterior_diag(G, wr_use, beta)
    prior = 1.0 / (beta * wr_use**2)
    return dm, 1.0 - np.sqrt(var) / np.sqrt(prior)

# --- beta sweep: two decades around the discrepancy point --------------------
betas = beta_star * np.array([0.1, 0.316, 1.0, 3.16, 10.0])
sweep = {}
for b in betas:
    dm, informed = solve_all(wr, b)
    rms = float(np.sqrt(np.mean((G @ dm - r) ** 2)))
    sweep[b] = dict(dm=dm, informed=informed, rms=rms)

dm_star = sweep[beta_star * 1.0 if beta_star in sweep else betas[2]]["dm"]
inf_star = sweep[betas[2]]["informed"]
pattern_corr = {f"{b/beta_star:.2g}x": float(np.corrcoef(
    inf_star, sweep[b]["informed"])[0, 1]) for b in betas}
model_corr = {f"{b/beta_star:.2g}x": float(np.corrcoef(
    sweep[betas[2]]["dm"], sweep[b]["dm"])[0, 1]) for b in betas}
print("informed-pattern corr across sweep:", pattern_corr)
print("model corr across sweep:", model_corr)

# --- depth weighting on vs off ----------------------------------------------
flat = np.ones_like(wr)
beta_flat = inversion.tune_beta(G, r, flat)
dm_flat, inf_flat = solve_all(flat, beta_flat)
dm_dw, inf_dw = solve_all(wr, beta_star)

st_all = forge_data.load_stations()
from scipy.spatial import cKDTree
cc = tree.cell_centers[active]
kd = cKDTree(np.column_stack([st_all.Easting, st_all.Northing]))
d2, j = kd.query(cc[:, :2])
depth = st_all.NAVD88.to_numpy()[j] - cc[:, 2]
w_dw = np.abs(dm_dw); w_fl = np.abs(dm_flat)
com_dw = float(np.sum(depth * w_dw) / w_dw.sum())
com_fl = float(np.sum(depth * w_fl) / w_fl.sum())
print(f"centre-of-mass depth of |dm|: with dw {com_dw:.0f} m | without {com_fl:.0f} m")

# ---------------------------------------------------------------- figures
inv_st = forge_data.inversion_stations()
y_sec = float(inv_st.Northing.median())
h = tree.h_gridded[active]
band = np.abs(cc[:, 1] - y_sec) < 400

fig, axes = plt.subplots(2, 2, figsize=(13.5, 9), dpi=150)

ax = axes[0, 0]
bb = np.geomspace(beta_star / 30, beta_star * 30, 25)
rmss = [float(np.sqrt(np.mean((G @ inversion.solve_map(G, r, wr, b) - r) ** 2)))
        for b in bb]
ax.semilogx(bb, rmss, "-o", ms=3, color="#3a86a8")
ax.axhline(inversion.TARGET_RMS, color="r", ls="--", lw=1.2,
           label="measured floor 0.76")
ax.axvline(beta_star, color="k", ls=":", lw=1.2, label="β* (discrepancy)")
ax.set_xlabel("β (prior strength)"); ax.set_ylabel("misfit RMS [mGal]")
ax.set_title("Σd provenance: β chosen by the MEASURED floor, not taste",
             fontsize=10)
ax.legend(fontsize=8)

ax = axes[0, 1]
for b, c in zip(betas, ["#c9d3e8", "#8ba7d4", "#3a5fa8", "#8ba7d4", "#c9d3e8"]):
    near = d2 < 1500
    bins = np.linspace(0, 2500, 22)
    mid = 0.5 * (bins[1:] + bins[:-1])
    med = [np.median(sweep[b]["informed"][near & (depth >= lo) & (depth < hi)])
           for lo, hi in zip(bins[:-1], bins[1:])]
    ax.plot(mid, med, "-", color=c, lw=2.2 if b == betas[2] else 1.2,
            label=f"β = {b/beta_star:.2g}·β*")
ax.set_xlabel("depth below ground [m]")
ax.set_ylabel("median data-informed fraction")
ax.set_title("what survives a 100× prior sweep: the SHAPE of knowledge",
             fontsize=10)
ax.legend(fontsize=7)

for ax, m, title in [
    (axes[1, 0], dm_dw, f"with depth weighting — |Δρ| mass at {com_dw:.0f} m"),
    (axes[1, 1], dm_flat,
     f"WITHOUT — mass at {com_fl:.0f} m: NO surface collapse on this mesh "
     "(volume-scaled octree + mref neutralize the textbook pathology)"),
]:
    sc = ax.scatter(cc[band, 0] / 1000, cc[band, 2], c=m[band], cmap="RdBu_r",
                    vmin=-0.15, vmax=0.15, marker="s",
                    s=np.sqrt(h[band, 0]) * 1.6, linewidths=0)
    fig.colorbar(sc, ax=ax, label="Δρ update [g/cc]")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Easting [km]"); ax.set_ylabel("elevation [m]")
    ax.set_xlim(331.4, 340.6); ax.set_ylim(-1400, 2300)

fig.tight_layout()
fig.savefig(ROOT / "figures" / "provenance.png")

stats = {"beta_star": beta_star, "pattern_corr": pattern_corr,
         "model_corr": model_corr,
         "com_depth_with_dw_m": com_dw, "com_depth_without_dw_m": com_fl,
         "sigma_d_stack_mgal": {"mesh_floor": 0.755, "published_fit": 0.03,
                                "bridge_approx": "~0.1 (declared)"}}
(ROOT / "figures" / "provenance_stats.json").write_text(json.dumps(stats, indent=1))
print(json.dumps(stats, indent=1))
print("figure -> figures/provenance.png")
