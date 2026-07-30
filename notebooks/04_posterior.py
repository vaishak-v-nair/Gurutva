"""THE deliverable: the posterior uncertainty map, fully validated.

Run: py -3 notebooks/04_posterior.py
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
from src import forge_data, inversion, posterior

ROOT = Path(__file__).resolve().parents[1]

res = inversion.run()
G, wr, beta = res["G"], res["wr"], res["beta"]
tree, active = res["tree"], res["active"]

var_cf = posterior.posterior_diag(G, wr, beta)
sig = np.sqrt(var_cf)
sig_prior = np.sqrt(1.0 / (beta * wr**2))
informed = 1.0 - sig / sig_prior              # 0 = prior-dominated

t0 = time.time()
N = 30_000
var_mc = posterior.mc_sample_variance(G, wr, beta, n_samples=N)
mc_stats, mc_ok = posterior.mc_gates(var_mc, var_cf, N)
print(f"MC gate ({N} samples, {time.time()-t0:.0f}s): "
      f"{'PASS' if mc_ok else 'FAIL'} {mc_stats}")

cov, per_rep = posterior.coverage(G, wr, beta, res["mref"], reps=20)
print(f"coverage gate (20 reps): {cov:.4f} "
      f"({'PASS' if 0.93 < cov < 0.97 else 'FAIL'})")

np.save(ROOT / "figures" / "posterior_sigma.npy", sig)

# ---------------------------------------------------------------- figures
inv_st = forge_data.inversion_stations()
basement = forge_data.basement_surface("Original")
y_sec = float(inv_st.Northing.median())
cc = tree.cell_centers[active]
h = tree.h_gridded[active]
band = np.abs(cc[:, 1] - y_sec) < 400

fig, axes = plt.subplots(2, 2, figsize=(13.5, 9), dpi=150)

ax = axes[0, 0]
sc = ax.scatter(cc[band, 0] / 1000, cc[band, 2], c=informed[band],
                cmap="magma", vmin=0, vmax=informed.max(), marker="s",
                s=np.sqrt(h[band, 0]) * 1.7, linewidths=0)
fig.colorbar(sc, ax=ax, label="data-informed fraction (1 - σ_post/σ_prior)")
sl = np.abs(basement[:, 1] - y_sec) < 300
order = np.argsort(basement[sl, 0])
ax.plot(basement[sl, 0][order] / 1000, basement[sl, 2][order], "c-", lw=1.2,
        label="top of granite")
zs = inv_st[np.abs(inv_st.Northing - y_sec) < 400]
ax.plot(zs.Easting / 1000, zs.z_sensor, "wv", ms=4, label="stations")
ax.set_title("WHAT THE DATA ACTUALLY KNOW — E-W section", fontsize=10)
ax.set_xlabel("Easting [km]"); ax.set_ylabel("elevation [m]")
ax.set_xlim(331.4, 340.6); ax.set_ylim(-1400, 2300)
ax.legend(loc="lower left", fontsize=7)

ax = axes[0, 1]
st = forge_data.load_stations()
from scipy.spatial import cKDTree
kd = cKDTree(np.column_stack([st.Easting, st.Northing]))
d2, j = kd.query(cc[:, :2])
depth = st.NAVD88.to_numpy()[j] - cc[:, 2]
near = d2 < 1500
ax.scatter(depth[near], informed[near], s=3, alpha=0.25, linewidths=0,
           color="#6b4fd8")
bins = np.linspace(0, depth[near].max(), 25)
mid = 0.5 * (bins[1:] + bins[:-1])
med = [np.median(informed[near & (depth >= a) & (depth < b)])
       for a, b in zip(bins[:-1], bins[1:])]
ax.plot(mid, med, "k-", lw=2, label="median")
ax.set_title("knowledge fades with depth — the exclusion-limit logic",
             fontsize=10)
ax.set_xlabel("depth below ground [m]")
ax.set_ylabel("data-informed fraction")
ax.legend(fontsize=8)

ax = axes[1, 0]
ax.loglog(var_cf, var_mc, ".", ms=2, alpha=0.3, color="#3a86a8")
lim = [var_cf.min() * 0.8, var_cf.max() * 1.3]
ax.loglog(lim, lim, "k--", lw=1)
ax.set_title(f"MC gate: {N:,} samples vs closed form — "
             f"median err {mc_stats['median_rel_err']*100:.2f}%", fontsize=10)
ax.set_xlabel("closed-form variance"); ax.set_ylabel("Monte-Carlo variance")

ax = axes[1, 1]
ax.bar(range(1, 21), per_rep, color="#3a86a8")
ax.axhline(0.95, color="r", ls="--", lw=1.2, label="nominal 95%")
ax.set_ylim(0.9, 1.0)
ax.set_title(f"coverage gate: pooled {cov*100:.2f}% (prior-drawn truths)",
             fontsize=10)
ax.set_xlabel("repetition"); ax.set_ylabel("95%-interval coverage")
ax.legend(fontsize=8)

fig.tight_layout()
fig.savefig(ROOT / "figures" / "posterior_uncertainty.png")

stats = {"mc": mc_stats, "mc_pass": bool(mc_ok),
         "coverage_pooled": cov, "coverage_pass": bool(0.93 < cov < 0.97),
         "sigma_min": float(sig.min()), "sigma_max": float(sig.max()),
         "informed_max": float(informed.max()),
         "beta": beta, "rms_mgal": res["rms"]}
(ROOT / "figures" / "posterior_stats.json").write_text(json.dumps(stats, indent=1))
print(json.dumps(stats, indent=1))
print("figure -> figures/posterior_uncertainty.png")
