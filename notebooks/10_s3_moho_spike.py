"""S3: the Moho spike — measure the kill-criterion, then rule.

Declared BEFORE measuring (src/moho.py, from the Phase-2 design note):
  ACCURACY  forward error at working resolution < 20% of the data-noise scale
  SPEED     <= 0.5 s per forward, so 10^5 simulations fit in ~14 h

Run: py -3 notebooks/10_s3_moho_spike.py
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
from src import moho, moho_fast

ROOT = Path(__file__).resolve().parents[1]
SPEED_BUDGET = 0.5          # s per forward (declared)
ACC_FRACTION = 0.20         # of the noise scale (declared)


def timed(fn, repeats=3):
    fn()                                     # warm the JIT
    ts = []
    for _ in range(repeats):
        t0 = time.time(); fn(); ts.append(time.time() - t0)
    return float(np.median(ts))


# ---------------------------------------------------------------- yardstick
g8 = moho.load_gravity(step=8)
dep8 = moho.moho_on_grid(g8["lat"], g8["lon"])
d8 = moho.forward(g8["lat"], g8["lon"], dep8, 1.6, 1.6)
r = g8["sedfree_bouguer"] - d8
noise = float(np.std(r - r.mean()))
print(f"yardstick: published Moho explains {100*(1-np.var(r-r.mean())/np.var(g8['sedfree_bouguer'])):.1f}% "
      f"of the signal; residual scale = {noise:.1f} mGal")
acc_budget = ACC_FRACTION * noise
print(f"accuracy budget = {acc_budget:.1f} mGal | speed budget = {SPEED_BUDGET} s\n")

# ------------------------------------------------- route A: direct tesseroid
direct = []
for step in (16, 12, 8, 6):
    gg = moho.load_gravity(step=step)
    dd = moho.moho_on_grid(gg["lat"], gg["lon"])
    sp = 0.2 * step
    t = timed(lambda: moho.forward(gg["lat"], gg["lon"], dd, sp, sp))
    direct.append(dict(step=step, n=len(gg["lat"]), shape=list(gg["shape"]),
                       deg=sp, t=t))
    print(f"direct  step={step:2d}  {gg['shape']}  n={len(gg['lat']):5d}  "
          f"{t:6.3f} s/forward  {'PASS' if t <= SPEED_BUDGET else 'over budget'}")

# ------------------------------------------- route B: layered sensitivity
step = 8
gg = moho.load_gravity(step=step)
glat, glon, sp = gg["lat"], gg["lon"], 0.2 * step
dd = moho.moho_on_grid(glat, glon)
edges = moho_fast.layer_edges(n_layers=24)

t0 = time.time()
A = moho_fast.build_sensitivity(glat, glon, sp, sp, edges, verbose=False)
t_pre = time.time() - t0
mem = A.nbytes / 1e6
t_mv = timed(lambda: moho_fast.forward(A, dd, edges), repeats=10)
d_fast = moho_fast.forward(A, dd, edges)
d_exact = moho.forward(glat, glon, dd, sp, sp)
disc_err = float(np.sqrt(np.mean((d_fast - d_exact) ** 2)))
print(f"\nlayered  n={len(glat)} x {len(edges)-1} layers: precompute {t_pre:.0f} s, "
      f"{mem:.0f} MB, {t_mv*1e3:.1f} ms/forward")
print(f"         discretisation error vs exact tesseroid: {disc_err:.2f} mGal "
      f"({'PASS' if disc_err <= acc_budget else 'FAIL'} vs {acc_budget:.1f} budget)")

sims = 100_000
print(f"\n10^5 simulations: direct(step8) {direct[2]['t']*sims/3600:.1f} h | "
      f"layered {(t_pre + t_mv*sims)/3600:.2f} h")

verdict = ("PROCEED" if (t_mv <= SPEED_BUDGET and disc_err <= acc_budget)
           else "DEFER")
print(f"\nS3 KILL-CRITERION: {verdict}")

# ---------------------------------------------------------------- figures
fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), dpi=150)

ax = axes[0]
ns = [d["n"] for d in direct]
ts = [d["t"] for d in direct]
ax.loglog(ns, ts, "-o", ms=5, color="#3a86a8", label="direct tesseroid")
ax.axhline(t_mv, color="#4aa86b", lw=2, label=f"layered matvec ({t_mv*1e3:.0f} ms)")
ax.axhline(SPEED_BUDGET, color="r", ls="--", lw=1.3, label="declared budget 0.5 s")
ax.set_xlabel("grid points"); ax.set_ylabel("seconds per forward")
ax.set_title("(A) the compute wall, and the way through", fontsize=10)
ax.legend(fontsize=7)

ax = axes[1]
sc = ax.scatter(glon, glat, c=dep8 / 1000, s=14, cmap="viridis_r", linewidths=0)
fig.colorbar(sc, ax=ax, label="Moho depth [km]")
ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
ax.set_title("(B) the published South American Moho\n(no uncertainties — the gap)",
             fontsize=10)

ax = axes[2]
ax.scatter(d_exact, d_fast, s=8, alpha=0.5, linewidths=0, color="#6b4fd8")
lim = [min(d_exact.min(), d_fast.min()), max(d_exact.max(), d_fast.max())]
ax.plot(lim, lim, "k--", lw=1)
ax.set_xlabel("exact tesseroid forward [mGal]")
ax.set_ylabel("layered matvec forward [mGal]")
ax.set_title(f"(C) the approximation's price: {disc_err:.2f} mGal\n"
             f"(budget {acc_budget:.1f}, data residual {noise:.0f})", fontsize=10)

fig.tight_layout()
fig.savefig(ROOT / "figures" / "s3_moho_spike.png")

stats = {"noise_scale_mgal": noise, "accuracy_budget_mgal": acc_budget,
         "speed_budget_s": SPEED_BUDGET, "direct": direct,
         "layered": {"n_points": len(glat), "n_layers": len(edges) - 1,
                     "precompute_s": t_pre, "memory_mb": mem,
                     "per_forward_s": t_mv, "disc_error_mgal": disc_err},
         "sims_1e5_hours": {"direct_step8": direct[2]["t"] * sims / 3600,
                            "layered": (t_pre + t_mv * sims) / 3600},
         "verdict": verdict}
(ROOT / "figures" / "s3_stats.json").write_text(json.dumps(stats, indent=1))
print("figure -> figures/s3_moho_spike.png")
