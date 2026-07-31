"""Gurutva in space: WHERE deterministic astrodynamics can be trusted.

The pitch this example was written to demonstrate was: "standard tooling
reports one path and a linearised covariance; the truth is a much larger
corridor." We measured it. **In most cislunar regimes that pitch is wrong** —
linearisation is adequate, and Gurutva does not claim otherwise.

What IS defensible, and what this example delivers, is the map: a sweep of
dynamical regimes showing where the linearised (Kalman/STM) answer holds and
where it under-reports. That is a smaller claim than the strategy assumed
and a truer one — and knowing the boundary is the product.

Run: py -3 examples/space_trajectory_uncertainty.py
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
from src.gurutva_core import astro

ROOT = Path(__file__).resolve().parents[1]

# Declared regimes — chosen for dynamical character, not for a nice answer.
REGIMES = [
    ("benign transfer",      [0.72, 0.0, 0.02, 0.05, 0.42, 0.0], 12),
    ("L1 lingering",         [astro.L1_X + 0.002, 0, 0, 0, 0.008, 0], 12),
    ("L1 lingering (20 d)",  [astro.L1_X + 0.002, 0, 0, 0, 0.008, 0], 20),
    ("close lunar flyby",    [0.90, 0.0, 0.005, 0.10, 0.30, 0.0], 12),
    ("deep lunar encounter", [0.95, 0.0, 0.0, 0.05, 0.20, 0.0], 10),
]
N_ENS, POS_KM, VEL_MMS = 600, 1.0, 10.0


def assess(state0, days, n=N_ENS, n_out=180):
    t_span = (0.0, days * 86_400.0 / astro.TU_S)
    t, ens = astro.ensemble(state0, t_span, n=n, pos_km=POS_KM,
                            vel_mms=VEL_MMS, n_out=n_out)
    mean, cov, vol_mc = astro.covariance_tube(ens)
    P0 = np.diag([(POS_KM / astro.LU_KM) ** 2] * 3
                 + [((VEL_MMS * 1e-6) / (astro.LU_KM / astro.TU_S)) ** 2] * 3)
    _, ref, cov_lin, vol_lin = astro.linear_covariance(state0, t_span, P0,
                                                       n_out=n_out)
    sig_mc = np.sqrt(np.trace(cov, axis1=1, axis2=2)) * astro.LU_KM
    sig_lin = np.sqrt(np.trace(cov_lin[:, :3, :3], axis1=1, axis2=2)) * astro.LU_KM
    r_moon = np.linalg.norm(ref[:3].T - np.array([1 - astro.MU, 0, 0]),
                            axis=1) * astro.LU_KM
    return dict(t=t, ens=ens, ref=ref, sig_mc=sig_mc, sig_lin=sig_lin,
                ratio=float(vol_mc[-1] / max(vol_lin[-1], 1e-30)),
                min_moon_km=float(r_moon.min()), days=days)


results = []
for name, s0, days in REGIMES:
    t0 = time.time()
    r = assess(s0, days)
    r["name"] = name
    results.append(r)
    print(f"{name:22s} sigma MC {r['sig_mc'][-1]:8.1f} km | lin "
          f"{r['sig_lin'][-1]:8.1f} | volume ratio {r['ratio']:6.2f}x | "
          f"min lunar dist {r['min_moon_km']:8.0f} km  ({time.time()-t0:.0f}s)")

# ---- what DRIVES the breakdown: horizon and navigation quality --------
DEEP = [0.95, 0.0, 0.0, 0.05, 0.20, 0.0]
drivers = []
for tag, days, pos, vel in [("30 d, 1 km", 30, 1.0, 10.0),
                            ("20 d, 10 km", 20, 10.0, 100.0),
                            ("20 d, 50 km", 20, 50.0, 500.0)]:
    span = (0.0, days * 86_400.0 / astro.TU_S)
    t_, ens_ = astro.ensemble(DEEP, span, n=300, pos_km=pos, vel_mms=vel,
                              n_out=120)
    _, _, vmc = astro.covariance_tube(ens_)
    P0d = np.diag([(pos / astro.LU_KM) ** 2] * 3
                  + [((vel * 1e-6) / (astro.LU_KM / astro.TU_S)) ** 2] * 3)
    _, _, _, vlin = astro.linear_covariance(DEEP, span, P0d, n_out=120)
    drivers.append(dict(tag=tag, ratio=float(vmc[-1] / max(vlin[-1], 1e-30))))
    print(f"  driver {tag:14s} volume ratio {drivers[-1]['ratio']:8.2f}x")

worst = max(results, key=lambda r: r["ratio"])
adequate = [r for r in results if r["ratio"] < 1.25]
print(f"\nlinearisation adequate (<1.25x) in {len(adequate)}/{len(results)} "
      f"regimes; worst case '{worst['name']}' at {worst['ratio']:.2f}x "
      f"(closest approach {worst['min_moon_km']:,.0f} km)")

# ---------------------------------------------------------------- figures
fig = plt.figure(figsize=(16.5, 4.8), dpi=150)

ax = fig.add_subplot(1, 3, 1)
demo = worst
step = max(1, demo["ens"].shape[0] // 250)
for i in range(0, demo["ens"].shape[0], step):
    ax.plot(demo["ens"][i, 0], demo["ens"][i, 1], lw=0.3, alpha=0.15,
            color="#6b4fd8")
ax.plot(demo["ref"][0], demo["ref"][1], "k-", lw=1.6, label="deterministic")
ax.plot(-astro.MU, 0, "o", ms=9, color="#3a86a8", label="Earth")
ax.plot(1 - astro.MU, 0, "o", ms=5, color="#999", label="Moon")
ax.plot(astro.L1_X, 0, "x", ms=7, color="#b3543f", label="L1")
ax.set_xlabel("x [Earth-Moon units, rotating frame]"); ax.set_ylabel("y")
ax.set_title(f"{demo['name']}: {demo['ens'].shape[0]} futures\n"
             f"from one {POS_KM} km navigation error", fontsize=9)
ax.legend(fontsize=6.5); ax.set_aspect("equal")

ax = fig.add_subplot(1, 3, 2)
for r, c in zip(results, ["#c9d3e8", "#8ba7d4", "#5f7fb8", "#c98a3d", "#b3543f"]):
    ax.semilogy(r["t"] * astro.TU_S / 86_400, r["sig_mc"], color=c, lw=1.8,
                label=r["name"])
    ax.semilogy(r["t"] * astro.TU_S / 86_400, r["sig_lin"], color=c, lw=1.0,
                ls="--")
ax.set_xlabel("days"); ax.set_ylabel("1σ position spread [km]")
ax.set_title("solid = true ensemble · dashed = linearised\n"
             "(they track each other almost everywhere)", fontsize=9)
ax.legend(fontsize=6)

ax = fig.add_subplot(1, 3, 3)
names = [r["name"].replace(" (", "\n(") for r in results]
ratios = [r["ratio"] for r in results]
cols = ["#4aa86b" if x < 1.25 else "#b3543f" for x in ratios]
ax.barh(range(len(results)), ratios, color=cols)
ax.axvline(1.0, color="k", ls="--", lw=1.2)
ax.axvline(1.25, color="#b3543f", ls=":", lw=1.2)
ax.set_yticks(range(len(results)))
ax.set_yticklabels(names, fontsize=7)
ax.set_xlabel("corridor volume: true / linearised")
ax.set_title("the honest map: where linearisation\nholds (green) and slips (red)",
             fontsize=9)

fig.tight_layout()
fig.savefig(ROOT / "figures" / "space_uncertainty_corridor.png")

stats = {"nav_pos_km": POS_KM, "nav_vel_mms": VEL_MMS, "members": N_ENS,
         "regimes": [{"name": r["name"], "days": r["days"],
                      "sigma_mc_km": float(r["sig_mc"][-1]),
                      "sigma_linear_km": float(r["sig_lin"][-1]),
                      "volume_ratio": r["ratio"],
                      "min_lunar_km": r["min_moon_km"]} for r in results],
         "finding": ("linearised covariance is adequate in most cislunar "
                     "regimes at 1 km / 10 mm/s over 10-20 days; it "
                     "under-reports only during deep lunar encounters"),
         "drivers": drivers,
         "boundary": ("linearisation is adequate at short horizons (10-20 d) "
                      "with good navigation (1 km/10 mm/s); it under-reports "
                      "3-48x once a deep lunar encounter is combined with a "
                      "30-day horizon or degraded navigation (10-50 km)"),
         "claim_not_made": ("Gurutva does NOT claim standard astrodynamics "
                            "tooling is broken everywhere — measured, and it "
                            "mostly is not. The product is the boundary.")}
(ROOT / "figures" / "space_corridor_stats.json").write_text(
    json.dumps(stats, indent=1))
print("figure -> figures/space_uncertainty_corridor.png")
