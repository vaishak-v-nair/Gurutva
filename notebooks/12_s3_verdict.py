"""S3 verdict: why three passing gates were not enough.

Uses the saved posterior samples (no retraining). Produces the honest
figure: the model-class ceiling, the gate scoreboard, and the uncertainty
field labelled for what it is — a diagnostic, not a claim.

Run: py -3 notebooks/12_s3_verdict.py
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import moho, moho_infer as mi

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"

S = mi.build()
mi.fit_projection(S)
glat, glon, obs = S["glat"], S["glon"], S["obs"]
saved = np.load(FIG / "s3_posterior_samples.npz")
depth_mean = saved["depth_mean"] / 1000.0
depth_sd = saved["depth_sd"] / 1000.0
dep_pub = moho.moho_on_grid(glat, glon)


def dc_rms(d):
    r = d - obs
    return float(np.sqrt(np.mean((r - r.mean()) ** 2)))


def rbf(nlon, nlat, ell):
    lons = np.linspace(glon.min(), glon.max(), nlon)
    lats = np.linspace(glat.min(), glat.max(), nlat)
    C = np.array([(lo, la) for lo in lons for la in lats])
    d2 = ((glon[:, None] - C[None, :, 0]) ** 2
          + (glat[:, None] - C[None, :, 1]) ** 2)
    return np.exp(-0.5 * d2 / ell**2)


# ------------------------------------------------------- the ceiling ladder
EXACT = dc_rms(moho.forward(glat, glon, dep_pub, S["deg"], S["deg"]))
ladder = [dict(label="exact model\n(30,351 params)", n=30351, rms=EXACT)]
for nlon, nlat, ell in [(6, 8, 7.0), (10, 13, 4.0), (16, 21, 2.5), (24, 31, 1.6)]:
    P = rbf(nlon, nlat, ell)
    w, *_ = np.linalg.lstsq(P, dep_pub - mi.MEAN_DEPTH, rcond=None)
    dfit = np.clip(mi.MEAN_DEPTH + P @ w, 6_000, 79_000)
    ladder.append(dict(label=f"{P.shape[1]} RBF\n@{ell:.1f}°",
                       n=P.shape[1],
                       rms=dc_rms(moho.forward(glat, glon, dfit, S["deg"], S["deg"])),
                       depth_err_km=float(np.median(np.abs(dfit - dep_pub)) / 1000)))
OURS_PP = 120.7           # measured, notebooks/11_s3_full.py
OUR_CEILING = ladder[1]["rms"]
for l in ladder:
    print(f"{l['label'].replace(chr(10),' '):24s} {l['rms']:6.1f} mGal")
print(f"our posterior achieved:  {OURS_PP:6.1f} mGal (own class ceiling {OUR_CEILING:.1f})")

# ---------------------------------------------------------------- figures
fig, axes = plt.subplots(1, 4, figsize=(20, 4.8), dpi=150)

ax = axes[0]
xs = [l["n"] for l in ladder[1:]]
ys = [l["rms"] for l in ladder[1:]]
ax.semilogx(xs, ys, "-o", ms=6, color="#3a86a8", label="best possible with N coefficients")
ax.axhline(EXACT, color="#4aa86b", ls="--", lw=1.5, label=f"exact model ({EXACT:.0f} mGal)")
ax.axhline(OURS_PP, color="#b3543f", lw=2, label=f"our posterior ({OURS_PP:.0f})")
ax.scatter([48], [OUR_CEILING], s=90, color="#c98a3d", zorder=5,
           label=f"our class ceiling ({OUR_CEILING:.0f})")
ax.set_xlabel("basis coefficients"); ax.set_ylabel("gravity misfit [mGal, DC-removed]")
ax.set_title("(A) the ceiling: even the CORRECT Moho,\nexpressed in our basis, misfits by 62 mGal",
             fontsize=9)
ax.legend(fontsize=6.5)

ax = axes[1]
sc = ax.scatter(glon, glat, c=depth_mean, s=22, cmap="viridis_r", marker="s", linewidths=0)
fig.colorbar(sc, ax=ax, label="km")
ax.set_title("(B) posterior mean Moho — DIAGNOSTIC ONLY\n(model class inadequate; not a map)",
             fontsize=9)
ax.set_xlabel("longitude"); ax.set_ylabel("latitude")

ax = axes[2]
sc = ax.scatter(glon, glat, c=depth_sd, s=22, cmap="magma", marker="s", linewidths=0)
fig.colorbar(sc, ax=ax, label="km")
ax.set_title("(C) posterior σ — NOT CLAIMED\ncalibrated for a model that can't fit the data",
             fontsize=9)
ax.set_xlabel("longitude"); ax.set_ylabel("latitude")

ax = axes[3]
names = ["licensing\n(OOD 0.0%)", "calibration\n(SBC 5.1/88.7)",
         "stability\n(0.62 km)", "ADEQUACY\n(121 vs 19 mGal)"]
vals = [1, 1, 1, 0]
ax.barh(range(4), [1, 1, 1, 1], color="#e8e8ee")
ax.barh(range(4), vals, color=["#4aa86b", "#4aa86b", "#4aa86b", "#b3543f"])
for i, (n_, v) in enumerate(zip(names, vals)):
    ax.text(0.02, i, n_ + ("  PASS" if v else "  FAIL"), va="center",
            fontsize=8, color="white" if v else "#3a1010", fontweight="bold")
ax.set_yticks([]); ax.set_xticks([])
ax.set_title("(D) three gates passed.\nThe fourth is why nothing is claimed.", fontsize=9)

fig.tight_layout()
fig.savefig(FIG / "s3_verdict.png")

stats = {"ceiling_ladder": ladder, "exact_model_mgal": EXACT,
         "our_class_ceiling_mgal": OUR_CEILING, "our_posterior_mgal": OURS_PP,
         "gates": {"licensing": "PASS (0.0 pct)", "sbc": "PASS (5.1/88.7)",
                   "stability": "PASS (0.62 km, sd ratio 0.989)",
                   "adequacy": "FAIL (120.7 vs 19.3 mGal)"},
         "verdict": "NO MOHO UNCERTAINTY MAP CLAIMED — model class inadequate",
         "path_forward_coefficients": 744}
(FIG / "s3_verdict_stats.json").write_text(json.dumps(stats, indent=1))
print("figure -> figures/s3_verdict.png")
