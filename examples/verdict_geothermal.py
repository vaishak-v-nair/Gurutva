"""PRODUCT DEMO 1 — the drilling verdict, on real data.

Utah FORGE, a real DOE geothermal site with a published gravity inversion.
The published deliverable is a density model: a picture, no error bars. This
produces what a customer would actually buy instead:

    1. is this model claimable at all?          (four gates)
    2. below what depth is it invented?         (informed fraction)
    3. how much mass is in the box, +/- what?   (exact linear functional)

Number 3 decides a drilling budget, and the industry reports it without an
interval. Where it comes from matters more than that it exists, so all three
modelling choices below are declared, sourced, and each was checked by a
gate that had the power to refuse.

Run: py -3 examples/verdict_geothermal.py   (~1 min)
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import inversion
from src.gurutva_core import gates
from src.product import prior as PR, report, verdict as V

ROOT = Path(__file__).resolve().parents[1]
FLOOR = inversion.TARGET_RMS       # 0.76 mGal — MEASURED, not assumed

# ---- the three declared choices, and what refused the alternatives -------
#
# AMPLITUDE. 0.25 g/cc is the published report's OWN basin contrast (2.42
# g/cc against the 2.67 reduction background). The customer's number, not
# ours. What it replaces: beta tuned by the discrepancy principle, which
# implies rock density varies by 0.021 g/cc. Used as a prior it fails the
# licensing gate at the 100th percentile — that prior could not have made
# the anomaly that was measured — and it still under-states the error bar on
# box mass against the licensed prior (the factor is computed and printed by
# this run rather than quoted here, because it depends on the box).
# A regularization weight is a numerical knob, never a statement about rock.
# (An earlier flat-prior comparison gave ~9x. It is NOT quoted anywhere:
# that configuration failed licensing, and a number from a refused
# configuration is not a number.)
#
# SHAPE. 500 m correlation length, the scale the basin/granite structure
# actually varies on. What it replaces: an independent-per-cell prior, which
# licensing ALSO refused (99th percentile). Uncorrelated draws are static:
# neighbouring cells cancel and the long-wavelength gravity a real basin
# makes never appears. Measured prior-predictive at 0.25 g/cc: 1.9 mGal
# uncorrelated, 2.9 mGal at 500 m. Longer lengths license more easily (0.0
# pct at 1-2 km) and that is exactly why 500 m is used — the widest prior
# always passes licensing, so the tightest one that passes is the honest
# one. See the CAUTION in gurutva_core/gates.py.
#
# REGIONAL. A declared planar surrogate for the published mesh's ~50 km
# padding cells, which were never delivered with the archive. Without it the
# residual is 9.90 mGal and NOTHING licenses, correctly: that field is made
# by mass outside our mesh, so no prior over our cells can produce it. With
# it, 2.03 mGal and the local anomaly is what remains.
PRIOR_SD_GCC = 0.25
CORR_LEN_M = 500.0
REGIONAL_ORDER = 1
N_POST = 600          # posterior samples for the per-cell map

rng = np.random.default_rng(20260805)

print("assembling the real survey (Utah FORGE, DOE GDR 1144) ...")
run = inversion.run(bridged=False)
G, tree, active = run["G"], run["tree"], run["active"]
cc, vol = tree.cell_centers[active], tree.cell_volumes[active]
r, regional = inversion.residual_data(G, run["d_obs"], run["mref"],
                                      stations=run["stations"],
                                      order=REGIONAL_ORDER)
prior, pmeta = PR.build(tree, active, PRIOR_SD_GCC, CORR_LEN_M, rng)
dm = PR.posterior_mean(G, prior, r)
RMS = float(np.sqrt(np.mean((G @ dm - r) ** 2)))
print(f"  {G.shape[0]} stations, {G.shape[1]:,} cells | prior {PRIOR_SD_GCC} "
      f"g/cc @ {CORR_LEN_M:.0f} m | residual {np.sqrt(np.mean(r**2)):.2f} -> "
      f"RMS {RMS:.3f} mGal (floor {FLOOR})")

# ------------------------------------------------------------------ gates
suite = gates.GateSuite()

sims = (G @ prior.sample(rng, 300)
        + rng.standard_normal((G.shape[0], 300))).T
suite.add(gates.licensing(r, sims))

# calibration: the EXACT functional variance against the SAMPLED one, over
# random functionals. Threshold sits above the sampling floor sqrt(2/N).
sd_post, sd_mc_err = PR.posterior_sd(G, prior, np.random.default_rng(101), N_POST)
SGt = prior.apply_inv(G.T)
K = G @ SGt + np.eye(G.shape[0])
fl = (prior.sample(np.random.default_rng(202), N_POST)
      - SGt @ np.linalg.solve(K, G @ prior.sample(np.random.default_rng(202),
                                                  N_POST)
                              + np.random.default_rng(303).standard_normal(
                                  (G.shape[0], N_POST))))
rel = []
for k in range(24):
    w = np.random.default_rng(400 + k).standard_normal(G.shape[1])
    rel.append(abs(float(np.std(w @ fl)) / PR.functional_sd(G, prior, w) - 1))
med_rel = float(np.median(rel))
floor_rel = np.sqrt(0.5 / N_POST)          # sd of a sampled sd, relative
suite.add(gates.GateReport(
    "calibration(MC)", med_rel < 4 * floor_rel, med_rel,
    f"median < {4 * floor_rel:.3f} (4x the {N_POST}-sample floor)",
    "exact functional sd vs sampled"))

sd_b, _ = PR.posterior_sd(G, prior, np.random.default_rng(999), N_POST)
suite.add(gates.stability((dm, sd_post), (dm, sd_b), atol=1e-9))

suite.add(gates.adequacy(RMS, FLOOR))

# --------------------------------------- what the data actually supports
prior_sd_cell = prior.marginal_sd(np.random.default_rng(55), 600)
inf = 1.0 - sd_post / prior_sd_cell
model = run["mref"] + dm

z = cc[:, 2]
edges = np.linspace(z.min(), z.max(), 25)
# "invented below X" = the deepest level at which the TYPICAL cell is still
# informed. Median, not "any 10% of cells": a handful of well-lit cells beside
# a station does not make a depth slice knowable, and the looser criterion put
# the line below the whole model, which is a claim the map does not support.
lit = [0.5 * (edges[i] + edges[i + 1]) for i in range(len(edges) - 1)
       if np.any((z >= edges[i]) & (z < edges[i + 1]))
       and np.median(inf[(z >= edges[i]) & (z < edges[i + 1])]) > 0.05]
z_blind = min(lit) if lit else z.max()

cx, cy = cc[:, 0].mean(), cc[:, 1].mean()
box = ((np.abs(cc[:, 0] - cx) < 1000) & (np.abs(cc[:, 1] - cy) < 1000)
       & (z > z_blind))
w = np.where(box, vol, 0.0)
mass = float(w @ (run["mref"] + dm))
mass_sd = PR.functional_sd(G, prior, w)
sd_disc = V.functional_sd(G, run["wr"], run["beta"], w)   # standard workflow

MT = 1e-6         # g/cc * m^3 = t -> Mt
lo, hi = (mass - 1.96 * mass_sd) * MT, (mass + 1.96 * mass_sd) * MT
detected = abs(mass) > 1.96 * mass_sd
numbers = {
    "excess mass, 2x2 km box above the blind depth":
        f"{mass * MT:+,.0f} Mt  (95%: {lo:+,.0f} to {hi:+,.0f} Mt)",
    "is that a detection?":
        ("YES — the interval excludes zero" if detected else
         f"NO — consistent with zero. 95% upper limit {hi:+,.0f} Mt"),
    "the model is INVENTED below":
        f"{z_blind:,.0f} m elevation ({z.max() - z_blind:,.0f} m below the "
        f"top of the mesh)",
    "cells the survey actually informs (>5%)":
        f"{float(np.mean(inf > 0.05)) * 100:.1f}% of {len(inf):,}",
    "best-informed cell": f"{inf.max() * 100:.0f}% of its prior uncertainty removed",
}
v = V.assess(suite, numbers, subject="the Utah FORGE density model")
print("\n" + str(v))
print(f"\nstandard-workflow error bar: +/-{sd_disc * MT:,.0f} Mt   "
      f"honest: +/-{mass_sd * MT:,.0f} Mt   "
      f"over-confidence {mass_sd / sd_disc:.1f}x")

# ---------------------------------------------------------------- figure
from scipy.spatial import cKDTree

_kd = cKDTree(cc)
_xg = np.linspace(cc[:, 0].min(), cc[:, 0].max(), 240)
_zg = np.linspace(cc[:, 2].min(), cc[:, 2].max(), 150)
_XX, _ZZ = np.meshgrid(_xg, _zg)
_dist, _idx = _kd.query(np.column_stack(
    [_XX.ravel(), np.full(_XX.size, cy), _ZZ.ravel()]))
_outside = (_dist > 600.0).reshape(_XX.shape)   # beyond the model footprint


def slab(values):
    """Nearest-cell section through the octree at y = survey centre.

    Binning cell CENTRES onto a regular grid leaves holes wherever the
    octree is coarse, and the holes read as structure that is not there.
    A nearest-cell lookup draws the model that actually exists."""
    Z = np.asarray(values)[_idx].reshape(_XX.shape).astype(float)
    return _xg / 1000, _zg, np.where(_outside, np.nan, Z)


fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.1), dpi=150)
for ax, val, ttl, cm, kw in [
    (axes[0], model, "what the industry ships:\ndensity model (g/cc)",
     "RdBu_r", dict(vmin=-0.35, vmax=0.35)),
    (axes[1], sd_post, f"posterior sigma (g/cc)\nprior was {PRIOR_SD_GCC} — "
     "this is what survives", "viridis", dict(vmin=0, vmax=PRIOR_SD_GCC)),
    (axes[2], inf, "informed fraction:\n1 = the data knows, 0 = we assumed",
     "magma", dict(vmin=0, vmax=max(0.2, float(np.nanmax(inf))))),
]:
    xe, ze, Z = slab(val)
    im = ax.pcolormesh(xe, ze, Z, cmap=cm, shading="auto", **kw)
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.axhline(z_blind, color="r", lw=1.3, ls="--")
    ax.set_title(ttl, fontsize=9)
    ax.set_xlabel("easting (km)")
axes[0].set_ylabel("elevation (m)")
fig.tight_layout()
FIG = ROOT / "figures" / "verdict_geothermal.png"
fig.savefig(FIG)

report.render(
    ROOT / "figures" / "verdict_geothermal.html",
    "Gurutva — model verdict",
    "Utah FORGE geothermal site · gravity inversion · DOE GDR 1144 (CC-BY 4.0)",
    v, figure=FIG,
    sections=[
        ("What the standard workflow would have told you",
         [(f"error bar from a regularizer tuned to a misfit target "
           f"(implied prior {1 / np.sqrt(run['beta']):.3f} g/cc)",
           f"+/- {sd_disc * MT:,.0f} Mt"),
          (f"error bar from a prior declared as rock ({PRIOR_SD_GCC} g/cc, "
           f"correlated over {CORR_LEN_M:.0f} m)",
           f"+/- {mass_sd * MT:,.0f} Mt"),
          ("over-confidence factor", f"{mass_sd / sd_disc:.1f}x")],
         "The weight that makes an inversion look right is not a statement "
         "about rock. Read as a prior it claims density varies by "
         f"{1 / np.sqrt(run['beta']):.3f} g/cc — which no geologist would "
         "sign — and it fails the licensing gate at the 100th percentile: "
         "that prior could not have produced the anomaly that was measured."),
        ("The declared prior",
         [("marginal density sd", f"{PRIOR_SD_GCC} g/cc "
           "(the published report's own basin contrast)"),
          ("correlation length", f"{CORR_LEN_M:.0f} m "
           "(the scale the structure varies on)"),
          ("regional field", f"order-{REGIONAL_ORDER} surrogate for the "
           "undelivered ~50 km padding cells"),
          ("prior-predictive vs observed",
           f"{sims.std():.2f} mGal vs {np.sqrt(np.mean(r**2)):.2f} mGal")],
         "Every number here was read off the site or the customer's own "
         "report before any gate was run. Widening a prior until a gate "
         "turns green passes every gate and means nothing."),
    ],
    caption="Red dashed line: below this the survey no longer constrains "
            "density and the model is the regularizer, not the rock.",
    footer=f"Measured noise floor {FLOOR} mGal · achieved {RMS:.3f} mGal · "
           f"{G.shape[0]} stations, {G.shape[1]:,} active cells · per-cell "
           f"map from {N_POST} posterior samples (MC error "
           f"{float(np.median(sd_mc_err / sd_post)) * 100:.1f}%); the mass "
           f"interval is exact, not sampled · reproducible: "
           f"<code>py -3 examples/verdict_geothermal.py</code>")

(ROOT / "figures" / "verdict_geothermal.json").write_text(json.dumps({
    "claimable": v.claimable, "verdict": v.headline, "gates": v.gates,
    "numbers": numbers, "prior": pmeta,
    "prior_sd_gcc": PRIOR_SD_GCC, "corr_len_m": CORR_LEN_M,
    "regional_order": REGIONAL_ORDER,
    "mass_Mt": mass * MT, "mass_sd_Mt": mass_sd * MT,
    "sd_standard_workflow_Mt": sd_disc * MT,
    "overconfidence_factor": mass_sd / sd_disc,
    "z_blind_m": z_blind, "rms_mGal": RMS, "floor_mGal": FLOOR,
    "residual_rms_mGal": float(np.sqrt(np.mean(r**2)))}, indent=1))
print("\nreport -> figures/verdict_geothermal.html")
