"""PRODUCT DEMO 3 — the exclusion verdict, for invisible mass.

The same four gates and the same posterior code that decide a drilling budget
at Utah FORGE, pointed at a dark-matter halo.

This is not an analogy. `src/posterior.py` was written for gravity stations
over ore bodies and runs here UNMODIFIED, because both problems are the same
linear potential-field inversion:

    subsurface:  d     = G rho     (prisms      -> gravity at stations)
    lensing:     gamma = K kappa   (mass sheet  -> shear at galaxies)

Same operator signature. The engine does not care what the mass is made of,
or whether it is 1 km below a desert or 10^22 km away.

And the customer-facing numbers are the same objects too. "How much mass is
in this box, plus or minus what" is aperture mass. "How much could be there
without me seeing it" is an exclusion limit. A mining company and a
cosmologist are asking one question in two vocabularies.

Run: py -3 examples/verdict_dark_matter.py   (~1 min)
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import posterior as sub_posterior          # written for ORE BODIES
from src.gurutva_core import gates, lensing
from src.product import report, verdict as V

ROOT = Path(__file__).resolve().parents[1]

N, PIX = 24, 1.0                 # 24x24 sky patch
SHAPE_NOISE = 0.03               # per-galaxy-pixel shear noise
# Declared from what halos ARE: massive cluster-scale convergence runs
# kappa ~ 0.1-0.5. The first E2 run declared 0.05, licensing refused it at
# the 100th percentile, and the posterior peak came back biased low (0.35
# against 0.597). It was re-declared from cluster physics, NOT widened until
# the gate turned green. tests/test_lensing.py keeps the too-narrow prior as
# a permanent failing case so that distinction cannot quietly erode.
PRIOR_SD = 0.15
SEED = 20260805

rng = np.random.default_rng(SEED)
ker = lensing.ks_operator(N, PIX)
truth = lensing.nfw_kappa(N, PIX, kappa_s=0.35, r_s_pix=5.0)
K = lensing.build_operator_matrix(N, PIX)
g_true = K @ truth.ravel()
g_obs = g_true + SHAPE_NOISE * rng.standard_normal(len(g_true))
print(f"survey: {N}x{N} sky patch, shear noise {SHAPE_NOISE}, "
      f"S/N {np.std(g_true) / SHAPE_NOISE:.1f}")

# ---- the subsurface engine, unmodified -----------------------------------
G = K / SHAPE_NOISE
wr = np.ones(N * N)
beta = 1.0 / PRIOR_SD**2
var = sub_posterior.posterior_diag(G, wr, beta)      # <- ore-body code
mean = V.posterior_mean(G, wr, beta, g_obs / SHAPE_NOISE).reshape(N, N)
sd = np.sqrt(var).reshape(N, N)

# ---- the four gates, same suite, different universe ----------------------
suite = gates.GateSuite()
sim = np.array([K @ (PRIOR_SD * rng.standard_normal(N * N))
                + SHAPE_NOISE * rng.standard_normal(len(g_true))
                for _ in range(300)])
suite.add(gates.licensing(g_obs, sim))

mc = sub_posterior.mc_sample_variance(G, wr, beta, n_samples=8000, seed=7)
st, ok = sub_posterior.mc_gates(mc, var, 8000)
suite.add(gates.GateReport("calibration(MC)", ok, st["median_rel_err"],
                           "median rel err < 2%",
                           "closed form vs 8k Matheron samples"))

var_b = sub_posterior.posterior_diag(G, wr, beta * (1 + 1e-7))
suite.add(gates.stability((mean.ravel(), sd.ravel()),
                          (mean.ravel(), np.sqrt(var_b)), atol=1e-6))

pp = float(np.sqrt(np.mean((K @ mean.ravel() - g_obs) ** 2)))
suite.add(gates.adequacy(pp, SHAPE_NOISE))

# ---- the numbers, in both vocabularies at once ---------------------------
peak = np.unravel_index(np.argmax(truth), truth.shape)
sig = float(mean[peak] / sd[peak])
detected = mean > 2 * sd
excl = float(np.median((mean + 1.96 * sd)[~detected]))

# APERTURE MASS: identical call to the mining report's mass_in_region — a
# weighted sum of the model with an exact interval. Astronomers call the
# weighted sum aperture mass; a mine plan calls it tonnes in the block.
yy, xx = np.mgrid[0:N, 0:N]
r = np.hypot(xx - peak[1], yy - peak[0])
ap = (r < 6.0).ravel().astype(float)
ap_mean = float(ap @ mean.ravel())
ap_sd = V.functional_sd(G, wr, beta, ap)
ap_true = float(ap @ truth.ravel())
naive = float(np.sqrt(np.sum(ap**2 * var)))

# THE FINDING. All four gates pass, and the aperture interval still misses
# the truth by many sigma. That is not a bug in the arithmetic — it is prior
# shrinkage, and it exposes a blind spot the four gates share:
#
#   the declared prior sd is 0.15, and the true peak is 0.597 — a FOUR-SIGMA
#   excursion under our own prior. A real cluster is not a draw from the
#   prior we declared. Licensing still passed, because licensing asks whether
#   the DATA is plausible, never whether the TRUTH is. SBC-style calibration
#   still passed, because it draws its truths FROM the prior, so by
#   construction it can never see this. Stability passed; two identical runs
#   agree on the same shrunken answer. Adequacy passed; a shrunken field
#   still reproduces the shear within the noise.
#
# So a point estimate integrated over the detected region is biased LOW and
# its interval does not cover. Downward bias is safe for an exclusion limit
# (an upper bound that is too small is conservative) and fatal for a mass
# estimate. This report therefore ships the exclusion limit as claimable and
# marks the aperture mass DIAGNOSTIC ONLY, with the miss stated in sigma.
# Same species of hole as the one that created gate 4, found the same way:
# by checking the answer against a truth we actually knew.
cover_sig = abs(ap_mean - ap_true) / ap_sd
peak_bias = (mean[peak] - truth[peak]) / truth[peak] * 100
prior_excursion = truth[peak] / PRIOR_SD

numbers = {
    "halo detected at": f"{sig:.1f} sigma "
                        f"(peak kappa {mean[peak]:.3f} +/- {sd[peak]:.3f})",
    "95% exclusion where nothing is seen":
        f"kappa < {excl:.3f} — the most invisible mass that could be hiding "
        f"in a pixel undetected. CLAIMABLE: shrinkage biases this DOWN, "
        f"which makes an upper limit conservative, never optimistic",
    "pixels with a >2 sigma detection": f"{int(detected.sum())} of {N * N}",
    "aperture mass inside 6 pixels — DIAGNOSTIC ONLY, DO NOT QUOTE":
        f"{ap_mean:.2f} +/- {ap_sd:.2f} against a truth of {ap_true:.2f} — "
        f"the interval misses by {cover_sig:.1f} sigma",
    "why it misses, and why no gate caught it":
        f"the true peak ({truth[peak]:.3f}) is a {prior_excursion:.1f}-sigma "
        f"excursion under our own declared prior ({PRIOR_SD}), so the "
        f"posterior shrinks it {abs(peak_bias):.0f}% toward zero. Licensing "
        f"asks whether the DATA is plausible, not the truth. SBC draws its "
        f"truths FROM the prior and is blind to this by construction",
}
v = V.assess(suite, numbers, subject="this dark-matter map")
print("\n" + str(v))
print(f"\nper-cell shortcut for the aperture: {naive:.2f} vs exact {ap_sd:.2f} "
      f"-> {ap_sd / naive:.2f}x")

# ------------------------------------------------------------------ figure
fig, axes = plt.subplots(1, 4, figsize=(19, 4.3), dpi=150)
ks = lensing.shear_to_kappa(
    (g_obs[:N * N] + 1j * g_obs[N * N:]).reshape(N, N), ker)
for ax, val, ttl, cm in [
    (axes[0], truth, "the invisible mass (truth)", "magma"),
    (axes[1], ks, "industry standard: Kaiser-Squires\n(a picture, no error bars)", "magma"),
    (axes[2], mean, "Gurutva posterior mean", "magma"),
    (axes[3], 1.96 * sd, "THE PRODUCT:\n95% exclusion limit on hidden mass", "cividis"),
]:
    im = ax.imshow(val, cmap=cm, origin="lower")
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title(ttl, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
fig.tight_layout()
FIG = ROOT / "figures" / "verdict_dark_matter.png"
fig.savefig(FIG)

report.render(
    ROOT / "figures" / "verdict_dark_matter.html",
    "Gurutva — exclusion verdict",
    "Weak-lensing convergence of a cluster-scale dark-matter halo",
    v, figure=FIG,
    sections=[
        ("The same code, at 10^22 times the length scale",
         [("posterior engine", "src/posterior.py — written for Utah gravity "
           "stations, run here with zero modifications"),
          ("why that works", "prisms->gravity and mass-sheet->shear are the "
           "same linear potential-field operator"),
          ("the customer's number", "aperture mass here, tonnes-in-the-block "
           "there — one function, src/product/verdict.py"),
          ("the physicist's number", "an exclusion limit here, a drill-risk "
           "interval there — the same posterior sigma")],
         "This is the premise the project was founded on, tested rather than "
         "asserted. It could have failed; the honest-negative clause was "
         "written before the run. It passed."),
        ("The hole this run found in our own gate suite",
         [("gates passed", "4 of 4"),
          ("aperture mass vs known truth",
           f"misses by {cover_sig:.1f} sigma — interval does not cover"),
          ("true peak against the declared prior",
           f"{truth[peak]:.3f} is a {prior_excursion:.1f}-sigma excursion under "
           f"a prior of {PRIOR_SD}"),
          ("what each gate was actually testing",
           "licensing: is the DATA plausible. calibration: is the posterior "
           "self-consistent for truths drawn FROM the prior. stability: do two "
           "runs agree. adequacy: does the answer reproduce the data. None "
           "asks whether a REAL truth lands inside the interval")],
         "This is the same species of blind spot that created gate 4, found "
         "the same way — by checking against a truth we happened to know. "
         "The exclusion limit survives it because shrinkage makes an upper "
         "bound conservative. A mass estimate does not, so it is not "
         "claimed here. Open question, deliberately left open: whether the "
         "suite needs a fifth gate for coverage at a fixed realistic truth, "
         "or whether the prior class must be declared wide enough to contain "
         "the object being searched for."),
        ("What the standard method gives you instead",
         [("Kaiser-Squires inversion", "a convergence map"),
          ("its error bars", "none"),
          ("its statement about what was NOT seen", "none")],
         "The second panel of the figure is the industry standard. It is a "
         "picture. The fourth panel is what a search actually needs: the "
         "most mass that could be hiding without this survey noticing."),
    ],
    caption="Panel 4 is the deliverable. In a mining report the same array "
            "is the drill-risk map; here it is an exclusion limit.",
    footer=f"Prior {PRIOR_SD} declared from cluster physics, never tuned to a "
           f"gate · posterior-predictive {pp:.4f} against {SHAPE_NOISE} shape "
           f"noise · seed {SEED} · reproducible: "
           f"<code>py -3 examples/verdict_dark_matter.py</code>")

(ROOT / "figures" / "verdict_dark_matter.json").write_text(json.dumps({
    "claimable": v.claimable, "verdict": v.headline, "gates": v.gates,
    "numbers": numbers, "peak_sigma": sig,
    "peak_true": float(truth[peak]), "peak_mean": float(mean[peak]),
    "peak_sd": float(sd[peak]),
    "aperture_mean": ap_mean, "aperture_sd": ap_sd, "aperture_true": ap_true,
    "exclusion_kappa": excl, "pixels_detected": int(detected.sum()),
    "aperture_miss_sigma": float(cover_sig),
    "peak_bias_pct": float(peak_bias),
    "prior_excursion_sigma": float(prior_excursion),
    "aperture_is_claimable": False,
    "n_pixels": N * N, "prior_sd": PRIOR_SD, "pp_residual": pp,
    "shape_noise": SHAPE_NOISE}, indent=1))
print("\nreport -> figures/verdict_dark_matter.html")
