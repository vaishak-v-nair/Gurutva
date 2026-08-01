"""PRODUCT DEMO 2 — the containment verdict, for CO2 storage.

Storing CO2 underground is regulated. The operator must demonstrate
CONFORMANCE (the plume is where the model says) and CONTAINMENT (it has not
left the storage complex), and under the EU guidance that uncertainty is the
operator's risk to carry. So the filing question is not "where is the CO2".
It is:

    "How much CO2 could be outside the complex without my survey seeing it?"

That is an exclusion limit, and it is the same object as demo 1's drill
interval and as a dark-matter upper limit — one engine, three markets.

WHAT THIS IS, EXACTLY. A survey-capability analysis with published Sleipner
monitoring parameters. The seafloor gravimetry programme at Sleipner is the
reference case for the method: 30 seafloor benchmarks at ~500 m spacing over
the plume, surveyed 2002/2005/2009/2013, intra-survey repeatability ~3 uGal
early and ~1.1 uGal by 2013 (Alnes et al. 2008; Nooner et al. 2007). Utsira
reservoir near ~1000 m depth, porosity ~0.36, in-situ CO2 ~700 kg/m3 against
~1050 kg/m3 brine.

WHAT THIS IS NOT. It is not Sleipner's measured gravity data. The Sleipner
2019 Benchmark Model is public on CO2DataShare but under a licence that must
be accepted per user and forbids selling the material, so it is deliberately
not used here. The observations below are simulated from a declared plume,
and every number is therefore a statement about WHAT THIS SURVEY DESIGN CAN
PROVE — which is exactly what an operator needs before committing to a
monitoring programme, and what a regulator needs to judge one. Swapping in a
licensed real dataset changes the inputs, not a line of the method.

Run: py -3 examples/verdict_carbon_storage.py   (~1 min)
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import prism_forward as pf
from src.gurutva_core import gates
from src.product import prior as PR, report, verdict as V

ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------- published parameters
WATER_DEPTH = 85.0          # m, station elevation is -WATER_DEPTH (z-up)
RES_TOP, RES_BOT = -950.0, -1050.0
PHI = 0.36                  # Utsira porosity
RHO_CO2, RHO_BRINE = 700.0, 1050.0        # kg/m3, in situ
NOISE_UGAL = 3.0            # intra-survey repeatability, 2002-2009 epochs
N_STATION_SIDE = (6, 5)     # 30 benchmarks
STATION_SPACING = 500.0
INJECTED_MT = 10.0          # declared inventory to reconcile against
COMPLEX_R = 1500.0          # licensed storage complex radius

# Bulk density contrast of a fully CO2-saturated cell. This is a hard
# physical ceiling: no cell can be lighter than brine fully replaced by CO2.
MAX_CONTRAST = PHI * (RHO_CO2 - RHO_BRINE) / 1000.0      # g/cc, negative
# Declared prior: amplitude is that ceiling (any cell COULD be saturated),
# correlation length is the benchmark spacing, which is also the scale the
# survey can resolve. Both read off the site, neither tuned to a gate.
PRIOR_SD = abs(MAX_CONTRAST)
CORR_LEN = 500.0
NOISE = NOISE_UGAL * 1e-3   # mGal

# --------------------------------------------------------------- the model
nx, ny, nz = 30, 30, 5
dx = dy = 200.0
dz = (RES_TOP - RES_BOT) / nz
xs = (np.arange(nx) - (nx - 1) / 2) * dx
ys = (np.arange(ny) - (ny - 1) / 2) * dy
zs = RES_BOT + dz / 2 + np.arange(nz) * dz
CX, CY, CZ = np.meshgrid(xs, ys, zs, indexing="ij")
centers = np.column_stack([CX.ravel(), CY.ravel(), CZ.ravel()])
dims = np.tile([dx, dy, dz], (len(centers), 1))
vol = np.full(len(centers), dx * dy * dz)
rad = np.hypot(centers[:, 0], centers[:, 1])


def top(a):
    """Vertical sum onto the map view: per-column totals, the way a plume
    inventory is actually reported."""
    return a.reshape(nx, ny, nz).sum(axis=2).T


sx = (np.arange(N_STATION_SIDE[0]) - (N_STATION_SIDE[0] - 1) / 2) * STATION_SPACING
sy = (np.arange(N_STATION_SIDE[1]) - (N_STATION_SIDE[1] - 1) / 2) * STATION_SPACING
SX, SY = np.meshgrid(sx, sy, indexing="ij")
stations = np.column_stack([SX.ravel(), SY.ravel(),
                            np.full(SX.size, -WATER_DEPTH)])

print(f"survey: {len(stations)} seafloor benchmarks at {STATION_SPACING:.0f} m, "
      f"{NOISE_UGAL} uGal repeatability")
G = pf.prism_matrix(stations, centers, dims)

# declared truth: a radially tapered plume inside the complex, scaled so the
# CO2 inventory equals the declared injected mass.
shape_ = np.exp(-(rad / 900.0) ** 2) * np.exp(-((centers[:, 2] + 990.0) / 45.0) ** 2)
co2_per_unit = (shape_ * vol * PHI * RHO_CO2).sum()          # kg per unit sat
sat = shape_ * (INJECTED_MT * 1e9 / co2_per_unit)
truth = sat * MAX_CONTRAST                                    # g/cc, negative
rng = np.random.default_rng(20260805)
d_obs = G @ truth + NOISE * rng.standard_normal(len(stations))
print(f"  plume: peak saturation {sat.max():.2f}, signal "
      f"{np.abs(G @ truth).max() * 1e3:.1f} uGal peak, "
      f"S/N {np.std(G @ truth) / NOISE:.1f}")

# ------------------------------------------------------------------- gates
prior, pmeta = PR.build_regular((nx, ny, nz), (dx, dy, dz), PRIOR_SD,
                                CORR_LEN, rng)
Gw = G / NOISE
dw = d_obs / NOISE
mean = PR.posterior_mean(Gw, prior, dw)
RMS = float(np.sqrt(np.mean((G @ mean - d_obs) ** 2)))

suite = gates.GateSuite()
sims = (Gw @ prior.sample(rng, 300)
        + rng.standard_normal((len(stations), 300))).T
suite.add(gates.licensing(dw, sims))

N_POST = 800
sd_post, sd_err = PR.posterior_sd(Gw, prior, np.random.default_rng(11), N_POST)
SGt = prior.apply_inv(Gw.T)
K = Gw @ SGt + np.eye(len(stations))
mm = prior.sample(np.random.default_rng(22), N_POST)
fl = mm - SGt @ np.linalg.solve(
    K, Gw @ mm + np.random.default_rng(33).standard_normal((len(stations), N_POST)))
rel = [abs(float(np.std(np.random.default_rng(40 + k).standard_normal(len(centers)) @ fl))
           / PR.functional_sd(Gw, prior,
                              np.random.default_rng(40 + k).standard_normal(len(centers))) - 1)
       for k in range(24)]
med_rel, floor_rel = float(np.median(rel)), np.sqrt(0.5 / N_POST)
suite.add(gates.GateReport("calibration(MC)", med_rel < 4 * floor_rel, med_rel,
                           f"median < {4 * floor_rel:.3f} ({N_POST}-sample floor)",
                           "exact functional sd vs sampled"))

sd_b, _ = PR.posterior_sd(Gw, prior, np.random.default_rng(77), N_POST)
suite.add(gates.stability((mean, sd_post), (mean, sd_b), atol=1e-9))
suite.add(gates.adequacy(RMS, NOISE))

# --------------------------------------------- the regulator's three numbers
# gravity measures the MASS DEFICIT (CO2 replacing brine). CO2 tonnage is
# that deficit scaled by rho_CO2 / (rho_brine - rho_CO2) — the standard
# conversion used in Sleipner gravity interpretation.
TO_CO2 = RHO_CO2 / (RHO_BRINE - RHO_CO2)
MT = 1e-6                                    # g/cc * m^3 = t -> Mt


def co2_mass(mask):
    w = np.where(mask, vol, 0.0)
    m = float(w @ mean) * TO_CO2 * MT
    s = PR.functional_sd(Gw, prior, w) * TO_CO2 * MT
    return abs(m), s          # deficit is negative; report CO2 as positive


inside = rad < COMPLEX_R
m_in, s_in = co2_mass(inside)


def leak_body(radius, offset=2000.0):
    """A compact accumulation just outside the complex, and the two numbers
    that decide whether the survey can say anything about it:

      detectable  1.96 * posterior sd of its mass — what the data can resolve
      ceiling     the most CO2 that body could physically hold (saturation 1)

    If detectable > ceiling the survey CANNOT see that leak even when it is
    completely full of CO2, and no containment claim about it is honest.
    This is the comparison a whole-area mass bound hides.
    """
    body = ((np.hypot(centers[:, 0] - offset, centers[:, 1]) < radius)
            & (centers[:, 2] > -1010.0))
    _, sd = co2_mass(body)
    ceiling = float((vol[body] * PHI * RHO_CO2).sum()) * 1e-9      # kg -> Mt
    return 1.96 * sd, ceiling, body


scan = [(rr,) + leak_body(rr)[:2] for rr in (400.0, 600.0, 800.0, 1200.0)]
visible = [(rr, dt, ce) for rr, dt, ce in scan if dt < ce]
min_leak, ceil_400, _ = leak_body(400.0)

# The area-wide bound, computed and then REFUSED. Reported because the
# refusal is the finding: the prior alone permits ~100 Mt spread thinly over
# the whole outside area, so a bound on it is a statement about the prior,
# not about the survey. Exclusion limits are only meaningful for a DEFINED
# body — the same lesson as the informed-fraction map.
m_out, s_out = co2_mass(~inside)
area_bound = m_out + 1.96 * s_out

# What better instrumentation buys — measured, not asserted. 1.1 uGal is the
# repeatability Sleipner actually reached by 2013.
G11 = G / (1.1e-3)
prior11 = prior
s_in_11 = PR.functional_sd(G11, prior11, np.where(inside, vol, 0.0)) * TO_CO2 * MT
b400 = ((np.hypot(centers[:, 0] - 2000.0, centers[:, 1]) < 400.0)
        & (centers[:, 2] > -1010.0))
leak_11 = 1.96 * PR.functional_sd(G11, prior11, np.where(b400, vol, 0.0)) * TO_CO2 * MT

numbers = {
    "CO2 accounted for inside the complex":
        f"{m_in:,.1f} +/- {s_in:,.1f} Mt   (declared injected: {INJECTED_MT:,.1f} Mt)",
    "inventory reconciles?":
        ("YES — the declared inventory sits inside the 95% interval"
         if abs(m_in - INJECTED_MT) < 1.96 * s_in else
         "NO — the survey and the injection record disagree"),
    "CONTAINMENT — smallest leak this survey can detect":
        (f"{visible[0][1]:,.2f} Mt in a {visible[0][0]:.0f} m body 2 km out — "
         f"but only if that body is at least "
         f"{visible[0][1] / visible[0][2] * 100:.0f}% saturated"
         if visible else "NOTHING in the sizes tested — see the table below"),
    "so an emptier leak of that size is invisible":
        f"a {visible[0][0]:.0f} m body below "
        f"{visible[0][1] / visible[0][2] * 100:.0f}% saturation cannot be "
        f"ruled out at all, and no containment claim covers it"
        if visible else "n/a",
    "the recovered picture, judged honestly":
        f"the posterior mean exceeds the physical saturation ceiling in "
        f"{float(np.mean(np.abs(mean) > abs(MAX_CONTRAST))) * 100:.0f}% of "
        f"cells — a Gaussian prior is unbounded, so the MAP picture is not "
        f"physical. The mass limits are integrals and stay valid; being "
        f"unbounded makes them conservative, never optimistic",
    "the honest limit of gravity ALONE":
        f"per-column sigma is ~{float(np.median(top(sd_post * vol))) * TO_CO2 * MT:,.1f} Mt "
        f"against a peak column of {float(np.abs(top(truth * vol)).max()) * TO_CO2 * MT:,.2f} Mt — "
        f"this survey constrains INTEGRALS, not maps. That is why Sleipner "
        f"runs gravity alongside seismic, and any claim to map a plume from "
        f"30 benchmarks alone would be the prior talking",
    "what a whole-area bound would have said":
        f"< {area_bound:,.0f} Mt outside — REFUSED, it is {area_bound / INJECTED_MT * 100:.0f}% "
        f"of the injected mass and is set by the prior, not the data",
    "upgrading 3.0 -> 1.1 uGal (Sleipner's 2013 repeatability)":
        f"inventory +/-{s_in:,.1f} -> +/-{s_in_11:,.1f} Mt; 400 m leak "
        f"threshold {min_leak:,.2f} -> {leak_11:,.2f} Mt "
        f"({(1 - leak_11 / min_leak) * 100:.0f}% better for a 2.7x quieter "
        f"instrument — this survey is limited by its geometry, not its noise)",
}
v = V.assess(suite, numbers, subject="this monitoring programme")
print("\n" + str(v))

# ------------------------------------------------------------------ figure
fig, axes = plt.subplots(1, 4, figsize=(19.5, 4.3), dpi=150)
ext = [xs[0] / 1000, xs[-1] / 1000, ys[0] / 1000, ys[-1] / 1000]
th = np.linspace(0, 2 * np.pi, 200)
t_col = -top(truth * vol) * TO_CO2 * MT        # CO2 registers as a DEFICIT
m_col = -top(mean * vol) * TO_CO2 * MT
vmax = float(np.abs(t_col).max())
for ax, val, ttl, cm, kw in [
    (axes[0], t_col, "the plume (declared truth)\nCO2, Mt per column",
     "magma", dict(vmin=-vmax, vmax=vmax)),
    (axes[1], m_col, "recovered mean, SAME SCALE\nthe picture is the weak part",
     "magma", dict(vmin=-vmax, vmax=vmax)),
    (axes[2], top(sd_post * vol) * TO_CO2 * MT,
     "posterior sigma\n(what the survey cannot see)", "viridis", {}),
    (axes[3], 1.96 * top(sd_post * vol) * TO_CO2 * MT,
     "THE FILING NUMBER:\n95% upper limit on hidden CO2", "cividis", {}),
]:
    im = ax.imshow(val, origin="lower", extent=ext, cmap=cm, **kw)
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.plot(COMPLEX_R / 1000 * np.cos(th), COMPLEX_R / 1000 * np.sin(th),
            "w--", lw=1.4)
    ax.plot(stations[:, 0] / 1000, stations[:, 1] / 1000, "k.", ms=3)
    ax.set_title(ttl, fontsize=9)
    ax.set_xlabel("km")
axes[0].set_ylabel("km")
fig.tight_layout()
FIG = ROOT / "figures" / "verdict_carbon_storage.png"
fig.savefig(FIG)

report.render(
    ROOT / "figures" / "verdict_carbon_storage.html",
    "Gurutva — containment verdict",
    "CO2 storage monitoring · seafloor gravimetry · Sleipner-class survey design",
    v, figure=FIG,
    sections=[
        ("The survey being judged",
         [("benchmarks", f"{len(stations)} on the seafloor, "
           f"{STATION_SPACING:.0f} m spacing"),
          ("repeatability", f"{NOISE_UGAL} uGal (Sleipner 2002-2009 epochs)"),
          ("reservoir", f"{abs(RES_TOP):.0f}-{abs(RES_BOT):.0f} m depth, "
           f"porosity {PHI}"),
          ("in-situ densities", f"CO2 {RHO_CO2:.0f} vs brine {RHO_BRINE:.0f} "
           f"kg/m3 -> bulk contrast {MAX_CONTRAST:.3f} g/cc when saturated"),
          ("peak signal", f"{np.abs(G @ truth).max() * 1e3:.1f} uGal against "
           f"{NOISE_UGAL} uGal noise")],
         "The 3.0 -> 1.1 uGal row above is computed, not assumed: that is "
         "the trade a monitoring budget turns on, and it is knowable before "
         "a single benchmark is deployed."),
        ("Leak-detection capability, by body size (2 km outside the complex)",
         [(f"{rr:.0f} m radius",
           f"detectable at {dt:,.2f} Mt · physical ceiling {ce:,.2f} Mt · "
           + (f"VISIBLE above {dt / ce * 100:.0f}% saturation" if dt < ce
              else "INVISIBLE — undetectable even when completely full"))
          for rr, dt, ce in scan],
         "A containment claim is only honest for bodies in the VISIBLE rows. "
         "For the rest the survey is silent, and saying otherwise would be "
         "reporting the prior as if it were a measurement."),
        ("What this analysis is, precisely",
         [("survey parameters", "published Sleipner monitoring programme"),
          ("observations", "SIMULATED from the declared plume above"),
          ("Sleipner measured data", "NOT used — licensed per user and "
           "not sellable; deliberately excluded"),
          ("what the numbers mean", "what this survey DESIGN can prove")],
         "A capability analysis is what an operator needs before committing "
         "to a monitoring programme and what a regulator needs to judge one. "
         "Substituting a licensed real dataset changes the inputs, not the "
         "method."),
    ],
    caption="White dashed circle: the licensed storage complex. Black dots: "
            "seafloor benchmarks. Panels 1 and 2 share a colour scale on "
            "purpose: 30 stations against 4,500 unknowns cannot resolve a "
            "picture, and the recovered mean is largely ringing. That is the "
            "honest state of the art, and it is why the deliverable is panel "
            "4 — the most CO2 that could be hiding in each column without "
            "this survey seeing it — and not the pretty map.",
    footer=f"Posterior mean fits the data to {RMS * 1e3:.2f} uGal against a "
           f"{NOISE_UGAL} uGal floor · prior {PRIOR_SD:.3f} g/cc "
           f"(fully-saturated ceiling) correlated over {CORR_LEN:.0f} m · "
           f"per-cell map from {N_POST} posterior samples; mass limits are "
           f"exact, not sampled · reproducible: "
           f"<code>py -3 examples/verdict_carbon_storage.py</code>")

(ROOT / "figures" / "verdict_carbon_storage.json").write_text(json.dumps({
    "claimable": v.claimable, "verdict": v.headline, "gates": v.gates,
    "numbers": numbers, "prior": pmeta,
    "co2_inside_Mt": m_in, "co2_inside_sd_Mt": s_in,
    "area_bound_REFUSED_Mt": area_bound,
    "leak_scan": [dict(radius_m=rr, detectable_Mt=dt, ceiling_Mt=ce,
                       visible=bool(dt < ce)) for rr, dt, ce in scan],
    "min_detectable_400m_Mt": min_leak, "ceiling_400m_Mt": ceil_400,
    "inventory_sd_at_1p1uGal_Mt": s_in_11,
    "injected_Mt": INJECTED_MT, "rms_uGal": RMS * 1e3,
    "noise_uGal": NOISE_UGAL, "prior_sd_gcc": PRIOR_SD,
    "data": "SIMULATED from a declared plume; Sleipner survey parameters only"},
    indent=1))
print("\nreport -> figures/verdict_carbon_storage.html")
