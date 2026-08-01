"""`gurutva` — the command a customer actually runs.

Everything else in this repo is machinery. This is the product: point it at a
survey you own and get back the verdict report.

    py -3 -m src.product.cli report   --survey mysurvey.csv --noise 0.05 \
        --prior-sd 0.25 --corr-len 500 --out report.html

    py -3 -m src.product.cli selftest --survey mysurvey.csv --noise 0.05 \
        --prior-sd 0.25 --corr-len 500 --body-depth 800 --out capability.html

Two modes, because a customer has two different questions.

  report    You have data. What does it support? Runs the four core gates and
            returns the interval on the quantity a decision turns on, the
            informed-fraction map, and the depth below which the model is
            invented. Gate 5 CANNOT run here — real data has no known truth —
            so the verdict says PROVISIONAL rather than letting silence read
            as success.

  selftest  You have a survey design, or are about to pay for one. We plant a
            body of known size and depth, simulate what YOUR stations and
            YOUR noise would record, and invert it. Gate 5 runs, because now
            the truth is known, and the verdict can reach CLAIMABLE.

INPUT. One CSV, four columns, header row required:

    x,y,z,gz            projected metres, or lon,lat in degrees
    331000,4263000,1650.2,-0.084

  x, y   easting/northing in METRES, or longitude/latitude in DEGREES —
         detected automatically and projected onto a local tangent plane
  z      station elevation in METRES, z-UP
  gz     gravity anomaly in mGal, or (with --field mag) total-field
         anomaly in nT. Background already removed.

Nothing is guessed. --noise, --prior-sd and --corr-len are mandatory: a tool
that invents a noise floor is inventing its own passing grade.
"""

import argparse
import json
import subprocess
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src import mag_forward as mf, prism_forward as pf
from src.gurutva_core import gates
from src.product import (joint as JT, prior as PR, report,
                         survey as SV, verdict as V)

MAX_CELLS = 60_000       # laptop guard; refuse rather than swap for an hour

# What changes between the two physics, and nothing else does.
FIELDS = {
    "gravity": dict(data_unit="mGal", model_unit="g/cc",
                    quantity="excess mass", out_unit="Mt", scale=1e-6,
                    weight="volume"),
    "mag": dict(data_unit="nT", model_unit="SI susceptibility",
                quantity="mean susceptibility", out_unit="SI", scale=1.0,
                weight="mean"),
}


def _version():
    """Stamp the report with the commit it came from. An auditable document
    that cannot say which code produced it is not auditable."""
    try:
        h = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                           cwd=Path(__file__).resolve().parents[2],
                           capture_output=True, text=True, timeout=5)
        return f"gurutva {h.stdout.strip()}" if h.returncode == 0 else "gurutva"
    except Exception:
        return "gurutva"


_VERSION = _version()


def load_survey(path):
    """Read x,y,z,gz and REFUSE anything it cannot honestly invert.

    Every guard is here because a hostile pass over the CLI hit it: a
    semicolon export died inside genfromtxt, a UTF-8 BOM turned the first
    column into a name starting with U+FEFF so the error blamed a missing x,
    a single NaN sailed through and was caught three gates later by luck, and
    a collinear survey crashed in the sparse factoriser.

    Returns (stations, data, meta). Geographic coordinates are detected and
    projected; meta records what happened so the report can say so.
    """
    raw = Path(path).read_text(encoding="utf-8-sig", errors="replace")
    head = raw.splitlines()[0] if raw.strip() else ""
    delim = max((",", ";", "\t"), key=lambda c: head.count(c))
    if head.count(delim) < 3:
        raise SystemExit(
            f"{path}: could not find 4 columns in the header line.\n"
            f"  got: {head[:70]!r}\n"
            f"  need: x,y,z,gz  (comma, semicolon or tab separated)")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")      # we report the errors ourselves
        rows = np.genfromtxt(raw.splitlines(), delimiter=delim, names=True,
                             dtype=float, invalid_raise=False)
    body = [ln for ln in raw.splitlines()[1:] if ln.strip()]
    if rows.size == 0 and body:
        others = {c: body[0].count(c) for c in (",", ";", "\t") if c != delim}
        alt = max(others, key=others.get) if others else None
        raise SystemExit(
            f"{path}: the header is separated by {delim!r} but no data row "
            f"could be read that way.\n"
            f"  first data line: {body[0][:70]!r}\n"
            + (f"  It looks {alt!r}-separated. Make the header match the data."
               if alt and others[alt] >= 3 else
               "  Header and data must use the same separator."))
    names = tuple(n.strip().lower() for n in (rows.dtype.names or ()))
    rows.dtype.names = names
    alias = {"lon": "x", "longitude": "x", "easting": "x",
             "lat": "y", "latitude": "y", "northing": "y",
             "elev": "z", "elevation": "z", "height": "z",
             "grav": "gz", "mgal": "gz", "tmi": "gz", "nt": "gz"}
    names = tuple(alias.get(n, n) for n in names)
    rows.dtype.names = names
    missing = [c for c in ("x", "y", "z", "gz") if c not in names]
    if missing:
        raise SystemExit(
            f"{path}: missing column(s) {', '.join(missing)}.\n"
            f"  found: {', '.join(names) or '(none)'}\n"
            f"  need:  x,y,z,gz  (metres or lon/lat degrees, metres, data)")

    x, y = np.asarray(rows["x"], float), np.asarray(rows["y"], float)
    z, d = np.asarray(rows["z"], float), np.asarray(rows["gz"], float)

    bad = ~np.isfinite(np.column_stack([x, y, z, d])).all(axis=1)
    if bad.any():
        first = int(np.argmax(bad)) + 2          # 1-based, past the header
        raise SystemExit(
            f"{path}: {int(bad.sum())} row(s) contain a blank or non-numeric "
            f"value; the first is line {first}.\n"
            f"  Gurutva will not guess a missing reading. Remove those rows "
            f"or fill them in.")
    if len(x) < 8:
        raise SystemExit(f"{path}: only {len(x)} stations. Too few to say "
                         "anything honest about a 3-D density field.")

    meta = {"geographic": False}
    if SV.looks_geographic(x, y):
        try:
            x, y, pm = SV.project(x, y)
        except ValueError as e:
            raise SystemExit(f"{path}: {e}")
        meta = {"geographic": True, **pm}
        print(f"  coordinates read as lon/lat and projected onto a local "
              f"tangent plane at {pm['lat0']:.4f}, {pm['lon0']:.4f} "
              f"(survey spans {pm['span_km']:.1f} km)")

    st = np.column_stack([x, y, z])
    ex, ey = x.max() - x.min(), y.max() - y.min()
    if min(ex, ey) <= 0:
        raise SystemExit(
            f"{path}: the stations span {ex:,.0f} m east-west and {ey:,.0f} m "
            f"north-south.\n"
            f"  A survey along a single line cannot constrain a 3-D field — "
            f"there is no second horizontal direction for the data to "
            f"resolve. Use a 2-D layout, or a profile-specific tool.")

    uniq = len({(round(a, 3), round(b, 3)) for a, b in st[:, :2]})
    if uniq < len(st):
        print(f"  warning: {len(st) - uniq} station(s) repeat an existing "
              f"x,y position. Repeats add no new information and will make "
              f"the survey look better resolved than it is.")
    return st, d, meta


def build_mesh(stations, cell, depth, top=None, drape=True, margin=0.25):
    """A regular block under the survey, optionally draped to topography.

    Returns (centers, dims, shape, volumes, active). `active` is False for
    cells above the ground: with a flat-topped block over rolling terrain the
    inversion is handed air to put density into, and it will.
    """
    x0, x1 = stations[:, 0].min(), stations[:, 0].max()
    y0, y1 = stations[:, 1].min(), stations[:, 1].max()
    ceiling = float(stations[:, 2].max()) if top is None else float(top)
    mx, my = (x1 - x0) * margin, (y1 - y0) * margin
    xs = np.arange(x0 - mx + cell / 2, x1 + mx, cell)
    ys = np.arange(y0 - my + cell / 2, y1 + my, cell)
    zs = np.arange(ceiling - depth + cell / 2, ceiling, cell)
    for n, span, hint in ((len(xs), x1 - x0, "--cell smaller than the survey width"),
                          (len(ys), y1 - y0, "--cell smaller than the survey height"),
                          (len(zs), depth, "--depth larger than --cell")):
        if n < 1:
            raise SystemExit(
                f"mesh has zero cells along one axis: that extent is "
                f"{span:,.0f} m and --cell is {cell:,.0f} m. Use {hint}.")
    if len(xs) * len(ys) * len(zs) > MAX_CELLS:
        raise SystemExit(
            f"mesh would be {len(xs) * len(ys) * len(zs):,} cells "
            f"(limit {MAX_CELLS:,}). Increase --cell or shrink --depth.")
    CX, CY, CZ = np.meshgrid(xs, ys, zs, indexing="ij")
    centers = np.column_stack([CX.ravel(), CY.ravel(), CZ.ravel()])
    dims = np.tile([cell, cell, cell], (len(centers), 1))
    vol = np.full(len(centers), cell ** 3)
    shape = (len(xs), len(ys), len(zs))

    if drape:
        active = ~SV.air_mask(centers, stations)
        if active.sum() < 8:
            raise SystemExit(
                "draping to topography left almost no cells below ground. "
                "Check that --depth is measured downward and that station "
                "elevations are in metres.")
    else:
        active = np.ones(len(centers), dtype=bool)
    return centers, dims, shape, vol, active


def core_gates(G, prior, d, rng, n_post, mean, sd, unit="", prior_sd=None):
    suite = gates.GateSuite()
    sims = (G @ prior.sample(rng, 300)
            + rng.standard_normal((G.shape[0], 300))).T
    suite.add(gates.licensing(d, sims))

    # SCALE ADVICE, because --prior-sd is a per-CELL marginal sd and thousands
    # of correlated cells add up. A user hunting a body of susceptibility 0.06
    # will type 0.06 and declare a prior that predicts 600 nT of signal over a
    # survey that reads 4. Licensing passes (an over-wide prior makes anything
    # look typical) and adequacy then fails as OVERFITS, which is a true but
    # unhelpful thing to be told. So say the useful version instead.
    pred, obs = float(np.std(sims)), float(np.std(d))
    ratio = pred / obs if obs > 0 else np.inf
    if prior_sd is not None and (ratio > 10 or ratio < 0.1):
        wide = "too wide" if ratio > 1 else "too tight"
        print(f"  WARNING: your declared prior predicts data with spread "
              f"{pred:.3g} but yours has {obs:.3g} ({ratio:.0f}x {wide}).")
        print(f"           --prior-sd is a PER-CELL sd and correlated cells "
              f"add up. For this survey try about "
              f"{prior_sd / ratio:.2g} {unit}.")

    SGt = prior.apply_inv(G.T)
    Kd = G @ SGt + np.eye(G.shape[0])
    m = prior.sample(np.random.default_rng(22), n_post)
    fl = m - SGt @ np.linalg.solve(
        Kd, G @ m + np.random.default_rng(33).standard_normal(
            (G.shape[0], n_post)))
    rel = []
    for k in range(16):
        w = np.random.default_rng(40 + k).standard_normal(G.shape[1])
        rel.append(abs(float(np.std(w @ fl))
                       / PR.functional_sd(G, prior, w) - 1))
    med, floor = float(np.median(rel)), np.sqrt(0.5 / n_post)
    suite.add(gates.GateReport("calibration(MC)", med < 4 * floor, med,
                               f"median < {4 * floor:.3f} ({n_post}-sample floor)",
                               "exact functional sd vs sampled"))
    sd_b, _ = PR.posterior_sd(G, prior, np.random.default_rng(77), n_post)
    suite.add(gates.stability((mean, sd), (mean, sd_b), atol=1e-9))
    return suite, dict(prior_pred=pred, observed=obs, ratio=ratio)


def run(args):
    cfg = FIELDS[args.field]
    stations, d_obs, geo = load_survey(args.survey)
    # bool(), not np.bool_: a numpy scalar leaks all the way to json.dumps
    # and dies with "Object of type bool is not JSON serializable" AFTER the
    # whole inversion has run. Found by running the tool, not by reading it.
    drape = bool(args.topography == "on" or
                 (args.topography == "auto"
                  and np.ptp(stations[:, 2]) > args.cell / 2))
    centers, dims, shape, vol, active = build_mesh(
        stations, args.cell, args.depth, args.top, drape)
    rng = np.random.default_rng(args.seed)
    print(f"survey: {len(stations)} stations | {args.field} | mesh "
          f"{shape[0]}x{shape[1]}x{shape[2]} at {args.cell:g} m, "
          f"{int(active.sum()):,} of {len(centers):,} cells below ground"
          f"{' (draped to topography)' if drape else ' (flat top)'}")

    if args.field == "gravity":
        G_raw = pf.prism_matrix(stations, centers[active], dims[active])
    else:
        G_raw = mf.mag_matrix(stations, centers[active], dims[active],
                              args.b0, args.inclination, args.declination,
                              q=args.remanence_q,
                              rem_inclination=args.remanence_inclination,
                              rem_declination=args.remanence_declination)
    G = G_raw / args.noise

    # PRIOR FROM THE TARGET, when one is declared. --prior-sd is a per-cell
    # sd and correlated cells add up, so typing the contrast of the body you
    # are hunting declares a prior hundreds of times too wide. Declaring the
    # BODY instead fixes the amplitude from physics, and touches no observed
    # data, so it cannot become a way of tuning until a gate turns green.
    prior_sd, prior_source = args.prior_sd, "declared directly"
    if args.target_contrast is not None:
        top_z = float(stations[:, 2].max()) if args.top is None else args.top
        amp, ncell = JT.target_anomaly_std(
            lambda m: G_raw @ m, stations, centers[active], dims[active],
            args.target_contrast, args.target_radius, args.target_depth, top_z)
        prior_sd = JT.derive_prior_sd(G, shape, (args.cell,) * 3,
                                      args.corr_len, rng, amp / args.noise,
                                      active=active)
        prior_source = (f"derived from a declared target: {args.target_contrast:+g} "
                        f"{cfg['model_unit']} over {ncell} cells at "
                        f"{args.target_depth:g} m depth, which would make a "
                        f"{amp:.3g} {cfg['data_unit']} anomaly")
        print(f"  prior sd {prior_sd:.4g} {cfg['model_unit']} {prior_source}")
    prior, pmeta = PR.build_regular(shape, (args.cell,) * 3, prior_sd,
                                    args.corr_len, rng, active=active)

    truth = None
    if args.mode == "selftest":
        cx, cy = stations[:, 0].mean(), stations[:, 1].mean()
        zb = float(stations[:, 2].max()) - args.body_depth
        ca = centers[active]
        inside = ((np.hypot(ca[:, 0] - cx, ca[:, 1] - cy) < args.body_radius)
                  & (np.abs(ca[:, 2] - zb) < args.body_radius))
        if not inside.any():
            raise SystemExit("planted body falls outside the mesh — increase "
                             "--depth or reduce --body-depth")
        truth = np.where(inside, args.body_contrast, 0.0)
        d_obs = G_raw @ truth + args.noise * rng.standard_normal(len(stations))
        print(f"  self-test: {int(inside.sum())} cells at "
              f"{args.body_contrast:+g} {cfg['model_unit']}, "
              f"{args.body_depth:g} m deep; peak signal "
              f"{np.abs(G_raw @ truth).max():.3f} {cfg['data_unit']} against "
              f"{args.noise:g}")

    dw = d_obs / args.noise
    mean = PR.posterior_mean(G, prior, dw)
    sd, sd_err = PR.posterior_sd(G, prior, np.random.default_rng(11),
                                 args.samples)
    rms = float(np.sqrt(np.mean((G_raw @ mean - d_obs) ** 2)))

    suite, scale_info = core_gates(G, prior, dw, rng, args.samples, mean, sd,
                                   unit=cfg["model_unit"], prior_sd=prior_sd)
    # Reference is what THIS model predicts of itself, not the raw noise: a
    # flexible model should fit better than the noise, and comparing to the
    # noise punishes it for being correct. See prior.expected_residual.
    ref = args.noise * PR.expected_residual(G, prior)
    suite.add(gates.adequacy(rms, ref))
    if truth is not None:
        suite.add(gates.recovery(mean, sd, truth))

    # ---- the numbers a decision turns on --------------------------------
    ca, va = centers[active], vol[active]
    prior_sd_cell = prior.marginal_sd(np.random.default_rng(55), 400)
    inf = 1.0 - sd / prior_sd_cell
    # BLIND DEPTH, on the 90th percentile rather than the median.
    #
    # The median asks "is the TYPICAL cell at this depth informed", which is
    # the right question for a broad gravity anomaly and the wrong one for a
    # compact magnetic body: a real magnetic survey lights up a handful of
    # cells to 87% while the median cell sits at 1.3%, so a median rule
    # declared the whole model blind and the report had nothing to say. The
    # question that works for both is "does this depth contain ANYTHING the
    # survey can see", which is a high quantile.
    z = ca[:, 2]
    edges = np.linspace(z.min(), z.max(), 20)
    lit, typical = [], []
    for i in range(len(edges) - 1):
        m = (z >= edges[i]) & (z < edges[i + 1])
        if not np.any(m):
            continue
        zc = 0.5 * (edges[i] + edges[i + 1])
        if np.percentile(inf[m], 90) > 0.05:
            lit.append(zc)
        if np.median(inf[m]) > 0.05:
            typical.append(zc)
    z_blind = min(lit) if lit else z.max()
    z_typical = min(typical) if typical else None

    if args.region_file:
        vx, vy = SV.read_polygon(args.region_file)
        if geo.get("geographic") and np.abs(vx).max() <= 180:
            vx, vy, _ = SV.project(vx, vy, geo["lon0"], geo["lat0"])
        box = SV.points_in_polygon(ca[:, 0], ca[:, 1], vx, vy) & (z > z_blind)
        region_desc = (f"your {SV.polygon_area(vx, vy) / 1e6:,.2f} km2 block "
                       f"from {Path(args.region_file).name}")
    else:
        cx, cy = stations[:, 0].mean(), stations[:, 1].mean()
        half = args.region / 2.0
        box = ((np.abs(ca[:, 0] - cx) < half) & (np.abs(ca[:, 1] - cy) < half)
               & (z > z_blind))
        region_desc = (f"the {args.region / 1000:g} km block at the centre of "
                       f"your survey")
    if not box.any():
        # Do not crash and do not quietly widen: say which of the two things
        # actually happened, because they call for different fixes.
        if not lit:
            raise SystemExit(
                "your survey does not constrain a single depth level: even "
                "the best-informed cells stay below 5% of their prior. "
                "Nothing here can be reported honestly. Either the data is "
                "too weak for this mesh, or --prior-sd is far too wide "
                "(check the scale warning above).")
        raise SystemExit(
            f"the reporting region contains no cells above the blind depth "
            f"({z_blind:,.0f} m). Widen --region, move --region-file, or "
            f"accept that the survey sees nothing under that block.")

    w = np.where(box, va, 0.0)
    if cfg["weight"] == "mean":
        w = w / w.sum()                      # volume-weighted average, not a sum
    m_box = float(w @ mean)
    sd_box = PR.functional_sd(G, prior, w)
    naive = float(np.sqrt(np.sum(w**2 * sd**2)))
    S = cfg["scale"]
    lo, hi = (m_box - 1.96 * sd_box) * S, (m_box + 1.96 * sd_box) * S
    # Susceptibility lives near 1e-5, so a fixed-point format prints "-0.0000"
    # for every real answer. Mass is read in Mt and wants thousands separators.
    fmt = ",.1f" if args.field == "gravity" else ".3e"

    numbers = {
        f"{cfg['quantity']}, {region_desc}, above the blind depth":
            f"{m_box * S:+{fmt}} {cfg['out_unit']}  "
            f"(95%: {lo:+{fmt}} to {hi:+{fmt}})",
        "is that a detection?":
            ("YES — the 95% interval excludes zero"
             if abs(m_box) > 1.96 * sd_box else
             f"NO — consistent with zero. 95% upper limit "
             f"{(abs(m_box) + 1.96 * sd_box) * S:{fmt}} {cfg['out_unit']}"),
        "your model is INVENTED below":
            f"{z_blind:,.0f} m elevation — deeper than this NOTHING in the "
            f"model is constrained by your data",
        "and the TYPICAL cell is informed only above":
            (f"{z_typical:,.0f} m elevation" if z_typical is not None else
             "no depth at all — your survey lights up a few cells strongly "
             "and leaves the rest to the prior, which is normal for a "
             "compact body and means per-cell values are not a map"),
        "cells your survey informs (>5%)":
            f"{float(np.mean(inf > 0.05)) * 100:.1f}% of {len(inf):,}",
        "per-cell shortcut would have said":
            f"+/-{naive * S:{fmt}} instead of +/-{sd_box * S:{fmt}} "
            f"({sd_box / naive:.2f}x) — neighbouring cells trade off, so the "
            f"variance of a sum is not the sum of variances",
    }
    if truth is not None:
        rec = suite.recovery_report
        numbers = {
            "CAN YOUR SURVEY SEE IT?":
                ("YES — the planted body is recovered inside its interval"
                 if rec.passed else
                 "NO — the planted body is NOT recovered inside its interval"),
            "recovery": f"{rec.value * 100:.0f}% of cells cover the truth, "
                        f"worst miss {float(rec.note.split()[1]):.1f} sigma",
            "planted body": f"{args.body_contrast:+g} {cfg['model_unit']}, "
                            f"radius {args.body_radius:g} m, "
                            f"{args.body_depth:g} m deep",
            **numbers,
        }

    v = V.assess(suite, numbers,
                 subject=("this survey design" if truth is not None
                          else Path(args.survey).name))
    print("\n" + str(v))

    # ---- the figure the footer promises ---------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2), dpi=140)
    cy_ = stations[:, 1].mean()
    kd = cKDTree(ca)
    xg = np.linspace(ca[:, 0].min(), ca[:, 0].max(), 200)
    zg = np.linspace(ca[:, 2].min(), ca[:, 2].max(), 120)
    XX, ZZ = np.meshgrid(xg, zg)
    dist, idx = kd.query(np.column_stack(
        [XX.ravel(), np.full(XX.size, cy_), ZZ.ravel()]))
    outside = (dist > args.cell * 1.5).reshape(XX.shape)
    for ax, val, ttl, cm, kw in [
        (axes[0], sd, f"how uncertain each cell still is ({cfg['model_unit']})",
         "viridis", {}),
        (axes[1], inf, "informed fraction: 1 = your data knows it, "
         "0 = it is your prior talking", "magma",
         dict(vmin=0, vmax=max(0.2, float(np.nanmax(inf))))),
    ]:
        Z = np.where(outside, np.nan, np.asarray(val)[idx].reshape(XX.shape))
        im = ax.pcolormesh(xg / 1000, zg, Z, cmap=cm, shading="auto", **kw)
        fig.colorbar(im, ax=ax, fraction=0.046)
        ax.axhline(z_blind, color="r", lw=1.3, ls="--")
        ax.plot(stations[:, 0] / 1000, stations[:, 2], "k.", ms=2)
        ax.set_title(ttl, fontsize=9)
        ax.set_xlabel("easting (km)")
    axes[0].set_ylabel("elevation (m)")
    fig.tight_layout()
    out = Path(args.out)
    figpath = out.with_suffix(".png")
    fig.savefig(figpath)
    plt.close(fig)

    declared = [("field", f"{args.field} ({cfg['data_unit']} in, "
                          f"{cfg['model_unit']} out)"),
                ("noise floor", f"{args.noise:g} {cfg['data_unit']} — YOUR "
                                f"measurement, not our guess"),
                ("prior sd", f"{prior_sd:.4g} {cfg['model_unit']} — "
                             f"{prior_source}"),
                ("prior correlation length", f"{args.corr_len:g} m"),
                ("mesh", f"{shape[0]}x{shape[1]}x{shape[2]} at {args.cell:g} m, "
                         f"{args.depth:g} m deep, "
                         f"{int(active.sum()):,} cells below ground"),
                ("topography", "draped from station elevations" if drape
                               else "flat top"),
                ("coordinates", f"lon/lat projected to a local tangent plane "
                                f"at {geo['lat0']:.4f}, {geo['lon0']:.4f}"
                                if geo.get("geographic") else
                                "projected metres, used as given"),
                ("posterior fit", f"{rms:.4g} {cfg['data_unit']} against an "
                                  f"expected {ref:.4g} for a model with this "
                                  f"much freedom"),
                ("prior predicts vs your data",
                 f"{scale_info['prior_pred']:.3g} vs "
                 f"{scale_info['observed']:.3g} (whitened) — "
                 f"{scale_info['ratio']:.1f}x. Far from 1 means --prior-sd is "
                 f"mis-scaled: it is a PER-CELL sd and correlated cells add "
                 f"up")]
    if args.field == "mag" and args.remanence_q:
        declared.insert(1, ("remanence",
                            f"Koenigsberger Q = {args.remanence_q:g}, "
                            f"remanent direction inclination "
                            f"{args.remanence_inclination if args.remanence_inclination is not None else args.inclination:g}, "
                            f"declination "
                            f"{args.remanence_declination if args.remanence_declination is not None else args.declination:g} "
                            f"— DECLARED, not solved for"))
    if args.field == "mag":
        declared.insert(1, ("ambient field",
                            f"{args.b0:,.0f} nT, inclination "
                            f"{args.inclination:g}, declination "
                            f"{args.declination:g}"
                            + ("" if args.remanence_q else
                               " (induced magnetisation only)")))

    report.render(
        out,
        "Gurutva — survey capability" if truth is not None
        else "Gurutva — model verdict",
        f"{Path(args.survey).name} · {len(stations)} stations · "
        f"{args.noise:g} {cfg['data_unit']} declared noise",
        v, figure=figpath,
        caption=(f"Section through the model at the centre of your survey; "
                 f"black dots are stations. Below the red line "
                 f"({z_blind:,.0f} m elevation) the typical cell is "
                 f"unconstrained: that part of any map you have been shown is "
                 f"the regularizer, not the rock."),
        headline=((f"{m_box * S:+{fmt}} {cfg['out_unit']}"),
                  (f"{cfg['quantity']} in {region_desc}, above the blind "
                   f"depth. 95% between {lo:+{fmt}} and {hi:+{fmt}}."))
        if truth is None else
        ((f"{'YES' if suite.recovery_report.passed else 'NO'}"),
         (f"can this survey see a {args.body_radius:g} m body at "
          f"{args.body_depth:g} m depth? "
          f"{suite.recovery_report.value * 100:.0f}% of cells recover the "
          f"planted truth inside their interval.")),
        version=_VERSION,
        sections=[("What you declared", declared,
                   "Change any of these and the answer changes. That is not a "
                   "weakness of the method, it is the part every other tool "
                   "hides. Declare them from your site, never tune them until "
                   "a gate turns green.")],
        footer=f"Generated by <code>gurutva {args.mode}</code> · per-cell map "
               f"from {args.samples} posterior samples (MC error "
               f"{float(np.median(sd_err / sd)) * 100:.1f}%); the interval on "
               f"{cfg['quantity']} is exact, not sampled.")
    out.with_suffix(".json").write_text(json.dumps({
        "mode": args.mode, "field": args.field, "survey": str(args.survey),
        "status": v.status, "claimable": v.claimable, "verdict": v.headline,
        "gates": v.gates, "numbers": numbers, "geographic": geo,
        "declared": {"noise": args.noise, "prior_sd": prior_sd,
                     "prior_source": prior_source,
                     "corr_len": args.corr_len, "cell": args.cell,
                     "depth": args.depth, "topography": drape},
        "value": m_box * S, "value_sd": sd_box * S, "unit": cfg["out_unit"],
        "prior_scale_check": scale_info,
        "z_blind_m": float(z_blind), "rms": rms,
        "cells_active": int(active.sum()), "cells_total": int(len(centers))},
        indent=1))
    print(f"\nreport -> {out}\njson   -> {out.with_suffix('.json')}")
    return 0 if v.claimable else 1


def run_joint(args):
    """Gravity AND magnetics, coupled through a declared petrophysical
    correlation rather than a structural penalty.

    The operator stays block diagonal; the coupling lives entirely in the
    prior covariance. That keeps the problem linear, which keeps the
    closed-form posterior, the exact functional interval, and every gate that
    depends on them. A cross-gradient structural term would be a stronger
    coupling and would cost all three.
    """
    st_g, d_g, geo = load_survey(args.survey)
    st_m, d_m, _ = load_survey(args.survey_mag)
    if len(st_g) != len(st_m) or not np.allclose(st_g[:, :2], st_m[:, :2],
                                                 atol=1.0):
        raise SystemExit(
            "joint mode needs both surveys at the SAME stations, in the same "
            "order. Interpolating one onto the other invents data, and the "
            "invented part would carry no error bar.")

    drape = bool(args.topography == "on" or
                 (args.topography == "auto"
                  and np.ptp(st_g[:, 2]) > args.cell / 2))
    centers, dims, shape, vol, active = build_mesh(
        st_g, args.cell, args.depth, args.top, drape)
    ca, va = centers[active], vol[active]
    n = int(active.sum())
    rng = np.random.default_rng(args.seed)
    print(f"joint: {len(st_g)} stations x 2 physics | mesh "
          f"{shape[0]}x{shape[1]}x{shape[2]}, {n:,} cells | "
          f"rho-chi correlation {args.rho_chi_correlation:+g}")

    gg = pf.prism_matrix(st_g, ca, dims[active])
    gm = mf.mag_matrix(st_m, ca, dims[active], args.b0, args.inclination,
                       args.declination, q=args.remanence_q,
                       rem_inclination=args.remanence_inclination,
                       rem_declination=args.remanence_declination)
    A = JT.block_operator(gg, gm, args.noise, args.noise_mag)
    dw = JT.stack_data(d_g, d_m, args.noise, args.noise_mag)
    prior, pmeta = JT.build_joint_prior(
        shape, (args.cell,) * 3, args.prior_sd, args.prior_sd_chi,
        args.rho_chi_correlation, args.corr_len, rng, active=active)

    mean = PR.posterior_mean(A, prior, dw)
    sd, sd_err = PR.posterior_sd(A, prior, np.random.default_rng(11),
                                 args.samples)
    rho, chi = JT.split(mean)
    rms_g = float(np.sqrt(np.mean((gg @ rho - d_g) ** 2)))
    rms_m = float(np.sqrt(np.mean((gm @ chi - d_m) ** 2)))

    suite, scale_info = core_gates(A, prior, dw, rng, args.samples, mean, sd,
                                   unit="joint", prior_sd=None)
    ref = PR.expected_residual(A, prior)
    pp = float(np.sqrt(np.mean((A @ mean - dw) ** 2)))
    suite.add(gates.adequacy(pp, ref))

    prior_sd_cell = prior.marginal_sd(np.random.default_rng(55), 400)
    inf_rho, inf_chi = JT.split(1.0 - sd / prior_sd_cell)
    z = ca[:, 2]
    edges = np.linspace(z.min(), z.max(), 20)
    lit = [0.5 * (edges[i] + edges[i + 1]) for i in range(len(edges) - 1)
           if np.any((z >= edges[i]) & (z < edges[i + 1]))
           and np.percentile(inf_rho[(z >= edges[i]) & (z < edges[i + 1])],
                             90) > 0.05]
    z_blind = min(lit) if lit else z.max()

    cx, cy = st_g[:, 0].mean(), st_g[:, 1].mean()
    half = args.region / 2.0
    box = ((np.abs(ca[:, 0] - cx) < half) & (np.abs(ca[:, 1] - cy) < half)
           & (z > z_blind))
    if not box.any():
        raise SystemExit("the reporting region contains no cells above the "
                         "blind depth. Widen --region.")
    w_rho = np.concatenate([np.where(box, va, 0.0), np.zeros(n)])
    m_box = float(w_rho @ mean)
    sd_box = PR.functional_sd(A, prior, w_rho)
    MT = 1e-6

    # What the second dataset actually bought, in the customer's own units.
    p_g, _ = PR.build_regular(shape, (args.cell,) * 3, args.prior_sd,
                              args.corr_len, rng, active=active)
    sd_alone = PR.functional_sd(gg / args.noise, p_g, np.where(box, va, 0.0))
    tighter = (1 - sd_box / sd_alone) * 100 if sd_alone > 0 else 0.0

    numbers = {
        "excess mass in the block, from BOTH datasets":
            f"{m_box * MT:+,.1f} Mt  (95%: {(m_box - 1.96 * sd_box) * MT:+,.1f} "
            f"to {(m_box + 1.96 * sd_box) * MT:+,.1f})",
        "what the magnetics bought":
            f"the interval went from +/-{sd_alone * MT:,.1f} to "
            f"+/-{sd_box * MT:,.1f} Mt ({tighter:.1f}% tighter)",
        "how it got there":
            f"a DECLARED petrophysical correlation of "
            f"{args.rho_chi_correlation:+g} between density and "
            f"susceptibility, and nothing else. At 0 the two inversions are "
            f"exactly independent, which is pinned by a test. This number is "
            f"an assumption about your rocks, not a measurement",
        "gravity fit / magnetic fit":
            f"{rms_g:.4g} mGal against {args.noise:g} | "
            f"{rms_m:.4g} nT against {args.noise_mag:g}",
        "your model is INVENTED below": f"{z_blind:,.0f} m elevation",
        "cells informed (density / susceptibility)":
            f"{float(np.mean(inf_rho > 0.05)) * 100:.1f}% / "
            f"{float(np.mean(inf_chi > 0.05)) * 100:.1f}% of {n:,}",
    }
    v = V.assess(suite, numbers, subject=f"{Path(args.survey).name} + "
                                        f"{Path(args.survey_mag).name}")
    print(chr(10) + str(v))

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2), dpi=140)
    kd = cKDTree(ca)
    xg = np.linspace(ca[:, 0].min(), ca[:, 0].max(), 200)
    zg = np.linspace(ca[:, 2].min(), ca[:, 2].max(), 120)
    XX, ZZ = np.meshgrid(xg, zg)
    dist, idx = kd.query(np.column_stack(
        [XX.ravel(), np.full(XX.size, cy), ZZ.ravel()]))
    outside = (dist > args.cell * 1.5).reshape(XX.shape)
    for ax, val, ttl, cm in [
        (axes[0], rho, "density contrast (g/cc)", "RdBu_r"),
        (axes[1], chi, "susceptibility (SI)", "magma"),
    ]:
        Z = np.where(outside, np.nan, np.asarray(val)[idx].reshape(XX.shape))
        im = ax.pcolormesh(xg / 1000, zg, Z, cmap=cm, shading="auto")
        fig.colorbar(im, ax=ax, fraction=0.046)
        ax.axhline(z_blind, color="r", lw=1.3, ls="--")
        ax.set_title(ttl + " - one rock, two properties", fontsize=9)
        ax.set_xlabel("easting (km)")
    axes[0].set_ylabel("elevation (m)")
    fig.tight_layout()
    out = Path(args.out)
    figpath = out.with_suffix(".png")
    fig.savefig(figpath)
    plt.close(fig)

    report.render(
        out, "Gurutva - joint verdict",
        f"{Path(args.survey).name} + {Path(args.survey_mag).name} - "
        f"{len(st_g)} stations, two physics",
        v, figure=figpath,
        caption="Density and susceptibility recovered together. They are "
                "linked only by the declared correlation; the operator itself "
                "stays block diagonal.",
        headline=(f"{m_box * MT:+,.1f} Mt",
                  f"excess mass from gravity AND magnetics, {tighter:.1f}% "
                  f"tighter than gravity alone."),
        version=_VERSION,
        sections=[("What you declared",
                   [("gravity noise", f"{args.noise:g} mGal"),
                    ("magnetic noise", f"{args.noise_mag:g} nT"),
                    ("density prior sd", f"{args.prior_sd:g} g/cc"),
                    ("susceptibility prior sd", f"{args.prior_sd_chi:g} SI"),
                    ("petrophysical correlation",
                     f"{args.rho_chi_correlation:+g} - AN ASSUMPTION about "
                     f"your rocks. At 0 the two inversions are independent."),
                    ("correlation length", f"{args.corr_len:g} m"),
                    ("unknowns", f"{pmeta['n_unknowns']:,} "
                                 f"({n:,} cells x 2 properties)")],
                   "The coupling lives in the prior, not in a penalty term, "
                   "so the posterior stays closed-form and every gate still "
                   "applies. A cross-gradient structural term would couple "
                   "them more strongly and would cost all of that.")],
        footer=f"Generated by <code>gurutva joint</code> - per-cell maps from "
               f"{args.samples} posterior samples; the mass interval is "
               f"exact, not sampled.")
    out.with_suffix(".json").write_text(json.dumps({
        "mode": "joint", "status": v.status, "claimable": v.claimable,
        "verdict": v.headline, "gates": v.gates, "numbers": numbers,
        "value": m_box * MT, "value_sd": sd_box * MT,
        "value_sd_gravity_alone": sd_alone * MT, "tighter_pct": tighter,
        "correlation": args.rho_chi_correlation,
        "rms_gravity": rms_g, "rms_mag": rms_m,
        "z_blind_m": float(z_blind), "cells": n}, indent=1))
    print(chr(10) + f"report -> {out}")
    return 0 if v.claimable else 1


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="gurutva", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=("report", "selftest", "joint"))
    p.add_argument("--survey", required=True, help="CSV: x,y,z,gz")
    p.add_argument("--field", choices=tuple(FIELDS), default="gravity")
    p.add_argument("--noise", type=float, required=True,
                   help="MEASURED repeatability, mGal or nT. Mandatory.")
    p.add_argument("--prior-sd", type=float, default=None,
                   help="declared per-cell model sd: g/cc, or SI "
                        "susceptibility. Required unless --target-contrast "
                        "is given.")
    p.add_argument("--target-contrast", type=float, default=None,
                   help="derive the prior from the body you are looking for")
    p.add_argument("--target-radius", type=float, default=300.0)
    p.add_argument("--target-depth", type=float, default=800.0)
    p.add_argument("--remanence-q", type=float, default=0.0,
                   help="mag: Koenigsberger ratio, remanent over induced")
    p.add_argument("--remanence-inclination", type=float, default=None)
    p.add_argument("--remanence-declination", type=float, default=None)
    p.add_argument("--survey-mag", default=None,
                   help="joint: the magnetic survey (--survey is the gravity)")
    p.add_argument("--noise-mag", type=float, default=None,
                   help="joint: measured repeatability of the magnetics, nT")
    p.add_argument("--prior-sd-chi", type=float, default=None,
                   help="joint: declared per-cell susceptibility sd")
    p.add_argument("--rho-chi-correlation", type=float, default=0.0,
                   help="joint: declared petrophysical correlation, -0.99..0.99")
    p.add_argument("--corr-len", type=float, required=True,
                   help="declared correlation length, m")
    p.add_argument("--b0", type=float, default=50000.0,
                   help="mag: ambient field strength, nT")
    p.add_argument("--inclination", type=float, default=60.0,
                   help="mag: field inclination, degrees positive down")
    p.add_argument("--declination", type=float, default=0.0,
                   help="mag: field declination, degrees east of north")
    p.add_argument("--cell", type=float, default=200.0)
    p.add_argument("--depth", type=float, default=2000.0,
                   help="mesh depth below the highest station")
    p.add_argument("--top", type=float, default=None)
    p.add_argument("--topography", choices=("auto", "on", "off"),
                   default="auto",
                   help="drape the mesh top to station elevations")
    p.add_argument("--region", type=float, default=2000.0,
                   help="side of the block the result is reported for, m")
    p.add_argument("--region-file", default=None,
                   help="CSV of x,y (or lon,lat) vertices: your lease block")
    p.add_argument("--samples", type=int, default=600)
    p.add_argument("--seed", type=int, default=20260805)
    p.add_argument("--body-depth", type=float, default=800.0)
    p.add_argument("--body-radius", type=float, default=300.0)
    p.add_argument("--body-contrast", type=float, default=-0.3,
                   help="selftest: contrast of the planted body")
    p.add_argument("--out", default="gurutva_report.html")
    args = p.parse_args(argv)
    if args.prior_sd is None and args.target_contrast is None:
        p.error("give --prior-sd, or --target-contrast to derive it")
    if args.mode == "joint":
        for need, why in (("survey_mag", "--survey-mag"),
                          ("noise_mag", "--noise-mag"),
                          ("prior_sd_chi", "--prior-sd-chi")):
            if getattr(args, need) is None:
                p.error(f"joint mode needs {why}")
        return run_joint(args)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
