"""`gurutva` — the command a customer actually runs.

Everything else in this repo is machinery. This is the product: point it at a
gravity survey you own and get back the verdict report.

    py -3 -m src.product.cli report   --survey mysurvey.csv --noise 0.05 \
        --prior-sd 0.25 --corr-len 500 --out report.html

    py -3 -m src.product.cli selftest --survey mysurvey.csv --noise 0.05 \
        --prior-sd 0.25 --corr-len 500 --body-depth 800 --out capability.html

Two modes, because a customer has two different questions.

  report    You have data. What does it support? Runs the four core gates and
            returns the interval on mass in a region, the informed-fraction
            map, and the depth below which the model is invented. Gate 5
            CANNOT run here — there is no known truth on real data — and the
            verdict says so rather than letting silence read as success.

  selftest  You have a survey design, or you are about to pay for one. We
            plant a body of known size and depth, simulate what YOUR station
            layout and YOUR noise would record, and invert it. Gate 5 runs,
            because now the truth is known. This answers "what can my survey
            actually prove", which is the question worth asking before the
            money is spent, not after.

INPUT FORMAT. One CSV, four columns, header row required:

    x,y,z,gz
    331000,4263000,1650.2,-112.44

    x, y   horizontal position in METRES, any consistent projected system
    z      station elevation in METRES, z-UP (a mass below reads negative)
    gz     observed gravity anomaly in mGal, background already removed

Nothing is guessed. `--noise` is mandatory and must be YOUR measured
repeatability, because the adequacy gate is meaningless without it, and a
tool that invents a noise floor is inventing its own passing grade.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src import prism_forward as pf
from src.gurutva_core import gates
from src.product import prior as PR, report, verdict as V

MAX_CELLS = 60_000       # laptop guard; refuse rather than swap for an hour


def load_survey(path):
    """Read x,y,z,gz and REFUSE anything it cannot honestly invert.

    Every guard below is here because a hostile pass over the CLI hit it:
    a semicolon-delimited export (the European default) died inside
    genfromtxt, a UTF-8 BOM turned the first column into "﻿x" so the
    error blamed a missing x, a single NaN sailed through and was caught
    three gates later by luck, and a collinear survey crashed in the sparse
    factoriser with "zero-size array to reduction operation minimum".

    A tool aimed at people who are not programmers has to fail at the door,
    in their language, naming the row.
    """
    raw = Path(path).read_text(encoding="utf-8-sig", errors="replace")  # eats BOM
    head = raw.splitlines()[0] if raw.strip() else ""
    delim = max((",", ";", "\t"), key=lambda c: head.count(c))
    if head.count(delim) < 3:
        raise SystemExit(
            f"{path}: could not find 4 columns in the header line.\n"
            f"  got: {head[:70]!r}\n"
            f"  need: x,y,z,gz  (comma, semicolon or tab separated)")
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")          # we report the errors ourselves
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
    missing = [c for c in ("x", "y", "z", "gz") if c not in names]
    if missing:
        raise SystemExit(
            f"{path}: missing column(s) {', '.join(missing)}.\n"
            f"  found: {', '.join(names) or '(none)'}\n"
            f"  need:  x,y,z,gz  (metres, metres, metres, mGal)")

    st = np.column_stack([rows["x"], rows["y"], rows["z"]]).astype(float)
    d = np.asarray(rows["gz"], dtype=float)

    bad = ~np.isfinite(np.column_stack([st, d[:, None]])).all(axis=1)
    if bad.any():
        first = int(np.argmax(bad)) + 2          # +2: 1-based, past the header
        raise SystemExit(
            f"{path}: {int(bad.sum())} row(s) contain a blank or non-numeric "
            f"value; the first is line {first}.\n"
            f"  Gurutva will not guess a missing reading. Remove those rows "
            f"or fill them in.")
    if len(st) < 8:
        raise SystemExit(f"{path}: only {len(st)} stations. Too few to say "
                         "anything honest about a 3-D density field.")

    ext_x = st[:, 0].max() - st[:, 0].min()
    ext_y = st[:, 1].max() - st[:, 1].min()
    if min(ext_x, ext_y) <= 0:
        raise SystemExit(
            f"{path}: the stations span {ext_x:,.0f} m east-west and "
            f"{ext_y:,.0f} m north-south.\n"
            f"  A survey along a single line cannot constrain a 3-D density "
            f"field — there is no second horizontal direction for the data to "
            f"resolve. Use a 2-D layout, or a profile-specific tool.")

    uniq = len({(round(a, 3), round(b, 3)) for a, b in st[:, :2]})
    if uniq < len(st):
        print(f"  warning: {len(st) - uniq} station(s) repeat an existing "
              f"x,y position. Repeats add no new information and will make "
              f"the survey look better resolved than it is.")
    return st, d


def build_mesh(stations, cell, top, bottom, margin=0.25):
    """A regular block under the survey. Extent from the data, not from taste."""
    x0, x1 = stations[:, 0].min(), stations[:, 0].max()
    y0, y1 = stations[:, 1].min(), stations[:, 1].max()
    mx, my = (x1 - x0) * margin, (y1 - y0) * margin
    xs = np.arange(x0 - mx + cell / 2, x1 + mx, cell)
    ys = np.arange(y0 - my + cell / 2, y1 + my, cell)
    zs = np.arange(bottom + cell / 2, top, cell)
    # A zero-length axis used to reach the sparse factoriser and die there
    # with "zero-size array to reduction operation minimum", which tells a
    # geologist nothing. Catch it here, in their units.
    for n, span, hint in ((len(xs), x1 - x0, "--cell smaller than the survey width"),
                          (len(ys), y1 - y0, "--cell smaller than the survey height"),
                          (len(zs), top - bottom, "--depth larger than --cell")):
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
    return centers, dims, (len(xs), len(ys), len(zs)), np.full(len(centers),
                                                              cell ** 3)


def core_gates(G, prior, d, noise, rng, n_post, mean, sd):
    suite = gates.GateSuite()
    sims = (G @ prior.sample(rng, 300)
            + rng.standard_normal((G.shape[0], 300))).T
    suite.add(gates.licensing(d, sims))

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
    return suite


def run(args):
    stations, d_obs = load_survey(args.survey)
    top = args.top if args.top is not None else float(stations[:, 2].min())
    centers, dims, shape, vol = build_mesh(stations, args.cell, top,
                                           top - args.depth)
    rng = np.random.default_rng(args.seed)
    print(f"survey: {len(stations)} stations | mesh {shape[0]}x{shape[1]}x"
          f"{shape[2]} = {len(centers):,} cells at {args.cell:g} m")

    G_raw = pf.prism_matrix(stations, centers, dims)
    G = G_raw / args.noise
    prior, pmeta = PR.build_regular(shape, (args.cell,) * 3, args.prior_sd,
                                    args.corr_len, rng)

    truth = None
    if args.mode == "selftest":
        # plant a compact body at the declared depth, directly under the
        # survey centre — the thing the customer wants to know they'd see
        cx, cy = stations[:, 0].mean(), stations[:, 1].mean()
        zb = top - args.body_depth
        inside = ((np.hypot(centers[:, 0] - cx, centers[:, 1] - cy) < args.body_radius)
                  & (np.abs(centers[:, 2] - zb) < args.body_radius))
        if not inside.any():
            raise SystemExit("planted body falls outside the mesh — increase "
                             "--depth or reduce --body-depth")
        truth = np.where(inside, args.body_contrast, 0.0)
        d_obs = G_raw @ truth + args.noise * rng.standard_normal(len(stations))
        print(f"  self-test: {int(inside.sum())} cells at {args.body_contrast:+g} "
              f"g/cc, {args.body_depth:g} m below the surface; "
              f"peak signal {np.abs(G_raw @ truth).max():.3f} mGal against "
              f"{args.noise:g} mGal noise")

    dw = d_obs / args.noise
    mean = PR.posterior_mean(G, prior, dw)
    sd, sd_err = PR.posterior_sd(G, prior, np.random.default_rng(11), args.samples)
    rms = float(np.sqrt(np.mean((G_raw @ mean - d_obs) ** 2)))

    suite = core_gates(G, prior, dw, args.noise, rng, args.samples, mean, sd)
    # Reference is what THIS model predicts of itself, not the raw noise —
    # a flexible model should fit better than the noise, and comparing to the
    # noise punishes it for being correct. See prior.expected_residual.
    ref = args.noise * PR.expected_residual(G, prior)
    suite.add(gates.adequacy(rms, ref))
    if truth is not None:
        suite.add(gates.recovery(mean, sd, truth))

    # ---- the numbers a decision turns on --------------------------------
    prior_sd_cell = prior.marginal_sd(np.random.default_rng(55), 400)
    inf = 1.0 - sd / prior_sd_cell
    z = centers[:, 2]
    edges = np.linspace(z.min(), z.max(), 20)
    lit = [0.5 * (edges[i] + edges[i + 1]) for i in range(len(edges) - 1)
           if np.any((z >= edges[i]) & (z < edges[i + 1]))
           and np.median(inf[(z >= edges[i]) & (z < edges[i + 1])]) > 0.05]
    z_blind = min(lit) if lit else z.max()

    cx, cy = stations[:, 0].mean(), stations[:, 1].mean()
    half = args.region / 2.0
    box = ((np.abs(centers[:, 0] - cx) < half)
           & (np.abs(centers[:, 1] - cy) < half) & (z > z_blind))
    w = np.where(box, vol, 0.0)
    m_box = float(w @ mean)
    sd_box = PR.functional_sd(G, prior, w)
    naive = float(np.sqrt(np.sum(w**2 * sd**2)))
    MT = 1e-6

    numbers = {
        f"excess mass, {args.region / 1000:g}x{args.region / 1000:g} km block above the blind depth":
            f"{m_box * MT:+,.1f} Mt  (95%: {(m_box - 1.96 * sd_box) * MT:+,.1f} "
            f"to {(m_box + 1.96 * sd_box) * MT:+,.1f} Mt)",
        "is that a detection?":
            ("YES — the 95% interval excludes zero"
             if abs(m_box) > 1.96 * sd_box else
             f"NO — consistent with zero. 95% upper limit "
             f"{(abs(m_box) + 1.96 * sd_box) * MT:,.1f} Mt"),
        "your model is INVENTED below":
            f"{z_blind:,.0f} m elevation ({z.max() - z_blind:,.0f} m below the "
            f"top of the mesh) — deeper than this, the typical cell is the "
            f"prior, not your data",
        "cells your survey informs (>5%)":
            f"{float(np.mean(inf > 0.05)) * 100:.1f}% of {len(inf):,}",
        "per-cell shortcut would have said":
            f"+/-{naive * MT:,.1f} Mt instead of +/-{sd_box * MT:,.1f} Mt "
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
            "planted body":
                f"{args.body_contrast:+g} g/cc, radius {args.body_radius:g} m, "
                f"{args.body_depth:g} m deep",
            **numbers,
        }

    v = V.assess(suite, numbers,
                 subject=("this survey design" if truth is not None
                          else Path(args.survey).name))
    print("\n" + str(v))

    out = Path(args.out)
    report.render(
        out,
        "Gurutva — survey capability" if truth is not None else "Gurutva — model verdict",
        f"{Path(args.survey).name} · {len(stations)} stations · "
        f"{args.noise:g} mGal declared noise",
        v,
        sections=[("What you declared",
                   [("noise floor", f"{args.noise:g} mGal — YOUR measurement, "
                     "not our guess"),
                    ("prior density sd", f"{args.prior_sd:g} g/cc"),
                    ("prior correlation length", f"{args.corr_len:g} m"),
                    ("mesh", f"{shape[0]}x{shape[1]}x{shape[2]} at "
                     f"{args.cell:g} m, {args.depth:g} m deep"),
                    ("posterior fit", f"{rms:.4g} mGal against an expected "
                     f"{ref:.4g} mGal for a model with this much freedom")],
                   "Change any of these and the answer changes. That is not a "
                   "weakness of the method, it is the part every other tool "
                   "hides. Declare them from your site, never tune them until "
                   "a gate turns green.")],
        footer=f"Generated by <code>gurutva {args.mode}</code> · per-cell map "
               f"from {args.samples} posterior samples (MC error "
               f"{float(np.median(sd_err / sd)) * 100:.1f}%); mass intervals "
               f"are exact, not sampled.")
    out.with_suffix(".json").write_text(json.dumps({
        "mode": args.mode, "survey": str(args.survey),
        "claimable": v.claimable, "verdict": v.headline,
        "gates": v.gates, "numbers": numbers,
        "declared": {"noise": args.noise, "prior_sd": args.prior_sd,
                     "corr_len": args.corr_len, "cell": args.cell,
                     "depth": args.depth},
        "mass_Mt": m_box * MT, "mass_sd_Mt": sd_box * MT,
        "z_blind_m": float(z_blind), "rms": rms}, indent=1))
    print(f"\nreport -> {out}\njson   -> {out.with_suffix('.json')}")
    return 0 if v.claimable else 1


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="gurutva", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=("report", "selftest"))
    p.add_argument("--survey", required=True, help="CSV with x,y,z,gz")
    p.add_argument("--noise", type=float, required=True,
                   help="MEASURED repeatability in mGal. Mandatory: the "
                        "adequacy gate is meaningless without it.")
    p.add_argument("--prior-sd", type=float, required=True,
                   help="declared density-contrast sd, g/cc (from your rocks)")
    p.add_argument("--corr-len", type=float, required=True,
                   help="declared correlation length, m (how your rock varies)")
    p.add_argument("--cell", type=float, default=200.0)
    p.add_argument("--depth", type=float, default=2000.0,
                   help="mesh depth extent below the shallowest station")
    p.add_argument("--top", type=float, default=None)
    p.add_argument("--region", type=float, default=2000.0,
                   help="side length of the block the mass is reported for, m")
    p.add_argument("--samples", type=int, default=600)
    p.add_argument("--seed", type=int, default=20260805)
    p.add_argument("--body-depth", type=float, default=800.0,
                   help="selftest: depth of the planted body")
    p.add_argument("--body-radius", type=float, default=300.0)
    p.add_argument("--body-contrast", type=float, default=-0.3,
                   help="selftest: density contrast of the planted body, g/cc")
    p.add_argument("--out", default="gurutva_report.html")
    return run(p.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
