# gravity-posterior

*What does a published gravity inversion actually know? Reproducing a published potential-field inversion and computing the posterior uncertainty almost nobody reports.*

> **Status: Day 1.** This README states the plan and the pass/fail gates before any results exist. Every claim below is a commitment, not a result, until its checkbox is ticked and its figure is in this repo.

## Purpose

Inversion software (SimPEG open-source, commercial packages) returns a single regularized model — one picture of the subsurface. The decisions those pictures support deserve a *posterior*: which parts of the model the data actually constrain, and which parts come from the regularization. For a linear forward operator with Gaussian noise and a Gaussian prior, that posterior exists in closed form. This repo computes it for a published inversion, validates it hard, and shows the result honestly.

I'm new to geophysics, coming from software/ML. This repo is deliberately built so that every number in it can be checked without trusting me.

## What this repo will contain (the deliverables)

1. **Published vs recovered** — qualitative comparison with the published section, plus the coarse-mesh re-inversion (own β, own misfit target) that all gates actually run against.
2. **Posterior-uncertainty map** — per-cell standard deviation of the recovered model, active cells only, prior declared (including depth weighting) in the open.
3. **Validation numbers** — the gates below, pass or fail, reported either way.
4. **Wall-clock and memory figures** — everything runs on a laptop.
5. **Provenance defense** — where Σd and Σm come from: noise-floor estimate, sensitivity-to-α figure, with/without-depth-weighting comparison. Pre-answering "isn't this just your regularization choice, replotted?"

## Validation gates (stated before the code exists)

- **G-vs-analytic (first pytest):** the sensitivity matrix's prediction for a single dense cell matches the analytic point-mass/prism solution at several stations to ~6 significant digits.
- **Mean-match:** the closed-form posterior mean equals a tightly-converged direct re-solve of the fixed quadratic at the extracted final β (tolerance stated relative to solver convergence).
- **Monte-Carlo vs closed-form:** 30,000 samples (Matheron's rule, float64, seeded): median relative error < 2%, 95th percentile < 5%, ≤1% of cells beyond 3× their MC standard error.
- **Coverage (SBC-style):** with truths drawn from the prior, the posterior covers at nominal rates across repetitions. A fixed realistic-body run is shown as an ungated diagnostic.

## Reproducibility

- Pinned environment (`requirements.txt`, SimPEG version fixed).
- Every stochastic step uses a seeded `numpy.random.Generator`; the seed is stated here.
- `make figures` regenerates every figure in this README bit-for-bit.

## Progress

- [x] Day 1: repo public ([github.com/vaishak-v-nair/Gurutva](https://github.com/vaishak-v-nair/Gurutva)), environment pinned, this README up
- [x] Week 0: dataset verified — **Utah FORGE 3D gravity** (see the verification note below)
- [ ] Week 1: introduction posted on the SimPEG forum
- [x] G-vs-analytic pytest passing (2026-07-29: SimPEG matches the analytic point mass to ~1e-8 relative; z-up sign convention established empirically — see `src/units_probe.py` and `tests/test_g_analytic.py`)
- [x] Coarse re-inversion running and gated (2026-07-31): own β=2191 by discrepancy at the measured 0.76 mGal floor; achieved RMS 0.760; DC −216.9 mGal fitted and reported; declared choices below
- [ ] Published-model comparison — **BLOCKED on the 2.55 re-reduction bridge, by measurement**: the published model misfits our delivered 2.67 data by 7.94 mGal RMS (10× our fit), because it was built to fit differently-reduced data; comparing models before matching data would be theater
- [x] Regularization choice defended in prose; α→prior mapping (precision form, depth weighting declared) — see "Declared inversion choices" below
- [x] Mean-match test passing (Woodbury vs whitened-CG, rel < 1e-7 — the unwhitened system stalls past 20k CG iterations, exactly the conditioning failure the plan's whitening instruction predicted)
- [x] Closed-form posterior diagonal computed (data-space solve, active cells only) — 2026-07-31
- [x] MC validation gates passed: N=30,000 Matheron samples vs closed form — median rel. err 0.55% (<2%), 95th pct 1.6% (<5%), 0.24% of cells beyond 3× MC standard error (≤1%)
- [x] SBC coverage gate passed: 95.01% pooled coverage of the 95% interval over 20 prior-drawn-truth repetitions (realistic-body diagnostic deferred with the model comparison — blocked on the 2.55 bridge)
- [ ] Provenance defense artifact added
- [ ] Week-4 burn checkpoint: actual hours recorded, windows recomputed

## Week-0 dataset verification note (2026-07-30)

**Chosen: Utah FORGE 3D gravity — DOE Geothermal Data Repository submission 1144** ([gdr.openei.org/submissions/1144](https://gdr.openei.org/submissions/1144), doi:[10.15121/1542061](https://doi.org/10.15121/1542061), CC-BY 4.0). 518 ground-gravity stations (323 used in the published inversion) over the Utah FORGE geothermal site, Milford Valley, Utah. The archive is self-contained: the gravity data, the published SimPEG inversion report (Witter / Innovate Geothermal, May 2019), TWO delivered 268,773-cell 3D density models, per-station misfits (published RMS ≈ 0.03 mGal), and the basement surfaces in a geoh5 workspace — meaning the reproduction can be checked **numerically, cell by cell**.

Verified by independent research agents and then adversarially reviewed (the refuter opened the geoh5 with h5py, recomputed the published RMS from delivered misfits, and cross-matched all 323 inversion stations against the delivered station table — 323/323 exact):

- Open license (CC-BY 4.0 at submission level, GDR + OSTI) — PASS
- Published inversion of this exact data, with numerical model files — PASS
- Ground gravity — PASS · True 3D density model (not gridding/interpolation) — PASS
- Smooth regularization, no active bounds (density histogram shows no clipping; ≤0.08% of cells at extremes) — PASS
- Reachable and laptop-feasible today (local copy under `data/forge/`) — PASS

**Honest caveats, stated before any code runs:**

1. **No settings table exists anywhere** — the report documents no alphas, no β, no norms. The claim of this repo is therefore *reproduce the published RESULT within a stated tolerance*, never "re-run their configuration." Every regularization choice here is ours, declared openly.
2. **"Smooth L2" is an evidence-supported inference, not a documented fact** (Li & Oldenburg lineage in the report; no bound-clipping; histogram modes drifted off the reference densities; the report itself attributes an artifact to "the smoothing regularization"). Our Monte-Carlo and mean-match gates are internal and valid regardless of what norm the original run used.
3. **Geometry comes from the delivered geoh5 only** (readable with h5py — no commercial software), never from utahforge.com (stale links). Reproduction target is **Model #1**; the archive's own README brands Model #2 a test case.
4. **Station height rule: Z = NAVD88 + 0.200 m, NAD83 zone 12.** Never the ellipsoid height column (≈2 mGal systematic error if confused).
5. **Their exact observed vector is unrecoverable** (quantized misfits; nonstandard anomaly-column conventions). Pipeline rule: first reproduce their gCBGA(2.67) exactly from the delivered columns, then re-reduce at 2.55 g/cc + 20 m upward continuation with the method stated, and treat the DC/background level explicitly against the two-layer reference model (2.42/2.65 g/cc).
6. **The coarse-mesh error floor gets measured, not assumed** — and now it is measured (2026-07-30): **floor RMS = 0.755 mGal** (p95 1.55, max 2.19) across the 323 stations — 25× the published 0.0298 mGal RMS, ~6% of the 12.5 mGal signal span. Method: the published 268,773-cell model forward-modeled two ways — exact prisms on its validated native geometry vs SimPEG on our 10,093-cell mesh (mass-preserving projection; 0.37% of |mass| sits above our station-derived topo and was excluded from both sides). Cross-validated: SimPEG and an independent exact-prism implementation agree on the coarse forward to 4×10⁻⁶ mGal. **Consequence, stated before any inversion runs: this repo's honest misfit target is ≈0.76 mGal (floor ⊕ published RMS) — never their 0.03, which no 10k-cell mesh can reach.** Map and numbers: `figures/error_floor.png`, `figures/floor_stats.json`.

**Backup:** Clear Lake Volcanic Field (Mitchell et al. 2023; five of six checks pass; bounds active in 0.07% of cells — usable only with a documented waiver). **Guaranteed fallback:** the SimPEG L2 tutorial (synthetic, all checks pass). **Phase-2 earmark — the space one:** the South American Moho from satellite gravity (Uieda & Barbosa 2017) — mathematically wrong-shaped for Phase 1 (nonlinear interface inversion), but published with *zero* uncertainty estimates, making it a genuine future target for the nonlinear engine: error bars on a continent's crust, from space data.

## Declared inversion choices (pass one, 2026-07-31)

Every choice is ours and stated, because no settings table exists for the original:
**Data**: the delivered, verified gCBGA(2.67 g/cc) — a declared deviation from the published run's 2.55+upward-continued data; the bridge is the next milestone and blocks any model comparison (measured consequence above). **Reference model**: the published two-layer geology as contrast vs the 2.67 background (−0.25 basin fill above the Top-of-Granite surface from the geoh5, −0.02 below). **DC**: one constant (−216.9 mGal) fitted and reported — a local mesh cannot produce the regional level. **Prior**: diagonal Gaussian in precision form, sensitivity-based depth weighting (wr = (Σ G²)^¼, normalized) as cell weights — the Li–Oldenburg role, declared as part of the prior per the plan. **β = 2191**: discrepancy principle bisected onto the *measured* coarse-mesh floor (0.76 mGal), never the unreachable published 0.03. **Solver**: closed-form data-space (Woodbury); independently verified each run by a whitened-CG solve of the same quadratic (mean-match gate, rel < 1e-7). **No bounds anywhere** — the Gaussian posterior stays exact.

## Honesty rules

This repo follows two rules. Nothing is claimed without a figure in this repo backing it, or it is explicitly labeled as aspiration. And the Phase-1 posterior uses a diagonal prior, honestly labeled as such — the smoothness-prior posterior matching the published regularizer is a stretch goal, never a silent substitution.

## Author

Vaishak — software/ML background, learning numerical geophysics in public. Mistakes will be visible here; that is the point.
