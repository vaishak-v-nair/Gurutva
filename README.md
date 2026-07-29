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

- [ ] Day 1: repo public, environment pinned, this README up
- [ ] Week 0: dataset verified (public license · published inversion · potential-field · L2/Tikhonov, not IRLS · no active bound constraints) — verification note added here
- [ ] Week 1: introduction posted on the SimPEG forum
- [x] G-vs-analytic pytest passing (2026-07-29: SimPEG matches the analytic point mass to ~1e-8 relative; z-up sign convention established empirically — see `src/units_probe.py` and `tests/test_g_analytic.py`)
- [ ] Published inversion reproduced (qualitative) + coarse-mesh re-inversion (gated object)
- [ ] Regularization choice defended in prose; α→prior mapping (precision form, depth weighting declared) documented
- [ ] Mean-match test passing
- [ ] Closed-form posterior diagonal computed (whitened data-space solve, active cells only)
- [ ] MC validation gates passed
- [ ] SBC coverage gate passed; realistic-body diagnostic figure added
- [ ] Provenance defense artifact added
- [ ] Week-4 burn checkpoint: actual hours recorded, windows recomputed

## Honesty rules

This repo follows two rules. Nothing is claimed without a figure in this repo backing it, or it is explicitly labeled as aspiration. And the Phase-1 posterior uses a diagonal prior, honestly labeled as such — the smoothness-prior posterior matching the published regularizer is a stretch goal, never a silent substitution.

## Author

Vaishak — software/ML background, learning numerical geophysics in public. Mistakes will be visible here; that is the point.
