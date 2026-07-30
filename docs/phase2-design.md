# Phase-2 design note — the nonlinear engine

Status: the gating artifact required before any Phase-2 spend. Written 2026-08-01.
House rule applies throughout: every gate calibrated to the numerics that produce its numbers; pins set from measurement, never taste.

## The measurement this design stands on

The Phase-1 machinery answers Phase-2's hardest question exactly. Eigen-analysis of the resolution operator for the bridged FORGE run (β\*=2918):

- **Effective resolved parameters: trace(R) = 32.4** (of 10,093 cells / 323 data)
- **72 modes carry 90% of all resolution; 112 carry 99%**; mode gains fall from 0.99 (mode 1) to 0.13 (mode 72)

The honest dimensionality of this inverse problem is ~10², not 10⁴. That fact, not fashion, dictates the method.

## Why go beyond linear-Gaussian at all

Phase 1's closed form is exact for its assumptions. Three real things break them, in increasing order of ambition:
1. **Non-Gaussian priors** — geology is bimodal (basin fill *or* basement, rarely between); a Gaussian prior smears what a mixture prior would sharpen.
2. **Outlier-robust noise** — field data has bad stations; Student-t likelihoods have no closed form.
3. **Nonlinear parameterizations** — interface geometry. This is the earmarked space debut: the South American Moho from satellite gravity (Uieda & Barbosa 2017), a nonlinear interface inversion published with **zero uncertainties**. Phase 2's endgame is putting error bars on a continent.

## Method commitment (satisfies the review's blocking D9)

**Neural Posterior Estimation (NPE) via the `sbi` package (version pinned at adoption), trained on the coefficients of the top k = 128 KL modes of the Phase-1 linear-Gaussian posterior** (covers >99% of measured resolution; margin above k99=112). Remaining modes stay at their prior — justified by measurement, not convenience. Vanilla NPE on 10⁴ raw cells is not attempted; field-native score-based methods are the recorded fallback if mode-space NPE fails its gates.

Known honest limitation, recorded now: the KL basis is derived from the *linear* operator. It is valid for stages S1–S2 (same forward, non-Gaussian priors/likelihoods). For the Moho (different, nonlinear forward), the reduced basis must be rebuilt from that problem's own linearization — the basis does not transfer and we will not pretend it does.

## Stages, each with an adjudicable gate

- **S1 — training wheels (FORGE, linear limit).** Simulator = the validated G (a matvec; simulations are free). Train NPE on (mode coefficients ← simulated data). **Gate:** per-mode posterior mean/σ must match the analytic Gaussian, with acceptance bands calibrated by SBC repetitions (never raw tolerances), plus SBC rank-histogram uniformity. The exact answer exists; NPE must find it.
- **S2 — the first beyond-Gaussian result (FORGE).** Replace the Gaussian prior with a declared basin/basement mixture in mode space; likelihood optionally Student-t. **Gate:** SBC calibration holds; the posterior visibly sharpens the interface relative to Phase 1 (figure, not adjective) with coverage intact.
- **S3 — the Moho spike (kill-criterion attached).** Prototype a fast tesseroid forward (harmonica / coarse grids). **Kill-criterion:** if the surrogate cannot reach forward accuracy consistent with the data noise within the compute budget, the Moho defers — recorded, not fudged — and Phase 2 completes on S2 + E2.
- **E2 — the dark-matter twin** (pulled forward per burn checkpoint): the same engine rendering a sensor-network exclusion figure; honest-negative clause stands.

## Compute budget

Laptop + free Colab/Kaggle GPU tiers. S1/S2 training: 10⁵–10⁶ simulated draws at k=128 — hours, not days (simulations are matvecs). S3 tesseroids are the only real cost driver — hence the spike-with-kill-criterion, budgeted before commitment.

## Risks

| Risk | Mitigation |
|---|---|
| NPE miscalibrated (the field's classic silent failure) | SBC gates at every stage; no calibration, no claim |
| KL basis bakes in linearity | Scoped: S1–S2 only; Moho gets its own basis; recorded above |
| `sbi` API churn | Version pinned at adoption, in requirements.txt |
| Founder canary (unaided command) | Self-test scheduled before S1 completes (month-1 review, 2026-09-01) |
| Community signal shifts priorities | Forum replies feed the month-1 review; this note bends to evidence |

## Success criteria

Phase 2 is done when: S1's gate passes (NPE = analytic in the linear limit), **one beyond-linear posterior ships with calibration intact (S2)**, and E2 ships (positive or honest-negative). Outer bound stays the plan's month 9; the burn record suggests far earlier, and we decline to promise it.
