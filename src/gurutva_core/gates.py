"""The gate suite — domain-independent, and the reason Gurutva is trustworthy.

Each gate was forged by a specific failure in this repo:

  licensing    S2 (2026-08-03): a neural posterior at an out-of-distribution
               observation gave 0.49 vs 1.07 across identical trainings. The
               claim was retracted. Gate: the observation must be something
               the model could plausibly produce.
  calibration  standard SBC — but note what it CANNOT do (see adequacy).
  stability    the gate that caught S2: two independently seeded runs must
               agree AT THE REAL OBSERVATION, not merely in simulation.
  adequacy     S3 (2026-08-04): licensing, calibration AND stability all
               passed for a model whose posterior missed the observed
               gravity by 120.7 mGal against a published 19.3. The first
               three are self-consistency tests; only this one asks whether
               the model can reproduce reality.

A model may be claimed only when all four hold. Anything less is reported
as a diagnostic, never as a map.
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class GateReport:
    name: str
    passed: bool
    value: float
    threshold: str
    note: str = ""

    def __str__(self):
        mark = "PASS" if self.passed else "FAIL"
        return f"[{mark}] {self.name}: {self.value:.4g} ({self.threshold}) {self.note}"


@dataclass
class GateSuite:
    reports: list = field(default_factory=list)

    def add(self, r: GateReport):
        self.reports.append(r)
        print(str(r), flush=True)
        return r

    @property
    def claimable(self) -> bool:
        return len(self.reports) == 4 and all(r.passed for r in self.reports)

    def verdict(self) -> str:
        if self.claimable:
            return "CLAIMABLE — all four gates pass"
        failed = [r.name for r in self.reports if not r.passed]
        if len(self.reports) < 4:
            return f"INCOMPLETE — only {len(self.reports)}/4 gates run"
        return f"NOT CLAIMED — failed: {', '.join(failed)}"


def licensing(x_obs, x_sim, max_percentile=95.0) -> GateReport:
    """Is the real observation something this model could have produced?

    CAUTION (measured, S3): this gate PASSES trivially when the prior
    predictive spread is enormous — an over-wide prior makes every
    observation look typical. Passing licensing is necessary, never
    sufficient. Pair it with `adequacy`, always.
    """
    mu, sd = x_sim.mean(axis=0), x_sim.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    d_sim = np.linalg.norm((x_sim - mu) / sd, axis=1)
    d_obs = float(np.linalg.norm((np.asarray(x_obs) - mu) / sd))
    pct = float((d_sim < d_obs).mean() * 100)
    return GateReport("licensing", pct < max_percentile, pct,
                      f"< {max_percentile} pct",
                      "observation must be plausible under the prior")


def calibration(ranks, tail_target=0.05, cover_target=0.90,
                tail_tol=0.03, cover_tol=0.06) -> GateReport:
    """SBC: rank statistics of truth within posterior samples.

    Tests self-consistency INSIDE the model's own world only.
    """
    ranks = np.asarray(ranks)
    tail = float(((ranks < tail_target / 2) | (ranks > 1 - tail_target / 2)).mean())
    cover = float(((ranks > 0.05) & (ranks < 0.95)).mean())
    ok = (abs(tail - tail_target) < tail_tol
          and abs(cover - cover_target) < cover_tol)
    return GateReport("calibration", ok, cover,
                      f"tails~{tail_target}, cover~{cover_target}",
                      f"tails={tail:.3f}")


def stability(est_a, est_b, atol, ratio_lo=0.8, ratio_hi=1.25) -> GateReport:
    """Do two independent runs agree AT THE REAL OBSERVATION?

    est_a/est_b: (mean, sd) arrays from independently seeded inferences.
    """
    (mA, sA), (mB, sB) = est_a, est_b
    gap = float(np.median(np.abs(np.asarray(mA) - np.asarray(mB))))
    ratio = float(np.median(np.asarray(sA) / np.asarray(sB)))
    ok = gap < atol and ratio_lo < ratio < ratio_hi
    return GateReport("stability", ok, gap, f"< {atol:g}",
                      f"sd ratio {ratio:.3f}")


def adequacy(pp_residual, reference_residual, factor=3.0) -> GateReport:
    """CAN THE MODEL REPRODUCE REALITY? The gate the other three cannot be.

    pp_residual: posterior-predictive misfit to the observation.
    reference_residual: what a credible model achieves on the same data.

    TWO-SIDED since 2026-08-05, and the second side was forced by a
    measurement, not by taste. S3 failed by MISSING the data (120.7 vs 19.3)
    so the gate was born one-sided. Then Utah FORGE, run with a prior
    declared from rock physics instead of from a tuned regularizer, came in
    at 0.37 mGal against a MEASURED noise floor of 0.76 — beating the noise.
    A model that explains the noise is not a better model; it has absorbed
    it, and its error bars are then fiction in the opposite direction.
    Both failures are inadequacy. The gate now says so.
    """
    hi = factor * reference_residual
    lo = reference_residual / factor
    ok = lo < pp_residual < hi
    if ok:
        side = ""
    elif pp_residual >= hi:
        side = " — UNDERFITS: cannot reproduce the data"
    else:
        side = " — OVERFITS: has absorbed the noise floor"
    return GateReport("adequacy", ok, pp_residual,
                      f"{lo:.3g} < pp < {hi:.3g}",
                      "the only gate that tests reality" + side)
