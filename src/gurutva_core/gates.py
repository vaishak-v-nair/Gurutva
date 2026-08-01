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
    """The four core gates, plus `recovery` whenever a truth is available.

    `recovery` is not a fifth core gate, because it CANNOT always run: it
    needs a known answer, and the whole point of an inversion is that you do
    not have one. So the suite tracks it separately and the verdict states
    plainly whether it was run. A claim that has never been checked against a
    known truth is still a claim, but the reader is told so.
    """

    CORE = ("licensing", "calibration", "stability", "adequacy")
    reports: list = field(default_factory=list)

    def add(self, r: GateReport):
        self.reports.append(r)
        print(str(r), flush=True)
        return r

    def _core_hits(self):
        # prefix match: "calibration(MC)" satisfies "calibration"
        return {c: next((r for r in self.reports if r.name.startswith(c)), None)
                for c in self.CORE}

    @property
    def recovery_report(self):
        return next((r for r in self.reports if r.name.startswith("recovery")),
                    None)

    @property
    def claimable(self) -> bool:
        hits = self._core_hits()
        if any(r is None or not r.passed for r in hits.values()):
            return False
        rec = self.recovery_report
        return rec.passed if rec is not None else True

    @property
    def status(self) -> str:
        """One of INCOMPLETE / NOT CLAIMED / PROVISIONAL / CLAIMABLE.

        PROVISIONAL exists because the old two-state wording argued with
        itself: a real-data run printed the green word CLAIMABLE and then, in
        the same breath, "nothing here has been checked against a right
        answer". A reader who scans stops at the green word. Three genuinely
        different situations now get three different words, and CLAIMABLE
        goes back to meaning something strong and rare.
        """
        hits = self._core_hits()
        if any(r is None for r in hits.values()):
            return "INCOMPLETE"
        if any(not r.passed for r in self.reports):
            return "NOT CLAIMED"
        return "CLAIMABLE" if self.recovery_report is not None else "PROVISIONAL"

    def verdict(self) -> str:
        st = self.status
        if st == "INCOMPLETE":
            missing = [c for c, r in self._core_hits().items() if r is None]
            return f"INCOMPLETE — core gate(s) not run: {', '.join(missing)}"
        if st == "NOT CLAIMED":
            failed = [r.name for r in self.reports if not r.passed]
            return f"NOT CLAIMED — failed: {', '.join(failed)}"
        if st == "PROVISIONAL":
            return ("PROVISIONAL — all four core gates pass, but nothing here "
                    "has been checked against a known right answer. Real data "
                    "has none. Run a self-test on this survey design to find "
                    "out what it can actually recover.")
        return ("CLAIMABLE — all four core gates pass AND the result was "
                "checked against a known right answer.")


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


def recovery(est, sd, truth, k=1.96, min_cover=0.90,
             max_miss_sigma=None) -> GateReport:
    """DOES THE INTERVAL CONTAIN A KNOWN RIGHT ANSWER?

    Born 2026-08-05 from the dark-matter run, which passed all four core
    gates and then missed a truth we happened to know by 11.6 sigma. The
    cause was prior shrinkage: the true peak was a 4-sigma excursion under
    the declared prior, so the posterior pulled it toward zero and the
    interval never covered it.

    Why none of the four could catch that, structurally:
      licensing   asks whether the DATA is plausible under the prior. It
                  says nothing about whether the TRUTH is.
      calibration (SBC) draws its test truths FROM the prior. A truth the
                  prior cannot produce is outside its universe by
                  construction, so coverage looks perfect while real
                  coverage fails.
      stability   two seeds agree on the same shrunken answer.
      adequacy    a shrunken field still reproduces the data within noise,
                  because the operator smooths.

    So this gate exists, and it is deliberately NOT a fifth core gate: it
    needs a known answer, and an inversion normally has none. Run it on
    synthetic truths, on a withheld reference model, or on a benchmark —
    and when you cannot run it, the suite says so out loud rather than
    letting silence read as success.

    est/sd/truth: arrays (a field) or scalars (a single functional).
    Reports the fraction covered and the WORST miss in sigma, because a
    field can cover 99% of cells and still be badly wrong where it matters.
    """
    est = np.atleast_1d(np.asarray(est, dtype=float))
    sd = np.atleast_1d(np.asarray(sd, dtype=float))
    truth = np.atleast_1d(np.asarray(truth, dtype=float))
    miss = np.abs(est - truth) / np.where(sd > 0, sd, np.inf)
    cover = float(np.mean(miss < k))
    worst = float(np.max(miss))
    n = len(miss)
    # The worst-miss threshold MUST scale with how many cells you looked at.
    # A perfectly calibrated posterior over n cells has an expected maximum
    # |z| of about sqrt(2 ln n) — 4.2 at n=8000 — so a fixed 3-sigma cap is a
    # max-over-cells gate that correct code fails by construction. This
    # repo's own README warns against exactly that, and the first version of
    # this gate did it anyway: a survey self-test covering 99.4% of cells was
    # failed by a lone 4.0-sigma cell that calibration predicts you will see.
    if max_miss_sigma is None:
        max_miss_sigma = max(3.0, 1.15 * float(np.sqrt(2.0 * np.log(max(n, 2)))))
    ok = cover >= min_cover and worst <= max_miss_sigma
    return GateReport("recovery", ok, cover,
                      f"cover >= {min_cover:.2f} and worst miss <= "
                      f"{max_miss_sigma:.2f} sigma (n={n:,})",
                      f"worst {worst:.1f} sigma — the gate the other four "
                      f"cannot be")
