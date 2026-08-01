"""The product: a verdict on a model, not another map.

Everyone in this business sells a picture of what is underground. The picture
has no error bars, and companies bet millions on it. Gurutva sells the
judgement on that picture:

    1. where the model is REAL and where it is INVENTED   (informed fraction)
    2. PASS or FAIL on four gates                          (gurutva_core.gates)
    3. ONE number the customer can act on                  (mass_in_region)

Item 3 is what is actually bought. "How many tonnes are in this box, plus or
minus what?" and "how much could be hiding outside the box without me seeing
it?" are the same question asked by a mining company and by a CO2 storage
operator filing with a regulator.

That number is a LINEAR FUNCTIONAL of the model, w^T m — total mass in a
region is a weighted sum of cell densities. Its variance is w^T Sigma w,
which needs the OFF-diagonal posterior covariance: neighbouring cells trade
off against each other, so summing per-cell variances would be wrong (and
would over-state confidence, which is the one direction we are never allowed
to be wrong in). Using the same data-space identity as posterior_diag:

    Sigma = S - S G^T K^-1 G S
    w^T Sigma w = w^T S w - (G S w)^T K^-1 (G S w)

Exact, one extra triangular solve, no n x n matrix ever formed.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import cho_solve

from ..posterior import _factor, posterior_diag


@dataclass
class Verdict:
    """What the customer receives. Deliberately small and quotable."""
    claimable: bool
    headline: str
    gates: list = field(default_factory=list)
    numbers: dict = field(default_factory=dict)

    def __str__(self):
        lines = [self.headline, ""]
        lines += [f"  {g}" for g in self.gates]
        if self.numbers:
            lines.append("")
            lines += [f"  {k}: {v}" for k, v in self.numbers.items()]
        return "\n".join(lines)


def posterior_mean(G, wr, beta, d, mref=None):
    """MAP/posterior mean for the whitened linear-Gaussian problem."""
    s2, cf = _factor(G, wr, beta)
    mref = np.zeros(G.shape[1]) if mref is None else np.asarray(mref)
    return mref + s2 * (G.T @ cho_solve(cf, d - G @ mref))


def functional_sd(G, wr, beta, w):
    """Exact posterior sd of w^T m — the number the customer acts on.

    Never approximate this by sqrt(sum(w^2 * diag)): that ignores the
    negative correlations between neighbouring cells and reports a tighter
    interval than the data supports. Over-confidence is the one failure
    mode this company exists to prevent.
    """
    s2, cf = _factor(G, wr, beta)
    w = np.asarray(w, dtype=float)
    Sw = s2 * w
    v = float(w @ Sw)
    GSw = G @ Sw
    v -= float(GSw @ cho_solve(cf, GSw))
    return np.sqrt(max(v, 0.0))


def mass_in_region(G, wr, beta, d, mask, volumes, mref=None):
    """Total mass inside a region, with an honest interval.

    mask     : bool array over cells — the box the customer cares about
               (an ore shell, a licensed CO2 storage complex, a lease block)
    volumes  : per-cell volume in the same length units as the density model

    Returns (mass, sd) in density-units x volume-units. For g/cc and m^3 the
    result is in tonnes (1 g/cc * 1 m^3 = 1 t), which is the unit both a
    mine plan and a CO2 inventory are written in.
    """
    w = np.where(np.asarray(mask, dtype=bool), np.asarray(volumes, float), 0.0)
    mean = posterior_mean(G, wr, beta, d, mref)
    return float(w @ mean), functional_sd(G, wr, beta, w)


def informed_fraction(G, wr, beta):
    """1 - sd_post/sd_prior per cell. 1 = the data knows, 0 = we assumed it.

    This is the map that stops a customer trusting the deep, pretty part of
    an inversion that is nothing but the regularizer talking.
    """
    var = posterior_diag(G, wr, beta)
    prior_var = 1.0 / (beta * np.asarray(wr, float) ** 2)
    return 1.0 - np.sqrt(var / prior_var)


def exclusion_limit(mean, sd, k=1.96):
    """Upper bound on what could be present where nothing was detected.

    The CO2 operator's sentence ("no more than X escaped the complex") and
    the physicist's exclusion limit are this same object. Reported only for
    cells with no detection, because where there IS a detection the interval,
    not the bound, is the answer.
    """
    mean, sd = np.asarray(mean, float), np.asarray(sd, float)
    detected = mean > k * sd
    return mean + k * sd, detected


def assess(suite, numbers=None, subject="this model"):
    """Turn a gate suite into the sentence the customer reads first."""
    numbers = numbers or {}
    # Delegate to the suite rather than re-deriving the wording here. When the
    # recovery gate was added, this function still said "all four gates pass"
    # for a run whose fifth gate had failed — a headline drifting out of sync
    # with the gates it summarises is exactly the failure mode this product
    # sells against.
    v = suite.verdict()
    if suite.claimable:
        head = f"{v} The numbers below for {subject} are supported by the data."
    else:
        head = (f"{v}. Numbers below are DIAGNOSTIC ONLY and must not be used "
                f"to support a decision about {subject}.")
    return Verdict(suite.claimable, head,
                   [str(g) for g in suite.reports], numbers)
