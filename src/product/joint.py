"""Joint gravity + magnetic inversion, and a prior derived from the target.

Two things that a single-physics, single-knob tool cannot do.

JOINT INVERSION. Gravity constrains density, magnetics constrains
susceptibility, and the same rock carries both. The coupling lives in the
PRIOR, not in a penalty term:

    d = [ d_grav / sigma_g ]   A = [ G_g   0  ]   m = [ rho ]
        [ d_mag  / sigma_m ]       [  0   G_m ]       [ chi ]

    Q_joint = C^-1 (x) Q_spatial          C = [[sr^2,      r sr sc],
                                               [r sr sc,   sc^2   ]]

with (x) the Kronecker product and Q_spatial the unit-amplitude spatial
precision the single-physics path already builds. That is genuinely joint:
because C is not diagonal, gravity data moves the susceptibility posterior
and magnetic data moves the density posterior, through the declared
petrophysical correlation r and nothing else.

The alternative in common use is a cross-gradient structural term, which is
nonlinear and would throw away the closed-form posterior, the exact
functional interval, and every gate that depends on them. A declared
correlation is weaker but it is honest, linear, and its one assumption is
visible in the report instead of buried in an objective function. r = 0
recovers two independent inversions exactly, which is the gate.

PRIOR FROM TARGET. `--prior-sd` is a per-cell marginal sd and thousands of
correlated cells add up, so a user hunting a 0.06 SI body types 0.06 and
declares a prior predicting 889 nT over a survey that reads 2. Warning them
was the first fix; deriving it is the real one. Declare the body you are
looking for — contrast, radius, depth — and the prior amplitude follows from
the anomaly THAT BODY would make. It uses only declared physics and never
looks at the observed data, so it cannot become a way of tuning the prior
until a gate turns green.
"""

import numpy as np
import scipy.sparse as sp

from . import prior as PR


def derive_prior_sd(G, shape, spacing, corr_len, rng, target_anomaly,
                    active=None, n_sample=120):
    """Per-cell prior sd whose prior-predictive matches a declared target.

    The prior-predictive spread is exactly linear in the prior sd, so one
    sample set at sd = 1 fixes the constant and the answer is closed-form.

    target_anomaly: the standard deviation of the anomaly the declared target
    body would produce at these stations, in data units.
    """
    p1, _ = PR.build_regular(shape, spacing, 1.0, corr_len, rng,
                             n_calib=200, active=active)
    spread_per_unit = float(np.std(G @ p1.sample(rng, n_sample)))
    if spread_per_unit <= 0:
        raise ValueError("the operator produces no signal from the prior; "
                         "check the mesh and station geometry")
    return float(target_anomaly / spread_per_unit)


def target_anomaly_std(forward, stations, centers, dims, contrast, radius,
                       depth, top, **kw):
    """Anomaly spread from a declared target body, in data units.

    A compact body of the declared contrast and radius, centred under the
    survey at the declared depth. This is a statement about what you are
    LOOKING FOR, made before any data is consulted.
    """
    cx, cy = stations[:, 0].mean(), stations[:, 1].mean()
    zb = top - depth
    inside = ((np.hypot(centers[:, 0] - cx, centers[:, 1] - cy) < radius)
              & (np.abs(centers[:, 2] - zb) < radius))
    if not inside.any():
        raise ValueError(
            f"a target of radius {radius:g} m at {depth:g} m depth falls "
            f"outside the mesh — increase --depth or --cell")
    m = np.where(inside, contrast, 0.0)
    return float(np.std(forward(m))), int(inside.sum())


def petrophysical_precision(sd_rho, sd_chi, r):
    """C^-1 for the 2x2 density/susceptibility covariance."""
    if not -0.99 <= r <= 0.99:
        raise ValueError(f"correlation {r} must be within +/-0.99: at +/-1 "
                         f"density and susceptibility become the same "
                         f"unknown and the system is singular")
    c = np.array([[sd_rho**2, r * sd_rho * sd_chi],
                  [r * sd_rho * sd_chi, sd_chi**2]])
    return np.linalg.inv(c)


def build_joint_prior(shape, spacing, sd_rho, sd_chi, r, corr_len, rng,
                      active=None):
    """Joint prior over [rho; chi] with a declared petrophysical correlation.

    Q_joint = C^-1 (x) Q_unit, where Q_unit is the spatial precision at unit
    marginal amplitude. The Kronecker ordering matches a block-stacked model
    vector: the first n entries are density, the next n susceptibility.
    """
    p1, meta = PR.build_regular(shape, spacing, 1.0, corr_len, rng,
                                n_calib=300, active=active)
    cinv = petrophysical_precision(sd_rho, sd_chi, r)
    q = sp.kron(sp.csc_matrix(cinv), p1.Q, format="csc")
    return PR.SmoothPrior(q), {**meta, "sd_rho": sd_rho, "sd_chi": sd_chi,
                               "correlation": r, "n_unknowns": q.shape[0]}


def block_operator(g_grav, g_mag, noise_grav, noise_mag):
    """Whitened block operator for the stacked [rho; chi] model.

    Each physics is divided by ITS OWN noise, which is what puts mGal and nT
    on the same footing. Getting that wrong would silently let whichever
    dataset has the larger raw numbers dominate the answer.
    """
    ng, nm = g_grav.shape[0], g_mag.shape[0]
    n = g_grav.shape[1]
    if g_mag.shape[1] != n:
        raise ValueError(f"the two operators disagree on cell count: "
                         f"{n} vs {g_mag.shape[1]}")
    a = np.zeros((ng + nm, 2 * n))
    a[:ng, :n] = g_grav / noise_grav
    a[ng:, n:] = g_mag / noise_mag
    return a


def stack_data(d_grav, d_mag, noise_grav, noise_mag):
    return np.concatenate([np.asarray(d_grav, float) / noise_grav,
                           np.asarray(d_mag, float) / noise_mag])


def split(model):
    """[rho; chi] -> (rho, chi)."""
    n = len(model) // 2
    return model[:n], model[n:]
