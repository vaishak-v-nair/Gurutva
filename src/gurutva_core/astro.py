"""Engine 2: state/trajectory uncertainty in the Earth-Moon system.

NOT a potential-field inversion. Here the gravitational field is KNOWN and
the *state* is uncertain; chaos in the three-body problem turns a small
navigation error into a large volume of possible futures. The product is
the same — gravity with error bars — but the mathematics is ensemble
propagation, not Poisson inversion. Keeping that distinction explicit is
the point of this module living beside, not inside, the inversion core.

Dynamics: the circular restricted three-body problem (CR3BP) in the
Earth-Moon rotating frame, nondimensional units, plus an optional
stochastic solar-radiation-pressure (SRP) acceleration.

    x'' - 2y' = dOmega/dx
    y'' + 2x' = dOmega/dy
    z''       = dOmega/dz
    Omega = (x^2 + y^2)/2 + (1-mu)/r1 + mu/r2

Earth sits at (-mu, 0, 0), the Moon at (1-mu, 0, 0). Distances are in
Earth-Moon separations (384,400 km); time in units where the synodic
period is 2*pi (~27.3 d / 2pi = 4.35 d per unit).
"""

import numpy as np
from scipy.integrate import solve_ivp

MU = 0.012150585609624       # Moon / (Earth+Moon) mass ratio
LU_KM = 384_400.0            # length unit, km
TU_S = 375_190.0             # time unit, s (so that the mean motion is 1)
L1_X = 0.8369151324           # nondimensional x of the Earth-Moon L1


def accel(t, s, srp=np.zeros(3)):
    x, y, z, vx, vy, vz = s
    r1 = np.sqrt((x + MU) ** 2 + y * y + z * z)
    r2 = np.sqrt((x - 1 + MU) ** 2 + y * y + z * z)
    c1 = (1 - MU) / r1**3
    c2 = MU / r2**3
    ax = x + 2 * vy - c1 * (x + MU) - c2 * (x - 1 + MU) + srp[0]
    ay = y - 2 * vx - c1 * y - c2 * y + srp[1]
    az = -c1 * z - c2 * z + srp[2]
    return [vx, vy, vz, ax, ay, az]


def propagate(state0, t_span, n_out=400, srp=np.zeros(3), rtol=1e-10):
    """One deterministic trajectory. Returns (t, states[6, n_out])."""
    t_eval = np.linspace(*t_span, n_out)
    sol = solve_ivp(accel, t_span, np.asarray(state0, float), args=(srp,),
                    t_eval=t_eval, rtol=rtol, atol=1e-12, method="DOP853")
    return sol.t, sol.y


def ensemble(state0, t_span, n=1000, pos_km=1.0, vel_mms=10.0,
             srp_frac=0.05, seed=20260804, n_out=400):
    """Monte-Carlo ensemble under realistic navigation uncertainty.

    pos_km   1-sigma position knowledge (km)
    vel_mms  1-sigma velocity knowledge (mm/s)
    srp_frac 1-sigma fractional uncertainty on solar radiation pressure,
             modelled as a constant acceleration of unknown magnitude
    """
    rng = np.random.default_rng(seed)
    sp = pos_km / LU_KM
    sv = (vel_mms * 1e-6) / (LU_KM / TU_S)          # mm/s -> nondimensional
    nominal_srp = 1e-7                               # ~ typical SRP magnitude
    out = np.empty((n, 6, n_out))
    for i in range(n):
        s0 = np.asarray(state0, float).copy()
        s0[:3] += sp * rng.standard_normal(3)
        s0[3:] += sv * rng.standard_normal(3)
        srp = nominal_srp * (1 + srp_frac * rng.standard_normal()) * np.array(
            [1.0, 0.0, 0.0])
        t, y = propagate(s0, t_span, n_out=n_out, srp=srp)
        out[i] = y
    return t, out


def covariance_tube(ens):
    """Per-epoch mean, 3x3 position covariance, and 95% ellipsoid volume."""
    mean = ens[:, :3, :].mean(axis=0)
    n_t = ens.shape[2]
    cov = np.empty((n_t, 3, 3))
    vol = np.empty(n_t)
    for k in range(n_t):
        c = np.cov(ens[:, :3, k].T)
        cov[k] = c
        ev = np.clip(np.linalg.eigvalsh(c), 1e-30, None)
        # 95% chi2_3 quantile = 7.815 -> semi-axes sqrt(7.815 * eigenvalue)
        vol[k] = 4 / 3 * np.pi * np.prod(np.sqrt(7.815 * ev))
    return mean, cov, vol


def linear_covariance(state0, t_span, P0, n_out=400, eps=1e-7):
    """Linearised (Kalman/STM-style) covariance propagation — what standard
    astrodynamics software reports. Compared against the true ensemble to
    show where linearisation under-reports risk."""
    t, ref = propagate(state0, t_span, n_out=n_out)
    n_t = len(t)
    stm = np.empty((n_t, 6, 6))
    for j in range(6):
        pert = np.asarray(state0, float).copy()
        pert[j] += eps
        _, yj = propagate(pert, t_span, n_out=n_out)
        stm[:, :, j] = ((yj - ref) / eps).T
    covs = np.einsum("kij,jl,kml->kim", stm, P0, stm)
    vol = np.array([4 / 3 * np.pi * np.prod(np.sqrt(
        7.815 * np.clip(np.linalg.eigvalsh(c[:3, :3]), 1e-30, None)))
        for c in covs])
    return t, ref, covs, vol
