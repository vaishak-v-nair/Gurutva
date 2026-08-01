"""Magnetics: total-field anomaly of a rectangular prism, from the gravity
gradient tensor.

Why this route rather than a separate magnetic formula. Poisson's relation
says a uniformly magnetised body and a uniformly dense body of the same shape
produce potentials that differ only by a directional derivative. Concretely,
with Phi = integral dv / |r - r'| (the shape integral this repo already
evaluates for gravity),

    A(r)   = -(mu0/4pi) M  m_hat . grad Phi          magnetic scalar potential
    B      = -grad A       = (mu0/4pi) M grad grad Phi . m_hat
    dT     ~= b_hat . B    = (mu0/4pi) M (b_hat^T GAMMA m_hat)

with GAMMA = grad grad Phi, the (dimensionless) gravity gradient tensor of the
same prism. For INDUCED magnetisation M = chi * B0 / mu0, so

    dT [nT] = (B0 [nT] / 4pi) * chi * (b_hat^T GAMMA m_hat)

One geometry kernel serves both physics. That matters here for a reason
beyond elegance: the prism corner sum in `prism_forward` has already been
burned once by the pi-leak (arctan2's extended range breaking the 8-corner
cancellation, a factor of -pi x 10^4). Deriving a second, independent
magnetic formula would mean a second chance to make that class of error in a
place no existing test looks. Instead GAMMA is gated three ways:

  1. Laplace: GAMMA_xx + GAMMA_yy + GAMMA_zz = 0 outside the body. Free, and
     it catches almost every sign or branch mistake.
  2. Against the already-gated gz: d(gz)/dx must equal the corresponding
     tensor row, checked by central differences.
  3. Symmetry: GAMMA_xy == GAMMA_yx, and so on.

The sign convention is NOT asserted from the algebra. It is pinned by a
physical case in tests/test_mag.py: at the magnetic pole (field straight
down), a susceptible body produces a POSITIVE total-field anomaly directly
above it. Same discipline as `units_probe.py` used for gravity.

REMANENCE is supported through the Koenigsberger ratio Q. Total magnetisation
is chi * H0 * (m_ind + Q m_rem): once Q and the remanent direction are
declared, the bracket is a known vector, the unknown stays a scalar per cell,
and the problem stays linear. A reversely magnetised body (Q > 1, direction
opposed to the field) then produces a NEGATIVE anomaly over positive
susceptibility, which is what reversed bodies actually do and what an
induced-only model gets badly wrong.
"""

import numpy as np

MU0_OVER_4PI = 1e-7          # T m / A, exact by definition of mu0


def _corner_sums(stations, centers, dims):
    """The six independent GAMMA components, per prism, per station.

    Returns (n_stations, n_cells, 6): xx, yy, zz, xy, xz, yz.
    Dimensionless: the log terms carry length units that cancel inside the
    alternating 8-corner sum, exactly as they do for gz.
    """
    ns, nc = len(stations), len(centers)
    out = np.zeros((ns, nc, 6))
    for s_idx, (sx, sy, sz) in enumerate(stations):
        x = np.stack([centers[:, 0] - dims[:, 0] / 2 - sx,
                      centers[:, 0] + dims[:, 0] / 2 - sx], axis=1)
        y = np.stack([centers[:, 1] - dims[:, 1] / 2 - sy,
                      centers[:, 1] + dims[:, 1] / 2 - sy], axis=1)
        z = np.stack([centers[:, 2] - dims[:, 2] / 2 - sz,
                      centers[:, 2] + dims[:, 2] / 2 - sz], axis=1)
        acc = np.zeros((nc, 6))
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    sgn = (-1.0) ** (i + j + k)
                    xi, yj, zk = x[:, i], y[:, j], z[:, k]
                    rr = np.sqrt(xi**2 + yj**2 + zk**2)
                    rr = np.maximum(rr, 1e-12)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        # PRINCIPAL-BRANCH arctan, never arctan2 — the same
                        # trap that cost this repo a factor of -pi x 10^4.
                        axx = np.arctan(yj * zk / (xi * rr))
                        ayy = np.arctan(xi * zk / (yj * rr))
                        azz = np.arctan(xi * yj / (zk * rr))
                    # MINUS on the three arctan (diagonal) terms. Laplace
                    # cannot see this: it only constrains the diagonal, so a
                    # sign shared by all three passes it while disagreeing
                    # with the log (off-diagonal) terms. Caught by checking
                    # the RATIO to d(gz)/dx across components — xz and yz
                    # gave -G*1000*1e5 while zz gave +G*1000*1e5.
                    acc[:, 0] -= sgn * np.nan_to_num(axx)
                    acc[:, 1] -= sgn * np.nan_to_num(ayy)
                    acc[:, 2] -= sgn * np.nan_to_num(azz)
                    acc[:, 3] += sgn * np.log(np.maximum(rr + zk, 1e-300))
                    acc[:, 4] += sgn * np.log(np.maximum(rr + yj, 1e-300))
                    acc[:, 5] += sgn * np.log(np.maximum(rr + xi, 1e-300))
        out[s_idx] = acc
    return out


def direction(inclination_deg, declination_deg):
    """Unit vector for a field direction, in the repo's z-UP frame.

    Inclination is positive DOWNWARD (the geophysical convention), so the
    vertical component is negated to land in a z-up frame. Declination is
    measured east of north, and north is +y.
    """
    inc = np.radians(inclination_deg)
    dec = np.radians(declination_deg)
    return np.array([np.cos(inc) * np.sin(dec),      # east
                     np.cos(inc) * np.cos(dec),      # north
                     -np.sin(inc)])                  # z-up


def prism_tensor(stations, centers, dims):
    """Full symmetric GAMMA per station and cell: (n_stations, n_cells, 3, 3)."""
    c = _corner_sums(stations, centers, dims)
    T = np.empty(c.shape[:2] + (3, 3))
    T[..., 0, 0] = c[..., 0]
    T[..., 1, 1] = c[..., 1]
    T[..., 2, 2] = c[..., 2]
    T[..., 0, 1] = T[..., 1, 0] = c[..., 3]
    T[..., 0, 2] = T[..., 2, 0] = c[..., 4]
    T[..., 1, 2] = T[..., 2, 1] = c[..., 5]
    return T


def magnetisation_direction(inclination, declination, q=0.0,
                            rem_inclination=None, rem_declination=None):
    """Unit magnetisation direction and its amplitude factor.

    Total magnetisation is induced plus remanent:

        M = chi * H0 * m_ind  +  M_rem * m_rem
          = chi * H0 * (m_ind + Q * m_rem)

    where Q is the Koenigsberger ratio, remanent over induced magnitude. The
    bracket is a KNOWN vector once Q and the remanent direction are declared,
    so the unknown stays a scalar per cell and the problem stays linear. That
    is the whole reason to parameterise remanence this way rather than
    inverting for a direction per cell.

    Returns (unit vector, amplitude), where amplitude is |m_ind + Q m_rem| —
    a reversely magnetised body with Q > 1 flips the sign of the anomaly,
    which is exactly what a real reversed body does.
    """
    m_ind = direction(inclination, declination)
    if q == 0.0:
        return m_ind, 1.0
    if rem_inclination is None:
        rem_inclination = inclination
    if rem_declination is None:
        rem_declination = declination
    v = m_ind + q * direction(rem_inclination, rem_declination)
    amp = float(np.linalg.norm(v))
    if amp < 1e-12:
        raise ValueError(
            "induced and remanent magnetisation cancel exactly (Q = 1 and "
            "exactly antiparallel): the body would be magnetically invisible "
            "and susceptibility is unidentifiable.")
    return v / amp, amp


def project_tensor(c, b, m):
    """b^T GAMMA m from the six stored components, for ANY b and m.

    The induced-only path had b == m and could use the symmetric shortcut.
    Remanence breaks that: the field direction and the magnetisation
    direction differ, so the cross terms need both orderings.
    """
    return (b[0] * m[0] * c[..., 0] + b[1] * m[1] * c[..., 1]
            + b[2] * m[2] * c[..., 2]
            + (b[0] * m[1] + b[1] * m[0]) * c[..., 3]
            + (b[0] * m[2] + b[2] * m[0]) * c[..., 4]
            + (b[1] * m[2] + b[2] * m[1]) * c[..., 5])


def mag_matrix(stations, centers, dims, b0_nt, inclination, declination,
               sign=-1.0, q=0.0, rem_inclination=None, rem_declination=None):
    """Sensitivity matrix A: total-field anomaly in nT per unit susceptibility.

    dT = A @ chi, with chi in SI (dimensionless).

    With q = 0 this is induced magnetisation only: the body is magnetised
    along the ambient field, so b_hat and m_hat coincide. Give a Koenigsberger
    ratio q > 0 (and optionally a remanent direction) and the magnetisation
    tilts away from the field. Because the direction is DECLARED rather than
    solved for, the problem stays linear and the whole posterior machinery
    applies unchanged.

    `sign` carries the convention, fixed by the pole test in tests/test_mag.py
    rather than asserted from the algebra.
    """
    b = direction(inclination, declination)            # what the meter reads
    m, amp = magnetisation_direction(inclination, declination, q,
                                     rem_inclination, rem_declination)
    c = _corner_sums(stations, centers, dims)
    return sign * (b0_nt / (4.0 * np.pi)) * amp * project_tensor(c, b, m)


def mag_tf(stations, centers, dims, chi, b0_nt, inclination, declination,
           q=0.0, rem_inclination=None, rem_declination=None):
    """Total-field anomaly in nT from susceptibilities chi (SI)."""
    return mag_matrix(stations, centers, dims, b0_nt, inclination,
                      declination, q=q, rem_inclination=rem_inclination,
                      rem_declination=rem_declination) @ np.asarray(chi, float)
