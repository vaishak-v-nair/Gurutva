"""Gates on the magnetic forward, and on the tensor it is built from.

The sign convention is NOT asserted from algebra here. It is pinned by a
physical case, the same way src/units_probe.py pinned gravity's z-up sign.
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import mag_forward as mf, prism_forward as pf


def _prism(cx=0.0, cy=0.0, cz=-600.0, size=300.0):
    return (np.array([[cx, cy, cz]]), np.array([[size, size, size]]))


def test_tensor_obeys_laplace_outside_the_body():
    """GAMMA_xx + GAMMA_yy + GAMMA_zz = 0 in free space. Free, and it catches
    almost every sign or branch mistake in the corner sum."""
    c, d = _prism()
    rng = np.random.default_rng(0)
    st = np.column_stack([rng.uniform(-900, 900, 25),
                          rng.uniform(-900, 900, 25),
                          np.zeros(25)])
    T = mf.prism_tensor(st, c, d)
    trace = T[..., 0, 0] + T[..., 1, 1] + T[..., 2, 2]
    scale = np.abs(T).max()
    assert np.max(np.abs(trace)) < 1e-9 * scale, "Laplace violated"


def test_tensor_is_symmetric():
    c, d = _prism()
    st = np.array([[120.0, -80.0, 0.0], [400.0, 250.0, 10.0]])
    T = mf.prism_tensor(st, c, d)
    for a, b in ((0, 1), (0, 2), (1, 2)):
        assert np.allclose(T[..., a, b], T[..., b, a], atol=1e-12)


def test_tensor_matches_finite_differences_of_the_gated_gravity():
    """Pin the tensor to the ALREADY-GATED gz rather than to fresh algebra.

    gz = -G rho * d(Phi)/dz * unit factors, so d(gz)/dx must be proportional
    to GAMMA_xz with the same constant for every component. Checking the
    RATIO across components is what makes this a real test: a wrong sign or a
    swapped index shows up as an inconsistent ratio.
    """
    c, d = _prism()
    st = np.array([[150.0, -120.0, 0.0]])
    h = 0.5
    rho = np.array([1.0])

    def gz(p):
        return float(pf.prism_gz(np.array([p]), c, d, rho)[0])

    num = {}
    for axis, key in ((0, "xz"), (1, "yz"), (2, "zz")):
        p1, p2 = st[0].copy(), st[0].copy()
        p1[axis] += h
        p2[axis] -= h
        num[key] = (gz(p1) - gz(p2)) / (2 * h)

    T = mf.prism_tensor(st, c, d)[0, 0]
    ana = {"xz": T[0, 2], "yz": T[1, 2], "zz": T[2, 2]}

    ratios = [num[k] / ana[k] for k in ("xz", "yz", "zz") if abs(ana[k]) > 1e-9]
    assert len(ratios) >= 2
    assert np.allclose(ratios, ratios[0], rtol=2e-3), (
        f"tensor components disagree with d(gz)/dx: ratios {ratios}")


def test_at_the_pole_a_susceptible_body_reads_positive():
    """THE SIGN CONVENTION, pinned by physics rather than by algebra.

    Field straight down (inclination 90), station directly above a
    susceptible prism: the body concentrates flux and the total-field
    anomaly must be POSITIVE. This is the magnetic twin of the z-up sign
    probe that gravity got in src/units_probe.py.
    """
    c, d = _prism()
    st = np.array([[0.0, 0.0, 0.0]])
    t = mf.mag_tf(st, c, d, [0.05], b0_nt=55000.0, inclination=90.0,
                  declination=0.0)
    assert t[0] > 0, f"expected a positive anomaly over the body, got {t[0]:.3f}"


def test_a_negative_susceptibility_flips_the_sign():
    c, d = _prism()
    st = np.array([[0.0, 0.0, 0.0]])
    pos = mf.mag_tf(st, c, d, [0.05], 55000.0, 90.0, 0.0)[0]
    neg = mf.mag_tf(st, c, d, [-0.05], 55000.0, 90.0, 0.0)[0]
    assert np.isclose(pos, -neg, rtol=1e-12)


def test_anomaly_is_linear_in_susceptibility_and_field_strength():
    """dT = (B0/4pi) chi (...): doubling either must double the anomaly."""
    c, d = _prism()
    st = np.array([[40.0, 20.0, 0.0]])
    a = mf.mag_tf(st, c, d, [0.02], 50000.0, 60.0, 5.0)[0]
    b = mf.mag_tf(st, c, d, [0.04], 50000.0, 60.0, 5.0)[0]
    e = mf.mag_tf(st, c, d, [0.02], 100000.0, 60.0, 5.0)[0]
    assert np.isclose(b, 2 * a, rtol=1e-12)
    assert np.isclose(e, 2 * a, rtol=1e-12)


def test_at_the_equator_the_anomaly_is_symmetric_with_a_reversed_centre():
    """Field horizontal to the north (inclination 0), profile along north.

    Then b_hat == m_hat == north, so dT is proportional to GAMMA_yy alone,
    which is SYMMETRIC about the body. The classic equatorial signature is a
    central low with flanking highs — not the antisymmetric pair you get at
    mid-latitude. An earlier version of this test asserted antisymmetry and
    was simply wrong about the physics; the code was right.
    """
    c, d = _prism()
    ys = np.linspace(-800, 800, 41)
    st = np.column_stack([np.zeros_like(ys), ys, np.zeros_like(ys)])
    t = mf.mag_tf(st, c, d, [0.05], 55000.0, inclination=0.0, declination=0.0)
    assert np.allclose(t, t[::-1], rtol=1e-9), "must be symmetric about the body"
    centre, flank = t[len(t) // 2], t[0]
    assert centre * flank < 0, "centre and flanks must have opposite sign"
    assert abs(centre) > abs(flank), "the central lobe should dominate"


def test_at_mid_latitude_the_anomaly_becomes_asymmetric():
    """At intermediate inclination the projection mixes GAMMA_yy with
    GAMMA_yz, and the north-south profile loses its symmetry. That asymmetry
    is the fingerprint of the directional derivative actually being applied;
    a body treated as a monopole would stay symmetric at every latitude."""
    c, d = _prism()
    ys = np.linspace(-800, 800, 41)
    st = np.column_stack([np.zeros_like(ys), ys, np.zeros_like(ys)])
    t = mf.mag_tf(st, c, d, [0.05], 55000.0, inclination=45.0, declination=0.0)
    asym = np.max(np.abs(t - t[::-1])) / np.max(np.abs(t))
    assert asym > 0.2, f"expected a clearly asymmetric profile, got {asym:.3f}"
    assert t.max() > 0 and t.min() < 0, "expected both a high and a low"


def test_direction_vector_uses_the_geophysical_convention():
    """Inclination positive DOWN, declination east of north, +y is north,
    and the repo's frame is z-UP."""
    assert np.allclose(mf.direction(90, 0), [0, 0, -1], atol=1e-12)
    assert np.allclose(mf.direction(0, 0), [0, 1, 0], atol=1e-12)
    assert np.allclose(mf.direction(0, 90), [1, 0, 0], atol=1e-12)


def test_magnitude_is_in_a_believable_range_for_a_real_target():
    """A 300 m cube of chi=0.05 magnetite-bearing rock at 600 m depth in a
    55,000 nT field should read in the tens of nT, not micro- or mega-nT.
    An order-of-magnitude slip in the constant would sail past every
    dimensionless check above."""
    c, d = _prism()
    st = np.array([[0.0, 0.0, 0.0]])
    t = abs(mf.mag_tf(st, c, d, [0.05], 55000.0, 90.0, 0.0)[0])
    assert 1.0 < t < 500.0, f"{t:.3f} nT is not a plausible magnitude"
