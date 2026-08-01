"""Gates on survey geometry: projection, topography, and lease blocks.

None of these uses pyproj or shapely, because neither is installed and a tool
people must run on a locked-down work laptop should not need them. That makes
gating them mandatory rather than optional: a hand-rolled projection quietly
wrong by a factor is worse than no projection at all.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.product import survey as SV


def _geodesic(lon1, lat1, lon2, lat2, lat0):
    """Short-line distance on the WGS84 ellipsoid via Euler's radius.

    A spherical haversine is NOT the right yardstick here: at 38.5 degrees
    the sphere and the ellipsoid disagree by ~2900 ppm, which swamps any
    error the projection could plausibly have. Euler's formula gives the
    radius of curvature in the direction actually travelled,

        R(az) = 1 / (cos^2(az)/M + sin^2(az)/N)

    which combines M and N through the azimuth — something project() never
    does, since it decomposes into east and north separately. So this does
    test the decomposition rather than restate it.
    """
    s = np.sin(np.radians(lat0))
    w = 1.0 - SV.WGS84_E2 * s * s
    n_rad = SV.WGS84_A / np.sqrt(w)
    m_rad = SV.WGS84_A * (1.0 - SV.WGS84_E2) / w**1.5
    dlon, dlat = np.radians(lon2 - lon1), np.radians(lat2 - lat1)
    de = dlon * n_rad * np.cos(np.radians(lat0))
    az = np.arctan2(de, dlat * m_rad)
    r_az = 1.0 / (np.cos(az) ** 2 / m_rad + np.sin(az) ** 2 / n_rad)
    ang = np.hypot(dlat, dlon * np.cos(np.radians(lat0)))
    return r_az * ang


# ------------------------------------------------------------- projection
def test_projected_distances_match_an_independent_yardstick():
    """Ground distance from the tangent plane must match a true ellipsoid
    geodesic to a few parts per million across a survey-sized area."""
    lon0, lat0 = -112.85, 38.5
    rng = np.random.default_rng(0)
    lon = lon0 + rng.uniform(-0.05, 0.05, 60)
    lat = lat0 + rng.uniform(-0.04, 0.04, 60)
    x, y, meta = SV.project(lon, lat, lon0, lat0)
    worst = 0.0
    for i in range(59):
        d_proj = np.hypot(x[i] - x[i + 1], y[i] - y[i + 1])
        d_true = _geodesic(lon[i], lat[i], lon[i + 1], lat[i + 1], lat0)
        worst = max(worst, abs(d_proj - d_true) / max(d_true, 1e-9))
    assert worst < 1e-5, f"worst relative error {worst * 1e6:.1f} ppm"


def test_projection_is_centred_and_oriented_east_north():
    lon0, lat0 = 77.5, 12.9        # Bengaluru, a northern-hemisphere site
    x, y, _ = SV.project([lon0, lon0 + 0.01, lon0],
                         [lat0, lat0, lat0 + 0.01], lon0, lat0)
    assert abs(x[0]) < 1e-9 and abs(y[0]) < 1e-9, "centre maps to the origin"
    assert x[1] > 0 and abs(y[1]) < 1e-9, "+lon must be +x (east)"
    assert y[2] > 0 and abs(x[2]) < 1e-9, "+lat must be +y (north)"


def test_projection_uses_latitude_dependent_scale():
    """A degree of longitude is ~111 km at the equator and ~half that at 60
    degrees. A spherical constant would miss this."""
    x_eq, _, _ = SV.project([0.0, 0.5], [0.0, 0.0], 0.0, 0.0)
    x_60, _, _ = SV.project([0.0, 0.5], [60.0, 60.0], 0.0, 60.0)
    ratio = (x_60[1] - x_60[0]) / (x_eq[1] - x_eq[0])
    assert 0.48 < ratio < 0.52, f"expected ~cos(60)=0.5, got {ratio:.3f}"


def test_an_oversized_survey_is_refused_not_distorted():
    with pytest.raises(ValueError) as e:
        SV.project([-2.0, 2.0], [40.0, 42.0])
    assert "tangent plane" in str(e.value)


def test_geographic_detection_never_mistakes_utm_for_degrees():
    """Guessing wrong here is a silent factor of 111,000."""
    utm_x = np.array([331000.0, 331250.0, 331500.0])
    utm_y = np.array([4263000.0, 4263250.0, 4263500.0])
    assert not SV.looks_geographic(utm_x, utm_y)
    assert SV.looks_geographic(np.array([-112.85, -112.84]),
                               np.array([38.50, 38.51]))
    # a continent-sized spread of degrees is not a survey either
    assert not SV.looks_geographic(np.array([-120.0, 20.0]),
                                   np.array([-30.0, 50.0]))


# ------------------------------------------------------------- topography
def test_topography_reproduces_station_elevations_exactly():
    """At a station's own position the interpolant must return that station's
    elevation, not a smoothed version of its neighbours."""
    rng = np.random.default_rng(2)
    st = np.column_stack([rng.uniform(0, 2000, 40), rng.uniform(0, 2000, 40),
                          rng.uniform(1400, 1900, 40)])
    z = SV.topography(st, st[:, :2])
    assert np.allclose(z, st[:, 2], atol=1e-6)


def test_topography_stays_inside_the_measured_range():
    """Inverse distance, not a spline, precisely so it cannot invent a hill
    nobody measured — and every invented metre becomes air or rock."""
    rng = np.random.default_rng(3)
    st = np.column_stack([rng.uniform(0, 2000, 30), rng.uniform(0, 2000, 30),
                          rng.uniform(1400, 1900, 30)])
    q = np.column_stack([rng.uniform(-500, 2500, 200),
                         rng.uniform(-500, 2500, 200)])
    z = SV.topography(st, q)
    assert z.min() >= st[:, 2].min() - 1e-9
    assert z.max() <= st[:, 2].max() + 1e-9


def test_air_cells_are_the_ones_above_the_ground():
    st = np.array([[0.0, 0.0, 1000.0], [1000.0, 0.0, 1000.0],
                   [0.0, 1000.0, 1000.0], [1000.0, 1000.0, 1000.0]])
    centers = np.array([[500.0, 500.0, 1200.0],    # air
                        [500.0, 500.0, 900.0],     # rock
                        [500.0, 500.0, 1000.5]])   # just above ground
    air = SV.air_mask(centers, st)
    assert list(air) == [True, False, True]


# ----------------------------------------------------------- lease blocks
def test_point_in_polygon_on_a_known_shape():
    vx = np.array([0.0, 10.0, 10.0, 0.0])
    vy = np.array([0.0, 0.0, 10.0, 10.0])
    px = np.array([5.0, -1.0, 11.0, 9.9, 5.0])
    py = np.array([5.0, 5.0, 5.0, 9.9, -0.1])
    assert list(SV.points_in_polygon(px, py, vx, vy)) == [
        True, False, False, True, False]


def test_point_in_polygon_handles_a_concave_shape():
    """An L-shape: the notch must be OUTSIDE. A convex-hull shortcut would
    quietly include a customer's neighbour's ground."""
    vx = np.array([0.0, 10.0, 10.0, 5.0, 5.0, 0.0])
    vy = np.array([0.0, 0.0, 4.0, 4.0, 10.0, 10.0])
    inside = SV.points_in_polygon(np.array([2.0, 8.0]), np.array([8.0, 8.0]),
                                  vx, vy)
    assert list(inside) == [True, False], "the notch must be outside"


def test_polygon_closes_itself():
    open_ring = (np.array([0.0, 4.0, 4.0, 0.0]), np.array([0.0, 0.0, 4.0, 4.0]))
    closed = (np.append(open_ring[0], 0.0), np.append(open_ring[1], 0.0))
    p = np.array([2.0]), np.array([2.0])
    assert SV.points_in_polygon(*p, *open_ring)[0]
    assert SV.points_in_polygon(*p, *closed)[0]


def test_polygon_area_matches_a_known_rectangle():
    a = SV.polygon_area(np.array([0.0, 300.0, 300.0, 0.0]),
                        np.array([0.0, 0.0, 200.0, 200.0]))
    assert np.isclose(a, 60000.0)


def test_polygon_area_is_orientation_free():
    vx, vy = np.array([0.0, 300.0, 300.0, 0.0]), np.array([0.0, 0.0, 200.0, 200.0])
    assert np.isclose(SV.polygon_area(vx, vy),
                      SV.polygon_area(vx[::-1], vy[::-1]))


def test_read_polygon_accepts_metres_or_degrees(tmp_path):
    nl = chr(10)
    p1 = tmp_path / "m.csv"
    p1.write_text("x,y" + nl + "0,0" + nl + "10,0" + nl + "10,10" + nl,
                  encoding="utf-8")
    vx, vy = SV.read_polygon(str(p1))
    assert list(vx) == [0, 10, 10]

    p2 = tmp_path / "d.csv"
    p2.write_text("lon;lat" + nl + "-112.85;38.5" + nl + "-112.84;38.5"
                  + nl + "-112.84;38.51" + nl, encoding="utf-8")
    vx, vy = SV.read_polygon(str(p2))
    assert np.isclose(vx[0], -112.85) and np.isclose(vy[0], 38.5)


def test_read_polygon_refuses_a_line(tmp_path):
    p = tmp_path / "line.csv"
    p.write_text("x,y" + chr(10) + "0,0" + chr(10) + "1,1" + chr(10),
                 encoding="utf-8")
    with pytest.raises(ValueError) as e:
        SV.read_polygon(str(p))
    assert "3 vertices" in str(e.value)
