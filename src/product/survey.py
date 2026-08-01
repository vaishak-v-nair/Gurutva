"""Survey geometry: geographic coordinates, topography, and lease blocks.

Three things a real customer needs that the first CLI did not have. None of
them uses pyproj or shapely, because neither is installed here and adding a
dependency to a tool people must be able to run on a locked-down work laptop
is a real cost. Each is therefore implemented directly, with its accuracy
stated and gated rather than assumed.
"""

import numpy as np

WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)

# Beyond this the tangent plane stops being honest (see project()).
MAX_SPAN_KM = 100.0


def looks_geographic(x, y):
    """True when the numbers are plainly degrees rather than metres.

    A projected survey in UTM has easting ~1e5-1e6 and northing up to 1e7.
    Degrees never leave [-180, 180] and [-90, 90]. Guessing wrong by a factor
    of 111,000 would be silent and catastrophic, so the test is deliberately
    conservative: BOTH columns must be in range, and the span must be small
    enough to be a survey rather than a continent.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    in_range = (np.all(np.abs(x) <= 180.0) and np.all(np.abs(y) <= 90.0))
    tiny_span = (x.max() - x.min()) < 10.0 and (y.max() - y.min()) < 10.0
    return bool(in_range and tiny_span)


def project(lon, lat, lon0=None, lat0=None):
    """Longitude/latitude in degrees -> local metres, east-north, WGS84.

    A local tangent plane about the survey centre, using the true meridional
    and prime-vertical radii of curvature at that latitude rather than a
    spherical earth:

        x = N(lat0) cos(lat0) * dlon      N = a / sqrt(1 - e^2 sin^2 lat0)
        y = M(lat0) * dlat                M = a (1-e^2) / (1 - e^2 sin^2)^1.5

    Accurate to well under a metre across a few tens of km, which is far
    inside any cell size this tool will use. It is NOT a substitute for a
    real projection over large areas, so a survey spanning more than
    MAX_SPAN_KM is refused rather than quietly distorted.

    Returns (x, y, meta).
    """
    lon = np.asarray(lon, float)
    lat = np.asarray(lat, float)
    lat0 = float(np.mean(lat)) if lat0 is None else lat0
    lon0 = float(np.mean(lon)) if lon0 is None else lon0

    s = np.sin(np.radians(lat0))
    w = 1.0 - WGS84_E2 * s * s
    n_rad = WGS84_A / np.sqrt(w)                       # prime vertical
    m_rad = WGS84_A * (1.0 - WGS84_E2) / w**1.5        # meridional

    x = np.radians(lon - lon0) * n_rad * np.cos(np.radians(lat0))
    y = np.radians(lat - lat0) * m_rad

    span_km = max(x.max() - x.min(), y.max() - y.min()) / 1000.0
    if span_km > MAX_SPAN_KM:
        raise ValueError(
            f"the survey spans {span_km:,.0f} km. A local tangent plane is "
            f"only honest below {MAX_SPAN_KM:.0f} km — project your "
            f"coordinates to a proper grid (UTM) and pass metres instead.")
    return x, y, {"lat0": lat0, "lon0": lon0, "span_km": span_km,
                  "n_rad": n_rad, "m_rad": m_rad}


def topography(stations, xy, power=2.0, n_near=8):
    """Ground elevation at each (x, y), interpolated from the stations.

    Inverse-distance weighting over the nearest few stations. Deliberately
    not a spline: a spline will overshoot past the edge of a survey and
    invent a hill nobody measured, and every cell it invents becomes air or
    rock in the mesh below.
    """
    from scipy.spatial import cKDTree

    st = np.asarray(stations, float)
    xy = np.atleast_2d(np.asarray(xy, float))
    kd = cKDTree(st[:, :2])
    k = min(n_near, len(st))
    dist, idx = kd.query(xy, k=k)
    dist = np.atleast_2d(dist)
    idx = np.atleast_2d(idx)
    exact = dist[:, 0] < 1e-9
    with np.errstate(divide="ignore"):
        w = 1.0 / np.maximum(dist, 1e-12) ** power
    z = np.sum(w * st[idx, 2], axis=1) / np.sum(w, axis=1)
    z[exact] = st[idx[exact, 0], 2]
    return z


def air_mask(centers, stations, drape=0.0):
    """True where a cell centre sits ABOVE the ground and is therefore air.

    Without this the mesh is a flat-topped block: over a valley it fills the
    air with rock the survey never saw, and the inversion happily puts
    density there to fit the data.
    """
    ground = topography(stations, centers[:, :2])
    return centers[:, 2] > (ground + drape)


def read_polygon(path):
    """Vertices of a lease block or licence boundary, from a CSV of x,y.

    Accepts the same delimiters and header casing as a survey file, and
    either projected metres or lon/lat (detected the same way).
    """
    raw = __import__("pathlib").Path(path).read_text(encoding="utf-8-sig",
                                                     errors="replace")
    lines = [ln for ln in raw.splitlines() if ln.strip()]
    if len(lines) < 4:
        raise ValueError(f"{path}: a polygon needs at least 3 vertices")
    head = lines[0]
    delim = max((",", ";", "\t"), key=lambda c: head.count(c))
    names = [n.strip().lower() for n in head.split(delim)]
    rows = np.array([[float(v) for v in ln.split(delim)] for ln in lines[1:]])
    try:
        ix, iy = names.index("x"), names.index("y")
    except ValueError:
        try:
            ix, iy = names.index("lon"), names.index("lat")
        except ValueError:
            raise ValueError(
                f"{path}: need columns x,y (metres) or lon,lat (degrees); "
                f"found {', '.join(names)}")
    return rows[:, ix], rows[:, iy]


def points_in_polygon(px, py, vx, vy):
    """Ray-casting point-in-polygon, vectorised over the points.

    Closes the ring automatically. Points exactly on an edge are not
    guaranteed either way — for a lease block whose cells are hundreds of
    metres across, that ambiguity is far below the resolution of anything
    this tool claims.
    """
    px, py = np.asarray(px, float), np.asarray(py, float)
    vx, vy = np.asarray(vx, float), np.asarray(vy, float)
    if vx[0] != vx[-1] or vy[0] != vy[-1]:
        vx, vy = np.append(vx, vx[0]), np.append(vy, vy[0])
    inside = np.zeros(px.shape, dtype=bool)
    for i in range(len(vx) - 1):
        x1, y1, x2, y2 = vx[i], vy[i], vx[i + 1], vy[i + 1]
        straddles = (y1 > py) != (y2 > py)
        with np.errstate(divide="ignore", invalid="ignore"):
            xint = (x2 - x1) * (py - y1) / (y2 - y1) + x1
        inside ^= straddles & (px < xint)
    return inside


def polygon_area(vx, vy):
    """Shoelace area in the units of the vertices squared. Sign-free."""
    vx, vy = np.asarray(vx, float), np.asarray(vy, float)
    if vx[0] != vx[-1] or vy[0] != vy[-1]:
        vx, vy = np.append(vx, vx[0]), np.append(vy, vy[0])
    return 0.5 * abs(np.dot(vx[:-1], vy[1:]) - np.dot(vx[1:], vy[:-1]))
