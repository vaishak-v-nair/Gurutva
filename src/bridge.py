"""The 2.55 bridge: re-reduce delivered gCBGA(2.67) to the published run's
gCBGA(2.55) + 20 m upward continuation — with every step declared.

Physics: both density-dependent reduction terms scale linearly in the
reduction density rho:
- simple Bouguer slab: 2*pi*G*rho*h  ->  d(slab) = 2piG * (2.67-2.55) * h
  (h = NAVD88 orthometric height, the standard Bouguer datum height);
- terrain corrections (iztc + oztc, delivered at 2.67): scale by 2.55/2.67.

Declared approximations, bounded and folded into the error budget:
- Bullard-B curvature difference is NOT recomputed (~<0.1 mGal here);
- the published 20 m upward continuation is approximated by evaluating the
  FORWARD at z_sensor + 20 m against un-continued point data — at 20 m the
  continuation's smoothing acts below the ~250 m station spacing.

The refuter's trap is avoided by construction: the slab difference comes
from first principles, never from the nonstandard gFA/gSBGA columns.
"""

import numpy as np

TWO_PI_G = 0.041930          # mGal per metre per (g/cc)
RHO_DELIVERED = 2.67
RHO_PUBLISHED = 2.55
UPWARD_M = 20.0


def bridge_data(inv_stations) -> np.ndarray:
    """gCBGA(2.55) at each station from the delivered 2.67 columns."""
    h = inv_stations.NAVD88.to_numpy(float)
    tc = (inv_stations.iztc + inv_stations.oztc).to_numpy(float)
    d267 = inv_stations.gCBGA.to_numpy(float)

    slab_diff = TWO_PI_G * (RHO_DELIVERED - RHO_PUBLISHED) * h
    tc_diff = (RHO_PUBLISHED / RHO_DELIVERED - 1.0) * tc
    return d267 + slab_diff + tc_diff


def bridged_stations(inv_stations) -> np.ndarray:
    """Station coordinates for the bridged inversion: sensor + 20 m."""
    return np.column_stack([
        inv_stations.Easting.to_numpy(float),
        inv_stations.Northing.to_numpy(float),
        inv_stations.z_sensor.to_numpy(float) + UPWARD_M,
    ])
