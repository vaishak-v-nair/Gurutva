"""Exact rectangular-prism gravity forward (Nagy 1966 corner formula).

Convention: matches SimPEG as established by src/units_probe.py — output in
mGal, density in g/cc, z-UP sign (a mass below the station reads negative).
The sign and units are not assumed: tests/test_prism.py pins this forward
against SimPEG on the same cell to ~1e-5 relative.
"""

import numpy as np

from scipy.constants import G as G_SI   # same constant as tests + SimPEG probe
SI_TO_MGAL = 1e5
GCC_TO_SI = 1000.0


def prism_gz(stations: np.ndarray, centers: np.ndarray, dims: np.ndarray,
             drho: np.ndarray, chunk: int = 20000) -> np.ndarray:
    """gz (mGal, z-up) at each station from prisms (centers, dims, drho g/cc)."""
    out = np.zeros(len(stations))
    for s_idx, (sx, sy, sz) in enumerate(stations):
        acc = 0.0
        for a in range(0, len(centers), chunk):
            c = centers[a:a + chunk]
            d = dims[a:a + chunk]
            r = drho[a:a + chunk]
            x = np.stack([c[:, 0] - d[:, 0] / 2 - sx,
                          c[:, 0] + d[:, 0] / 2 - sx], axis=1)
            y = np.stack([c[:, 1] - d[:, 1] / 2 - sy,
                          c[:, 1] + d[:, 1] / 2 - sy], axis=1)
            z = np.stack([c[:, 2] - d[:, 2] / 2 - sz,
                          c[:, 2] + d[:, 2] / 2 - sz], axis=1)
            g = np.zeros(len(c))
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        sgn = (-1.0) ** (i + j + k)
                        xi, yj, zk = x[:, i], y[:, j], z[:, k]
                        rr = np.sqrt(xi**2 + yj**2 + zk**2)
                        # Blakely (1995) grouping with PRINCIPAL-BRANCH arctan:
                        # arctan2's extended range breaks the 8-corner
                        # cancellation (each corner leaks +/-pi; measured as a
                        # pi*1e4 scale error before this fix).
                        with np.errstate(divide="ignore", invalid="ignore"):
                            at = np.arctan(xi * yj / (zk * rr))
                        at = np.nan_to_num(at)
                        term = (zk * at
                                - xi * np.log(np.maximum(rr + yj, 1e-300))
                                - yj * np.log(np.maximum(rr + xi, 1e-300)))
                        g += sgn * term
            acc += float(np.sum(G_SI * (r * GCC_TO_SI) * g))
        # sign: Blakely corner form returns down-positive; our convention
        # is z-up (pinned against SimPEG by tests/test_prism.py)
        out[s_idx] = -acc * SI_TO_MGAL
    return out


def prism_matrix(stations: np.ndarray, centers: np.ndarray,
                 dims: np.ndarray) -> np.ndarray:
    """Sensitivity matrix G (n_stations x n_cells), mGal per g/cc.

    Same Nagy corner formula and the same principal-branch arctan as
    prism_gz — kept as one code path in spirit and pinned to it by test, so
    the pi-leak bug (arctan2's extended range breaking the 8-corner
    cancellation, a factor of -pi x 10^4) cannot reappear in only one of
    them.
    """
    G = np.empty((len(stations), len(centers)))
    for s_idx, (sx, sy, sz) in enumerate(stations):
        x = np.stack([centers[:, 0] - dims[:, 0] / 2 - sx,
                      centers[:, 0] + dims[:, 0] / 2 - sx], axis=1)
        y = np.stack([centers[:, 1] - dims[:, 1] / 2 - sy,
                      centers[:, 1] + dims[:, 1] / 2 - sy], axis=1)
        z = np.stack([centers[:, 2] - dims[:, 2] / 2 - sz,
                      centers[:, 2] + dims[:, 2] / 2 - sz], axis=1)
        acc = np.zeros(len(centers))
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    sgn = (-1.0) ** (i + j + k)
                    xi, yj, zk = x[:, i], y[:, j], z[:, k]
                    rr = np.sqrt(xi**2 + yj**2 + zk**2)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        at = np.arctan(xi * yj / (zk * rr))
                    at = np.nan_to_num(at)
                    term = (zk * at
                            - xi * np.log(np.maximum(rr + yj, 1e-300))
                            - yj * np.log(np.maximum(rr + xi, 1e-300)))
                    acc += sgn * term
        G[s_idx] = -acc * G_SI * GCC_TO_SI * SI_TO_MGAL
    return G
