"""Coarse inversion mesh for the FORGE reproduction.

Design constraints (from the plan + adversarial review):
- 5-10k ACTIVE cells total (laptop budget for explicit G and dense posterior);
- finest refinement at topography and at the basement interface, because
  interface aliasing, not data noise, sets the coarse-mesh misfit floor;
- domain covers the published model volume with coarse padding for edge
  effects (the published run used huge padding; ours is honest-but-modest
  and gets its error measured, not assumed).
"""

import numpy as np
from discretize import TreeMesh
from discretize.utils import active_from_xyz

from . import forge_data


def station_topo(stations) -> np.ndarray:
    """Scattered topography points: station ground elevations (NAVD88)."""
    return np.column_stack([stations.Easting, stations.Northing,
                            stations.NAVD88]).astype(float)


def build_mesh(pad: float = 4000.0, pad_z: float = 1500.0,
               base_xy: float = 256.0, base_z: float = 32.0):
    """TreeMesh refined at topo + basement; returns (mesh, active, meta).

    Budget discipline (plan: 5-10k active cells): the finest level
    (256 x 256 x 32 m — the 32 m VERTICAL resolution at the interface is the
    aliasing-critical dimension; 256 m lateral is what the budget affords,
    and the error-floor measurement prices that choice) is spent ONLY on
    the basement interface inside the
    model footprint — that is where the 0.23 g/cc contrast makes aliasing
    bite. Topography gets one level coarser; everything else coarsens
    naturally with distance. tests/test_mesh.py enforces the budget.
    """
    model = forge_data.load_density_model1()
    x0, x1 = model.X_UTMNAD83z12_m.min(), model.X_UTMNAD83z12_m.max()
    y0, y1 = model.Y_UTMNAD83z12_m.min(), model.Y_UTMNAD83z12_m.max()
    z0, z1 = model.Z_Elevation_m.min(), model.Z_Elevation_m.max()

    def po2(n):
        return int(2 ** np.ceil(np.log2(n)))

    nx = po2((x1 - x0 + 2 * pad) / base_xy)
    ny = po2((y1 - y0 + 2 * pad) / base_xy)
    nz = po2((z1 - z0 + 2 * pad_z) / base_z)
    origin = [ (x0 + x1) / 2 - nx * base_xy / 2,
               (y0 + y1) / 2 - ny * base_xy / 2,
               (z0 + z1) / 2 - nz * base_z / 2 ]
    mesh = TreeMesh([[(base_xy, nx)], [(base_xy, ny)], [(base_z, nz)]],
                    origin=origin, diagonal_balance=True)

    st = forge_data.load_stations()
    topo = station_topo(st)
    basement = forge_data.basement_surface("Original")

    # interface: finest level, but only inside the model footprint (+margin);
    # the delivered surface extends far beyond the inverted volume.
    m = 500.0
    infoot = ((basement[:, 0] > x0 - m) & (basement[:, 0] < x1 + m)
              & (basement[:, 1] > y0 - m) & (basement[:, 1] < y1 + m))
    # thin finest band at the interface: one point per finest column, zero
    # padding — refine_surface's default vertical padding costs ~7 cells of
    # band thickness (measured), which alone breaks the 10k budget.
    bi = basement[infoot]
    gx = np.round(bi[:, 0] / base_xy).astype(int)
    gy = np.round(bi[:, 1] / base_xy).astype(int)
    import pandas as pd
    binned = (pd.DataFrame({"gx": gx, "gy": gy, "z": bi[:, 2]})
              .groupby(["gx", "gy"], as_index=False).z.mean())
    pts = np.column_stack([binned.gx * base_xy, binned.gy * base_xy, binned.z])
    # insert_cells is the stable TreeMesh primitive: force the containing
    # cell of each point to an explicit level (max_level = finest).
    mesh.insert_cells(pts, np.full(len(pts), mesh.max_level), finalize=False)
    mesh.insert_cells(topo, np.full(len(topo), mesh.max_level - 3),
                      finalize=False)
    mesh.finalize()

    active = active_from_xyz(mesh, topo, method="nearest")
    meta = {
        "model_extent": ((x0, x1), (y0, y1), (z0, z1)),
        "n_total": mesh.n_cells,
        "n_active": int(active.sum()),
        "finest_cell": (base_xy, base_xy, base_z),
    }
    return mesh, active, meta
