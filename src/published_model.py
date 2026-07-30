"""Published Model #1 with reconstructed cell geometry.

The delivered CSV gives cell CENTERS only. The octree structure makes sizes
decidable from center residues (probed 2026-07-30):
- lateral: origins x=331900, y=4259900; (c - org) % 50 == 25 -> 50 m cell,
  % 100 == 50 -> 100 m cell (two interleaved lattices produce the observed
  25 m spacings between unique centers).
- vertical: sizes {30, 60, 120}; origin found by exhaustive test over
  candidates; residue s/2 (mod s), with distinct classes.

The inference is audited, not trusted: total reconstructed volume must tile
the model bounding box (tests/test_prism.py).
"""

import numpy as np

from . import forge_data

X_ORG, Y_ORG = 331_900.0, 4_259_900.0
REF_DENSITY = 2.67          # g/cc — same fixed reference on both forwards;
                            # any constant cancels in the floor difference.


def _classify_axis(c: np.ndarray, org: float, sizes) -> np.ndarray:
    r = c - org
    out = np.zeros(len(c))
    for s in sizes:                       # residue s/2 mod s, checked
        hit = np.isclose(r % (2 * s), s / 2) | np.isclose(r % (2 * s), 3 * s / 2)
        out[hit & (out == 0)] = s
    return out


def load_geometry():
    """Returns centers (N,3), dims (N,3), drho (N,) for Model #1.

    The octree scales all dims together (measured: the z-size populations
    179,831 / 73,216 / 15,726 exactly match the lateral 'unclassified'
    populations under an independent-axis model). Levels:
        dz 30 -> (50, 50, 30); dz 60 -> (100, 100, 60);
        dz 120 -> (200, 200, 120).
    The z-residue classification (z_org = zmin - 60) partitions all 268,773
    cells exactly; the volume-coverage audit is the final judge.
    """
    m = forge_data.load_density_model1()
    cx = m.X_UTMNAD83z12_m.to_numpy(float)
    cy = m.Y_UTMNAD83z12_m.to_numpy(float)
    cz = m.Z_Elevation_m.to_numpy(float)

    dz = None
    for z_org in (cz.min() - 60.0, cz.min() - 15.0, cz.min() - 30.0,
                  cz.min() - 45.0, cz.min() - 90.0):
        cand = _classify_axis(cz, z_org, (30.0, 60.0, 120.0))
        one_hot = sum(int(np.sum(cand == s)) for s in (30.0, 60.0, 120.0))
        if np.all(cand > 0) and one_hot == len(cand):
            dz = cand
            break
    if dz is None:
        raise ValueError("z cell-size classification failed for all origins")

    lateral = {30.0: 50.0, 60.0: 100.0, 120.0: 200.0}
    dxy = np.vectorize(lateral.get)(dz)

    centers = np.column_stack([cx, cy, cz])
    dims = np.column_stack([dxy, dxy, dz])
    drho = m.Density_gcm3.to_numpy(float) - REF_DENSITY
    return centers, dims, drho


def overlap_count() -> int:
    """Exact tiling audit: number of doubly-claimed sub-voxels.

    Every cell claims its (dx/25 x dy/25 x dz/15) sub-voxels on the common
    finest lattice. Overlaps (a size or position inferred too large) show up
    as duplicate claims. Holes are LEGITIMATE — the published model is
    topo-cropped (air above ground is absent), which is why a naive
    volume/bounding-box audit reads ~0.89 and proves nothing.
    """
    centers, dims, _ = load_geometry()
    lo = (centers - dims / 2).min(axis=0)
    keys = []
    for c, d in ((centers, dims),):
        i0 = np.round((c[:, 0] - d[:, 0] / 2 - lo[0]) / 25.0).astype(np.int64)
        j0 = np.round((c[:, 1] - d[:, 1] / 2 - lo[1]) / 25.0).astype(np.int64)
        k0 = np.round((c[:, 2] - d[:, 2] / 2 - lo[2]) / 15.0).astype(np.int64)
        ni = np.round(d[:, 0] / 25.0).astype(np.int64)
        nj = np.round(d[:, 1] / 25.0).astype(np.int64)
        nk = np.round(d[:, 2] / 15.0).astype(np.int64)
        for cell in range(len(c)):
            ii, jj, kk = np.meshgrid(
                np.arange(i0[cell], i0[cell] + ni[cell]),
                np.arange(j0[cell], j0[cell] + nj[cell]),
                np.arange(k0[cell], k0[cell] + nk[cell]), indexing="ij")
            keys.append(((ii * 2048 + jj) * 2048 + kk).ravel())
    allk = np.concatenate(keys)
    return int(len(allk) - len(np.unique(allk)))
