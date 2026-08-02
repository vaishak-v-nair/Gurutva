"""Mesh budget gate: the laptop budget from the plan, made executable."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# The research stack is not needed to check the product, and pinning it into
# the default install broke `pip install -r requirements.txt` on every Python
# below 3.14. It lives in requirements-research.txt now, so this module skips
# rather than erroring at collection when it is absent.
pytest.importorskip("discretize",
                    reason="pip install -r requirements-research.txt")

from src import forge_data, mesh as mesh_mod


def test_mesh_budget_and_coverage():
    mesh, active, meta = mesh_mod.build_mesh()
    n = meta["n_active"]
    # plan budget 5-10k with honest slack either side
    assert 4_000 <= n <= 12_000, f"active cells {n} outside laptop budget"

    # every published-inversion station must sit inside the mesh footprint
    inv = forge_data.inversion_stations()
    o = mesh.origin
    top = [o[i] + mesh.h[i].sum() for i in range(3)]
    assert inv.Easting.min() > o[0] and inv.Easting.max() < top[0]
    assert inv.Northing.min() > o[1] and inv.Northing.max() < top[1]

    # finest cells must exist (interface refinement actually happened)
    h = mesh.h_gridded[active]
    assert np.isclose(h[:, 2].min(), meta["finest_cell"][2])
