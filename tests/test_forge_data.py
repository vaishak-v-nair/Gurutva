"""Week-0/1 gates: the dataset facts the whole project stands on.

Each test pins a claim from the README verification note to executable truth.
"""

import numpy as np
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import forge_data


def test_station_table_has_518_rows():
    assert len(forge_data.load_stations()) == 518


def test_cbga_identity_holds_at_rounding_level():
    """gCBGA = gSBGA + iztc + oztc, at the columns' own precision.

    Columns are quantized to 0.001 mGal; four independently rounded values
    can disagree by up to 2 quanta, so the noise-floor-calibrated tolerance
    is 0.0021 — never tighter (3 of 518 stations sit at exactly 0.002).
    Same lesson as the MC gates: calibrate to the numerics, or correct
    data fails a correct identity.
    """
    resid = forge_data.cbga_identity_residual(forge_data.load_stations())
    assert resid <= 0.0021, f"delivered-column identity broken: {resid} mGal"


def test_published_inversion_used_323_matched_stations():
    inv = forge_data.inversion_stations()
    assert len(inv) == 323


def test_sensor_height_rule_navd88_plus_20cm():
    """Misfit-file Z equals NAVD88 + 0.200 m (values quantized to 0.01 m)."""
    inv = forge_data.inversion_stations()
    dz = inv.X_UTMNAD83z12_m * 0 + (inv.Z_Elevation_m - inv.NAVD88)
    assert np.all(np.abs(dz - forge_data.SENSOR_HEIGHT) <= 0.011), (
        f"height rule violated: dz range [{dz.min():.3f}, {dz.max():.3f}]"
    )


def test_basement_surface_loads_from_geoh5():
    """Pins the adversarial reviewer's count: 103,455 vertices. The surface
    extends below the inverted volume floor (z to ~-2112 m) — wider footprint
    than the model, which is expected and fine."""
    v = forge_data.basement_surface("Original")
    assert v.shape == (103_455, 3)
    # z spans ~-2112 m (deep basin) to ~+2723 m (granite outcrops at the
    # surface in the Mineral Mountains east of the valley)
    assert -2500 < v[:, 2].min() < v[:, 2].max() < 2800


def test_repeat_station_twins_documented():
    """FGA127/FGB139: one location, two campaigns, readings 0.072 mGal apart.
    We keep FGA127; this test keeps the fact executable."""
    st = forge_data.load_stations()
    dup = st[st.duplicated(subset=["Easting", "Northing"], keep=False)]
    assert sorted(dup.name) == ["FGA127", "FGB139"]
    assert abs(dup.gCBGA.iloc[0] - dup.gCBGA.iloc[1]) == pytest.approx(0.072, abs=0.001)


def test_published_model_cell_count():
    assert len(forge_data.load_density_model1()) == 268_773
