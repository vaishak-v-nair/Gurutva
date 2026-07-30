"""Utah FORGE data access — stations, published misfits, basement surface.

Every rule in here traces to the Week-0 verification note in the README:
- station height: Z = NAVD88 + 0.200 m (NAD83 zone 12); never HAE.
- inversion subset: the 323 stations of the published Model #1 run, matched
  to the delivered station table by exact Easting/Northing.
- geometry: the Original Top of Granite surface is read from the delivered
  geoh5 workspace (plain HDF5), never from loose files or the web.
- delivered-column identity: gCBGA = gSBGA + iztc + oztc (checked in tests).
"""

from pathlib import Path

import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FORGE = ROOT / "data" / "forge"
STATION_FILE = FORGE / "FORGE_2C_finalCBGA.txt"
MISFIT_FILE = (FORGE / "DensityModel_Original_Top_Granite"
               / "UtahFORGE_1_Original_Top_Granite_Density_Model_MISFIT.csv")
MODEL_FILE = (FORGE / "DensityModel_Original_Top_Granite"
              / "UtahFORGE_1_Original_Top_Granite_Density_Model.csv")
GEOH5_FILE = FORGE / "UtahFORGE_3D_Gravity_Models_May2019.geoh5"

SENSOR_HEIGHT = 0.200          # m above NAVD88, pinned by the delivered misfit Z


def load_stations() -> pd.DataFrame:
    """All 518 delivered stations (tab-separated table, verified columns)."""
    df = pd.read_csv(STATION_FILE, sep="\t")
    expected = {"name", "lon", "lat", "HAE", "Easting", "Northing",
                "NGVD29", "NAVD88", "obs", "errg", "iztc", "oztc",
                "gFA", "gSBGA", "gCBGA"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"station table missing columns: {missing}")
    return df


def cbga_identity_residual(df: pd.DataFrame) -> float:
    """Max |gSBGA + iztc + oztc - gCBGA| over all stations (mGal).

    This is the first trap-check from the verification note: reproduce the
    delivered 2.67 g/cc reduction arithmetic exactly before touching it.
    """
    return float(np.max(np.abs(df.gSBGA + df.iztc + df.oztc - df.gCBGA)))


def load_misfit_model1() -> pd.DataFrame:
    """Per-station misfit of the published Model #1 inversion (323 rows)."""
    return pd.read_csv(MISFIT_FILE)


def inversion_stations() -> pd.DataFrame:
    """The 323 stations used in the published inversion, joined to the full
    table by exact Easting/Northing, with the sensor elevation applied.

    Known ambiguity (documented, not hidden): FGA127 and FGB139 are the same
    location re-occupied in two campaigns, with gCBGA readings 0.072 mGal
    apart, and the published run used exactly one row there. The delivered
    files cannot say which; we keep the first (FGA127) and carry the 0.072
    mGal as a known single-station uncertainty in the error budget.
    """
    st = load_stations().drop_duplicates(
        subset=["Easting", "Northing"], keep="first")
    mis = load_misfit_model1()
    merged = mis.merge(
        st, left_on=["X_UTMNAD83z12_m", "Y_UTMNAD83z12_m"],
        right_on=["Easting", "Northing"], how="left", validate="one_to_one",
    )
    if merged.name.isna().any():
        n = int(merged.name.isna().sum())
        raise ValueError(f"{n} misfit stations failed to match the table")
    merged["z_sensor"] = merged.NAVD88 + SENSOR_HEIGHT
    return merged


def load_density_model1() -> pd.DataFrame:
    """The published 268,773-cell Model #1 density model (g/cc)."""
    return pd.read_csv(MODEL_FILE)


def basement_surface(which: str = "Original") -> np.ndarray:
    """(N, 3) vertices of the Top of Granite surface from the geoh5.

    Reads the delivered workspace as plain HDF5; finds the Surface object by
    its stored Name attribute so GUID churn can never break the lookup.
    """
    target = f"{which} Top of Granite Surface"
    hits = []

    def visit(name, obj):
        if isinstance(obj, h5py.Group):
            n = obj.attrs.get("Name")
            if isinstance(n, bytes):
                n = n.decode()
            if n == target and "Vertices" in obj:
                hits.append(obj["Vertices"][...])

    with h5py.File(GEOH5_FILE, "r") as f:
        f.visititems(visit)
    if not hits:
        raise KeyError(f"surface {target!r} not found in {GEOH5_FILE.name}")
    v = hits[0]
    return np.column_stack([v["x"], v["y"], v["z"]]).astype(float)
