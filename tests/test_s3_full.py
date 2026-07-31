"""S3-full gates: pin the verdict that stopped the map from shipping.

Measured 2026-08-04 (figures/s3_verdict_stats.json): exact model 19.3 mGal;
our 48-RBF class ceiling 62.3; our posterior 120.7. Gates 1-3 (licensing
0.0 pct, SBC 5.1/88.7, stability 0.62 km) ALL PASSED for a model that
cannot reproduce the data — which is why gate 4 (posterior-predictive
adequacy) now exists and is mandatory.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import moho, moho_infer as mi

pytestmark = pytest.mark.skipif(
    not mi.CACHE.exists(),
    reason="Moho sensitivity cache not built (see notebooks/11_s3_full.py)")


@pytest.fixture(scope="module")
def setup():
    S = mi.build()
    mi.fit_projection(S)
    return S


def _dc_rms(S, d):
    r = d - S["obs"]
    return float(np.sqrt(np.mean((r - r.mean()) ** 2)))


def test_declared_basis_cannot_express_the_published_moho(setup):
    """The ceiling: fit the PUBLISHED (correct) Moho with our own basis and
    forward it. If this ever drops near the exact model's 19.3 mGal, the
    class became adequate and S3 may be reopened."""
    S = setup
    dep_pub = moho.moho_on_grid(S["glat"], S["glon"])
    w, *_ = np.linalg.lstsq(S["Phi"], dep_pub - mi.MEAN_DEPTH, rcond=None)
    dfit = np.clip(mi.MEAN_DEPTH + S["Phi"] @ w, 6_000, 79_000)
    ceiling = _dc_rms(S, moho.forward(S["glat"], S["glon"], dfit,
                                      S["deg"], S["deg"]))
    exact = _dc_rms(S, moho.forward(S["glat"], S["glon"], dep_pub,
                                    S["deg"], S["deg"]))
    assert exact < 25.0, f"exact-model misfit drifted: {exact:.1f}"
    assert ceiling > 2 * exact, (
        f"class ceiling {ceiling:.1f} is no longer far above the exact "
        f"{exact:.1f} — the S3 negative result is stale, reopen it")


def test_finer_bases_approach_the_exact_model(setup):
    """The measured path forward: ~744 coefficients reach 21.5 mGal."""
    S = setup
    glat, glon = S["glat"], S["glon"]
    dep_pub = moho.moho_on_grid(glat, glon)
    prev = np.inf
    for nlon, nlat, ell in [(6, 8, 7.0), (16, 21, 2.5), (24, 31, 1.6)]:
        lons = np.linspace(glon.min(), glon.max(), nlon)
        lats = np.linspace(glat.min(), glat.max(), nlat)
        C = np.array([(lo, la) for lo in lons for la in lats])
        d2 = ((glon[:, None] - C[None, :, 0]) ** 2
              + (glat[:, None] - C[None, :, 1]) ** 2)
        P = np.exp(-0.5 * d2 / ell**2)
        w, *_ = np.linalg.lstsq(P, dep_pub - mi.MEAN_DEPTH, rcond=None)
        dfit = np.clip(mi.MEAN_DEPTH + P @ w, 6_000, 79_000)
        rms = _dc_rms(S, moho.forward(glat, glon, dfit, S["deg"], S["deg"]))
        assert rms < prev, "richer bases must do better"
        prev = rms
    assert prev < 25.0, f"finest tested basis should approach exact: {prev:.1f}"


def test_ood_gate_alone_is_insufficient(setup):
    """The methodological finding, executable: the licensing gate PASSES
    (observation looks typical) for a model class that cannot fit the data.
    Passing gate 1 must never again be read as permission to claim."""
    S = setup
    pct, _, _ = mi.ood_percentile(S, n_ref=300, seed=5)
    assert pct < 95.0, "licensing gate is expected to PASS here"
    dep_pub = moho.moho_on_grid(S["glat"], S["glon"])
    w, *_ = np.linalg.lstsq(S["Phi"], dep_pub - mi.MEAN_DEPTH, rcond=None)
    dfit = np.clip(mi.MEAN_DEPTH + S["Phi"] @ w, 6_000, 79_000)
    ceiling = _dc_rms(S, moho.forward(S["glat"], S["glon"], dfit,
                                      S["deg"], S["deg"]))
    assert ceiling > 40.0, (
        "…while the same model class misfits badly — the two facts together "
        "are the reason gate 4 exists")
