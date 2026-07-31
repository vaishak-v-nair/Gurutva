"""S2.5 gates: pin the diagnosis that closed the retracted claim.

Measured 2026-08-03 (figures/s25_stats.json): the enrichment ladder
(65.4 / 83.1 / 88.4% explained at |w| 4438 / 982 / 488 m), the unphysical
heterogeneity demand (1.53 g/cc vs a 0.02 prior), and the OOD sweep
(in-distribution only at sigma_z >~ 1200 m, which this valley cannot
support). These tests fail if a future change quietly makes the README's
negative result untrue.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import s2_model, s25_model


@pytest.fixture(scope="module")
def setup():
    S = s25_model.build()
    return S, s2_model.x_observed(S)


def _fwd(S, dm):
    return (dm @ S["G"].T) @ S["U"][:, :s2_model.K_DATA]


def _interface_cols(S, Phi, dz=30.0):
    cols = []
    for i in range(Phi.shape[1]):
        zp = S["z_base"] + Phi[:, i] * dz
        m1 = np.where(S["cc"][:, 2] > zp, -0.25, -0.02)
        cols.append(_fwd(S, m1 - S["mref"]) / dz)
    return np.column_stack(cols)


def test_finer_interface_explains_more_at_lower_amplitude(setup):
    """The ladder's shape is the argument: shorter wavelengths explain more
    of the misfit AND need less relief to do it."""
    S, x_obs = setup
    prev_pct, prev_amp = 0.0, np.inf
    for nx, ny, ell in [(4, 3, 2000.), (6, 4, 1200.), (8, 5, 900.)]:
        C = s25_model._grid(S["cc"], nx, ny)
        A = np.column_stack([_interface_cols(S, s25_model._rbf(S["cc"], C, ell)),
                             _fwd(S, np.where(S["cc"][:, 2] > S["z_base"], 1.0, 0.0)),
                             _fwd(S, np.where(S["cc"][:, 2] > S["z_base"], 0.0, 1.0))])
        c, *_ = np.linalg.lstsq(A, x_obs, rcond=None)
        pct = 100 * (1 - np.linalg.norm(x_obs - A @ c) ** 2
                     / np.linalg.norm(x_obs) ** 2)
        amp = float(np.median(np.abs(c[:A.shape[1] - 2])))
        assert pct > prev_pct, "finer basis must explain more"
        assert amp < prev_amp, "finer basis must need less relief"
        prev_pct, prev_amp = pct, amp
    assert prev_pct > 85.0, f"finest class should explain >85%, got {prev_pct:.1f}"


def test_heterogeneity_explanation_is_unphysical(setup):
    """Broad heterogeneity 'explains' the misfit only at amplitudes larger
    than the entire basin-basement contrast — the reason it was rejected."""
    S, x_obs = setup
    A = np.column_stack([_fwd(S, S["Psi"][:, i])
                         for i in range(S["Psi"].shape[1])])
    c, *_ = np.linalg.lstsq(A, x_obs, rcond=None)
    demanded = float(np.median(np.abs(c)))
    contrast = abs(s2_model.RHO_BASIN[0] - s2_model.RHO_BASEMENT[0])
    # measured 2026-08-03: 1.14 g/cc alone (4.97x the contrast), 1.53 g/cc
    # when fitted jointly with interface+densities. Gate at 4x — below the
    # measurement, above any physically arguable heterogeneity.
    assert demanded > 4 * contrast, (
        f"heterogeneity demand {demanded:.2f} should dwarf the {contrast:.2f} "
        "g/cc unit contrast")


def test_real_data_stays_out_of_distribution_at_physical_priors(setup):
    """THE gate that licenses (or refuses) simulation-based inference here."""
    S, x_obs = setup
    old = s25_model.SIGMA_Z
    try:
        s25_model.SIGMA_Z = 300.0            # physically defensible relief
        pct, _, _ = s25_model.ood_percentile(S, x_obs, n_ref=400, seed=5)
        assert pct > 98.0, (
            f"OOD percentile {pct:.1f}: if this ever drops, the S2.5 negative "
            "result is stale and the interface question may be reopened")
    finally:
        s25_model.SIGMA_Z = old
