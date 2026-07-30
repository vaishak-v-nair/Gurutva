"""Provenance pins: the robustness claims, kept honest by regression.

Measured 2026-08-01 (figures/provenance_stats.json): informed-pattern corr
0.951 / 0.914 at beta*/10 and 10*beta*; model corr 0.905 / 0.966. Pins sit
below measurements with slack — if a change erodes the stability story,
these fail before the README lies.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import inversion, posterior


def test_conclusions_survive_two_decades_of_beta():
    a = inversion.assemble(bridged=True)
    G, mref, stations = a["G"], a["mref"], a["stations"]
    wr = inversion.depth_weights(G)
    r, _ = inversion.residual_data(G, a["d_obs"], mref,
                                   stations=stations, order=2)
    b0 = inversion.tune_beta(G, r, wr)

    def informed(beta):
        var = posterior.posterior_diag(G, wr, beta)
        return 1.0 - np.sqrt(var) / np.sqrt(1.0 / (beta * wr**2))

    def dm(beta):
        return inversion.solve_map(G, r, wr, beta)

    i0, m0 = informed(b0), dm(b0)
    for factor, pin_pattern, pin_model in [(0.1, 0.92, 0.85),
                                           (10.0, 0.88, 0.90)]:
        b = b0 * factor
        cp = float(np.corrcoef(i0, informed(b))[0, 1])
        cm = float(np.corrcoef(m0, dm(b))[0, 1])
        assert cp > pin_pattern, f"informed pattern unstable at {factor}x: {cp:.3f}"
        assert cm > pin_model, f"model unstable at {factor}x: {cm:.3f}"
