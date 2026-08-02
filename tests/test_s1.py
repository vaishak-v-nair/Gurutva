"""S1 gate: NPE must find the exact answer in the linear limit.

Full-run measurements (2026-08-02, figures/s1_stats.json, n_train=60k):
z_median 0.080, z_p95 0.268, sig_median 0.029, tail 5.08%, cover90 88.8%.
This test uses a lighter config for suite runtime; pins carry proportional
slack — measured, never taste (house rule).
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytest.importorskip("sbi", reason="pip install -r requirements-research.txt")

from src import modes, npe


@pytest.fixture(scope="module")
def trained():
    M = modes.build(bridged=True)
    post = npe.train(M["S"], k=M["k"], n_train=20_000, max_epochs=40)
    return M, post


def test_npe_agrees_with_exact_answer(trained):
    M, post = trained
    a = npe.agreement_stats(post, M["S"], M["k"], n_obs=40, n_samp=500)
    assert a["z_median"] < 0.25, f"posterior means drifting: {a}"
    assert a["sig_median"] < 0.10, f"posterior widths drifting: {a}"


def test_npe_calibration(trained):
    M, post = trained
    c = npe.sbc(post, M["S"], M["k"], rounds=120, n_samp=150)
    assert 0.02 < c["tail_frac"] < 0.10, f"tails miscalibrated: {c}"
    assert 0.82 < c["cover90"] < 0.96, f"coverage miscalibrated: {c}"
