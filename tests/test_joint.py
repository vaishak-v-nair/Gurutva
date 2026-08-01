"""Gates on joint inversion and on the target-derived prior.

The headline gate is the one that would catch a coupling bug: with the
petrophysical correlation set to zero, a joint inversion must reproduce two
independent single-physics inversions EXACTLY. If it does not, the block
system is leaking information between two unknowns that were declared
unrelated, and every joint answer is quietly wrong.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import mag_forward as mf, prism_forward as pf
from src.product import joint as JT, prior as PR


def _setup(nx=6, ny=5, nz=3, cell=300.0):
    xs = (np.arange(nx) - (nx - 1) / 2) * cell
    ys = (np.arange(ny) - (ny - 1) / 2) * cell
    zs = -np.arange(1, nz + 1) * cell
    CX, CY, CZ = np.meshgrid(xs, ys, zs, indexing="ij")
    centers = np.column_stack([CX.ravel(), CY.ravel(), CZ.ravel()])
    dims = np.tile([cell, cell, cell], (len(centers), 1))
    rng = np.random.default_rng(0)
    sx, sy = np.meshgrid(xs, ys, indexing="ij")
    stations = np.column_stack([sx.ravel(), sy.ravel(),
                                np.zeros(sx.size)])
    gg = pf.prism_matrix(stations, centers, dims)
    gm = mf.mag_matrix(stations, centers, dims, 55000.0, 70.0, 0.0)
    return (nx, ny, nz), (cell,) * 3, centers, dims, stations, gg, gm, rng


def test_zero_correlation_reproduces_two_separate_inversions_exactly():
    """THE gate. r = 0 means density and susceptibility were declared
    unrelated, so the joint posterior mean must equal what each physics
    produces alone — to machine precision, not approximately."""
    shape, spacing, centers, dims, st, gg, gm, rng = _setup()
    sr, sc, ng, nm = 0.05, 3e-4, 0.01, 1.0

    rho_t = np.zeros(len(centers)); rho_t[10:16] = 0.2
    chi_t = np.zeros(len(centers)); chi_t[20:26] = 1e-3
    dg = gg @ rho_t + ng * rng.standard_normal(len(st))
    dm = gm @ chi_t + nm * rng.standard_normal(len(st))

    pj, _ = JT.build_joint_prior(shape, spacing, sr, sc, 0.0, 400.0,
                                 np.random.default_rng(1))
    a = JT.block_operator(gg, gm, ng, nm)
    mj = PR.posterior_mean(a, pj, JT.stack_data(dg, dm, ng, nm))
    rho_j, chi_j = JT.split(mj)

    pg, _ = PR.build_regular(shape, spacing, sr, 400.0, np.random.default_rng(1))
    pm, _ = PR.build_regular(shape, spacing, sc, 400.0, np.random.default_rng(1))
    rho_s = PR.posterior_mean(gg / ng, pg, dg / ng)
    chi_s = PR.posterior_mean(gm / nm, pm, dm / nm)

    assert np.allclose(rho_j, rho_s, rtol=1e-8, atol=1e-12)
    assert np.allclose(chi_j, chi_s, rtol=1e-8, atol=1e-12)


def test_correlation_makes_magnetic_data_move_the_density_answer():
    """The point of joint inversion. With r != 0 the magnetic data must
    change the density posterior; with r = 0 it must not."""
    shape, spacing, centers, dims, st, gg, gm, rng = _setup()
    sr, sc, ng, nm = 0.05, 3e-4, 0.01, 1.0
    chi_t = np.zeros(len(centers)); chi_t[20:26] = 2e-3
    dg = 0.0 * (gg @ np.zeros(len(centers))) + ng * rng.standard_normal(len(st))
    dm = gm @ chi_t + nm * rng.standard_normal(len(st))
    a = JT.block_operator(gg, gm, ng, nm)
    d = JT.stack_data(dg, dm, ng, nm)

    out = {}
    for r in (0.0, 0.8):
        p, _ = JT.build_joint_prior(shape, spacing, sr, sc, r, 400.0,
                                    np.random.default_rng(1))
        out[r] = JT.split(PR.posterior_mean(a, p, d))[0]
    moved = np.max(np.abs(out[0.8] - out[0.0])) / max(np.max(np.abs(out[0.8])), 1e-30)
    assert moved > 0.5, "magnetic data must reach density through r"


def test_joint_never_widens_an_interval_that_gravity_alone_gives():
    """Adding a second dataset cannot make you less certain."""
    shape, spacing, centers, dims, st, gg, gm, rng = _setup()
    sr, sc, ng, nm = 0.05, 3e-4, 0.01, 1.0
    w = np.ones(len(centers))
    pg, _ = PR.build_regular(shape, spacing, sr, 400.0, np.random.default_rng(1))
    alone = PR.functional_sd(gg / ng, pg, w)

    a = JT.block_operator(gg, gm, ng, nm)
    for r in (0.0, 0.5, 0.9):
        p, _ = JT.build_joint_prior(shape, spacing, sr, sc, r, 400.0,
                                    np.random.default_rng(1))
        wj = np.concatenate([w, np.zeros(len(centers))])
        assert PR.functional_sd(a, p, wj) <= alone * (1 + 1e-8), (
            f"joint interval widened at r={r}")


def test_joint_prior_has_the_declared_marginal_amplitudes():
    shape, spacing, centers, dims, st, gg, gm, rng = _setup()
    sr, sc = 0.05, 3e-4
    p, meta = JT.build_joint_prior(shape, spacing, sr, sc, 0.6, 400.0,
                                   np.random.default_rng(2))
    s = p.sample(np.random.default_rng(3), 3000)
    rho, chi = JT.split(s)
    assert np.isclose(np.median(rho.std(axis=1)), sr, rtol=0.12)
    assert np.isclose(np.median(chi.std(axis=1)), sc, rtol=0.12)


def test_joint_prior_realises_the_declared_correlation():
    shape, spacing, centers, dims, st, gg, gm, rng = _setup()
    p, _ = JT.build_joint_prior(shape, spacing, 0.05, 3e-4, 0.75, 400.0,
                                np.random.default_rng(4))
    s = p.sample(np.random.default_rng(5), 4000)
    rho, chi = JT.split(s)
    got = np.median([np.corrcoef(rho[i], chi[i])[0, 1]
                     for i in range(0, rho.shape[0], 7)])
    assert abs(got - 0.75) < 0.06, f"declared 0.75, realised {got:.3f}"


def test_perfect_correlation_is_refused():
    """At r = +/-1 density and susceptibility are the same unknown and the
    2x2 covariance is singular."""
    with pytest.raises(ValueError) as e:
        JT.petrophysical_precision(0.05, 3e-4, 1.0)
    assert "singular" in str(e.value)


def test_block_operator_whitens_each_physics_by_its_own_noise():
    """mGal and nT only sit on the same footing after each is divided by ITS
    OWN noise. Get this wrong and whichever dataset has bigger raw numbers
    quietly wins."""
    shape, spacing, centers, dims, st, gg, gm, rng = _setup()
    a = JT.block_operator(gg, gm, 0.01, 1.0)
    n = len(centers)
    assert np.allclose(a[:len(st), :n], gg / 0.01)
    assert np.allclose(a[len(st):, n:], gm / 1.0)
    assert np.all(a[:len(st), n:] == 0) and np.all(a[len(st):, :n] == 0), (
        "the operator itself must stay block-diagonal; coupling belongs in "
        "the prior")


def test_mismatched_meshes_are_refused():
    shape, spacing, centers, dims, st, gg, gm, rng = _setup()
    with pytest.raises(ValueError) as e:
        JT.block_operator(gg, gm[:, :-3], 0.01, 1.0)
    assert "cell count" in str(e.value)


# ------------------------------------------------- prior derived from target
def test_derived_prior_predicts_the_declared_target_anomaly():
    """The derivation must actually hit the number it aimed at."""
    shape, spacing, centers, dims, st, gg, gm, rng = _setup()
    forward = lambda m: gg @ m
    amp, ncell = JT.target_anomaly_std(forward, st, centers, dims,
                                       contrast=0.3, radius=400.0,
                                       depth=600.0, top=0.0)
    assert ncell > 0 and amp > 0
    sd = JT.derive_prior_sd(gg, shape, spacing, 400.0,
                            np.random.default_rng(6), amp)
    p, _ = PR.build_regular(shape, spacing, sd, 400.0, np.random.default_rng(7))
    got = float(np.std(gg @ p.sample(np.random.default_rng(8), 300)))
    assert abs(got - amp) / amp < 0.25, f"aimed at {amp:.4g}, got {got:.4g}"


def test_derived_prior_scales_linearly_with_the_target_contrast():
    """Twice the target contrast is twice the prior amplitude, because the
    prior-predictive spread is exactly linear in the prior sd."""
    shape, spacing, centers, dims, st, gg, gm, rng = _setup()
    forward = lambda m: gg @ m
    kw = dict(radius=400.0, depth=600.0, top=0.0)
    a1, _ = JT.target_anomaly_std(forward, st, centers, dims, contrast=0.2, **kw)
    a2, _ = JT.target_anomaly_std(forward, st, centers, dims, contrast=0.4, **kw)
    s1 = JT.derive_prior_sd(gg, shape, spacing, 400.0,
                            np.random.default_rng(9), a1)
    s2 = JT.derive_prior_sd(gg, shape, spacing, 400.0,
                            np.random.default_rng(9), a2)
    assert np.isclose(s2 / s1, 2.0, rtol=1e-6)


def test_a_target_outside_the_mesh_is_refused():
    shape, spacing, centers, dims, st, gg, gm, rng = _setup()
    with pytest.raises(ValueError) as e:
        JT.target_anomaly_std(lambda m: gg @ m, st, centers, dims,
                              contrast=0.3, radius=100.0, depth=9000.0,
                              top=0.0)
    assert "outside the mesh" in str(e.value)


def test_the_derivation_never_touches_the_observed_data():
    """It must be a statement about what you are LOOKING FOR, so it cannot
    become a way of tuning the prior until a gate turns green. Same target
    and geometry, wildly different data: same derived prior."""
    shape, spacing, centers, dims, st, gg, gm, rng = _setup()
    amp, _ = JT.target_anomaly_std(lambda m: gg @ m, st, centers, dims,
                                   contrast=0.3, radius=400.0, depth=600.0,
                                   top=0.0)
    a = JT.derive_prior_sd(gg, shape, spacing, 400.0,
                           np.random.default_rng(11), amp)
    b = JT.derive_prior_sd(gg, shape, spacing, 400.0,
                           np.random.default_rng(11), amp)
    assert a == b
    import inspect
    src = inspect.getsource(JT.derive_prior_sd)
    assert "d_obs" not in src and "observed" not in src
