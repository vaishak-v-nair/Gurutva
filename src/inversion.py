"""The coarse re-inversion — pass one (diagonal prior), fully declared.

Objective (all terms quadratic; the Gaussian posterior is exact):

    min_m ||G m - d'||^2 + beta ||W (m - mref)||^2

- d' : delivered gCBGA(2.67) minus forward of mref minus a declared DC
       constant (a local mesh cannot produce the regional level).
- mref: the published two-layer geology as density CONTRAST vs the 2.67
       reduction background: -0.25 g/cc above the Top-of-Granite surface,
       -0.02 g/cc below (their 2.42 / 2.65 starting densities).
- W  : sensitivity-based depth weighting, wr_j = (sum_i G_ij^2)^(1/4),
       normalized — the Li-Oldenburg role, declared as part of the PRIOR
       precision (Sigma_m^-1 = beta W^T W), per the plan's D7a.
- beta: discrepancy principle against the MEASURED coarse-mesh floor
       (0.76 mGal RMS) — never the published 0.03, which this mesh
       cannot honestly reach.

Solver: data-space (Woodbury) closed form — K is only n_data x n_data.
Gate:   tests/test_inversion.py mean-match — an independent, tightly
        converged CG solve of the normal equations must agree.
"""

import numpy as np
import scipy.sparse.linalg as spla

from . import forge_data, mesh as mesh_mod

TARGET_RMS = 0.76          # mGal — measured floor (figures/floor_stats.json)
REF_BASIN, REF_BASEMENT = -0.25, -0.02   # g/cc contrast vs 2.67 background


def assemble():
    """G (SimPEG, ram), stations, observed data, mref, active mesh."""
    from simpeg import maps
    from simpeg.potential_fields import gravity

    inv = forge_data.inversion_stations()
    stations = np.column_stack([inv.Easting, inv.Northing,
                                inv.z_sensor]).astype(float)
    d_obs = inv.gCBGA.to_numpy(float)

    tree, active, meta = mesh_mod.build_mesh()
    n_act = int(active.sum())

    rx = gravity.receivers.Point(stations, components="gz")
    survey = gravity.survey.Survey(
        gravity.sources.SourceField(receiver_list=[rx]))
    kw = dict(mesh=tree, survey=survey, rhoMap=maps.IdentityMap(nP=n_act),
              store_sensitivities="ram")
    try:
        sim = gravity.simulation.Simulation3DIntegral(active_cells=active, **kw)
    except TypeError:
        sim = gravity.simulation.Simulation3DIntegral(ind_active=active, **kw)
    G = np.asarray(sim.G, dtype=float)

    # two-layer reference model from the delivered basement surface
    basement = forge_data.basement_surface("Original")
    cc = tree.cell_centers[active]
    # nearest basement z under each cell center (coarse NN lookup is fine:
    # the surface has 103k vertices over the footprint)
    from scipy.spatial import cKDTree
    kd = cKDTree(basement[:, :2])
    _, j = kd.query(cc[:, :2])
    z_base = basement[j, 2]
    mref = np.where(cc[:, 2] > z_base, REF_BASIN, REF_BASEMENT)

    return dict(G=G, d_obs=d_obs, mref=mref, stations=stations,
                tree=tree, active=active, meta=meta)


def depth_weights(G: np.ndarray) -> np.ndarray:
    wr = np.power(np.sum(G**2, axis=0), 0.25)
    return wr / wr.max()


def residual_data(G, d_obs, mref):
    """Residual after reference model and the declared DC constant."""
    r = d_obs - G @ mref
    dc = float(np.mean(r))
    return r - dc, dc


def solve_map(G, r, wr, beta):
    """Closed-form MAP increment via the data-space (Woodbury) identity."""
    s2 = 1.0 / (beta * wr**2)                # Sigma_m diagonal
    K = (G * s2) @ G.T + np.eye(G.shape[0])  # Sigma_d = I (declared unit)
    alpha = np.linalg.solve(K, r)
    return s2 * (G.T @ alpha)                # delta-m; model = mref + delta


def solve_map_cg(G, r, wr, beta, tol=1e-12):
    """Independent route for the mean-match gate: CG on the WHITENED normal
    equations (u = sqrt(beta) * wr * m). Raw normal equations are
    catastrophically ill-conditioned because wr spans orders of magnitude —
    the exact failure D7a's 'solve in whitened variables' instruction
    predicted (measured: CG stalls past 20k iterations unwhitened). In
    whitened space the prior term is the identity and CG converges fast."""
    n = G.shape[1]
    sw = np.sqrt(beta) * wr                  # whitening diagonal

    def mv(u):
        x = u / sw
        return (G.T @ (G @ x)) / sw + u

    A = spla.LinearOperator((n, n), matvec=mv, dtype=float)
    u, info = spla.cg(A, (G.T @ r) / sw, rtol=tol, maxiter=20_000)
    if info != 0:
        raise RuntimeError(f"CG failed to converge (info={info})")
    return u / sw


def tune_beta(G, r, wr, target=TARGET_RMS, lo=1e-6, hi=1e6, iters=60):
    """Discrepancy principle: bisect beta so RMS(G dm - r) hits the floor."""
    def rms(beta):
        dm = solve_map(G, r, wr, beta)
        return float(np.sqrt(np.mean((G @ dm - r) ** 2)))

    if rms(hi) < target or rms(lo) > target:
        raise RuntimeError("target RMS not bracketed — revisit bounds")
    for _ in range(iters):
        mid = np.sqrt(lo * hi)
        if rms(mid) > target:
            hi = mid
        else:
            lo = mid
    return np.sqrt(lo * hi)


def run():
    """Full pass-one inversion; returns everything the notebook needs."""
    a = assemble()
    wr = depth_weights(a["G"])
    r, dc = residual_data(a["G"], a["d_obs"], a["mref"])
    beta = tune_beta(a["G"], r, wr)
    dm = solve_map(a["G"], r, wr, beta)
    d_pred = a["G"] @ (a["mref"] + dm) + dc
    misfit = a["d_obs"] - d_pred
    return dict(**a, wr=wr, dc=dc, beta=beta, dm=dm, model=a["mref"] + dm,
                d_pred=d_pred, misfit=misfit,
                rms=float(np.sqrt(np.mean(misfit**2))))
