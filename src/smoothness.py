"""Smoothness prior — opening block of the stretch goal.

Assembles the sparse precision matrix Q of a smallness+smoothness Gaussian
prior on the ACTIVE TreeMesh cells, reusing SimPEG's regularization
machinery (never hand-rolling octree gradient stencils).

Convention pin: SimPEG's phi(m) = ||W(m - mref)||^2 with deriv2 = 2 W^T W.
We define Q := 0.5 * deriv2 so that phi(m) == m^T Q m exactly — and that
identity IS the gate (tests/test_smoothness.py), not an assumption.

Next blocks (not yet built): fold depth weights into the smallness term,
posterior mean/diag under sparse Q (splu-based), mean-match + MC gates for
the smooth posterior.
"""

import numpy as np
import scipy.sparse as sp

from . import mesh as mesh_mod


def build_precision(tree=None, active=None,
                    alpha_s: float = 1.0, alpha_x: float = 1.0,
                    alpha_y: float = 1.0, alpha_z: float = 1.0):
    """Sparse Q (n_active x n_active) for the declared alphas."""
    from simpeg import regularization

    if tree is None or active is None:
        tree, active, _ = mesh_mod.build_mesh()
    n_act = int(active.sum())

    reg = regularization.WeightedLeastSquares(
        tree, active_cells=active,
        alpha_s=alpha_s, alpha_x=alpha_x, alpha_y=alpha_y, alpha_z=alpha_z,
        reference_model=np.zeros(n_act),
    )
    Q = 0.5 * sp.csr_matrix(reg.deriv2(np.zeros(n_act)))
    return Q, reg, tree, active
