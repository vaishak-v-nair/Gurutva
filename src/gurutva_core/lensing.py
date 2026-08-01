"""E2: the dark-matter twin — the same engine, pointed at the invisible.

The claim this module exists to TEST, not to assert:

    P4 (from the office-hours design doc): "impossible and practical are the
    same math." Inverting gravity for an ore body and inverting gravity for
    unseen mass are the same Bayesian inverse problem — so a dark-matter
    exclusion limit and a drill-decision posterior are the same object.

That has been Gurutva's romance since day one. It has never been tested.
Here it is tested, with the honest-negative clause standing: if the mapping
fails, the mission arm becomes explicitly aspirational in all public
material until a real mapping exists.

The physics: weak gravitational lensing. A mass distribution Sigma(x) along
the line of sight deflects background galaxy images; the measurable is the
shear field gamma, related to the convergence kappa = Sigma / Sigma_crit by
a linear convolution (Kaiser-Squires). So:

    subsurface:  d = G rho      (prisms -> gravity at stations)
    lensing:     gamma = K kappa (mass sheet -> shear at galaxies)

Both are LINEAR potential-field operators. Same inversion machinery, same
declared prior, same four gates. That is the P4 test: not a metaphor, a
shared operator signature.

Convention: flat-sky, Fourier-space Kaiser-Squires,
    gamma_hat(l) = ((l1^2 - l2^2) + 2i l1 l2) / |l|^2 * kappa_hat(l)
"""

import numpy as np


def ks_operator(n, pixel_deg=1.0):
    """Kaiser-Squires kernel on an n x n flat-sky grid (Fourier space)."""
    l1 = np.fft.fftfreq(n, d=pixel_deg)[:, None]
    l2 = np.fft.fftfreq(n, d=pixel_deg)[None, :]
    l2sq = l1**2 + l2**2
    l2sq[0, 0] = 1.0
    ker = ((l1**2 - l2**2) + 2j * l1 * l2) / l2sq
    ker[0, 0] = 0.0
    return ker


def kappa_to_shear(kappa, ker):
    """Convergence -> complex shear (gamma1 + i gamma2)."""
    return np.fft.ifft2(ker * np.fft.fft2(kappa))


def shear_to_kappa(gamma, ker):
    """The classic inverse — included to show what it does NOT give you:
    a point estimate with no uncertainty and no way to state what the data
    could not see. This is the industry-standard answer Gurutva improves on."""
    with np.errstate(invalid="ignore", divide="ignore"):
        inv = np.conj(ker) / np.maximum(np.abs(ker) ** 2, 1e-12)
    return np.real(np.fft.ifft2(inv * np.fft.fft2(gamma)))


def build_operator_matrix(n, pixel_deg=1.0):
    """Explicit real matrix K mapping kappa (n^2,) -> [g1; g2] (2 n^2,).

    Built once so the SAME linear-Gaussian posterior machinery used for the
    subsurface applies verbatim. This is the P4 test in code: if the
    operator has the same signature, the engine does not care what the mass
    is made of.
    """
    ker = ks_operator(n, pixel_deg)
    K = np.empty((2 * n * n, n * n))
    basis = np.zeros((n, n))
    for j in range(n * n):
        basis.flat[j] = 1.0
        g = kappa_to_shear(basis, ker)
        K[:n * n, j] = g.real.ravel()
        K[n * n:, j] = g.imag.ravel()
        basis.flat[j] = 0.0
    return K


def nfw_kappa(n, pixel_deg, kappa_s=0.35, r_s_pix=6.0, centre=None):
    """A declared halo: NFW-like convergence profile (the 'invisible mass')."""
    c = centre if centre is not None else ((n - 1) / 2, (n - 1) / 2)
    y, x = np.mgrid[0:n, 0:n]
    r = np.sqrt((x - c[1]) ** 2 + (y - c[0]) ** 2) / r_s_pix
    r = np.maximum(r, 1e-3)
    inner = r < 1
    f = np.empty_like(r)
    ri = np.clip(r[inner], 1e-3, 0.999999)
    f[inner] = 1 - 2 / np.sqrt(1 - ri**2) * np.arctanh(
        np.sqrt((1 - ri) / (1 + ri)))
    ro = np.clip(r[~inner], 1.000001, None)
    f[~inner] = 1 - 2 / np.sqrt(ro**2 - 1) * np.arctan(
        np.sqrt((ro - 1) / (ro + 1)))
    return kappa_s * f / (r**2 - 1 + 1e-9)
