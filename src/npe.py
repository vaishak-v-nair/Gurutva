"""S1 neural engine: NPE on the mode coefficients, gated against the exact
answer.

sbi pinned at 0.26.1 (requirements.txt), torch 2.13 CPU. float32 training
is fine — NPE's approximation error dwarfs float32 eps, and the gates are
calibrated to the measured NPE error, per house rule.
"""

import numpy as np
import torch

from . import modes


def train(S, k=modes.K_DEFAULT, n_train=60_000, seed=20260802,
          max_epochs=80, batch=512):
    """Train NPE(theta=c | x=t) on the linear-limit mode simulator."""
    from sbi.inference import NPE
    from sbi.utils.user_input_checks import process_prior

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    c, t = modes.simulate(S, n_train, k, rng)

    prior_dist = torch.distributions.MultivariateNormal(
        torch.zeros(k), torch.eye(k))
    prior, *_ = process_prior(prior_dist)
    inference = NPE(prior=prior, density_estimator="maf",
                    show_progress_bars=False)
    de = inference.append_simulations(
        torch.as_tensor(c, dtype=torch.float32),
        torch.as_tensor(t, dtype=torch.float32),
    ).train(max_num_epochs=max_epochs, training_batch_size=batch,
            show_train_summary=False)
    return inference.build_posterior(de)


def sample(posterior, t_vec, n=1_000):
    """Posterior samples (n, k) for one observation t_vec."""
    x = torch.as_tensor(np.asarray(t_vec, dtype=np.float32))
    with torch.no_grad():
        s = posterior.sample((n,), x=x, show_progress_bars=False)
    return s.numpy().astype(float)


def agreement_stats(posterior, S, k, n_obs=100, n_samp=800, seed=20260803):
    """NPE vs exact analytic posterior over simulated observations.

    Returns median/p95 of |mu_err|/sigma_exact (z-error) and of
    |sigma_ratio - 1| pooled over modes and observations.
    """
    rng = np.random.default_rng(seed)
    _, t_test = modes.simulate(S, n_obs, k, rng)
    z_errs, s_errs = [], []
    for t in t_test:
        smp = sample(posterior, t, n=n_samp)
        mu_e, var_e = modes.analytic_posterior(S[:k], t)
        z_errs.append(np.abs(smp.mean(axis=0) - mu_e) / np.sqrt(var_e))
        s_errs.append(np.abs(smp.std(axis=0) / np.sqrt(var_e) - 1.0))
    z, s = np.concatenate(z_errs), np.concatenate(s_errs)
    return {"z_median": float(np.median(z)), "z_p95": float(np.percentile(z, 95)),
            "sig_median": float(np.median(s)),
            "sig_p95": float(np.percentile(s, 95))}


def sbc(posterior, S, k, rounds=300, n_samp=200, seed=20260804):
    """Simulation-based calibration: pooled rank statistics.

    Returns fraction of ranks in the extreme 5% tails (nominal 0.05) and
    the pooled 90%-interval coverage (nominal 0.90).
    """
    rng = np.random.default_rng(seed)
    c_true, t_all = modes.simulate(S, rounds, k, rng)
    tail, cover = [], []
    for c_star, t in zip(c_true, t_all):
        smp = sample(posterior, t, n=n_samp)
        rank = (smp < c_star).mean(axis=0)
        tail.append((rank < 0.025) | (rank > 0.975))
        cover.append((rank > 0.05) & (rank < 0.95))
    tail = np.concatenate(tail)
    cover = np.concatenate(cover)
    return {"tail_frac": float(tail.mean()), "cover90": float(cover.mean()),
            "rounds": rounds}
