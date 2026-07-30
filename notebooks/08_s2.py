"""S2 full run: the granite surface with error bars.

Run: py -3 notebooks/08_s2.py
"""

import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import forge_data, inversion, modes, posterior, s2_model

ROOT = Path(__file__).resolve().parents[1]

S2 = s2_model.build()
rng = np.random.default_rng(20260803)

# ---- train NPE on the geological generative model ---------------------------
from sbi.inference import NPE
from sbi.utils.user_input_checks import process_prior

t0 = time.time()
theta, xs = s2_model.simulate(S2, 50_000, rng)
print(f"simulated 50k geological draws in {time.time()-t0:.0f}s")

torch.manual_seed(20260803)
prior_t = torch.distributions.MultivariateNormal(
    torch.as_tensor(np.r_[np.zeros(12), s2_model.RHO_BASIN[0],
                          s2_model.RHO_BASEMENT[0]], dtype=torch.float32),
    torch.diag(torch.as_tensor(
        np.r_[np.full(12, s2_model.SIGMA_Z**2),
              s2_model.RHO_BASIN[1]**2, s2_model.RHO_BASEMENT[1]**2],
        dtype=torch.float32)))
prior, *_ = process_prior(prior_t)
inf = NPE(prior=prior, density_estimator="maf", show_progress_bars=False)
t0 = time.time()
de = inf.append_simulations(
    torch.as_tensor(theta, dtype=torch.float32),
    torch.as_tensor(xs, dtype=torch.float32),
).train(max_num_epochs=80, training_batch_size=512, show_train_summary=False)
post = inf.build_posterior(de)
print(f"NPE trained in {time.time()-t0:.0f}s")

# ---- SBC gate ---------------------------------------------------------------
t0 = time.time()
rounds, nsamp = 300, 200
th_true, x_all = s2_model.simulate(S2, rounds, np.random.default_rng(7))
tails, covers = [], []
for th, x in zip(th_true, x_all):
    with torch.no_grad():
        smp = post.sample((nsamp,), x=torch.as_tensor(x, dtype=torch.float32),
                          show_progress_bars=False).numpy()
    rank = (smp < th).mean(axis=0)
    tails.append((rank < 0.025) | (rank > 0.975))
    covers.append((rank > 0.05) & (rank < 0.95))
sbc = {"tail_frac": float(np.concatenate(tails).mean()),
       "cover90": float(np.concatenate(covers).mean()), "rounds": rounds}
print(f"SBC ({time.time()-t0:.0f}s):", sbc)

# ---- posterior at the REAL data --------------------------------------------
x_obs = s2_model.x_observed(S2)
with torch.no_grad():
    smp = post.sample((4_000,), x=torch.as_tensor(x_obs, dtype=torch.float32),
                      show_progress_bars=False).numpy()

# posterior predictive sanity
m_post = s2_model.model_from_theta(S2, smp[:200])
d_pred = (m_post - S2["mref"][None, :]) @ S2["G"].T
pp_rms = float(np.sqrt(np.mean((d_pred - S2["r"][None, :]) ** 2)))
print(f"posterior-predictive RMS vs observed: {pp_rms:.3f} mGal")

rho_basin = smp[:, 12]
rho_base = smp[:, 13]
print(f"rho_basin posterior: {rho_basin.mean():.3f} ± {rho_basin.std():.3f} "
      f"(prior -0.25 ± 0.03 — report's 2.42)")
print(f"rho_basement posterior: {rho_base.mean():.3f} ± {rho_base.std():.3f} "
      f"(prior -0.02 ± 0.02 — report's 2.65)")

# ---- figures ----------------------------------------------------------------
inv_st = forge_data.inversion_stations()
y_sec = float(inv_st.Northing.median())
cc, z_base, Phi = S2["cc"], S2["z_base"], S2["Phi"]

xs_line = np.linspace(cc[:, 0].min() + 300, cc[:, 0].max() - 300, 160)
line = np.column_stack([xs_line, np.full_like(xs_line, y_sec)])
d2l = ((line[:, None, 0] - S2["centers"][None, :, 0]) ** 2
       + (line[:, None, 1] - S2["centers"][None, :, 1]) ** 2)
Phi_line = np.exp(-0.5 * d2l / s2_model.RBF_LEN**2)
from scipy.spatial import cKDTree
bs = forge_data.basement_surface("Original")
kd = cKDTree(bs[:, :2])
_, jj = kd.query(line)
z0_line = bs[jj, 2]
z_post = z0_line[None, :] + smp[:, :12] @ Phi_line.T
z_prior = z0_line[None, :] + (s2_model.SIGMA_Z
                              * np.random.default_rng(1).standard_normal(
                                  (2000, 12))) @ Phi_line.T

fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), dpi=150)

ax = axes[0]
for band, color, lab in [(z_prior, "#d9c9a8", "prior (declared ±150 m field)"),
                         (z_post, "#3a86a8", "posterior (data-constrained)")]:
    lo, hi = np.percentile(band, [5, 95], axis=0)
    ax.fill_between(xs_line / 1000, lo, hi, alpha=0.45, color=color, label=lab)
ax.plot(xs_line / 1000, z0_line, "r-", lw=1.3, label="delivered surface")
ax.set_xlabel("Easting [km]"); ax.set_ylabel("elevation [m]")
ax.set_title("interface band at x_obs — UNSTABLE (OOD diagnostic only:\n"
             "run-to-run 0.49 vs 1.07; see README retraction)", fontsize=9)
ax.legend(fontsize=7, loc="lower right")

ax = axes[1]
sec_mask = (np.abs(cc[:, 1] - y_sec) < 400)
probe_ids = np.argsort(np.abs(cc[:, 2] - (z_base - 60))
                       + 1e6 * ~sec_mask)[:2]
res = inversion.run(bridged=True)
var_g = posterior.posterior_diag(res["G"], res["wr"], res["beta"])
for pid, color in zip(probe_ids, ["#6b4fd8", "#b3543f"]):
    vals = s2_model.model_from_theta(S2, smp[:2000])[:, pid]
    ax.hist(vals, bins=40, density=True, alpha=0.5, color=color,
            label=f"S2 posterior, cell @z={cc[pid,2]:.0f} m")
    g_mu, g_sd = res["model"][pid], np.sqrt(var_g[pid])
    xx = np.linspace(-0.45, 0.15, 300)
    ax.plot(xx, np.exp(-0.5 * ((xx - g_mu) / g_sd) ** 2)
            / (g_sd * np.sqrt(2 * np.pi)), "--", color=color, lw=1.2)
ax.set_xlabel("density contrast [g/cc]")
ax.set_title("bimodal geology vs the Gaussian smear (dashed = Phase 1)",
             fontsize=10)
ax.legend(fontsize=7)

ax = axes[2]
vals = [sbc["tail_frac"] * 100, sbc["cover90"] * 100]
ax.bar([0, 1], vals, width=0.5, color=["#b3543f", "#4aa86b"])
for x, nv in zip([0, 1], [5, 90]):
    ax.plot([x - 0.3, x + 0.3], [nv, nv], "k--", lw=1.4)
ax.set_xticks([0, 1])
ax.set_xticklabels(["tail frac\n(nominal 5%)", "90% coverage\n(nominal 90%)"],
                   fontsize=8)
ax.set_ylabel("%")
ax.set_title(f"SBC, {rounds} rounds × 14 params", fontsize=10)

fig.tight_layout()
fig.savefig(ROOT / "figures" / "s2_interface.png")

stats = {"sbc": sbc, "pp_rms_mgal": pp_rms,
         "rho_basin_post": [float(rho_basin.mean()), float(rho_basin.std())],
         "rho_basement_post": [float(rho_base.mean()), float(rho_base.std())],
         "interface_prior_band_med_m": float(np.median(
             np.percentile(z_prior, 95, axis=0)
             - np.percentile(z_prior, 5, axis=0))),
         "interface_post_band_med_m_UNSTABLE_OOD": float(np.median(
             np.percentile(z_post, 95, axis=0)
             - np.percentile(z_post, 5, axis=0)))}
(ROOT / "figures" / "s2_stats.json").write_text(json.dumps(stats, indent=1))
print(json.dumps(stats, indent=1))
print("NOTE: the x_obs interface band is OOD-unstable across retrainings;")
print("stable findings = density constraints + calibration. See README.")
print("figure -> figures/s2_interface.png")
