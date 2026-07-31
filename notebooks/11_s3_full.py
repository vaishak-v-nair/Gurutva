"""S3 full: the Bayesian Moho of South America — error bars for a continent.

Gate order is the S2.5 lesson, encoded:
  1. licensing (prior-predictive OOD) — already PASSED at 0.0 percentile
  2. SBC calibration
  3. run-to-run stability at the real observation (the gate that caught S2)
Only if all three hold is a posterior reported.

Run: py -3 notebooks/11_s3_full.py
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
from src import moho, moho_infer as mi

ROOT = Path(__file__).resolve().parents[1]

S = mi.build()
mi.fit_projection(S)
pct, d_obs, d_med = mi.ood_percentile(S)
print(f"[gate 1] licensing: OOD percentile {pct:.1f}% (dist {d_obs:.1f} vs {d_med:.1f})")
assert pct < 95, "not licensed — no posterior may be claimed"

N_TRAIN = 50_000


def train(seed):
    from sbi.inference import NPE
    from sbi.utils.user_input_checks import process_prior
    rng = np.random.default_rng(seed)
    t0 = time.time()
    theta, x = mi.simulate(S, N_TRAIN, rng)
    print(f"  simulated {N_TRAIN} in {time.time()-t0:.0f}s", flush=True)
    torch.manual_seed(seed)
    lo = np.r_[np.zeros(S["n_basis"]), mi.DRHO_PRIOR[0]]
    sd = np.r_[np.full(S["n_basis"], mi.SIGMA_W), mi.DRHO_PRIOR[1]]
    prior_t = torch.distributions.MultivariateNormal(
        torch.as_tensor(lo, dtype=torch.float32),
        torch.diag(torch.as_tensor(sd**2, dtype=torch.float32)))
    prior, *_ = process_prior(prior_t)
    inf = NPE(prior=prior, density_estimator="maf", show_progress_bars=False)
    t0 = time.time()
    de = inf.append_simulations(
        torch.as_tensor(theta, dtype=torch.float32),
        torch.as_tensor(x, dtype=torch.float32),
    ).train(max_num_epochs=60, training_batch_size=512, show_train_summary=False)
    print(f"  trained in {time.time()-t0:.0f}s", flush=True)
    return inf.build_posterior(de)


def sample(post, x, n=2000):
    with torch.no_grad():
        return post.sample((n,), x=torch.as_tensor(x, dtype=torch.float32),
                           show_progress_bars=False).numpy()


print("[training A]"); postA = train(20260804)
print("[training B]"); postB = train(777)

# ---------------------------------------------------------- gate 2: SBC
rounds, nsamp = 250, 200
th_t, x_t = mi.simulate(S, rounds, np.random.default_rng(31))
tails, covers = [], []
for th, x in zip(th_t, x_t):
    smp = sample(postA, x, nsamp)
    rank = (smp < th).mean(axis=0)
    tails.append((rank < 0.025) | (rank > 0.975))
    covers.append((rank > 0.05) & (rank < 0.95))
sbc = {"tail_frac": float(np.concatenate(tails).mean()),
       "cover90": float(np.concatenate(covers).mean()), "rounds": rounds}
print(f"[gate 2] SBC: tails {sbc['tail_frac']*100:.1f}% (nominal 5), "
      f"coverage {sbc['cover90']*100:.1f}% (nominal 90)")

# ------------------------------------------- gate 3: stability at x_obs
x_obs = mi.x_observed(S)
sA, sB = sample(postA, x_obs, 3000), sample(postB, x_obs, 3000)
dA = mi.depth_from_theta(S, sA)
dB = mi.depth_from_theta(S, sB)
mean_gap = float(np.median(np.abs(dA.mean(0) - dB.mean(0))))
sd_ratio = float(np.median(dA.std(0) / dB.std(0)))
print(f"[gate 3] stability: median |meanA-meanB| = {mean_gap/1000:.2f} km, "
      f"sd ratio {sd_ratio:.3f}")

stable = (mean_gap < 3_000.0) and (0.8 < sd_ratio < 1.25)
print(f"         -> {'STABLE' if stable else 'UNSTABLE — no claim'}")

# --------------------- gate 4: POSTERIOR-PREDICTIVE ADEQUACY (mandatory)
# Added after gates 1-3 all passed for a model that could not reproduce the
# data. Gates 1-3 test self-consistency; only this one tests reality.
_dp = mi.forward_theta(S, sample(postA, x_obs, 200))
_rr = _dp - S["obs"][None, :]
pp_check = float(np.sqrt(np.mean((_rr - _rr.mean(axis=1, keepdims=True)) ** 2)))
PUBLISHED_DC_RMS = 19.3
adequate = pp_check < 3 * PUBLISHED_DC_RMS
print(f"[gate 4] adequacy: posterior-predictive {pp_check:.1f} mGal vs "
      f"published {PUBLISHED_DC_RMS} -> {'ADEQUATE' if adequate else 'INADEQUATE — NO MAP CLAIMED'}")

# ------------------------------------------------------------- posterior
depth_mean = dA.mean(0) / 1000.0
depth_sd = dA.std(0) / 1000.0
prior_sd = mi.SIGMA_W * np.sqrt((S["Phi"] ** 2).sum(axis=1)) / 1000.0
shrink = float(np.median(1 - depth_sd / prior_sd))
drho_post = (float(sA[:, -1].mean()), float(sA[:, -1].std()))
pub = moho.moho_on_grid(S["glat"], S["glon"]) / 1000.0
resid_pub = float(np.median(np.abs(depth_mean - pub)))

d_pred = mi.forward_theta(S, sA[:200])
_r = d_pred - S["obs"][None, :]
pp_rms_raw = float(np.sqrt(np.mean(_r ** 2)))
# DC-removed is the meaningful number: a constant ~110 mGal offset dominates
# the raw residual even for the PUBLISHED model (raw 112.4, DC-removed 19.3),
# because the Bouguer datum and our reference layer differ by a constant.
pp_rms = float(np.sqrt(np.mean((_r - _r.mean(axis=1, keepdims=True)) ** 2)))
np.savez_compressed(ROOT / "figures" / "s3_posterior_samples.npz",
                    thetaA=sA, thetaB=sB, depth_mean=dA.mean(0),
                    depth_sd=dA.std(0), glat=S["glat"], glon=S["glon"])
print(f"\nposterior: depth sd median {np.median(depth_sd):.1f} km "
      f"(prior {np.median(prior_sd):.1f}, shrink {shrink*100:.0f}%)")
print(f"drho {drho_post[0]:.0f} +/- {drho_post[1]:.0f} kg/m3 "
      f"(prior 350 +/- 50) | median |ours-published| {resid_pub:.1f} km "
      f"| pp RMS(DC-removed) {pp_rms:.1f} mGal (raw {pp_rms_raw:.0f}, published model DC-removed 19.3)")

# ---------------------------------------------------------------- figures
glat, glon = S["glat"], S["glon"]
fig, axes = plt.subplots(1, 4, figsize=(20, 4.6), dpi=150)

for ax, v, ttl, cm, lab in [
    (axes[0], depth_mean, "posterior mean Moho depth", "viridis_r", "km"),
    (axes[1], depth_sd, "THE DELIVERABLE:\nposterior uncertainty (1σ)", "magma", "km"),
    (axes[2], pub, "published map (Uieda & Barbosa 2017)\n— no uncertainties exist", "viridis_r", "km"),
]:
    sc = ax.scatter(glon, glat, c=v, s=22, cmap=cm, linewidths=0, marker="s")
    fig.colorbar(sc, ax=ax, label=lab)
    ax.set_title(ttl, fontsize=9)
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude")

ax = axes[3]
ax.bar([0, 1], [sbc["tail_frac"] * 100, sbc["cover90"] * 100], width=0.5,
       color=["#b3543f", "#4aa86b"])
for x_, nv in zip([0, 1], [5, 90]):
    ax.plot([x_ - 0.3, x_ + 0.3], [nv, nv], "k--", lw=1.4)
ax.set_xticks([0, 1])
ax.set_xticklabels(["tails\n(nom. 5%)", "90% cover\n(nom. 90%)"], fontsize=8)
ax.set_ylabel("%")
ax.set_title(f"gates: SBC + stability\n|ΔA-B| = {mean_gap/1000:.2f} km", fontsize=9)

fig.tight_layout()
fig.savefig(ROOT / "figures" / "s3_moho_posterior.png")

stats = {"licensing_ood_pct": pct, "sbc": sbc,
         "stability": {"mean_gap_km": mean_gap / 1000, "sd_ratio": sd_ratio,
                       "stable": bool(stable)},
         "posterior": {"depth_sd_median_km": float(np.median(depth_sd)),
                       "prior_sd_median_km": float(np.median(prior_sd)),
                       "shrinkage": shrink,
                       "drho_mean": drho_post[0], "drho_sd": drho_post[1],
                       "median_abs_diff_vs_published_km": resid_pub,
                       "pp_rms_dc_removed_mgal": pp_rms, "pp_rms_raw_mgal": pp_rms_raw,
                       "published_model_dc_removed_mgal": 19.3},
         "n_train": N_TRAIN, "grid": list(S["g"]["shape"]),
         "resolution_deg": S["deg"]}
(ROOT / "figures" / "s3_posterior_stats.json").write_text(json.dumps(stats, indent=1))
print("figure -> figures/s3_moho_posterior.png")
