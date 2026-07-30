"""S1 full run: train NPE, gate against the exact answer, render.

Run: py -3 notebooks/07_s1_npe.py
"""

import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import modes, npe

ROOT = Path(__file__).resolve().parents[1]

M = modes.build(bridged=True)
S, t_obs, k = M["S"], M["t_obs"], M["k"]
print(f"modes built: k={k}, trace(R)={modes.gains(S).sum():.1f}")

t0 = time.time()
post = npe.train(S, k=k, n_train=60_000, max_epochs=80)
print(f"NPE trained in {time.time()-t0:.0f}s")

t0 = time.time()
agree = npe.agreement_stats(post, S, k, n_obs=100)
print(f"agreement ({time.time()-t0:.0f}s):", agree)

t0 = time.time()
cal = npe.sbc(post, S, k, rounds=300)
print(f"SBC ({time.time()-t0:.0f}s):", cal)

# at the REAL observation
smp = npe.sample(post, t_obs[:k], n=4_000)
mu_e, var_e = modes.analytic_posterior(S[:k], t_obs[:k])
mu_n, sd_n = smp.mean(axis=0), smp.std(axis=0)

fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6), dpi=150)

ax = axes[0]
ax.errorbar(mu_e, mu_n, yerr=sd_n, fmt=".", ms=4, alpha=0.6,
            color="#6b4fd8", ecolor="#c9c2e8", elinewidth=0.8)
lim = [min(mu_e.min(), mu_n.min()) - 0.3, max(mu_e.max(), mu_n.max()) + 0.3]
ax.plot(lim, lim, "k--", lw=1)
ax.set_xlabel("exact posterior mean (analytic)")
ax.set_ylabel("NPE posterior mean ± σ")
ax.set_title("the real observation: neural vs exact, 128 modes", fontsize=10)

ax = axes[1]
ax.plot(np.sqrt(var_e), sd_n, ".", ms=4, alpha=0.6, color="#3a86a8")
lim = [0, max(np.sqrt(var_e).max(), sd_n.max()) * 1.1]
ax.plot(lim, lim, "k--", lw=1)
ax.set_xlabel("exact posterior σ"); ax.set_ylabel("NPE posterior σ")
ax.set_title(f"uncertainty fidelity — median |σ ratio−1| "
             f"= {agree['sig_median']*100:.1f}%", fontsize=10)

ax = axes[2]
labels = ["tail frac\n(nominal 5%)", "90% coverage\n(nominal 90%)"]
vals = [cal["tail_frac"] * 100, cal["cover90"] * 100]
noms = [5, 90]
xpos = [0, 1]
ax.bar(xpos, vals, width=0.5, color=["#b3543f", "#4aa86b"])
for x, nv in zip(xpos, noms):
    ax.plot([x - 0.3, x + 0.3], [nv, nv], "k--", lw=1.4)
ax.set_xticks(xpos); ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("%")
ax.set_title(f"SBC calibration, {cal['rounds']} rounds × {k} modes",
             fontsize=10)

fig.tight_layout()
fig.savefig(ROOT / "figures" / "s1_npe.png")

stats = {"k": k, "n_train": 60_000, "agreement": agree, "sbc": cal}
(ROOT / "figures" / "s1_stats.json").write_text(json.dumps(stats, indent=1))
print(json.dumps(stats, indent=1))
print("figure -> figures/s1_npe.png")
