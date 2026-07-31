"""S2.5: why the retracted claim could not be rescued — measured.

Two experiments:
  (A) model-class ladder — how much of the real misfit each physically
      interpretable model class can explain, and at what amplitude;
  (B) prior-width sweep — how wide the declared relief prior must be for
      the real observation to become a plausible draw (the precondition
      for trusting ANY simulation-based posterior at that observation).

Run: py -3 notebooks/09_s25_diagnostic.py
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import inversion, posterior, s2_model, s25_model

ROOT = Path(__file__).resolve().parents[1]

S = s25_model.build()
Uk = S["U"][:, :s2_model.K_DATA]
cc, z_base, G, mref = S["cc"], S["z_base"], S["G"], S["mref"]
x_obs = s2_model.x_observed(S)
res = x_obs
basin_ref = cc[:, 2] > z_base


def fwd_dm(dm):
    return (dm @ G.T) @ Uk


def rbf(nx, ny, ell):
    C = s25_model._grid(cc, nx, ny)
    return s25_model._rbf(cc, C, ell)


def interface_cols(Phi, dz=30.0):
    cols = []
    for i in range(Phi.shape[1]):
        zp = z_base + Phi[:, i] * dz
        m1 = np.where(cc[:, 2] > zp, -0.25, -0.02)
        cols.append(fwd_dm(m1 - mref) / dz)
    return np.column_stack(cols)


rho_cols = np.column_stack([fwd_dm(np.where(basin_ref, 1.0, 0.0)),
                            fwd_dm(np.where(basin_ref, 0.0, 1.0))])
het_cols = np.column_stack([fwd_dm(S["Psi"][:, i])
                            for i in range(S["Psi"].shape[1])])

# ---------------------------------------------------------------- (A) ladder
ladder = []
for label, cols, nw, phys in [
    ("2-unit + 8 broad heterogeneity",
     np.column_stack([interface_cols(rbf(4, 3, 2000.)), rho_cols, het_cols]),
     12, "het |c| vs 0.02 g/cc prior"),
    ("interface 12 RBF @2.0 km", np.column_stack([interface_cols(rbf(4, 3, 2000.)), rho_cols]), 12, "±300 m"),
    ("interface 24 RBF @1.2 km", np.column_stack([interface_cols(rbf(6, 4, 1200.)), rho_cols]), 24, "±300 m"),
    ("interface 40 RBF @0.9 km", np.column_stack([interface_cols(rbf(8, 5, 900.)), rho_cols]), 40, "±300 m"),
]:
    c, *_ = np.linalg.lstsq(cols, res, rcond=None)
    rem = float(np.linalg.norm(res - cols @ c))
    pct = 100 * (1 - rem**2 / float(np.linalg.norm(res)) ** 2)
    w_med = float(np.median(np.abs(c[:nw])))
    extra = (float(np.median(np.abs(c[nw + 2:]))) if cols.shape[1] > nw + 2
             else None)
    ladder.append(dict(label=label, pct=pct, rem=rem, w_med=w_med,
                       het_med=extra, phys=phys))
    print(f"{label:34s} {pct:5.1f}% | rem {rem:5.1f} | |w|med {w_med:7.0f} m"
          + (f" | |het|med {extra:.3f} g/cc" if extra else ""))

NOISE = float(np.sqrt(s2_model.K_DATA))
print(f"(noise floor of the summary: |r| = {NOISE:.1f})")

# ------------------------------------------------------------ (B) OOD sweep
sweep = []
for sz in (150., 300., 500., 800., 1200., 1800.):
    s25_model.SIGMA_Z = sz
    _, xs = s25_model.simulate(S, 1200, np.random.default_rng(99))
    mu, sd = xs.mean(0), xs.std(0)
    d_sim = np.linalg.norm((xs - mu) / sd, axis=1)
    d_obs = float(np.linalg.norm((x_obs - mu) / sd))
    pct = float(100 * (d_sim < d_obs).mean())
    sweep.append(dict(sigma_z=sz, d_obs=d_obs, d_med=float(np.median(d_sim)),
                      pct=pct))
    print(f"sigma_z={sz:6.0f} m -> OOD percentile {pct:5.1f}%")
s25_model.SIGMA_Z = 300.0

# ---------------------------------------------------------------- figures
fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), dpi=150)

ax = axes[0]
labels = [l["label"].replace(" RBF", "\nRBF").replace(" + 8", "\n+ 8") for l in ladder]
pcts = [l["pct"] for l in ladder]
bars = ax.bar(range(len(ladder)), pcts,
              color=["#b3543f", "#c98a3d", "#7a9fc4", "#3a86a8"])
for i, l in enumerate(ladder):
    txt = (f"|w|~{l['w_med']:.0f} m" if l["het_med"] is None
           else f"het {l['het_med']:.2f} g/cc\n(prior 0.02)")
    ax.text(i, l["pct"] + 1.5, txt, ha="center", fontsize=7)
ax.set_xticks(range(len(ladder)))
ax.set_xticklabels(labels, fontsize=7)
ax.set_ylabel("% of real misfit explained")
ax.set_ylim(0, 108)
ax.set_title("(A) what each physical model class can explain", fontsize=10)

ax = axes[1]
szs = [s["sigma_z"] for s in sweep]
ax.plot(szs, [s["d_obs"] for s in sweep], "-o", ms=4, color="#3a86a8",
        label="real observation")
ax.axhline(sweep[0]["d_med"], color="k", ls="--", lw=1.2,
           label="typical simulated draw")
ax.axvspan(0, 400, color="#4aa86b", alpha=0.15)
ax.text(200, ax.get_ylim()[1] * 0.93, "physically\ndefensible", fontsize=7,
        ha="center", color="#2c6b45")
ax.set_xlabel("declared relief prior σ_z [m]")
ax.set_ylabel("distance of x_obs from simulator's world")
ax.set_title("(B) the prior needed to make the data 'normal'", fontsize=10)
ax.legend(fontsize=7)

ax = axes[2]
res_run = inversion.run(bridged=True)
var = posterior.posterior_diag(res_run["G"], res_run["wr"], res_run["beta"])
informed = 1.0 - np.sqrt(var) / np.sqrt(1.0 / (res_run["beta"] * res_run["wr"]**2))
y_sec = float(np.median(cc[:, 1]))
band = np.abs(cc[:, 1] - y_sec) < 400
h = res_run["tree"].h_gridded[res_run["active"]]
sc = ax.scatter(cc[band, 0] / 1000, cc[band, 2], c=informed[band], cmap="magma",
                marker="s", s=np.sqrt(h[band, 0]) * 1.6, linewidths=0)
fig.colorbar(sc, ax=ax, label="data-informed fraction")
ax.set_xlabel("Easting [km]"); ax.set_ylabel("elevation [m]")
ax.set_title("(C) what still stands: the Phase-1 posterior\n"
             "(closed-form, no OOD problem)", fontsize=10)

fig.tight_layout()
fig.savefig(ROOT / "figures" / "s25_diagnostic.png")

stats = {"ladder": ladder, "ood_sweep": sweep, "noise_floor": NOISE,
         "verdict": ("no physically-defensible 2-unit parametric class brings "
                     "the real observation in-distribution; simulation-based "
                     "inference is not licensed at this observation for this "
                     "model class")}
(ROOT / "figures" / "s25_stats.json").write_text(json.dumps(stats, indent=1))
print("\nfigure -> figures/s25_diagnostic.png")
