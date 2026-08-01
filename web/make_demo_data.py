"""Real numbers for the interactive demo. Nothing on the page is invented.

A visitor who has never heard of a gravity survey needs to SEE the problem in
three seconds, not read about it. The problem is non-uniqueness: wildly
different arrangements of buried rock produce readings that are identical to
within the accuracy of the instrument. Software picks one of them and draws
it confidently.

So the demo shows four such worlds side by side, with their readings on top
of each other. They are not artists' impressions — they are posterior samples
from the actual engine, which is exactly the set of worlds consistent with
one measurement. Then a slider lets the visitor spend money on a better
survey and watch which parts of the picture stop being guesses.

Run: py -3 web/make_demo_data.py   (~1 min)
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import prism_forward as pf
from src.gurutva_core import gates
from src.product import prior as PR

# A deliberately small 2-D scene so the page can draw it honestly: one line of
# sensors across the surface, a slab of ground beneath, cells 100 m across.
NX, NZ = 28, 12
CELL = 100.0
N_STATION = 24                # the default scene; the sweep varies this
TRUE_CONTRAST = 0.45          # g/cc, a dense body
PRIOR_SD = 0.20               # declared: how much rock varies here
CORR_LEN = 250.0

xs = (np.arange(NX) - (NX - 1) / 2) * CELL
zs = -(np.arange(NZ) + 0.5) * CELL
CX, CZ = np.meshgrid(xs, zs, indexing="ij")
centers = np.column_stack([CX.ravel(), np.zeros(CX.size), CZ.ravel()])
dims = np.tile([CELL, 4000.0, CELL], (len(centers), 1))   # long in y: 2-D
def make_stations(n):
    sx = np.linspace(xs[0], xs[-1], n)
    return sx, np.column_stack([sx, np.zeros(n), np.full(n, 5.0)])

# the truth: one compact body, off-centre and a few hundred metres down
truth = np.zeros(len(centers))
body = ((np.abs(centers[:, 0] - 300.0) < 250.0)
        & (centers[:, 2] < -350.0) & (centers[:, 2] > -650.0))
truth[body] = TRUE_CONTRAST

rng = np.random.default_rng(20260805)


def run(n_station, noise=0.05):
    """One survey EFFORT: the worlds, the readings, and the verdict.

    The variable is how many stations you paid for, not how good the meter
    is. Measured first, then chosen: sweeping instrument precision moved the
    reported interval from 8.8 to 8.6 Mt, i.e. not at all, because the
    posterior is prior-dominated at every noise level. Sweeping the number of
    stations moves it a lot. That is the same finding the CO2 report already
    carries — a survey is usually limited by its geometry, not its noise —
    and a slider that changes nothing would have taught the visitor the
    opposite.
    """
    sx, stations = make_stations(n_station)
    G = pf.prism_matrix(stations, centers, dims)
    d_obs = G @ truth + noise * rng.standard_normal(n_station)
    Gw = G / noise
    prior, _ = PR.build_regular((NX, 1, NZ), (CELL, 4000.0, CELL), PRIOR_SD,
                                CORR_LEN, np.random.default_rng(1))
    mean = PR.posterior_mean(Gw, prior, d_obs / noise)
    sd, _ = PR.posterior_sd(Gw, prior, np.random.default_rng(2), 800)

    # FOUR WORLDS. Matheron's rule: mean + (prior draw minus its own data
    # projection). Every one of these fits the observed readings as well as
    # the answer the software would have shipped.
    SGt = prior.apply_inv(Gw.T)
    K = Gw @ SGt + np.eye(n_station)
    r2 = np.random.default_rng(7)
    m = prior.sample(r2, 4)
    eps = r2.standard_normal((n_station, 4))
    worlds = (mean[:, None] + m - SGt @ np.linalg.solve(K, Gw @ m + eps)).T

    prior_sd_cell = prior.marginal_sd(np.random.default_rng(5), 400)
    informed = 1.0 - sd / prior_sd_cell

    suite = gates.GateSuite()
    sims = (Gw @ prior.sample(rng, 200)
            + rng.standard_normal((n_station, 200))).T
    suite.add(gates.licensing(d_obs / noise, sims))
    fl = prior.sample(np.random.default_rng(22), 400)
    fl = fl - SGt @ np.linalg.solve(
        K, Gw @ fl + np.random.default_rng(33).standard_normal((n_station, 400)))
    rel = [abs(float(np.std(w @ fl)) / PR.functional_sd(Gw, prior, w) - 1)
           for w in np.random.default_rng(40).standard_normal((12, len(centers)))]
    suite.add(gates.GateReport("calibration(MC)",
                               float(np.median(rel)) < 4 * np.sqrt(0.5 / 400),
                               float(np.median(rel)), "vs the sampling floor"))
    sd_b, _ = PR.posterior_sd(Gw, prior, np.random.default_rng(77), 800)
    suite.add(gates.stability((mean, sd), (mean, sd_b), atol=1e-9))
    rms = float(np.sqrt(np.mean((G @ mean - d_obs) ** 2)))
    suite.add(gates.adequacy(rms, noise * PR.expected_residual(Gw, prior)))
    suite.add(gates.recovery(mean, sd, truth))

    # the number a decision turns on: mass under the middle of the survey
    w = np.zeros(len(centers))
    w[(np.abs(centers[:, 0] - 300.0) < 400.0) & (centers[:, 2] > -800.0)] = \
        CELL * CELL * 500.0                       # 500 m of strike length
    mass = float(w @ mean) * 1e-6
    mass_sd = PR.functional_sd(Gw, prior, w) * 1e-6
    true_mass = float(w @ truth) * 1e-6

    return dict(
        noise=noise, n_station=n_station,
        station_x=[round(v, 1) for v in sx.tolist()],
        readings=[round(v, 4) for v in (G @ truth).tolist()],
        observed=[round(v, 4) for v in d_obs.tolist()],
        worlds=[[round(v, 3) for v in w_] for w_ in worlds.tolist()],
        world_readings=[[round(v, 4) for v in (G @ np.array(w_)).tolist()]
                        for w_ in worlds.tolist()],
        mean=[round(v, 3) for v in mean.tolist()],
        informed=[round(v, 3) for v in informed.tolist()],
        mass=round(mass, 1), mass_sd=round(mass_sd, 1),
        true_mass=round(true_mass, 1),
        status=suite.status,
        gates=[dict(name=r.name.split("(")[0], passed=bool(r.passed))
               for r in suite.reports],
    )


levels = [4, 6, 9, 14, 20, 30]
out = dict(nx=NX, nz=NZ, cell=CELL,
           x0=float(xs[0]), z0=float(zs[0]),
           truth=[round(v, 3) for v in truth.tolist()],
           true_contrast=TRUE_CONTRAST, prior_sd=PRIOR_SD,
           levels=[])
for n_ in levels:
    r = run(n_)
    out["levels"].append(r)
    print(f"{n_:3d} stations -> {r['status']:12s} "
          f"mass {r['mass']:+7.1f} +/- {r['mass_sd']:5.1f} Mt "
          f"(true {r['true_mass']:.1f})  informed "
          f"{100 * np.mean(np.array(r['informed']) > 0.05):.0f}%")

p = ROOT / "web" / "demo_data.json"
p.write_text(json.dumps(out, separators=(",", ":")))
print(f"\nwrote {p}  ({p.stat().st_size / 1024:.0f} KB)")
