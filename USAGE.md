# How to use Gurutva

You have a gravity or magnetic survey. You want to know what it actually supports before you spend money on it. That is the whole product.

## 1. Put your survey in a CSV

Four columns, header row required:

```
x,y,z,gz
331000,4263000,1650.2,-0.084
331250,4263000,1652.7,-0.091
```

| column | meaning |
|---|---|
| `x`, `y` | position in **metres** (UTM is fine) **or `lon`,`lat` in degrees** — detected automatically and projected onto a local tangent plane, accurate to a few parts per million across a survey |
| `z` | station elevation in **metres**, z-up |
| `gz` | your gravity anomaly in **mGal**, background already removed |

Column names are case-insensitive and common aliases work: `lon`/`longitude`/`easting`, `lat`/`latitude`/`northing`, `elev`/`elevation`/`height`, `gz`/`mgal`/`tmi`/`nt`.

Examples you can run right now:

| file | what it is |
|---|---|
| `examples/demo_survey.csv` | 120 gravity stations, projected metres |
| `examples/demo_survey_latlon.csv` | the same survey in lon/lat degrees |
| `examples/demo_survey_topo.csv` | the same survey over 667 m of relief |
| `examples/demo_magnetics.csv` | a magnetic survey, total-field nT |
| `examples/demo_lease_block.csv` | a 5-vertex lease block to report on |

## 2. Ask what your survey can see — BEFORE you trust it

```bash
py -3 -m src.product.cli selftest --survey examples/demo_survey.csv --noise 0.05 --prior-sd 0.10 --corr-len 400 --body-depth 900 --body-radius 300 --body-contrast -0.4 --out capability.html
```

This plants a body of a size and depth **you choose**, simulates what *your* station layout and *your* noise would record, and inverts it. Because the answer is known, gate 5 runs and tells you whether your survey could have found that body at all.

Change `--body-depth` and run it again. The depth where it stops recovering is the depth your survey goes blind. That number is worth knowing before a drilling programme, not after.

## 3. Run it on your real data

```bash
py -3 -m src.product.cli report --survey examples/demo_survey.csv --noise 0.05 --prior-sd 0.10 --corr-len 400 --out report.html
```

You get `report.html` (open it in a browser) and `report.json` (for your own scripts). The exit code is `0` when the result is claimable and `1` when it is not, so it drops straight into a pipeline.

## 4. What you must declare, and why nothing is guessed for you

| flag | what it is | why you, not us |
|---|---|---|
| `--noise` | your **measured** repeatability, mGal | a tool that invents a noise floor is inventing its own passing grade |
| `--prior-sd` | how much density varies at your site, g/cc | this is geology, not software. Take it from your own logs or report |
| `--corr-len` | the scale your rock varies over, m | rock is connected; a prior that treats each cell as a stranger cannot make the signal a real basin makes |

Optional: `--cell` (default 200 m), `--depth` (2000 m), `--region` (side of the block the result is reported for, 2000 m), `--samples` (600).

### Magnetics

```bash
py -3 -m src.product.cli report --survey examples/demo_magnetics.csv --field mag --noise 2.0 --prior-sd 3e-4 --corr-len 400 --b0 55000 --inclination 70 --declination 2 --out mag.html
```

Data in nT, model out in SI susceptibility. `--b0`, `--inclination` (positive **down**) and `--declination` (east of north) describe your ambient field.

**Remanence.** If the rock carries remanent magnetisation, add `--remanence-q` (the Koenigsberger ratio, remanent over induced) and optionally `--remanence-inclination` / `--remanence-declination`. Because the direction is *declared* rather than solved for, the problem stays linear and every gate still applies. A reversely magnetised body then reads negative over positive susceptibility, which is what reversed bodies actually do — an induced-only model fits that by inventing negative susceptibility, which is not a rock. The report states the declared values; without `--remanence-q` it says "induced magnetisation only".

### Do not guess the prior amplitude — declare the target

```bash
py -3 -m src.product.cli report --survey mysurvey.csv --field mag --noise 2.0 --corr-len 400 --target-contrast 0.06 --target-radius 300 --target-depth 700 --out r.html
```

Instead of `--prior-sd`, describe **the body you are looking for**. The prior amplitude is then derived from the anomaly that body would make:

```
prior sd 0.001549 SI susceptibility derived from a declared target:
+0.06 SI over 21 cells at 700 m depth, which would make a 46.9 nT anomaly
```

This uses only declared physics and never looks at your data, so it cannot become a way of tuning until a gate turns green. If adequacy then fails, that is the tool telling you the body you are hunting is not in this data.

### Joint gravity + magnetic inversion

```bash
py -3 -m src.product.cli joint --survey grav.csv --survey-mag mag.csv --noise 0.05 --noise-mag 2.0 --prior-sd 0.10 --prior-sd-chi 3e-4 --rho-chi-correlation 0.7 --corr-len 400 --out joint.html
```

Both surveys must be at the **same stations, in the same order** — interpolating one onto the other invents data, and the invented part would carry no error bar.

The two physics are coupled through one declared number: `--rho-chi-correlation`, the petrophysical correlation between density and susceptibility. That coupling lives in the prior, not in a penalty term, so the posterior stays closed-form and every gate still applies.

**At `0.0` the two inversions are exactly independent** — pinned by a test, and the report will say the magnetics bought 0.0%. Turn it up and the magnetics start informing density. On the example data, `0.7` narrows the mass interval by 5.3% and moves the estimate from +25.1 to +47.8 Mt. **That movement is your assumption doing work, not a measurement**, and the report labels it as such.

### Your own ground, not a box we chose

```bash
py -3 -m src.product.cli report --survey mysurvey.csv --region-file mylease.csv --noise 0.05 --prior-sd 0.1 --corr-len 400 --out r.html
```

`mylease.csv` is just vertices: a header of `x,y` (metres) or `lon,lat` (degrees), then one row per corner. Concave shapes work and the ring closes itself. The report names the block and its area.

### Topography

The mesh top is draped to your station elevations by default whenever the relief is more than half a cell. Force it with `--topography on` or `off`.

This matters more than it sounds. With a flat-topped mesh over a valley the inversion is handed air to put density into, and it will. On the 667 m relief example, draping drops 535 of 2,856 cells and **tightens** the reported interval from ±46 to ±36 Mt, because mass that could have hidden in the air is gone.

### If the prior-scale warning fires

`--prior-sd` is a **per-cell** standard deviation, and thousands of correlated cells add up. Hunting a body of susceptibility 0.06 and typing `--prior-sd 0.06` declares a prior predicting hundreds of nT over a survey that reads single digits. The tool measures this and tells you what to use instead:

```
WARNING: your declared prior predicts data with spread 889 but yours has 2.12 (419x too wide).
         --prior-sd is a PER-CELL sd and correlated cells add up.
         For this survey try about 0.00014 SI susceptibility.
```

Take that as a starting point, then check it against your rocks. It is a scale hint, not a licence to tune until a gate turns green.

**Declare these from your site. Never tune them until a gate turns green.** That is the one rule. If a gate fails, the honest move is to report the failure, not to adjust the inputs until it passes.

## 5. Reading the report

**The verdict line, first.** One of:

- **`PROVISIONAL`** (amber) — the normal result on real data. All four core gates passed, and nothing was checked against a known right answer, because real data has none. Run `selftest` to close that gap.
- **`CLAIMABLE`** (green) — all four core gates passed **and** the result was verified against a known truth. Only `selftest` can reach this.
- **`NOT CLAIMED`** (red) — a gate failed. Everything below it is a diagnostic. Do not put it in a decision.

**Then the numbers:**

1. **The quantity in your block, with a 95% interval** — excess mass in Mt for gravity, mean susceptibility for magnetics. If the interval spans zero you have no detection, and the upper limit tells you the most that could be hiding there.
2. **The depth below which nothing is constrained**, and separately the depth below which the *typical* cell is not. Those differ: a compact magnetic body lights a handful of cells to 87% while the median cell sits at 1.3%, which means per-cell values are not a map.
3. **What the per-cell shortcut would have said.** Usually 2 to 3 times too tight. Neighbouring cells trade off against each other, so the variance of a sum is not the sum of variances.

## 6. The gates, one line each

| gate | question |
|---|---|
| licensing | could my assumptions ever have produced this data? |
| calibration | when I say 95% sure, am I right 95% of the time? |
| stability | run it twice from scratch — same answer? |
| adequacy | does my answer reproduce the data it came from? |
| recovery | does my interval contain a known right answer? *(only when a truth exists)* |

The first three ask whether the model is self-consistent. They have all passed together on models that were wrong. Gates 4 and 5 are the ones that ask whether it is *right*.

## 7. When it fails

- **`adequacy — OVERFITS`**: your prior is loose enough that the model explains your data too easily. Tighten `--prior-sd` to something you can defend from your rocks.
- **`adequacy — UNDERFITS`**: your model class cannot reproduce your data. Try a finer `--cell`, a deeper `--depth`, or accept that you need a different model.
- **`licensing`**: the data is not something your declared prior could produce. Usually a regional field from mass outside your mesh, or a prior that is far too tight.
- **`recovery`** (selftest only): your survey cannot see the body you planted. That is an answer, not an error.
