# How to use Gurutva

You have a gravity survey. You want to know what it actually supports before you spend money on it. That is the whole product.

## 1. Put your survey in a CSV

Four columns, header row required:

```
x,y,z,gz
331000,4263000,1650.2,-0.084
331250,4263000,1652.7,-0.091
```

| column | meaning |
|---|---|
| `x`, `y` | position in **metres**, any consistent projected system (UTM is fine) |
| `z` | station elevation in **metres**, z-up |
| `gz` | your gravity anomaly in **mGal**, background already removed |

An example you can run right now: `examples/demo_survey.csv` (120 stations).

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

Optional: `--cell` (default 200 m), `--depth` (2000 m), `--region` (side of the block the mass is reported for, 2000 m), `--samples` (600).

**Declare these from your site. Never tune them until a gate turns green.** That is the one rule. If a gate fails, the honest move is to report the failure, not to adjust the inputs until it passes.

## 5. Reading the report

**The verdict line, first.** One of:

- `CLAIMABLE — four core gates pass. RECOVERY UNTESTED` — the normal result on real data. The four checks passed, and nothing was compared against a known right answer, because on real data there isn't one. Run `selftest` to cover that gap.
- `CLAIMABLE — ... recovery verified against a known truth` — only from `selftest`.
- `NOT CLAIMED — failed: <gate>` — everything below it is a diagnostic. Do not put it in a decision.

**Then the three numbers:**

1. **Excess mass in the block, with a 95% interval.** If the interval spans zero you have no detection — and the upper limit tells you the most that could be hiding there.
2. **The depth below which your model is invented.** Deeper than that, the typical cell is your prior, not your data.
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
