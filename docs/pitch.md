# Gurutva — the pitch

One page. Plain sentences. Every number below is generated from a shipped run, not written by hand.

Live page: https://gurutva.vercel.app/
Code: https://github.com/vaishak-v-nair/Gurutva (public, Apache-2.0)
Download: https://github.com/vaishak-v-nair/Gurutva/releases/latest

---

## The problem, in your language

You measure gravity at the surface. Software turns it into a picture of what is underground. You then bet real money on that picture.

But many different arrangements of rock make the exact same measurement. The software picks one of them and draws it. It does not tell you which parts came from your data and which parts came from its own settings.

So the picture looks confident. It has no error bars. And the decision it supports is worth millions.

## What we give back

Three things, on one page.

1. **Where the model is real, and where it is invented.** A map of what your survey actually constrains.
2. **Pass or fail, on four checks.** Can my assumptions have produced this data? Are my error bars honest? Do I get the same answer twice? And does my answer reproduce the data it came from?
3. **The one number you act on, with an interval.** How much mass is in this block. How much CO₂ could be outside the box. Not a point estimate — a range.

## Two proofs, both reproducible

**Utah FORGE, real DOE gravity data.** All four checks pass. The survey constrains 30% of the model, and below 1,322 m elevation the typical cell is not constrained at all. Excess mass in a 2×2 km block: **−346 Mt, 95% between −484 and −207.** The standard workflow would have quoted that interval **2.6× too tight**.

**CO₂ containment, Sleipner-class survey design.** Inventory reconciles at 7.4 ± 7.1 Mt against 10 Mt injected. A 400 m accumulation 2 km outside the storage complex is detectable **only if it is at least 72% saturated** — emptier than that and no containment claim covers it. And upgrading the gravimeter from 3 µGal to 1.1 µGal buys **6%**: that survey is limited by its geometry, not its noise.

*(The observations in the second case are simulated from published Sleipner survey parameters. It says so on the page, in the code, and twice in the report. What it computes is what a survey **design** can prove.)*

## Why this is hard to copy

Anyone can add error bars. Almost nobody will ship a tool that tells a paying customer "not today."

Ours does. On a satellite-gravity model of the South American Moho, three checks passed — and the fourth showed the model could not reproduce the gravity it was fitted to. We published the failure and withheld the map.

That refusal is the product. It cannot be faked by a competitor who has not built the thing that says no.

It also caught two priors that the industry uses every day:

- Tuning the regularization weight until the fit looks right implies rock density varies by 0.021 g/cc. Real rock varies by ten times that. That prior fails the first check at the 100th percentile.
- Treating each cell as independent also fails. Rock is connected over hundreds of metres; independent cells cancel each other out and cannot make the signal a real basin makes.

Both were caught by our own gates, on our own work, before anyone else saw it.

## What we want from you

**If you run surveys** — give us one dataset and the model you already produced from it. We return the verdict report you saw above, on your data. No charge for the first one. If the answer is "your survey cannot support this decision," that is what you will get, and it is worth knowing before the drill moves.

**If you work on this problem** — the repository is public, Apache-2.0, and has 172 passing tests. The retractions are in there too: a headline we withdrew within four hours, a fix that failed on physical grounds, six separate times a threshold was set below its own sampling floor. Tear it apart. We would rather be corrected than agreed with.

---

Vaishak V Nair · https://github.com/vaishak-v-nair/Gurutva/issues
