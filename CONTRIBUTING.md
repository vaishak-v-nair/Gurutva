# Contributing

The most valuable contribution is not a pull request. It is **running this on
your own survey and telling me it is wrong.** Nobody has done that yet.

## Reporting that it gave a wrong answer

Use the issue template. The three declared numbers — noise, prior sd,
correlation length — are the whole reproducibility story, because they are
statements about *your* site that the software refuses to invent. Without
them nobody can tell a bug from a correctly-declared prior that you disagree
with.

## Running the tests

```bash
pip install -r requirements.txt && pytest
```

That is the product layer: numpy, scipy, matplotlib, pytest. It runs on
Python 3.11 through 3.14, and CI proves it on all four.

Tests that gate the research results (the SimPEG re-inversion, the S1/S2/S3
neural work) skip themselves unless you also install the reproduction
environment:

```bash
pip install -r requirements-research.txt
```

Those pins are exact on purpose and are validated against Python 3.14.

## House rules, learned the hard way

These are not style preferences. Each one came from a specific failure and is
enforced somewhere in code.

1. **Calibrate every gate to the numerics that produce its numbers.** A test
   tuned to hope fails correct code. Six separate times a threshold was set
   below its own sampling floor. **Never a max-over-many-cells criterion** —
   it sits at the noise floor by construction.
2. **Measure the limit before claiming the result.** The mesh error floor
   (0.755 mGal) was measured before the first inversion, so the honest target
   existed before any temptation to hit a prettier one.
3. **Publish the confession beside the claim.** Every qualification lives next
   to the result it qualifies, in the README, not in a footnote.
4. **Report surprises against your own expectation**, including when the
   surprise makes you look wrong.
5. **No claim without a figure**, or it is labelled aspiration.
6. **Run it, do not reason about it.** Nearly every real defect in this repo
   was found by running the thing and looking, not by reading the code. The
   released binary stamped every report with no version for a full release
   because nobody ran the released binary and read its output.

## Pull requests

Add a test for the behaviour you are changing. If you are fixing a bug, the
test should fail before your fix. Every bug ever found in this repo has a test
named after it, and that is why the count keeps going up.

Keep the diff to what you are changing. If you think the foundation is wrong,
open an issue and say so — that is a more useful conversation than a large PR.

## Licence

Apache-2.0. By contributing you agree your contribution is licensed under it.
