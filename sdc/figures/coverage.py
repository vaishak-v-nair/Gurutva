"""The Phase 1 deliverable: what L0 can and cannot see.

Read next to ``difficulty.png`` this states the thesis in two charts — there is
what corruption does, and here is what it costs to see it.

The dominant structure is not a gradient, it is a partition. Sites where an
uncorrupted reference copy exists are caught completely, at every bit position,
for the price of two passes over memory. Sites where no reference exists are
invisible to this layer at any price, because there is nothing to compare
against. That is a structural fact about the layer, not a shortfall in it, and it
is the entire argument for L1 and L2.

Run:  python -m sdc.figures.coverage
"""

from __future__ import annotations

import json
from pathlib import Path

from sdc.detect import REFERENCED_SITES, UNREFERENCED_SITES, Guards
from sdc.inject import Injector
from sdc.runner import TrainConfig, train

SITES = list(REFERENCED_SITES) + list(UNREFERENCED_SITES)
BITS = [0, 6, 12, 18, 22, 23, 27, 30, 31]
SEEDS = [3, 5, 7]
STEPS = 12
COUNT = 2
OUT = Path(__file__).parent


def sweep() -> dict:
    cells = {}
    for site in SITES:
        for bit in BITS:
            caught_by = []
            injected = 0
            for seed in SEEDS:
                guards = Guards()
                record = train(
                    TrainConfig(steps=STEPS),
                    Injector(seed=seed, bits=(bit,), count=COUNT, sites=(site,)),
                    guards=guards,
                )
                injected += record.n_injections
                if record.fired("checksum"):
                    caught_by.append("checksum")
                elif record.fired("norm"):
                    caught_by.append("norm")
                else:
                    caught_by.append(None)

            hits = sum(1 for c in caught_by if c is not None)
            by = next((c for c in caught_by if c), None)
            cells[f"{site}/{bit}"] = {
                "site": site,
                "bit": bit,
                "rate": hits / len(SEEDS),
                "detector": by,
                "injections": injected,
            }
            mark = {"checksum": "C", "norm": "N", None: "."}[by]
            print(f"  {site:12s} bit {bit:2d}  rate {hits}/{len(SEEDS)}  [{mark}]")
    return cells


def plot(cells: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    grid = np.zeros((len(SITES), len(BITS)))
    labels = [["" for _ in BITS] for _ in SITES]
    for i, site in enumerate(SITES):
        for j, bit in enumerate(BITS):
            cell = cells[f"{site}/{bit}"]
            grid[i, j] = cell["rate"]
            labels[i][j] = {"checksum": "C", "norm": "N", None: "·"}[cell["detector"]]

    fig, ax = plt.subplots(figsize=(11.5, 4.2))
    ax.imshow(grid, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    fig.subplots_adjust(right=0.72)

    for i in range(len(SITES)):
        for j in range(len(BITS)):
            ax.text(
                j, i, labels[i][j], ha="center", va="center",
                fontsize=13, fontweight="bold",
                color="black" if grid[i, j] > 0.5 else "dimgrey",
            )

    ax.set_xticks(range(len(BITS)), [str(b) for b in BITS])
    ax.set_yticks(range(len(SITES)), SITES)
    ax.set_xlabel("flipped bit index (fp32)")

    # The partition is the result; draw it, and label it outside the cells so
    # nothing overlaps the data.
    split = len(REFERENCED_SITES) - 0.5
    ax.axhline(split, color="black", lw=2.5)
    right = len(BITS) - 0.4
    ax.text(
        right, (len(REFERENCED_SITES) - 1) / 2,
        "reference copy exists\n→ caught at every bit",
        fontsize=9, ha="left", va="center", style="italic", clip_on=False,
    )
    ax.text(
        right, split + 1 + (len(UNREFERENCED_SITES) - 2) / 2,
        "no reference\n→ L0 cannot help at any price",
        fontsize=9, ha="left", va="center", style="italic", clip_on=False,
    )

    ax.set_title(
        "Phase 1 — L0 coverage is a partition, not a gradient\n"
        "C = caught by integer checksum   N = caught by norm monitor (NaN)   "
        "· = missed by everything",
        fontsize=10.5,
    )
    fig.savefig(OUT / "coverage.png", dpi=150, bbox_inches="tight")
    print(f"wrote {OUT / 'coverage.png'}")


if __name__ == "__main__":
    cells = sweep()
    (OUT / "coverage.json").write_text(json.dumps(cells, indent=2))
    print(f"wrote {OUT / 'coverage.json'}")
    try:
        plot(cells)
    except ImportError:
        print("matplotlib unavailable — JSON written, plot skipped")
