"""The Phase 0 deliverable: divergence against bit position and injection volume.

This surface is the problem statement for Phase 1. It says which corruption a
detector must actually work for, and it is the axis every later detection number
gets reported against — a single scalar "detection rate" marginalises exactly the
variable that decides whether the product is useful.

Run:  python -m sdc.figures.difficulty
"""

from __future__ import annotations

import json
from pathlib import Path

from sdc.inject import Injector, field_of
from sdc.runner import TrainConfig, train

import torch

STEPS = 30
SEED = 11
BITS = [31, 30, 27, 24, 23, 22, 18, 12, 6, 2, 0]
COUNTS = [1, 4, 16]
OUT = Path(__file__).parent


def sweep() -> dict:
    cfg = TrainConfig(steps=STEPS)
    clean = train(cfg)
    rows = []
    for count in COUNTS:
        for bit in BITS:
            injected = train(
                cfg,
                Injector(seed=SEED, bits=(bit,), count=count, sites=("gemm_out",)),
            )
            gap = injected.loss_divergence(clean)
            rows.append(
                {
                    "bit": bit,
                    "field": field_of(bit, torch.float32),
                    "count": count,
                    "divergence": gap,
                    "n_injections": injected.n_injections,
                    "blew_up": injected.diverged,
                }
            )
            print(
                f"bit {bit:2d} ({rows[-1]['field']:8s}) x{count:3d} "
                f"-> gap {gap:.3e}{'  [NaN/inf]' if injected.diverged else ''}"
            )
    return {
        "clean_final_loss": clean.final_loss,
        "steps": STEPS,
        "seed": SEED,
        "rows": rows,
    }


def plot(data: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5.5))
    CAP = 1e3  # where inf/NaN runs get drawn
    ULP = 2.384e-07  # one ULP of the loss — the floor of what loss can resolve

    for count, colour in zip(COUNTS, ("tab:blue", "tab:purple", "tab:brown")):
        rows = sorted(
            (r for r in data["rows"] if r["count"] == count), key=lambda r: r["bit"]
        )
        xs = [r["bit"] for r in rows]
        ys = [CAP if r["blew_up"] else max(r["divergence"], 1e-12) for r in rows]
        ax.plot(xs, ys, marker="o", ms=5, color=colour, label=f"{count} flip(s)/step")
        blown = [(x, y) for x, y, r in zip(xs, ys, rows) if r["blew_up"]]
        if blown:
            ax.scatter(
                *zip(*blown), marker="X", s=140, color=colour, zorder=5,
                edgecolors="black", linewidths=0.6,
            )

    ax.axhline(ULP, color="black", ls=":", lw=1.2)
    ax.text(
        0.3, ULP * 1.4, "one ULP of the loss — nothing below this is resolvable",
        fontsize=8, style="italic",
    )
    ax.axhspan(1e-12, ULP * 2, color="tab:green", alpha=0.07)

    ax.axvspan(-0.5, 22.5, color="tab:green", alpha=0.06)
    ax.axvspan(22.5, 30.5, color="tab:orange", alpha=0.09)
    ax.axvspan(30.5, 31.5, color="tab:red", alpha=0.09)
    for x, label, colour in (
        (11, "mantissa", "tab:green"),
        (26.5, "exponent", "tab:orange"),
    ):
        ax.text(x, 2e4, label, ha="center", fontsize=9, color=colour)
    ax.text(31, 2e4, "sign", ha="center", fontsize=8, color="tab:red")

    ax.set_yscale("log")
    ax.set_ylim(1e-8, 3e5)
    ax.set_xlabel("flipped bit index  (fp32: 0–22 mantissa, 23–30 exponent, 31 sign)")
    ax.set_ylabel("max loss gap vs clean run (nats, log)")
    ax.set_title(
        "Phase 0 — what a detector actually has to see\n"
        f"{data['steps']} steps, clean final loss {data['clean_final_loss']:.3f}. "
        "X = run diverged to NaN/inf.",
        fontsize=11,
    )
    ax.legend(loc="center left")
    ax.grid(alpha=0.25, which="both")
    fig.tight_layout()
    fig.savefig(OUT / "difficulty.png", dpi=150)
    print(f"wrote {OUT / 'difficulty.png'}")


if __name__ == "__main__":
    data = sweep()
    (OUT / "difficulty.json").write_text(json.dumps(data, indent=2))
    print(f"wrote {OUT / 'difficulty.json'}")
    try:
        plot(data)
    except ImportError:
        print("matplotlib unavailable — JSON written, plot skipped")
