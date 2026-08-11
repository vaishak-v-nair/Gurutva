"""Gate 1.4 — what the L0 layer costs.

Reported two ways on purpose. The plan's target is <1% wall-clock, and a <1%
claim needs timing precision better than 1%, which a shared container does not
have. So:

  (a) analytic — bytes touched by the detectors per step, against bytes touched
      by the training step. Deterministic, and not a function of who else is on
      the machine.
  (b) wall-clock — A/B interleaved to cancel drift, medians over many
      repetitions, **with the measurement noise floor stated**. If the effect is
      below the floor, that is what gets reported. Not a number.

Run:  python -m sdc.figures.overhead
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import torch

from sdc.detect import Guards
from sdc.model import TinyTransformer
from sdc.runner import TrainConfig, train

REPS = 9
STEPS = 20
OUT = Path(__file__).parent


def _time_once(cfg: TrainConfig, guarded: bool, stride: int = 1) -> float:
    start = time.perf_counter()
    train(cfg, guards=Guards(optim_stride=stride) if guarded else None)
    return time.perf_counter() - start


def measure() -> dict:
    cfg = TrainConfig(steps=STEPS)

    # Warm up: first call pays for lazy init and the weight cache.
    _time_once(cfg, guarded=False)
    _time_once(cfg, guarded=True)

    plain: list[float] = []
    guarded: list[float] = []
    for i in range(REPS):
        # Interleaved, and order-swapped each rep, so any monotonic drift in
        # machine load cancels instead of loading onto one arm.
        if i % 2 == 0:
            plain.append(_time_once(cfg, False))
            guarded.append(_time_once(cfg, True))
        else:
            guarded.append(_time_once(cfg, True))
            plain.append(_time_once(cfg, False))
        print(f"  rep {i + 1}/{REPS}  plain {plain[-1]:.3f}s  guarded {guarded[-1]:.3f}s")

    p_med, g_med = statistics.median(plain), statistics.median(guarded)

    # The noise floor: spread within the *unguarded* arm, which by construction
    # measures the same work every time. Any A/B difference smaller than this is
    # not resolvable on this machine.
    p_lo, p_hi = min(plain), max(plain)
    noise_frac = (p_hi - p_lo) / p_med

    return {
        "reps": REPS,
        "steps": STEPS,
        "plain_median_s": p_med,
        "guarded_median_s": g_med,
        "overhead_frac": (g_med - p_med) / p_med,
        "noise_floor_frac": noise_frac,
        "plain_all_s": plain,
        "guarded_all_s": guarded,
    }


def analytic(cfg: TrainConfig, optim_stride: int = 1) -> dict:
    """Bytes the detectors touch per step, against the training step's own traffic.

    Deliberately counts the detectors generously — the int64 widening and the
    modular reduction both materialise temporaries, and pretending they are free
    would understate the cost.
    """
    model = TinyTransformer(cfg.model)
    P = sum(p.numel() for p in model.parameters())
    tokens = cfg.batch_size * cfg.model.block_size

    # Detector traffic per step:
    #   gradients   stamp + verify            = 2 passes over P, every step
    #   optim state stamp + verify, 2 moments = 4 passes over P, every Nth step
    # Each pass reads 4B (fp32) and materialises an 8B int64 temporary plus an
    # 8B weighted temporary => ~20B of traffic per element per pass.
    #
    # This is the trustworthy half of the gate. It is deterministic, it does not
    # depend on who else is on the machine, and it is what should be quoted.
    passes = 2 + 4 / optim_stride
    detector_bytes = passes * P * 20

    # Training step traffic, conservatively: fwd+bwd touch parameters and their
    # gradients several times over, and activations scale with tokens.
    step_bytes = 4 * (3 * P + 2 * P * 1) + 4 * tokens * cfg.model.n_embd * 8

    # FLOPs are the fairer denominator for the compute-bound part: the standard
    # 6*P*tokens for fwd+bwd. The detectors are ~O(P) elementwise ops.
    step_flops = 6 * P * tokens
    detector_flops = passes * P * 4

    return {
        "params": P,
        "optim_stride": optim_stride,
        "passes_over_params_per_step": passes,
        "tokens_per_step": tokens,
        "detector_bytes_per_step": detector_bytes,
        "step_bytes_per_step": step_bytes,
        "byte_ratio": detector_bytes / step_bytes,
        "detector_flops_per_step": detector_flops,
        "step_flops_per_step": step_flops,
        "flop_ratio": detector_flops / step_flops,
    }


def scaling() -> list[dict]:
    """Overhead against tokens per step.

    The detectors are O(params) per step; the training step is O(params x tokens).
    So the ratio should fall as 1/tokens, and the toy configuration used for the
    gates sits at the worst possible end of that curve. Measuring the trend turns
    a structural argument into evidence — or refutes it.
    """
    from sdc.model import ModelConfig

    from sdc.model import ModelConfig

    rows = []
    for batch, block in ((4, 32), (8, 64), (16, 64), (32, 128), (64, 128)):
        cfg = TrainConfig(
            steps=6, batch_size=batch, model=ModelConfig(block_size=block)
        )
        _time_once(cfg, guarded=False)  # warm
        # min-of-3: wall clock noise is one-sided, so the minimum is the least
        # contaminated estimate of the work actually done.
        plain = min(_time_once(cfg, False) for _ in range(3))
        full = min(_time_once(cfg, True, stride=1) for _ in range(3))
        strided = min(_time_once(cfg, True, stride=8) for _ in range(3))
        tokens = batch * block
        rows.append(
            {
                "batch": batch,
                "block": block,
                "tokens_per_step": tokens,
                "plain_s": plain,
                "guarded_s": full,
                "overhead_frac": (full - plain) / plain,
                "overhead_frac_stride8": (strided - plain) / plain,
            }
        )
        print(
            f"  {tokens:6d} tokens/step -> stride1 {rows[-1]['overhead_frac'] * 100:7.2f}%"
            f"   stride8 {rows[-1]['overhead_frac_stride8'] * 100:7.2f}%"
            f"   (plain {plain:.3f}s)"
        )
    return rows


def noise_at_scale(batch: int = 128, block: int = 128, reps: int = 6) -> dict:
    """Spread of the *same unguarded* configuration, repeated.

    Establishes where measurement stops meaning anything. Attempts to measure
    overhead above ~8k tokens/step on this container produced values including a
    negative one, which is impossible and therefore diagnostic: noise dominates.
    """
    from sdc.model import ModelConfig

    cfg = TrainConfig(steps=4, batch_size=batch, model=ModelConfig(block_size=block))
    _time_once(cfg, guarded=False)
    runs = [_time_once(cfg, guarded=False) for _ in range(reps)]
    med = statistics.median(runs)
    return {
        "tokens_per_step": batch * block,
        "reps": reps,
        "times_s": runs,
        "median_s": med,
        "spread_frac": (max(runs) - min(runs)) / med,
    }


def plot_scaling(rows: list[dict], noise: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5.5))
    xs = [r["tokens_per_step"] for r in rows]
    ys = [max(r["overhead_frac"] * 100, 0.05) for r in rows]
    ys8 = [max(r["overhead_frac_stride8"] * 100, 0.05) for r in rows]

    ax.plot(xs, ys, marker="o", color="tab:red", label="every step", zorder=4)
    ax.plot(
        xs, ys8, marker="s", color="tab:blue",
        label="optimizer state every 8th step", zorder=4,
    )

    ref_x = [xs[0] * 2 ** i for i in range(9)]
    ref_y = [ys[0] * xs[0] / x for x in ref_x]
    ax.plot(ref_x, ref_y, ls="--", color="grey", lw=1, label="1/tokens reference")

    ax.axhline(1.0, color="tab:green", ls=":", lw=1.5)
    ax.text(xs[0] * 1.1, 1.15, "1% target", fontsize=9, color="tab:green")

    floor = noise["spread_frac"] * 100
    ax.axhspan(0.04, floor, color="grey", alpha=0.18)
    ax.text(
        xs[0] * 1.1, floor * 0.55,
        f"below this machine's noise floor ({floor:.0f}% spread on identical runs)",
        fontsize=8.5, va="center",
    )

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_ylim(0.04, 400)
    ax.set_xlabel("tokens per step")
    ax.set_ylabel("L0 wall-clock overhead (%, log)")
    ax.set_title(
        "Gate 1.4 — fails at toy scale; the fix is fewer passes, not bigger batches\n"
        "The red curve leaves the 1/tokens reference: the checksum is memory-bound, "
        "and so is the large-batch step.",
        fontsize=10.5,
    )
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "overhead.png", dpi=150)
    print(f"wrote {OUT / 'overhead.png'}")


if __name__ == "__main__":
    cfg = TrainConfig(steps=STEPS)
    print(f"timing {REPS} interleaved reps of {STEPS} steps...")
    timing = measure()
    calc = analytic(cfg)
    print("\nscaling against tokens/step...")
    rows = scaling()
    print("\nnoise floor at scale (same config, repeated, unguarded)...")
    noise = noise_at_scale()
    print(
        f"  {noise['tokens_per_step']} tokens/step: spread "
        f"{noise['spread_frac'] * 100:.1f}% of median — anything smaller than this "
        "is not measurable here"
    )
    data = {
        "wall_clock": timing,
        "analytic": calc,
        "scaling": rows,
        "noise_at_scale": noise,
        "extrapolated_1pct_crossing_tokens": rows[0]["tokens_per_step"]
        * rows[0]["overhead_frac"]
        * 100,
    }
    (OUT / "overhead.json").write_text(json.dumps(data, indent=2))
    try:
        plot_scaling(rows, noise)
    except ImportError:
        print("matplotlib unavailable — JSON written, plot skipped")

    print()
    print("=== analytic ===")
    print(f"params                {calc['params']:,}")
    print(f"detector : step bytes {calc['byte_ratio'] * 100:6.2f}%")
    print(f"detector : step FLOPs {calc['flop_ratio'] * 100:6.2f}%")
    print()
    print("=== wall clock ===")
    print(f"plain     median      {timing['plain_median_s']:.4f}s")
    print(f"guarded   median      {timing['guarded_median_s']:.4f}s")
    print(f"measured overhead     {timing['overhead_frac'] * 100:+6.2f}%")
    print(f"noise floor           {timing['noise_floor_frac'] * 100:6.2f}%")
    if abs(timing["overhead_frac"]) < timing["noise_floor_frac"]:
        print("VERDICT: overhead is BELOW the measurement noise floor on this")
        print("         machine. Not reportable as a number — see analytic above.")
    else:
        print(f"VERDICT: overhead resolvable at {timing['overhead_frac'] * 100:+.2f}%")
    print(f"\nwrote {OUT / 'overhead.json'}")
