"""The image GitHub shows when someone shares the repo.

Drawn from `web/demo_data.json`, not illustrated. The four panels on the right
are the same posterior samples the landing page opens with, and the first one
is the block that was actually buried. A social card for this project that
used stock imagery would be the one dishonest artifact in it.

1280x640 is GitHub's social-preview size; it also serves as the og:image, so
a link pasted into Slack or Hacker News shows the argument rather than a URL.

Run: py -3 web/make_social.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
D = json.loads((ROOT / "web" / "demo_data.json").read_text())

PAPER, INK, MUT, LINE = "#faf8f4", "#12151a", "#5d6670", "#d8d3c9"
SERIF = ["Iowan Old Style", "Palatino Linotype", "Palatino", "Georgia",
         "DejaVu Serif", "serif"]
UI = ["Segoe UI", "Helvetica Neue", "Helvetica", "DejaVu Sans", "sans-serif"]


def ramp(v):
    """The page's own colour ramp, so the card and the site agree."""
    t = max(-1.0, min(1.0, v / 0.5))
    if t >= 0:
        return (1 - 60 * t / 255, 1 - 150 * t / 255, 1 - 165 * t / 255)
    a = -t
    return (1 - 165 * a / 255, 1 - 120 * a / 255, 1 - 40 * a / 255)


def panel(ax, vals, label, strong=False):
    nx, nz = D["nx"], D["nz"]
    grid = np.array(vals).reshape(nx, nz).T
    rgb = np.zeros(grid.shape + (3,))
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            rgb[i, j] = ramp(grid[i, j])
    ax.imshow(rgb, aspect="auto", interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(INK if strong else LINE)
        s.set_linewidth(2.0 if strong else 1.0)
    ax.set_title(label, fontsize=10.5, color=INK if strong else MUT,
                 fontfamily=UI, pad=5,
                 fontweight="600" if strong else "normal")


def build():
    fig = plt.figure(figsize=(12.8, 6.4), dpi=100)
    fig.patch.set_facecolor(PAPER)

    # ---- the words
    fig.text(0.055, 0.845, "गुरुत्व", fontsize=25, color=MUT,
             fontfamily=["Nirmala UI", "Mangal", "DejaVu Sans"], va="top")
    fig.text(0.052, 0.70, "Gurutva", fontsize=76, color=INK,
             fontfamily=SERIF, va="center")
    fig.add_artist(plt.Line2D([0.055, 0.30], [0.545, 0.545], color=INK,
                              lw=2.2, transform=fig.transFigure))
    fig.text(0.055, 0.455, "gravity, with error bars", fontsize=24,
             color=INK, fontfamily=SERIF, style="italic", va="center")
    fig.text(0.055, 0.315,
             "Which parts of your subsurface model the data\n"
             "actually support — and a verdict that can say no.",
             fontsize=15.5, color=MUT, fontfamily=UI, va="center",
             linespacing=1.6)
    fig.text(0.055, 0.115,
             "172 tests   ·   Apache-2.0   ·   runs on your laptop",
             fontsize=12.5, color=MUT, fontfamily=UI, va="center")

    # ---- the argument, in the real data
    L = D["levels"][-1]
    specs = [(D["truth"], "what is really there", True),
             (L["worlds"][0], "possibility 1", False),
             (L["worlds"][1], "possibility 2", False),
             (L["worlds"][2], "possibility 3", False)]
    # Explicit rows: an earlier version computed these and ran the bottom pair
    # off the canvas, under the caption. Laid out by hand and looked at.
    x0, w, gx = 0.485, 0.225, 0.035
    rows = (0.545, 0.215)                      # bottom edge of each row
    h = 0.235
    for k, (vals, label, strong) in enumerate(specs):
        ax = fig.add_axes([x0 + (k % 2) * (w + gx), rows[k // 2], w, h])
        panel(ax, vals, label, strong)

    fig.text(0.485, 0.085,
             "All four produce the same readings, to within twice the\n"
             "wobble of the instrument. Only one of them is the answer.",
             fontsize=12, color=MUT, fontfamily=UI, va="center",
             linespacing=1.6)

    out = ROOT / "docs" / "social-preview.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, facecolor=PAPER)
    plt.close(fig)
    print(f"wrote {out}  ({out.stat().st_size / 1024:.0f} KB, 1280x640)")
    return out


if __name__ == "__main__":
    build()
