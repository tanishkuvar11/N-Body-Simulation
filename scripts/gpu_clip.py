#!/usr/bin/env python3
"""Closing segment for results/animation.mp4: the four force methods at N = 262,144.

Times (ms per step, 8 CPU threads, RTX 4060) were measured on 27 Oct 2026 with
  bin/nbody.exe --ic plummer --n 262144 --steps 5 --threads 8 --pin --no-energy --force ...
(medians of 4 runs; the exact CPU method was run once with --steps 1 because it takes ~32 s per step).

    python scripts/gpu_clip.py            # writes results/animation_gpu.mp4 and appends it to animation.mp4
"""
import os
import shutil
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import animation  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RESULTS  # noqa: E402

BG, PANEL, INK, INK2, GRID = "#1a1a19", "#222220", "#f0efec", "#c3c2b7", "#3a3a37"
ACC, GREY = "#e5563a", "#8e8a82"
N = 262144
METHODS = [  # name, ms per step, live colour
    ("Exact, CPU only", 32147.0, GREY),
    ("Exact, graphics card", 248.5, GREY),
    ("Shortcut, CPU only", 274.5, GREY),
    ("Shortcut, CPU + graphics card (ours)", 215.5, ACC),
]
XMAX = 350.0                     # ms shown on the axis; the CPU exact bar runs off the chart
FPS, GROW, HOLD = 8, 2.5, 5.0    # seconds


def main():
    plt.rcParams.update({"text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                         "axes.edgecolor": GRID, "font.size": 13})
    fig = plt.figure(figsize=(16, 9), facecolor=BG)
    fig.suptitle(f"Computing the forces for {N:,} stars: four methods, 8 CPU threads + RTX 4060",
                 color=INK, fontsize=20, y=0.92)
    ax = fig.add_axes([0.31, 0.22, 0.61, 0.55], facecolor=BG)
    ax.set_axisbelow(True)
    names = [m[0] for m in METHODS]
    ax.set_yticks(range(len(METHODS)))
    ax.set_yticklabels(names, fontsize=14, color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0, XMAX)
    ax.set_xlabel("time per step [ms]  (shorter is faster)", fontsize=13)
    ax.grid(axis="x", color=GRID, lw=0.6)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    bars = ax.barh(range(len(METHODS)), [0] * len(METHODS), color=[m[2] for m in METHODS], height=0.6)
    labels = [ax.text(0, i, "", va="center", ha="left", fontsize=15, color=INK, fontweight="bold")
              for i in range(len(METHODS))]
    note = fig.text(0.5, 0.08, "", ha="center", fontsize=17, color=INK)
    foot = fig.text(0.5, 0.03, "Measured on our laptop. Shortcut = Barnes-Hut. Exact CPU method: 1 run (it takes 32 s per step); others: median of 4 runs.",
                    ha="center", fontsize=11, color=INK2)
    del foot

    n_grow, n_hold = int(GROW * FPS), int(HOLD * FPS)

    def update(k):
        f = min(1.0, (k + 1) / n_grow)
        for i, (bar, (name, ms, _)) in enumerate(zip(bars, METHODS)):
            shown = min(ms, XMAX * 1.02) * f
            bar.set_width(shown)
            if ms > XMAX:
                labels[i].set_text(f"  {ms * f:,.0f} ms  (off the chart)")
                labels[i].set_x(min(shown, XMAX * 0.55))
                labels[i].set_ha("left")
                labels[i].set_color(BG if shown > XMAX * 0.6 else INK)
                if shown >= XMAX * 0.6:
                    labels[i].set_x(XMAX * 0.02)
            else:
                labels[i].set_text(f"  {ms * f:,.0f} ms")
                labels[i].set_x(shown)
        if f >= 1.0:
            best = METHODS[3][1]
            note.set_text(f"Our CPU + graphics card method is the fastest: about {round(METHODS[0][1] / best, -1):,.0f}x faster than "
                          f"the exact CPU method, {METHODS[2][1] / best:.1f}x faster than the CPU shortcut")
        return list(bars) + labels + [note]

    anim = animation.FuncAnimation(fig, update, frames=n_grow + n_hold, interval=1000 / FPS, blit=False)
    out = os.path.join(RESULTS, "animation_gpu.mp4")
    anim.save(out, writer=animation.FFMpegWriter(fps=FPS, bitrate=4000), dpi=120,
              savefig_kwargs={"facecolor": BG})
    plt.close(fig)
    print("wrote", out)

    # append to the scheduling animation (keep a copy of the original once)
    main_mp4 = os.path.join(RESULTS, "animation.mp4")
    orig = os.path.join(RESULTS, "animation_scheduling.mp4")
    if not os.path.isfile(orig):
        shutil.copy2(main_mp4, orig)
    tmp = os.path.join(RESULTS, "animation_tmp.mp4")
    subprocess.check_call(["ffmpeg", "-loglevel", "error", "-y", "-i", orig, "-i", out, "-filter_complex",
                           "[0:v][1:v]concat=n=2:v=1:a=0[v]", "-map", "[v]", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                           "-r", str(FPS), "-b:v", "6M", tmp])
    os.replace(tmp, main_mp4)
    print("appended ->", main_mp4)


if __name__ == "__main__":
    main()
