"""Column-width IEEE figures for the report, from the project's result CSVs (same data as RESULTS.md)."""
import csv
import json
import os
import statistics as st
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402,F401
from PIL import Image  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results")
OUT = os.path.join(HERE, "fig")
os.makedirs(OUT, exist_ok=True)
D = json.load(open(os.path.join(ROOT, "presentation", "data.json")))

ACC, INK, GREY, LIGHT = "#D9472B", "#111111", "#7d7a73", "#b9b5ac"
COL = {"static": "#2a78d6", "dynamic": "#b9b5ac", "guided": "#1baf7a", "adaptive": ACC}
W = 3.45  # IEEE column width in inches

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"], "font.size": 8,
    "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "lines.linewidth": 1.2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e3e1dc",
    "grid.linewidth": 0.4, "legend.frameon": False, "savefig.dpi": 600, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02, "mathtext.fontset": "stix",
})


def rows(name):
    with open(os.path.join(RES, name), newline="") as f:
        return list(csv.DictReader(f))


def save(fig, name):
    fig.savefig(os.path.join(OUT, name))
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------- Fig. 1: timestep pipeline
fig, ax = plt.subplots(figsize=(W, 2.35))
ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 7.4)
steps = [("1  kick + drift (own block)", "per rank"), ("2  MPI_Allgatherv x, v", "MPI"),
         ("3  Morton sort (all bodies)", "per rank"), ("4  octree build", "per rank"),
         ("5  force loop, schedule(runtime)", "OpenMP + CUDA"), ("6  kick (own block)", "per rank"),
         ("7  controller: omp_set_schedule()", "every k steps")]
for i, (t, lvl) in enumerate(steps):
    y = 6.6 - i * 0.98
    hot = i in (4, 6)
    ax.add_patch(plt.Rectangle((0.2, y - 0.36), 6.3, 0.72, fill=False, lw=0.9 if hot else 0.6,
                               ec=ACC if hot else INK))
    ax.text(0.45, y, t, va="center", ha="left", fontsize=7.5, color=ACC if hot else INK)
    ax.text(6.8, y, lvl, va="center", ha="left", fontsize=7, color=GREY, style="italic")
    if i < len(steps) - 1:
        ax.annotate("", xy=(3.35, y - 0.6), xytext=(3.35, y - 0.38),
                    arrowprops=dict(arrowstyle="-|>", lw=0.6, color=GREY, mutation_scale=6))
save(fig, "fig1_pipeline.png")

# ---------------------------------------------------------------- Fig. 2: star snapshots
a = Image.open(os.path.join(ROOT, "presentation", "img", "stars_static.png")).convert("RGB")
b = Image.open(os.path.join(ROOT, "presentation", "img", "stars_adaptive.png")).convert("RGB")
fig, axs = plt.subplots(1, 2, figsize=(W, 1.85))
for ax, im, t in zip(axs, (a, b), ("(a) static", "(b) adaptive (dynamic)")):
    ax.imshow(im); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(t, fontsize=8, pad=2)
    for s in ax.spines.values():
        s.set_visible(True); s.set_linewidth(0.4); s.set_color(LIGHT)
fig.subplots_adjust(wspace=0.05)
save(fig, "fig2_stars.png")

# ---------------------------------------------------------------- Fig. 3: imbalance over time
tr = D["trace"]
fig, ax = plt.subplots(figsize=(W, 2.0))
ax.plot(tr["time"][1:], tr["static_lam"][1:], color=LIGHT, lw=1.0, label=r"static run, measured $\lambda$")
ax.plot(tr["time"][1:], tr["work_lam"][1:], color=INK, lw=1.0, label="work imbalance of static blocks")
ax.plot(tr["time"][1:], tr["adaptive_lam"][1:], color=ACC, lw=1.1, label=r"adaptive run, measured $\lambda$")
for t, lab, ha in ((tr["first_switch"], " brief switch", "left"), (tr["sustained_switch"], "sustained switch ", "right")):
    ax.axvline(t, color=ACC, lw=0.5, ls=":")
    ax.text(t, 0.69, lab, fontsize=6.5, color=ACC, va="top", ha=ha)
ax.axhline(0.15, color=GREY, lw=0.4, ls="--"); ax.axhline(0.05, color=GREY, lw=0.4, ls="--")
ax.set_xlim(0, 1.5); ax.set_ylim(0, 0.7)
ax.set_xlabel("simulation time"); ax.set_ylabel(r"$\lambda$ (5-step mean)")
ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.9), handlelength=1.4)
save(fig, "fig3_imbalance.png")

# ---------------------------------------------------------------- Fig. 4: time saved vs static
cases = [(ic, n) for n in (4096, 16384, 65536) for ic in ("plummer", "collide", "collapse")]
AB = {"plummer": "P", "collide": "CC", "collapse": "CL"}
labels = [f"{AB[ic]}\n{n // 1024}k" for ic, n in cases]
fig, ax = plt.subplots(figsize=(W, 1.95))
w = 0.27
for k, (sch, c) in enumerate((("dynamic", COL["dynamic"]), ("guided", COL["guided"]), ("adaptive", ACC))):
    vals = [100 * (1 - D["rel8"][f"{ic}_{n}"][sch]) for ic, n in cases]
    ax.bar([i + (k - 1) * w for i in range(len(cases))], vals, w * 0.92, color=c, label=sch, lw=0)
ax.set_xticks(range(len(cases))); ax.set_xticklabels(labels, fontsize=6.5)
ax.set_ylabel("time saved vs static [%]"); ax.set_ylim(0, 20)
ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0), handlelength=1.0, columnspacing=1.2)
save(fig, "fig4_saved.png")

# ---------------------------------------------------------------- Fig. 5: speedup vs threads (collide)
sp = D["speedup65536"]
fig, axs = plt.subplots(1, 2, figsize=(W, 1.75), sharey=True)
for ax, ic in zip(axs, ("uniform", "collide")):
    for sch in ("static", "dynamic", "guided", "adaptive"):
        ax.plot([1, 2, 4, 8], sp[ic][sch], marker="o", ms=2.5, color=COL[sch], label=sch, lw=1.0)
    ax.plot([1, 8], [1, 8], color=GREY, lw=0.5, ls="--")
    ax.set_xticks([1, 2, 4, 8]); ax.set_title(ic, fontsize=8, pad=2); ax.set_xlabel("threads")
axs[0].set_ylabel("speed-up S(p)")
axs[0].legend(loc="upper left", handlelength=1.2)
save(fig, "fig5_speedup.png")

# ---------------------------------------------------------------- Fig. 6: GPU direct sum
g = D["gpu_direct"]
fig, ax = plt.subplots(figsize=(W, 1.85))
ns = [x["n"] for x in g]
ax.plot(ns, [x["cpu_ms"] for x in g], marker="o", ms=3, color=GREY, label="CPU direct, 8 threads (fp64)")
ax.plot(ns, [x["gpu64_ms"] for x in g], marker="s", ms=3, color=LIGHT, label="GPU tiled kernel, fp64")
ax.plot(ns, [x["gpu32_ms"] for x in g], marker="^", ms=3, color=ACC, label="GPU tiled kernel, fp32")
ax.set_xscale("log", base=2); ax.set_yscale("log")
ax.set_xticks(ns); ax.set_xticklabels([f"{n:,}" for n in ns])
ax.set_xlabel("N"); ax.set_ylabel("time per step [ms]")
ax.legend(loc="upper left", handlelength=1.4)
save(fig, "fig6_gpu.png")
