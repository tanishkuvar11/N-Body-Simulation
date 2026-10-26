#!/usr/bin/env python3
"""Extra figures: what the inputs look like, and where the time goes.

  ic_gallery.png        every initial condition (incl. the rotating disk and a spinning collapse)
                        at t = 0, 0.75, 1.5: projected density, x-y plane
  imbalance_by_ic.png   static-schedule work imbalance over time, one line per input
  energy_drift.png      relative energy error over time, one line per input
  phase_breakdown.png   where a step's time goes for every ranks x threads split (results/mpi.csv)
  mpi_heatmap.png       speedup over the ranks x threads grid (results/mpi.csv)
  efficiency_heatmap.png  parallel efficiency, input x schedule x threads (results/sched.csv)

The first three run the simulator (7 runs, N = 16384, about a minute); the others only read the
committed CSVs.  Output: results/figures/extra/ (+ the traces used, in results/figures/extra/data/).

    python scripts/visualize.py [--n 16384] [--threads 8] [--skip-runs]
"""
import argparse
import csv
import os
import statistics
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from animate import read_snap  # noqa: E402
from common import ROOT, RESULTS, run_sim  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, LogNorm  # noqa: E402

# reference palette, light mode: categorical slots in fixed order, sequential blue ramp
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQ = LinearSegmentedColormap.from_list("seq_blue", BLUE_RAMP)
DENS = LinearSegmentedColormap.from_list("dens_blue", [SURFACE] + BLUE_RAMP)

# inputs shown in the gallery / line charts: (label, simulator args)
INPUTS = [
    ("uniform", ["--ic", "uniform"]),
    ("plummer", ["--ic", "plummer"]),
    ("collide", ["--ic", "collide"]),
    ("collapse", ["--ic", "collapse"]),
    ("collapse + spin 0.5", ["--ic", "collapse", "--spin", 0.5]),
    ("virial", ["--ic", "virial"]),
    ("disk", ["--ic", "disk"]),
]

OUT = os.path.join(RESULTS, "figures", "extra")
DATA = os.path.join(OUT, "data")
SNAPS = os.path.join(ROOT, "build", "visualize")       # large binaries stay out of git


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK, "text.color": INK,
        "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11,
        "axes.titleweight": "bold", "legend.frameon": False, "lines.linewidth": 2,
    })


def slug(label):
    return label.replace(" + ", "_").replace(" ", "_").replace(".", "")


def simulate(a):
    os.makedirs(SNAPS, exist_ok=True)
    os.makedirs(DATA, exist_ok=True)
    for label, args in INPUTS:
        s = slug(label)
        print(f"  running {label} ...", flush=True)
        run_sim(args + ["--n", a.n, "--steps", 150, "--dt", 0.01, "--threads", a.threads, "--sched", "static",
                        "--diag", 10, "--print", 0, "--quiet",
                        "--trace", os.path.join(DATA, f"trace_{s}.csv"),
                        "--snap", os.path.join(SNAPS, f"{s}.snap"), "--snap-every", 75])


def read_trace(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def fig_gallery():
    frames = {}
    for label, _ in INPUTS:
        _, _, fr = read_snap(os.path.join(SNAPS, f"{slug(label)}.snap"))
        frames[label] = fr
    nt = len(next(iter(frames.values())))
    fig, axes = plt.subplots(nt, len(INPUTS), figsize=(2.3 * len(INPUTS), 2.4 * nt + 0.4))
    lim = 2.2
    bins = np.linspace(-lim, lim, 161)
    for c, (label, _) in enumerate(INPUTS):
        for r, f in enumerate(frames[label]):
            ax = axes[r, c]
            h, _, _ = np.histogram2d(f["x"], f["y"], bins=[bins, bins])
            ax.imshow(h.T + 0.5, origin="lower", extent=[-lim, lim, -lim, lim], cmap=DENS,
                      norm=LogNorm(vmin=0.5, vmax=max(2.0, h.max())), interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            for sp in ax.spines.values():
                sp.set_visible(True); sp.set_color(GRID)
            if r == 0:
                ax.set_title(label, fontsize=10)
            if c == 0:
                ax.set_ylabel(f"t = {f['t']:.2f}", color=INK2)
    fig.suptitle("Initial conditions and their evolution (x-y projection, log density, N = "
                 f"{len(frames[INPUTS[0][0]][0]['x'])})", color=INK, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "ic_gallery.png"), dpi=130)
    plt.close(fig)


def fig_lines(col, fname, title, ylabel, logy=False, only_sampled=False):
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    for i, (label, _) in enumerate(INPUTS):
        rows = read_trace(os.path.join(DATA, f"trace_{slug(label)}.csv"))
        pts = [(float(r["time"]), float(r[col])) for r in rows if r[col] != ""]
        if only_sampled:
            pts = [p for p in pts if p[0] > 0]
        t, v = zip(*pts)
        ax.plot(t, v, color=SERIES[i], label=label, marker="o" if only_sampled else None, markersize=4)
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("simulation time")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left")
    ax.legend(ncol=2, fontsize=9, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, fname), dpi=130)
    plt.close(fig)


def load_csv(name):
    p = os.path.join(RESULTS, name)
    runs = defaultdict(list)
    with open(p, newline="") as f:
        for r in csv.DictReader(f):
            runs[r["label"].split(":")[1]].append(r)
    return runs


def med(rows, k):
    return statistics.median(float(r[k]) for r in rows)


def fig_phases():
    runs = load_csv("mpi.csv")
    combos = [(1, 1), (2, 1), (4, 1), (1, 8), (2, 4), (4, 2), (2, 8), (4, 4)]
    phases = [("t_force", "force loop"), ("t_comm", "communication"), ("t_sort", "Morton sort"),
              ("t_build", "tree build"), ("t_integ", "integration")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for ax, ic in zip(axes, ["uniform", "collide"]):
        labels, data = [], []
        for p, t in combos:
            rows = runs.get(f"{ic}_n65536_P{p}_T{t}_static")
            if not rows:
                continue
            labels.append(f"{p} x {t}")
            vals = [med(rows, k) for k, _ in phases]
            vals.append(max(0.0, med(rows, "t_total") - sum(vals)))
            data.append(vals)
        data = np.array(data)
        y = np.arange(len(labels))
        left = np.zeros(len(labels))
        names = [n for _, n in phases] + ["other"]
        colors = SERIES[:len(phases)] + [MUTED]
        for j in range(data.shape[1]):
            ax.barh(y, data[:, j], left=left, color=colors[j], label=names[j], height=0.62,
                    edgecolor=SURFACE, linewidth=2)
            left += data[:, j]
        for yy, tot in zip(y, left):
            ax.text(tot + 0.4, yy, f"{tot:.1f} s", va="center", fontsize=8, color=INK2)
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_xlabel("time for 100 steps [s] (static schedule, N = 65,536)")
        ax.set_title(f"{ic}: where the time goes", loc="left")
        ax.grid(axis="y", visible=False)
    axes[0].set_ylabel("MPI ranks x OpenMP threads")
    h, l = axes[1].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(l), fontsize=9, labelcolor=INK2, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(os.path.join(OUT, "phase_breakdown.png"), dpi=130)
    plt.close(fig)


def heat(ax, M, rows, cols, fmt, title, vmin=None, vmax=None):
    im = ax.imshow(np.ma.masked_invalid(M), cmap=SEQ, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows)
    ax.grid(False)
    lo, hi = np.nanmin(M), np.nanmax(M)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isnan(M[i, j]):
                ax.text(j, i, "-", ha="center", va="center", color=MUTED, fontsize=9)
            else:
                dark = (M[i, j] - (vmin if vmin is not None else lo)) / max(1e-9, (vmax or hi) - (vmin or lo)) > 0.55
                ax.text(j, i, fmt.format(M[i, j]), ha="center", va="center", fontsize=9,
                        color="#ffffff" if dark else INK)
    ax.set_title(title, loc="left")
    return im


def fig_mpi_heatmap():
    runs = load_csv("mpi.csv")
    ranks, threads = [1, 2, 4], [1, 2, 4, 8]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9), sharey=True, gridspec_kw={"wspace": 0.08})
    for ax, (ic, s) in zip(axes, [("uniform", "static"), ("collide", "static"), ("collide", "adaptive")]):
        base = med(runs[f"{ic}_n65536_P1_T1_static"], "t_total")
        M = np.full((len(ranks), len(threads)), np.nan)
        for i, p in enumerate(ranks):
            for j, t in enumerate(threads):
                key = f"{ic}_n65536_P{p}_T{t}_{s if t > 1 else 'static'}"
                if key in runs:
                    M[i, j] = base / med(runs[key], "t_total")
        im = heat(ax, M, [f"{p} rank{'s' if p > 1 else ''}" for p in ranks], [f"{t} thr" for t in threads],
                  "{:.1f}x", f"{ic}, {s}", vmin=1, vmax=11)
        ax.set_xlabel("OpenMP threads per rank")
    fig.colorbar(im, ax=axes, shrink=0.85, label="speedup vs 1 x 1")
    fig.suptitle("Hybrid MPI x OpenMP speedup (N = 65,536; '-' = not measured)", x=0.02, y=1.06, ha="left",
                 fontweight="bold")
    fig.savefig(os.path.join(OUT, "mpi_heatmap.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_efficiency():
    runs = load_csv("sched.csv")
    ics = ["uniform", "plummer", "collide", "collapse"]
    scheds = ["static", "dynamic", "guided", "adaptive"]
    threads = [2, 4, 8]
    fig, axes = plt.subplots(1, 3, figsize=(13, 5), sharey=True)
    for ax, n in zip(axes, [4096, 16384, 65536]):
        rows, M = [], []
        for ic in ics:
            t1 = med(runs[f"{ic}_n{n}_T1_static"], "t_total")
            for s in scheds:
                rows.append(f"{ic} / {s}")
                M.append([t1 / med(runs[f"{ic}_n{n}_T{t}_{s}"], "t_total") / t for t in threads])
        im = heat(ax, np.array(M), rows, [f"{t} thr" for t in threads], "{:.0%}", f"N = {n:,}", vmin=0.4, vmax=1.0)
        for k in range(1, len(ics)):
            ax.axhline(k * len(scheds) - 0.5, color=SURFACE, linewidth=3)
    fig.colorbar(im, ax=axes, shrink=0.8, label="parallel efficiency (speedup / threads)")
    fig.suptitle("OpenMP parallel efficiency by input, schedule and thread count", x=0.02, y=1.0, ha="left",
                 fontweight="bold")
    fig.savefig(os.path.join(OUT, "efficiency_heatmap.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=16384)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--skip-runs", action="store_true", help="reuse traces/snapshots from a previous run")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    style()
    if not a.skip_runs:
        simulate(a)
    fig_gallery()
    fig_lines("lam_static_work", "imbalance_by_ic.png",
              "Work imbalance a static schedule would have (8 threads)", "imbalance  max/mean - 1")
    fig_lines("de_rel", "energy_drift.png", "Energy conservation: |E(t) - E0| / |E0|  (|W0| when E0 ~ 0; Barnes-Hut, dt = 0.01)",
              "relative energy error", logy=True, only_sampled=True)
    fig_phases()
    fig_mpi_heatmap()
    fig_efficiency()
    print("wrote", ", ".join(sorted(f for f in os.listdir(OUT) if f.endswith(".png"))), "->", OUT)


if __name__ == "__main__":
    main()
