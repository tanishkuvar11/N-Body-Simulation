#!/usr/bin/env python3
"""Side-by-side animation: STATIC vs ADAPTIVE scheduling on a collapsing cloud.

Every star is coloured by the OpenMP thread that computed its force.  Under static each thread
owns one contiguous Morton block, i.e. one region of space; when the core forms, the thread that
owns it has far more work (its bar grows, lambda rises).  When the adaptive controller switches to
dynamic, the colours mix because work is handed out in small chunks, and the bars even out.

    python scripts/animate.py                 # run both simulations, write results/animation.mp4
    python scripts/animate.py --live          # show in a window instead of writing a video
    python scripts/animate.py --n 32768 --steps 150 --every 2 --threads 8 --ic collapse
"""
import argparse
import csv
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RESULTS, run_sim  # noqa: E402

# reference categorical palette, dark-mode steps (slots 1-8)
PALETTE = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]
BG, PANEL, INK, INK2, GRID = "#1a1a19", "#222220", "#f0efec", "#c3c2b7", "#3a3a37"
POLICY = {0: "static", 1: "dynamic", 2: "guided"}


def read_snap(path):
    with open(path, "rb") as f:
        data = f.read()
    if data[:8] != b"NBSNAP01":
        raise SystemExit(f"{path}: not a snapshot file")
    n, w, _ = struct.unpack_from("<3i", data, 8)
    off, frames = 20, []
    fsize = 16 + 4 * w + 13 * n
    while off + fsize <= len(data):
        step, t, pol, lam = struct.unpack_from("<ifif", data, off)
        wt = np.frombuffer(data, "<f4", w, off + 16)
        o = off + 16 + 4 * w
        xyz = np.frombuffer(data, "<f4", 3 * n, o).reshape(3, n)
        own = np.frombuffer(data, "u1", n, o + 12 * n)
        frames.append(dict(step=step, t=t, pol=pol, lam=lam, wt=wt.copy(), x=xyz[0].copy(), y=xyz[1].copy(),
                           own=own.copy()))
        off += fsize
    return n, w, frames


def read_trace(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    sw = [float(r["time"]) for r in rows if r["policy"] != "static"]
    return np.array([float(r["time"]) for r in rows]), np.array([float(r["t_force"]) for r in rows]),         (sw[0] if sw else None)


def simulate(a, sched, outdir):
    snap = os.path.join(outdir, f"{a.ic}_{sched}.snap")
    trace = os.path.join(outdir, f"{a.ic}_{sched}_trace.csv")
    args = ["--ic", a.ic, "--n", a.n, "--steps", a.steps, "--dt", a.dt, "--threads", a.threads, "--pin",
            "--sched", sched, "--no-energy", "--quiet", "--snap", snap, "--snap-every", a.every, "--trace", trace]
    print(f"running {sched} ...", flush=True)
    res, _ = run_sim(args)
    print(f"  {sched}: {float(res['t_total']):.2f} s, switches {res['switches']}", flush=True)
    return snap, trace


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ic", default="collapse")
    ap.add_argument("--n", type=int, default=32768)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--hold", type=float, default=3.0, help="seconds to hold the last frame")
    ap.add_argument("--live", action="store_true", help="show a window instead of writing a video")
    ap.add_argument("--reuse", action="store_true", help="reuse existing snapshot files")
    ap.add_argument("--out", default=os.path.join(RESULTS, "animation.mp4"))
    a = ap.parse_args()
    if a.threads > len(PALETTE):
        sys.exit(f"at most {len(PALETTE)} threads can be told apart by colour")

    outdir = os.path.join(RESULTS, "animation")
    os.makedirs(outdir, exist_ok=True)
    runs = {}
    for sched in ("static", "adaptive"):
        snap = os.path.join(outdir, f"{a.ic}_{sched}.snap")
        trace = os.path.join(outdir, f"{a.ic}_{sched}_trace.csv")
        if not (a.reuse and os.path.isfile(snap) and os.path.isfile(trace)):
            snap, trace = simulate(a, sched, outdir)
        n, w, frames = read_snap(snap)
        tt, tf, tsw = read_trace(trace)
        runs[sched] = dict(frames=frames, t=tt, cum=np.cumsum(tf) * 1e3, tf=tf, switch=tsw)

    import matplotlib
    if not a.live:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation
    from matplotlib.colors import to_rgba

    rgba = np.array([to_rgba(c, 0.75) for c in PALETTE[:w]])
    nfr = min(len(runs["static"]["frames"]), len(runs["adaptive"]["frames"]))
    wmax = max(float(f["wt"].max()) for r in runs.values() for f in r["frames"][1:nfr]) * 1e3 * 1.1
    cmax = max(r["cum"][-1] for r in runs.values()) * 1.05

    plt.rcParams.update({"text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                         "axes.edgecolor": GRID, "font.size": 10})
    fig = plt.figure(figsize=(16, 9), facecolor=BG)
    gs = fig.add_gridspec(3, 2, height_ratios=[5.2, 1.6, 1.6], hspace=0.38, wspace=0.12,
                          left=0.05, right=0.97, top=0.9, bottom=0.07)
    fig.suptitle(f"Adaptive OpenMP scheduling in a Barnes-Hut N-body simulation  -  {a.ic}, N = {n:,}, {w} threads"
                 "   (colour = thread that computes the star's force)", color=INK, fontsize=13)
    panels = {}
    for col, sched in enumerate(("static", "adaptive")):
        ax = fig.add_subplot(gs[0, col], facecolor=BG)
        ax.set_xlim(-1.15, 1.15); ax.set_ylim(-1.15, 1.15); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        f0 = runs[sched]["frames"][0]
        sc = ax.scatter(f0["x"], f0["y"], s=1.1, c=rgba[f0["own"]], linewidths=0, rasterized=True)
        title = ax.set_title("", color=INK, fontsize=13, loc="left")
        info = ax.text(0.01, 0.01, "", transform=ax.transAxes, color=INK2, fontsize=10, va="bottom", family="monospace")
        bx = fig.add_subplot(gs[1, col], facecolor=PANEL)
        bars = bx.bar(range(w), np.zeros(w), color=PALETTE[:w], width=0.7)
        mean_line = bx.axhline(0, color=INK, lw=1, ls="--")
        bx.set_ylim(0, wmax); bx.set_xticks(range(w)); bx.set_xticklabels([f"T{i}" for i in range(w)])
        bx.set_ylabel("thread time [ms]")
        bx.grid(axis="y", color=GRID, lw=0.5)
        panels[sched] = dict(sc=sc, title=title, info=info, bars=bars, mean=mean_line)

    cx = fig.add_subplot(gs[2, :], facecolor=PANEL)
    lines = {}
    for sched, c in (("static", "#3987e5"), ("adaptive", "#c98500")):
        (lines[sched],) = cx.plot([], [], color=c, lw=2, label=sched)
    cx.set_xlim(0, runs["static"]["t"][-1]); cx.set_ylim(0, cmax)
    cx.set_xlabel("simulation time"); cx.set_ylabel("cumulative force time [ms]")
    cx.grid(color=GRID, lw=0.5)
    cx.legend(loc="upper left", facecolor=PANEL, edgecolor=GRID, labelcolor=INK)
    gap = cx.text(0.99, 0.08, "", transform=cx.transAxes, ha="right", color=INK, fontsize=11)
    tsw = runs["adaptive"]["switch"]
    sw_line = cx.axvline(tsw if tsw is not None else 0, color="#c98500", lw=1, ls=":", visible=False)
    sw_text = cx.text(tsw if tsw is not None else 0, cmax * 0.92, "  adaptive switches to dynamic", color="#c98500",
                      fontsize=10, va="top", visible=False)

    def update(k):
        artists = []
        for sched, p in panels.items():
            fr = runs[sched]["frames"][k]
            p["sc"].set_offsets(np.column_stack([fr["x"], fr["y"]]))
            p["sc"].set_color(rgba[fr["own"]])
            pol = POLICY.get(fr["pol"], "?")
            head = "STATIC schedule" if sched == "static" else f"ADAPTIVE controller  -  now: {pol}"
            p["title"].set_text(head)
            p["info"].set_text(f"t = {fr['t']:.2f}   step {fr['step']:3d}   imbalance lambda = {fr['lam']:.2f}")
            ms = fr["wt"] * 1e3
            for b, h in zip(p["bars"], ms):
                b.set_height(h)
            p["mean"].set_ydata([ms.mean()] * 2)
            r = runs[sched]
            m = r["t"] <= fr["t"] + 1e-9
            lines[sched].set_data(r["t"][m], r["cum"][m])
            artists += [p["sc"], p["title"], p["info"], p["mean"], lines[sched], *p["bars"]]
        cs = runs["static"]["cum"][runs["static"]["t"] <= runs["static"]["frames"][k]["t"] + 1e-9][-1]
        ca = runs["adaptive"]["cum"][runs["adaptive"]["t"] <= runs["adaptive"]["frames"][k]["t"] + 1e-9][-1]
        gap.set_text(f"adaptive vs static so far: {100 * (cs - ca) / cs:+.1f} % less force time" if cs > 0 else "")
        shown = tsw is not None and runs["adaptive"]["frames"][k]["t"] >= tsw - 1e-9
        sw_line.set_visible(shown)
        sw_text.set_visible(shown)
        return artists + [gap, sw_line, sw_text]

    order = list(range(nfr)) + [nfr - 1] * int(a.hold * a.fps)
    anim = animation.FuncAnimation(fig, update, frames=order, interval=1000 / a.fps, blit=False)
    if a.live:
        plt.show()
    else:
        print(f"rendering {len(order)} frames -> {a.out}", flush=True)
        anim.save(a.out, writer=animation.FFMpegWriter(fps=a.fps, bitrate=6000), dpi=120, savefig_kwargs={"facecolor": BG})
        print("wrote", a.out)


if __name__ == "__main__":
    main()
