"""Extract every number the deck shows from the project's result files -> data.json,
and render the star visualisations in the deck palette -> img/*.png.

All values come from the same CSVs / snapshots that results/presentation.html is built from."""
import csv
import json
import os
import statistics as st
import struct
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results")
OUT = {}


def rows(name):
    with open(os.path.join(RES, name), newline="") as f:
        return list(csv.DictReader(f))


# ------------------------------------------------------------------ schedule experiment
g = defaultdict(list)
for r in rows("sched.csv"):
    g[(r["ic"], int(r["n"]), int(r["threads"]), r["sched"])].append(float(r["t_total"]))
M = lambda *k: st.median(g[k])  # noqa: E731
ICS = ["uniform", "plummer", "collide", "collapse"]
SCH = ["static", "dynamic", "guided", "adaptive"]

rel = {}            # time relative to static, 8 threads
for n in (4096, 16384, 65536):
    for ic in ICS:
        s = M(ic, n, 8, "static")
        rel[f"{ic}_{n}"] = {x: M(ic, n, 8, x) / s for x in SCH}
OUT["rel8"] = rel

spd = {}            # speedup vs 1-thread static, N = 65536
for ic in ICS:
    t1 = M(ic, 65536, 1, "static")
    spd[ic] = {x: [t1 / (M(ic, 65536, t, "static") if t == 1 else M(ic, 65536, t, x)) for t in (1, 2, 4, 8)] for x in SCH}
OUT["speedup65536"] = spd

clustered = [(ic, n) for n in (4096, 16384, 65536) for ic in ICS if ic != "uniform"]
gains = [1 - rel[f"{ic}_{n}"]["adaptive"] for ic, n in clustered]
OUT["adaptive_gain_clustered"] = {"min": min(gains), "max": max(gains), "count_faster": sum(x > 0 for x in gains),
                                  "count": len(gains)}
OUT["adaptive_uniform"] = {str(n): rel[f"uniform_{n}"]["adaptive"] for n in (4096, 16384, 65536)}
best = defaultdict(int)
for n in (4096, 16384, 65536):
    for ic in ICS:
        best[min(SCH, key=lambda x: rel[f"{ic}_{n}"][x])] += 1
OUT["best_counts"] = dict(best)
OUT["dynamic_small_uniform"] = rel["uniform_4096"]["dynamic"]

# ------------------------------------------------------------------ imbalance traces (benchmark, N=65536, 8 threads)
def trace(s):
    return rows(os.path.join("traces", f"sched_trace_collapse_n65536_T8_{s}.csv"))


ts, ta = trace("static"), trace("adaptive")
def roll(v, w=5):
    out = []
    for i in range(len(v)):
        a = v[max(1, i - w + 1): i + 1] if i >= 1 else v[:1]
        out.append(sum(a) / len(a))
    return out


OUT["trace"] = {
    "time": [float(r["time"]) for r in ts],
    "static_lam": roll([float(r["lam_time"]) for r in ts]),
    "adaptive_lam": roll([float(r["lam_time"]) for r in ta]),
    "work_lam": [float(r["lam_static_work"]) for r in ts],
    "first_switch": next(float(ta[i]["time"]) for i in range(1, len(ta))
                         if ta[i]["policy"] != "static" and ta[i - 1]["policy"] == "static"),
    "sustained_switch": [float(ta[i]["time"]) for i in range(1, len(ta))
                         if ta[i]["policy"] != "static" and ta[i - 1]["policy"] == "static"][-1],
    "work_lam_start": float(ts[1]["lam_static_work"]),
    "work_lam_max": max(float(r["lam_static_work"]) for r in ts),
}

# ------------------------------------------------------------------ GPU
gp = defaultdict(list)
for r in rows("gpu.csv"):
    gp[r["label"].split(":")[1]].append(r)
gm = lambda k, f="t_step": st.median(float(r[f]) for r in gp[k])  # noqa: E731
OUT["gpu_direct"] = [{"n": n, "cpu_ms": gm(f"direct_cpu_n{n}") * 1e3, "gpu64_ms": gm(f"direct_gpu_double_n{n}") * 1e3,
                      "gpu32_ms": gm(f"direct_gpu_float_n{n}") * 1e3, "err32": gm(f"direct_gpu_float_n{n}", "verify_rms")}
                     for n in (1024, 4096, 16384, 65536)]
OUT["gpu_bh"] = [{"n": n, "cpu_ms": min(gm(f"bh_cpu_leaf8_n{n}"), gm(f"bh_cpu_leaf16_n{n}")) * 1e3,
                  "hyb_ms": gm(f"bh_gpu_float_leaf256_n{n}") * 1e3} for n in (4096, 16384, 65536)]

# ------------------------------------------------------------------ MPI (conservative baseline: schedule-experiment T1)
mp = defaultdict(list)
for r in rows("mpi.csv"):
    mp[(r["ic"], r["ranks"], r["threads"], r["sched"])].append(float(r["t_total"]))
mpi = {}
for ic in ("uniform", "collide"):
    t1 = M(ic, 65536, 1, "static") * 100 / 150
    for p, t in [("4", "1"), ("1", "8"), ("2", "4"), ("4", "2"), ("4", "4")]:
        vals = {s: st.median(mp[(ic, p, t, s)]) for s in ("static", "dynamic", "adaptive") if mp.get((ic, p, t, s))}
        mpi[f"{ic}_{p}x{t}"] = {s: t1 / v for s, v in vals.items()}
OUT["mpi"] = mpi
rl = [float(r["lam_rank_mean"]) for r in rows("mpi.csv") if r["ic"] == "collide" and r["sched"] == "static"]
OUT["mpi_rank_lam_max"] = max(st.median([float(r["lam_rank_mean"]) for r in rows("mpi.csv")
                                         if r["ic"] == "collide" and r["sched"] == "static"
                                         and r["ranks"] == p and r["threads"] == t])
                              for p, t in [("2", "1"), ("4", "1"), ("2", "4"), ("4", "2"), ("2", "8"), ("4", "4")])

# ------------------------------------------------------------------ future work (Vishal)
fut = rows(os.path.join("future", "future.csv"))
OUT["future_rows"] = len(fut)

# ------------------------------------------------------------------ snapshots -> per-thread times + images
def read_snap(path):
    data = open(path, "rb").read()
    n, w, _ = struct.unpack_from("<3i", data, 8)
    off, fr = 20, []
    fs = 16 + 4 * w + 13 * n
    while off + fs <= len(data):
        step, t, pol, lam = struct.unpack_from("<ifif", data, off)
        wt = np.frombuffer(data, "<f4", w, off + 16)
        o = off + 16 + 4 * w
        xyz = np.frombuffer(data, "<f4", 3 * n, o).reshape(3, n)
        own = np.frombuffer(data, "u1", n, o + 12 * n)
        fr.append(dict(step=step, t=t, pol=pol, lam=lam, wt=wt.copy(), x=xyz[0].copy(), y=xyz[1].copy(), own=own.copy()))
        off += fs
    return n, w, fr


n_s, w_s, fs_ = read_snap(os.path.join(RES, "animation", "collapse_static.snap"))
_, _, fa_ = read_snap(os.path.join(RES, "animation", "collapse_adaptive.snap"))
ls_, la_ = fs_[-1], fa_[-1]
busiest = int(np.argmax(ls_["wt"]))
OUT["snap"] = {"n": n_s, "t": float(ls_["t"]), "static_wt_ms": [float(v) * 1e3 for v in ls_["wt"]],
               "adaptive_wt_ms": [float(v) * 1e3 for v in la_["wt"]], "static_lam": float(ls_["lam"]),
               "adaptive_lam": float(la_["lam"]), "busiest": busiest,
               "ratio_max_min": float(ls_["wt"].max() / ls_["wt"].min())}

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

os.makedirs(os.path.join(HERE, "img"), exist_ok=True)


HALF = 0.42


def stars(frame, path, bg, other, accent, size_in=5.0):
    fig = plt.figure(figsize=(size_in, size_in), dpi=400, facecolor=bg)
    ax = fig.add_axes([0, 0, 1, 1], facecolor=bg)
    hot = frame["own"] == busiest
    H, xe, ye = np.histogram2d(frame["x"], frame["y"], bins=120, range=[[-1.2, 1.2], [-1.2, 1.2]])
    i, j = np.unravel_index(np.argmax(H), H.shape)          # densest cell = centre of the core
    cx, cy = (xe[i] + xe[i + 1]) / 2, (ye[j] + ye[j + 1]) / 2
    ax.scatter(frame["x"][~hot], frame["y"][~hot], s=0.9, c=other, linewidths=0, alpha=0.5)
    ax.scatter(frame["x"][hot], frame["y"][hot], s=1.3, c=accent, linewidths=0, alpha=0.95)
    ax.set_xlim(cx - HALF, cx + HALF); ax.set_ylim(cy - HALF, cy + HALF); ax.set_aspect("equal"); ax.axis("off")
    fig.savefig(path, facecolor=bg)
    plt.close(fig)


stars(ls_, os.path.join(HERE, "img", "stars_static.png"), "#F5F3EE", "#77736B", "#D9472B")
stars(la_, os.path.join(HERE, "img", "stars_adaptive.png"), "#F5F3EE", "#77736B", "#D9472B")
stars(ls_, os.path.join(HERE, "img", "stars_static_dark.png"), "#111111", "#6E6B66", "#E5563A")

# ------------------------------------------------------------------ validation
OUT["validation_md"] = open(os.path.join(RES, "validation.md"), encoding="utf-8").read()

json.dump(OUT, open(os.path.join(HERE, "data.json"), "w"), indent=1)
print(json.dumps({k: v for k, v in OUT.items() if k not in ("trace", "validation_md", "speedup65536", "rel8")}, indent=1))
print("rel8", json.dumps({k: {s: round(x, 3) for s, x in v.items()} for k, v in rel.items()}))
print("speedup8", {ic: {s: round(v[s][3], 2) for s in SCH} for ic, v in spd.items()})
print("switch", OUT["trace"]["first_switch"], OUT["trace"]["sustained_switch"], "work lam start/max", OUT["trace"]["work_lam_start"], OUT["trace"]["work_lam_max"])
