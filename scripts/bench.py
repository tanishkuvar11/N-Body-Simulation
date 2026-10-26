#!/usr/bin/env python3
"""Benchmark driver for the evaluation plan.

Experiments (results/<exp>.csv, one row per run, appended; reruns resume where they stopped):
  sched   schedule comparison: static/dynamic/guided/adaptive x threads x N x input (OpenMP, 1 rank)
  mpi     hybrid MPI x OpenMP configurations (ranks x threads)
  gpu     CUDA: tiled direct kernel vs CPU direct sum; Barnes-Hut CPU vs BH+GPU hybrid
  ctl     controller sensitivity (check interval k, thresholds) and dynamic chunk size
  small   overhead of dynamic scheduling at small N on the balanced (uniform) input
  ext     controller switching to guided instead of dynamic
  future  extensions: --partition cost, --balanced auto, parallel sort + tree build
          (run on the second machine: python scripts/bench.py --exp future --gate-every 0
           --out results/future)

Laptop CPUs drift with temperature, so within each repetition the runs of an experiment
are executed in a shuffled order, with a short pause between runs; analysis uses medians.

Usage:  python scripts/bench.py [--exp sched,mpi,gpu,ctl,small] [--reps 3] [--quick] [--pause 1.0]
"""
import argparse
import csv
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RESULTS, run_sim  # noqa: E402

ICS = ["uniform", "plummer", "collide", "collapse"]
SCHEDS = ["static", "dynamic", "guided", "adaptive"]
BASE = ["--steps", 150, "--dt", 0.01, "--eps", 0.05, "--theta", 0.5, "--no-energy", "--print", 0]


class Run:
    def __init__(self, exp, key, args, ranks=1, trace=None):
        self.exp, self.key, self.args, self.ranks, self.trace = exp, key, list(args), ranks, trace


def exp_sched(quick):
    ns = [4096] if quick else [4096, 16384, 65536]
    threads = [2, 8] if quick else [2, 4, 8]
    runs = []
    for ic in ICS:
        for n in ns:
            base = BASE + ["--ic", ic, "--n", n, "--pin"]
            if quick:
                base[1] = 30
            runs.append(Run("sched", f"{ic}_n{n}_T1_static", base + ["--threads", 1, "--sched", "static"]))
            for t in threads:
                for s in SCHEDS:
                    tr = f"trace_{ic}_n{n}_T{t}_{s}.csv" if (t == max(threads) and n == max(ns)) else None
                    runs.append(Run("sched", f"{ic}_n{n}_T{t}_{s}", base + ["--threads", t, "--sched", s], trace=tr))
    return runs


def exp_mpi(quick):
    n = 16384 if quick else 65536
    combos = [(1, 1), (2, 1), (4, 1), (1, 8), (2, 4), (4, 2), (2, 8), (4, 4)]
    if quick:
        combos = [(1, 1), (2, 1), (2, 4), (4, 2)]
    runs = []
    for ic in ["uniform", "collide"]:
        for p, t in combos:
            base = BASE + ["--ic", ic, "--n", n, "--threads", t]
            base[1] = 30 if quick else 100
            # <= 8 threads in total: one per P-core; 16: both SMT threads of the 8 P-cores
            base += ["--pin"] if p * t <= 8 else ["--cpus", "0-15"]
            for s in (["static"] if t == 1 else ["static", "dynamic", "adaptive"]):
                runs.append(Run("mpi", f"{ic}_n{n}_P{p}_T{t}_{s}", base + ["--sched", s], ranks=p))
    return runs


def exp_gpu(quick):
    runs = []
    ns = [1024, 4096] if quick else [1024, 4096, 16384, 65536]
    for n in ns:
        base = ["--ic", "plummer", "--n", n, "--steps", 5 if quick else 10, "--no-energy", "--print", 0,
                "--pin", "--threads", 8, "--verify"]
        runs.append(Run("gpu", f"direct_cpu_n{n}", base + ["--force", "direct"]))
        runs.append(Run("gpu", f"direct_gpu_double_n{n}", base + ["--force", "gpu-direct"]))
        runs.append(Run("gpu", f"direct_gpu_float_n{n}", base + ["--force", "gpu-direct", "--gpu-float"]))
    for n in ([4096] if quick else [4096, 16384, 65536]):
        base = ["--ic", "plummer", "--n", n, "--steps", 5 if quick else 20, "--no-energy", "--print", 0,
                "--pin", "--threads", 8, "--verify", "--near-cap", 256]
        runs.append(Run("gpu", f"bh_cpu_leaf8_n{n}", base + ["--force", "bh", "--leaf", 8]))
        runs.append(Run("gpu", f"bh_cpu_leaf16_n{n}", base + ["--force", "bh", "--leaf", 16]))
        runs.append(Run("gpu", f"bh_gpu_double_leaf32_n{n}", base + ["--force", "bh-gpu", "--leaf", 32]))
        runs.append(Run("gpu", f"bh_gpu_float_leaf256_n{n}", base + ["--force", "bh-gpu", "--gpu-float", "--leaf", 256]))
    return runs


def exp_ctl(quick):
    n = 16384 if quick else 65536
    base = BASE + ["--ic", "collapse", "--n", n, "--threads", 8, "--pin"]
    runs = []
    for k in [1, 5, 10, 20]:
        runs.append(Run("ctl", f"k{k}", base + ["--sched", "adaptive", "--k", k]))
    for hi in [0.10, 0.25, 0.40]:          # 0.15 is the k=5 run above
        runs.append(Run("ctl", f"high{hi}", base + ["--sched", "adaptive", "--high", hi, "--low", round(hi / 3, 4)]))
    cbase = BASE + ["--ic", "collide", "--n", n, "--threads", 8, "--pin"]
    for c in [1, 8, 64, 512]:
        runs.append(Run("ctl", f"collide_dynamic_c{c}", cbase + ["--sched", "dynamic", "--chunk", c]))
    runs.append(Run("ctl", "collide_static", cbase + ["--sched", "static"]))
    return runs


def exp_small(quick):
    runs = []
    for n in [1024, 2048]:
        base = BASE + ["--ic", "uniform", "--n", n, "--threads", 8, "--pin"]
        base[1] = 100 if quick else 400
        runs.append(Run("small", f"n{n}_static", base + ["--sched", "static"]))
        for c in [1, 8]:
            runs.append(Run("small", f"n{n}_dynamic_c{c}", base + ["--sched", "dynamic", "--chunk", c]))
        runs.append(Run("small", f"n{n}_adaptive", base + ["--sched", "adaptive"]))
    return runs


def exp_ext(quick):
    """Extension: controller switching to guided instead of dynamic (run after the main suite)."""
    runs = []
    for ic in ICS:
        for n in ([4096] if quick else [16384, 65536]):
            base = BASE + ["--ic", ic, "--n", n, "--threads", 8, "--pin"]
            if quick:
                base[1] = 30
            runs.append(Run("ext", f"{ic}_n{n}_static", base + ["--sched", "static"]))
            runs.append(Run("ext", f"{ic}_n{n}_guided", base + ["--sched", "guided"]))
            runs.append(Run("ext", f"{ic}_n{n}_adaptive-dyn-k5", base + ["--sched", "adaptive"]))
            runs.append(Run("ext", f"{ic}_n{n}_adaptive-gui-k5", base + ["--sched", "adaptive", "--balanced", "guided"]))
            runs.append(Run("ext", f"{ic}_n{n}_adaptive-gui-k1", base + ["--sched", "adaptive", "--balanced", "guided",
                                                                         "--k", 1]))
    return runs


def exp_future(quick):
    """Future-work extensions: cost-weighted MPI partitioning, controller auto mode,
    parallel Morton sort + tree build."""
    runs = []
    n = 16384 if quick else 65536
    for ic in ["collide", "collapse"]:
        for p, t in [(4, 1), (4, 2), (2, 4)]:
            base = BASE + ["--ic", ic, "--n", n, "--threads", t]
            base[1] = 30 if quick else 100
            for s in (["static"] if t == 1 else ["static", "adaptive"]):
                for part in ["equal", "cost"]:
                    runs.append(Run("future", f"part_{ic}_P{p}_T{t}_{s}_{part}", base + ["--sched", s, "--partition", part],
                                    ranks=p))
    for ic in ICS:
        base = BASE + ["--ic", ic, "--n", n, "--threads", 8]
        if quick:
            base[1] = 30
        runs.append(Run("future", f"auto_{ic}_static", base + ["--sched", "static"]))
        runs.append(Run("future", f"auto_{ic}_guided", base + ["--sched", "guided"]))
        runs.append(Run("future", f"auto_{ic}_adaptive-dyn", base + ["--sched", "adaptive"]))
        runs.append(Run("future", f"auto_{ic}_adaptive-auto", base + ["--sched", "adaptive", "--balanced", "auto"]))
    for nn in ([65536] if quick else [65536, 262144]):
        for t in [1, 8]:
            runs.append(Run("future", f"build_n{nn}_T{t}",
                            ["--ic", "plummer", "--n", nn, "--steps", 20, "--theta", 2.0, "--no-energy", "--print", 0,
                             "--threads", t]))
    return runs


EXPERIMENTS = {"sched": exp_sched, "mpi": exp_mpi, "gpu": exp_gpu, "ctl": exp_ctl, "small": exp_small,
               "ext": exp_ext, "future": exp_future}


CALIB_ARGS = BASE + ["--ic", "uniform", "--n", 16384, "--threads", 8, "--pin", "--sched", "static"]
CALIB_KEY = "uniform_n16384_T8_static"          # identical to this configuration of the sched experiment


def calib_reference(out):
    """Median time of the calibration configuration in the (thermally stable) sched experiment."""
    p = os.path.join(out, "sched.csv")
    if not os.path.isfile(p):
        return None
    with open(p, newline="") as f:
        ts = sorted(float(r["t_total"]) for r in csv.DictReader(f) if r["label"].split(":")[1] == CALIB_KEY)
    return ts[len(ts) // 2] if ts else None


def thermal_gate(out, ref, tag, tol, wait, max_wait):
    """Run the calibration configuration until it is within tol of the reference (CPU cooled down)."""
    calib_csv = os.path.join(out, "calib.csv")
    waited = 0.0
    while True:
        res, _ = run_sim(CALIB_ARGS + ["--csv", calib_csv, "--label", f"calib:{tag}", "--quiet"])
        t = float(res["t_total"])
        if ref is None or t <= ref * (1 + tol):
            print(f"  [gate] calibration {t:.3f}s (reference {ref if ref else float('nan'):.3f}s) ok", flush=True)
            return
        if waited >= max_wait:
            print(f"  [gate] calibration {t:.3f}s still {t / ref - 1:+.0%} after {waited:.0f}s - continuing", flush=True)
            return
        print(f"  [gate] calibration {t:.3f}s is {t / ref - 1:+.0%} vs reference - cooling {wait:.0f}s", flush=True)
        time.sleep(wait)
        waited += wait


def done_labels(path):
    if not os.path.isfile(path):
        return set()
    with open(path, newline="") as f:
        return {row["label"] for row in csv.DictReader(f)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="sched,mpi,gpu,ctl,small")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--quick", action="store_true", help="tiny smoke-test version")
    ap.add_argument("--pause", type=float, default=1.0, help="seconds between runs")
    ap.add_argument("--out", default=RESULTS)
    ap.add_argument("--gate-every", type=int, default=8, help="thermal gate before every N runs (0 = off)")
    ap.add_argument("--gate-tol", type=float, default=0.08, help="allowed slowdown vs reference")
    ap.add_argument("--gate-wait", type=float, default=45.0, help="seconds to cool between gate attempts")
    ap.add_argument("--gate-max", type=float, default=900.0, help="give up waiting after this many seconds")
    a = ap.parse_args()
    a.out = os.path.abspath(a.out)          # the simulator runs with cwd=bin/
    ref = calib_reference(a.out) if a.gate_every > 0 else None
    if a.gate_every > 0:
        print(f"thermal gate: reference {ref}s for {CALIB_KEY} (from sched.csv)", flush=True)
    nrun = 0

    os.makedirs(a.out, exist_ok=True)
    tdir = os.path.join(a.out, "traces")
    os.makedirs(tdir, exist_ok=True)
    suffix = "_quick" if a.quick else ""
    for exp in a.exp.split(","):
        runs = EXPERIMENTS[exp](a.quick)
        csv_path = os.path.join(a.out, f"{exp}{suffix}.csv")
        done = done_labels(csv_path)
        total = len(runs) * a.reps
        print(f"== {exp}: {len(runs)} configurations x {a.reps} reps -> {csv_path}", flush=True)
        k = 0
        t_exp = time.time()
        for rep in range(a.reps):
            order = list(runs)
            random.Random(1000 * rep + len(exp)).shuffle(order)
            for r in order:
                k += 1
                label = f"{exp}:{r.key}:r{rep}"
                if label in done:
                    continue
                if a.gate_every > 0 and nrun % a.gate_every == 0:
                    thermal_gate(a.out, ref, f"{exp}:r{rep}:{nrun}", a.gate_tol, a.gate_wait, a.gate_max)
                nrun += 1
                args = r.args + ["--csv", csv_path, "--label", label, "--quiet"]
                if r.trace and rep == 0:
                    args += ["--trace", os.path.join(tdir, f"{exp}{suffix}_{r.trace}")]
                t0 = time.time()
                try:
                    res, _ = run_sim(args, ranks=r.ranks)
                except Exception as e:  # keep going; the failure is reported
                    print(f"  [{k}/{total}] FAILED {label}: {e}", flush=True)
                    continue
                print(f"  [{k}/{total}] {label}  t_total={float(res['t_total']):.3f}s  "
                      f"({time.time() - t0:.1f}s wall)", flush=True)
                time.sleep(a.pause)
        print(f"== {exp} finished in {(time.time() - t_exp) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
