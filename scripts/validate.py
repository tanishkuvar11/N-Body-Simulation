#!/usr/bin/env python3
"""Correctness validation (objective O1 checks + parallel-correctness checks).

  V1  reproducibility: identical final state (bitwise checksum) for every schedule,
      thread count, MPI rank count and pinning choice
  V2  CUDA direct kernel vs CPU direct sum (double ~1e-15, float ~1e-6)
  V3  BH+GPU hybrid gives the same forces as the CPU tree code
  V4  Barnes-Hut force error vs opening angle theta (monotone, < 1 % at theta = 0.5)
  V5  energy and momentum conservation (direct and Barnes-Hut)
  V6  leapfrog is second order: energy error ~ dt^2
  V7  Plummer initial conditions are in virial equilibrium (2KE/|W| ~ 1)

V1 also covers the extensions: cost-weighted MPI partitioning (--partition cost), the
controller's auto mode (--balanced auto); V1b the parallel sort / tree build (N >= PAR_MIN).

Writes results/validation.md (or --out FILE) and exits non-zero if any check fails.
GPU checks are reported as SKIP when no CUDA GPU is available.
Usage: python scripts/validate.py [--out FILE]
"""
import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import MPIEXEC, RESULTS, run_sim  # noqa: E402

lines = []
fails = 0


def report(name, ok, detail):
    """ok: True = PASS, False = FAIL, None = SKIP (not counted as a failure)."""
    global fails
    fails += 1 if ok is False else 0
    tag = {True: "PASS", False: "**FAIL**", None: "SKIP"}[ok]
    lines.append(f"| {name} | {tag} | {detail} |")
    print(tag.strip("*") + " " + name + " - " + detail, flush=True)


def f(res, key):
    return float(res[key])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(RESULTS, "validation.md"))
    out_path = ap.parse_args().out
    q = ["--quiet"]

    # V1 reproducibility --------------------------------------------------------------
    base = ["--ic", "collide", "--n", 6000, "--steps", 25, "--no-energy"] + q
    configs = [(1, 1, "static", []), (1, 4, "static", []), (1, 8, "dynamic", []), (1, 3, "guided", []),
               (1, 6, "adaptive", ["--k", 1, "--low", 0.0001, "--high", 0.0002]),   # forces switching
               (1, 8, "adaptive", ["--pin"])]
    configs += [(1, 5, "adaptive", ["--balanced", "auto", "--k", 1, "--low", 0.0001, "--high", 0.0002])]
    if MPIEXEC:
        configs += [(2, 3, "adaptive", []), (3, 2, "dynamic", []), (4, 2, "static", ["--pin"]),
                    (4, 1, "guided", []),
                    (4, 2, "adaptive", ["--partition", "cost", "--balanced", "auto"]),
                    (3, 3, "static", ["--partition", "cost"])]
    sums, sw = {}, 0
    for p, t, s, extra in configs:
        r, _ = run_sim(base + ["--threads", t, "--sched", s] + extra, ranks=p)
        sums[f"P{p}T{t}-{s}"] = r["checksum"]
        if s == "adaptive" and "--high" in extra and "--balanced" not in extra:
            sw = int(r["switches"])
    ok = len(set(sums.values())) == 1
    report("V1 reproducibility (bitwise)", ok,
           f"{len(sums)} configs (1-4 ranks, 1-8 threads, all schedules, forced controller switching: "
           f"{sw} switches; --balanced auto; --partition cost) -> {len(set(sums.values()))} distinct "
           f"checksum(s) `{next(iter(sums.values()))}`")

    # V1b parallel sort + tree build: same state for any thread count above PAR_MIN ---------------
    big = ["--ic", "collapse", "--n", 70000, "--steps", 10, "--no-energy"] + q
    cs = {t: run_sim(big + ["--threads", t])[0]["checksum"] for t in (1, 3, 8)}
    report("V1b parallel Morton sort + tree build (N = 70000)", len(set(cs.values())) == 1,
           ", ".join(f"{t} threads `{c}`" for t, c in cs.items()))

    # V2 GPU direct --------------------------------------------------------------------
    gpu_ok = True
    try:
        rd, _ = run_sim(["--n", 8192, "--steps", 0, "--force", "gpu-direct", "--verify", "--no-energy"] + q)
        rf, _ = run_sim(["--n", 8192, "--steps", 0, "--force", "gpu-direct", "--gpu-float", "--verify", "--no-energy"] + q)
        report("V2 CUDA tiled direct kernel vs CPU direct", f(rd, "verify_max") < 1e-12 and f(rf, "verify_max") < 1e-4,
               f"double: rms {f(rd, 'verify_rms'):.2e}, max {f(rd, 'verify_max'):.2e}; "
               f"float: rms {f(rf, 'verify_rms'):.2e}, max {f(rf, 'verify_max'):.2e}")
    except Exception as e:
        gpu_ok = False
        msg = str(e)
        unavailable = "GPU init failed" in msg
        report("V2 CUDA tiled direct kernel vs CPU direct", None if unavailable else False,
               "no CUDA GPU / NVRTC on this machine" if unavailable else f"error: {msg.splitlines()[0][:120]}")

    # V3 hybrid == CPU BH ----------------------------------------------------------------
    if not gpu_ok:
        report("V3 BH+GPU hybrid vs CPU Barnes-Hut", None, "no CUDA GPU / NVRTC on this machine")
    else:
        args = ["--ic", "plummer", "--n", 16384, "--steps", 0, "--verify", "--no-energy", "--threads", 8] + q
        res = {}
        for leaf in (8, 64):
            rc, _ = run_sim(args + ["--force", "bh", "--leaf", leaf])
            rg, _ = run_sim(args + ["--force", "bh-gpu", "--leaf", leaf])
            rcap, _ = run_sim(args + ["--force", "bh-gpu", "--leaf", leaf, "--near-cap", 4])   # forces CPU overflow path
            res[leaf] = (f(rc, "verify_rms"), f(rg, "verify_rms"), f(rcap, "verify_rms"), int(rcap["near_overflow"]))
        ok = all(abs(a - b) <= 1e-9 * a and abs(a - c) <= 1e-9 * a and o > 0 for a, b, c, o in res.values())
        report("V3 BH+GPU hybrid vs CPU Barnes-Hut", ok,
               "; ".join(f"leaf {lf}: CPU rms {a:.6e}, hybrid {b:.6e}, hybrid w/ list overflow {c:.6e} ({o} overflows)"
                         for lf, (a, b, c, o) in res.items()))

    # V4 BH accuracy vs theta ----------------------------------------------------------------
    errs = []
    for th in (0.3, 0.5, 0.7, 0.9):
        r, _ = run_sim(["--ic", "plummer", "--n", 16384, "--steps", 0, "--theta", th, "--verify", "--no-energy",
                        "--threads", 8] + q)
        errs.append((th, f(r, "verify_rms"), f(r, "verify_max")))
    mono = all(errs[i][1] < errs[i + 1][1] for i in range(len(errs) - 1))
    e05 = [e for e in errs if e[0] == 0.5][0][1]
    report("V4 Barnes-Hut force error vs theta", mono and e05 < 0.01,
           ", ".join(f"theta {t}: rms {a:.2e} (max {m:.2e})" for t, a, m in errs))

    # V5 conservation -------------------------------------------------------------------------
    cons = {}
    for force in ("direct", "bh"):
        r, _ = run_sim(["--ic", "plummer", "--n", 4096, "--steps", 400, "--dt", 0.005, "--force", force,
                        "--threads", 8, "--diag", 50] + q)
        cons[force] = (f(r, "de_rel"), f(r, "de_rel_max"), f(r, "p1"))
    ok = cons["direct"][1] < 1e-4 and cons["bh"][1] < 1e-3 and cons["direct"][2] < 1e-13
    report("V5 energy & momentum conservation (Plummer, N=4096, t=2)", ok,
           f"direct: max dE/E {cons['direct'][1]:.2e}, |P| {cons['direct'][2]:.1e}; "
           f"Barnes-Hut: max dE/E {cons['bh'][1]:.2e}, |P| {cons['bh'][2]:.1e} (tree forces are not pairwise symmetric)")

    # V6 leapfrog order -------------------------------------------------------------------------
    de = []
    for dt, steps in ((0.02, 100), (0.01, 200), (0.005, 400)):
        r, _ = run_sim(["--ic", "plummer", "--n", 2048, "--steps", steps, "--dt", dt, "--force", "direct",
                        "--threads", 8] + q)
        de.append(f(r, "de_rel"))
    ratios = [de[i] / de[i + 1] for i in range(2)]
    report("V6 leapfrog 2nd order (dE ~ dt^2)", all(2.5 < x < 6.0 for x in ratios),
           f"dE/E at dt=0.02,0.01,0.005: {de[0]:.2e}, {de[1]:.2e}, {de[2]:.2e}; ratios {ratios[0]:.2f}, {ratios[1]:.2f} (ideal 4)")

    # V7 virial equilibrium -------------------------------------------------------------------------
    _, out = run_sim(["--ic", "plummer", "--n", 16384, "--steps", 0, "--threads", 8])
    m = re.search(r"virial 2KE/\|W\|=([0-9.]+)", out)
    qv = float(m.group(1)) if m else -1
    report("V7 Plummer IC virial ratio", 0.95 < qv < 1.05, f"2KE/|W| = {qv:.4f}")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as fh:
        fh.write("# Validation results\n\nGenerated by `python scripts/validate.py`.\n\n")
        fh.write("| Check | Result | Details |\n|---|---|---|\n")
        fh.write("\n".join(lines) + "\n")
    skipped = sum("| SKIP |" in ln for ln in lines)
    print(f"\n{len(lines) - fails - skipped}/{len(lines) - skipped} checks passed ({skipped} skipped) -> {out_path}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
