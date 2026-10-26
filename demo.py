#!/usr/bin/env python3
"""Live demo (~5 minutes): correctness, adaptive scheduling, MPI, CUDA, results.

    python demo.py              # pauses between parts (press Enter)
    python demo.py --no-pause   # runs straight through
    python demo.py --part 2     # run only one part (1-5)
"""
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))
from common import EXE, MPIEXEC, RESULTS, env_with_dlls  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
PAUSE = "--no-pause" not in sys.argv
ONLY = int(sys.argv[sys.argv.index("--part") + 1]) if "--part" in sys.argv else None


def banner(n, title, text):
    print("\n" + "=" * 78)
    print(f"  PART {n}: {title}")
    print("=" * 78)
    print(text.strip() + "\n")


def run(args, ranks=1, keep=None):
    """Run the simulator, echoing the command; keep = substrings of lines to show (None = all)."""
    cmd = [EXE] + [str(a) for a in args]
    if ranks > 1:
        cmd = [MPIEXEC, "-n", str(ranks)] + cmd
    shown = " ".join(["mpiexec -n " + str(ranks)] if ranks > 1 else []) + " nbody.exe " + " ".join(str(a) for a in args)
    print(f"$ {shown.strip()}")
    p = subprocess.run(cmd, capture_output=True, text=True, env=env_with_dlls(), cwd=os.path.dirname(EXE))
    out = p.stdout.splitlines()
    for line in out:
        if line.startswith("RESULT"):
            continue
        if keep is None or any(k in line for k in keep):
            print("   " + line)
    if p.returncode != 0:
        print(p.stderr)
    res = {}
    for line in out:
        if line.startswith("RESULT "):
            res = dict(kv.split("=", 1) for kv in line[7:].split())
    return res


def wait():
    if PAUSE:
        input("\n   [Enter] to continue ...")


def part1():
    banner(1, "Correctness", """
Barnes-Hut forces are checked against exact direct summation; energy and momentum are
conserved; and because every rank and thread computes each body's force with the same
operation sequence, the final state is BIT-IDENTICAL for any schedule / thread / rank count.""")
    run(["--ic", "plummer", "--n", 16384, "--steps", 0, "--threads", 8, "--pin", "--verify"], keep=["VERIFY", "virial"])
    print()
    common = ["--ic", "collide", "--n", 8000, "--steps", 20, "--no-energy", "--quiet"]
    sums = []
    for t, s in [(1, "static"), (8, "dynamic"), (8, "adaptive")]:
        r = run(common + ["--threads", t, "--sched", s])
        sums.append(r.get("checksum"))
        print(f"   -> threads={t:<2} sched={s:<9} checksum {r.get('checksum')}")
    print(f"\n   identical: {len(set(sums)) == 1}")


def part2():
    banner(2, "Adaptive scheduling on a collapsing cloud", """
A cold uniform sphere collapses under gravity: the work per body is balanced at first and becomes
very unbalanced once a dense core forms (t ~ 1.1).  Watch lambda (= slowest thread / average - 1):
with STATIC it grows and the step time rises; ADAPTIVE switches to dynamic when lambda > 0.15.""")
    args = ["--ic", "collapse", "--n", 32768, "--steps", 150, "--threads", 8, "--pin", "--print", 15, "--no-energy"]
    a = run(args + ["--sched", "static"], keep=["step ", "done"])
    print()
    b = run(args + ["--sched", "adaptive"], keep=["step ", "done"])
    ta, tb = float(a["t_total"]), float(b["t_total"])
    print(f"\n   static {ta:.2f} s   adaptive {tb:.2f} s   ->  {100 * (ta - tb) / ta:+.1f} %  "
          f"(controller switches: {b['switches']})")


def part3():
    banner(3, "Hybrid MPI + OpenMP", """
Bodies are Morton-sorted and block-partitioned over MPI ranks; MPI_Allgatherv replicates
positions each step.  Same physics, same checksum - 1 rank x 8 threads vs 4 ranks x 2 threads.""")
    common = ["--ic", "collide", "--n", 32768, "--steps", 30, "--no-energy", "--quiet", "--pin", "--sched", "adaptive"]
    r1 = run(common + ["--threads", 8])
    print(f"   -> 1 rank x 8 threads : {float(r1['t_total']):.2f} s  checksum {r1['checksum']}")
    if MPIEXEC:
        r4 = run(common + ["--threads", 2], ranks=4)
        print(f"   -> 4 ranks x 2 threads: {float(r4['t_total']):.2f} s  checksum {r4['checksum']}  "
              f"(comm {float(r4['t_comm']):.2f} s)")
        print(f"\n   identical: {r1['checksum'] == r4['checksum']}")
    else:
        print("   (mpiexec not found - skipping the multi-rank run)")


def part4():
    banner(4, "CUDA on the GPU", """
Tiled O(N^2) direct-sum kernel (shared-memory tiles, Nyland et al.) compiled at run time with NVRTC,
compared with the exact CPU result; then the Barnes-Hut + GPU near-field hybrid vs the CPU tree code.""")
    run(["--force", "gpu-direct", "--gpu-float", "--n", 65536, "--steps", 10, "--no-energy", "--verify", "--threads", 8],
        keep=["GPU:", "VERIFY", "done"])
    print()
    a = run(["--force", "bh", "--leaf", 16, "--n", 65536, "--steps", 10, "--no-energy", "--threads", 8, "--pin"],
            keep=["done"])
    b = run(["--force", "bh-gpu", "--gpu-float", "--leaf", 256, "--n", 65536, "--steps", 10, "--no-energy",
             "--threads", 8, "--pin"], keep=["done"])
    print(f"\n   Barnes-Hut per step: CPU {float(a['t_step']) * 1e3:.1f} ms   CPU+GPU hybrid "
          f"{float(b['t_step']) * 1e3:.1f} ms  ->  {float(a['t_step']) / float(b['t_step']):.2f}x")


def part5():
    banner(5, "Animation and benchmark results", "Playing results/animation.mp4, then opening the figures and RESULTS.md ...")
    video = os.path.join(RESULTS, "animation.mp4")
    if os.path.isfile(video):
        print("   " + video)
        if os.name == "nt":
            os.startfile(video)  # noqa: S606 - default video player
            if PAUSE:
                print()
                input("   [Enter] after the video to open the figures ...")
    else:
        print("   (no animation yet - run: python scripts/animate.py)")
    figs = ["imbalance_traces.png", "relative_to_static.png", "speedup_n65536.png", "mpi_hybrid.png", "gpu.png"]
    for f in figs:
        p = os.path.join(RESULTS, "figures", f)
        if os.path.isfile(p):
            print("   " + p)
            if os.name == "nt":
                os.startfile(p)  # noqa: S606 - opens in the default image viewer
    md = os.path.join(ROOT, "RESULTS.md")
    if os.path.isfile(md) and os.name == "nt":
        os.startfile(md)


PARTS = [part1, part2, part3, part4, part5]


def main():
    if not os.path.isfile(EXE):
        sys.exit("bin/nbody.exe not found - run `python build.py` first")
    for i, p in enumerate(PARTS, 1):
        if ONLY and i != ONLY:
            continue
        p()
        if i < len(PARTS) and not ONLY:
            wait()
    print("\nDemo finished.")


if __name__ == "__main__":
    main()
