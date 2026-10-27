#!/usr/bin/env python3
"""Live demo (~6 minutes), explained in plain language.

    python demo.py              # pauses between parts (press Enter)
    python demo.py --no-pause   # runs straight through
    python demo.py --part 2     # run only one part (1-5)
    python demo.py --details    # also show the simulator commands and raw output
"""
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))
from common import EXE, MPIEXEC, RESULTS, env_with_dlls  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
PAUSE = "--no-pause" not in sys.argv
DETAILS = "--details" in sys.argv
ONLY = int(sys.argv[sys.argv.index("--part") + 1]) if "--part" in sys.argv else None


def banner(n, title, text):
    print()
    print(f"Part {n} of 5: {title}")
    print()
    print(text.strip("\n") + "\n")


class SimFailed(Exception):
    pass


def run(args, ranks=1, quiet_fail=False):
    """Run the simulator; returns (RESULT dict, stdout lines).  Raw output only with --details."""
    cmd = [EXE] + [str(a) for a in args]
    if ranks > 1:
        cmd = [MPIEXEC, "-n", str(ranks)] + cmd
    p = subprocess.run(cmd, capture_output=True, text=True, env=env_with_dlls(), cwd=os.path.dirname(EXE))
    out = p.stdout.splitlines()
    if DETAILS:
        print("   $ " + ("mpiexec -n %d " % ranks if ranks > 1 else "") + "nbody.exe " + " ".join(str(a) for a in args))
        for line in out:
            if not line.startswith("RESULT"):
                print("     | " + line)
    if p.returncode != 0:
        if quiet_fail:
            raise SimFailed(p.stdout + p.stderr)
        print(p.stdout + p.stderr)
        raise SystemExit("The simulator failed. See the message above.")
    res = {}
    for line in out:
        if line.startswith("RESULT "):
            res = dict(kv.split("=", 1) for kv in line[7:].split())
    return res, out


def wait():
    if PAUSE:
        input("\n   Press Enter to continue ...")


def ok(flag):
    return "YES" if flag else "NO"


# --------------------------------------------------------------------------------------------
def part1():
    banner(1, "Does it compute the right answer?", """
We simulate stars pulling on each other by gravity. To be fast we use a shortcut
(Barnes-Hut): far-away groups of stars are treated as one big star.
Check 1: how close is the shortcut to the exact calculation?
Check 2: do 1 worker and 8 workers give exactly the same result?""")
    r, _ = run(["--ic", "plummer", "--n", 16384, "--steps", 0, "--threads", 8, "--pin", "--verify"])
    err = float(r["verify_rms"]) * 100
    print(f"   Shortcut vs exact calculation, 16,384 stars:  average error {err:.2f} %")

    common = ["--ic", "collide", "--n", 8000, "--steps", 20, "--no-energy", "--quiet"]
    sums = []
    for t, s, words in [(1, "static", "1 worker"), (8, "dynamic", "8 workers, sharing on demand"),
                        (8, "adaptive", "8 workers, with our manager")]:
        r, _ = run(common + ["--threads", t, "--sched", s])
        sums.append(r["checksum"])
        print(f"   {words:<32} -> final result fingerprint {r['checksum']}")
    print(f"\n   Same result every time?  {ok(len(set(sums)) == 1)}")
    print("   (Splitting the work between workers changes the speed, never the answer.)")


# --------------------------------------------------------------------------------------------
STEP_RE = re.compile(r"step\s+(\d+)\s+t=\s*([0-9.]+)\s+step=\s*([0-9.]+)s\s+lambda=([0-9.]+).*policy=(\w+)")


def show_progress(out, label):
    print(f"   {label}")
    print("     step   how far the slowest worker is behind         way of sharing work")
    prev = None
    for line in out:
        m = STEP_RE.search(line)
        if not m:
            continue
        step, lam, pol = int(m.group(1)), float(m.group(4)), m.group(5)
        bar = "#" * min(30, int(round(lam * 50)))
        share = "equal shares (static)" if pol == "static" else "grab when free (dynamic)"
        note = "   <-- manager switched!" if prev is not None and pol != prev else ""
        print(f"     {step:4d}   {100 * lam:4.0f} %  {bar:<30}   {share}{note}")
        prev = pol


def part2():
    banner(2, "The main idea: our manager vs equal shares", """
8 workers (CPU cores) share the stars. A cloud of stars collapses into a dense clump.
Stars inside the clump take much longer to calculate, so with EQUAL SHARES the worker
holding the clump falls far behind while the others wait.

Watch the '#' bars: they show how far behind the slowest worker is.
Our MANAGER notices when it gets too big and switches to 'grab work when free'.""")
    args = ["--ic", "collapse", "--n", 32768, "--steps", 150, "--threads", 8, "--pin", "--print", 15, "--no-energy"]
    a, out_a = run(args + ["--sched", "static"])
    show_progress(out_a, "A) EQUAL SHARES the whole time (static):")
    print()
    b, out_b = run(args + ["--sched", "adaptive"])
    show_progress(out_b, "B) WITH OUR MANAGER (adaptive):")
    ta, tb = float(a["t_total"]), float(b["t_total"])
    print(f"\n   Total time:  equal shares {ta:.2f} s   |   with our manager {tb:.2f} s"
          f"   ->  {100 * (ta - tb) / ta:.0f} % faster")
    print("   (Exact percentages vary a bit from run to run on a laptop.)")


# --------------------------------------------------------------------------------------------
def part3():
    banner(3, "Splitting across separate programs (MPI)", """
Besides sharing work inside one program, the simulation can be split across separate
programs, the way it would run on several computers of a cluster.
Same job as 1 program x 8 workers, then as 4 programs x 2 workers each.""")
    common = ["--ic", "collide", "--n", 32768, "--steps", 30, "--no-energy", "--quiet", "--pin", "--sched", "adaptive"]
    r1, _ = run(common + ["--threads", 8])
    print(f"   1 program  x 8 workers : {float(r1['t_total']):.2f} s   fingerprint {r1['checksum']}")
    if MPIEXEC:
        r4, _ = run(common + ["--threads", 2], ranks=4)
        print(f"   4 programs x 2 workers : {float(r4['t_total']):.2f} s   fingerprint {r4['checksum']}")
        print(f"\n   Same result?  {ok(r1['checksum'] == r4['checksum'])}")
    else:
        print("   (mpiexec not found, so the 4-program run is skipped)")


# --------------------------------------------------------------------------------------------
def part4():
    N = 262144
    banner(4, "Using the graphics card (CUDA)", f"""
Four ways to compute the forces for {N:,} stars, 8 CPU threads in every case:
the exact method (every star against every other) and the Barnes-Hut shortcut,
each on the CPU alone and with the graphics card helping.""")
    common = ["--ic", "plummer", "--n", N, "--steps", 5, "--no-energy", "--threads", 8, "--pin"]
    print("   Running the three fast methods live (a few seconds each) ...")
    try:
        g, out = run(common + ["--force", "gpu-direct", "--gpu-float"], quiet_fail=True)
    except SimFailed:
        print("   The NVIDIA graphics card is not available right now (Windows only shows the integrated")
        print("   graphics). Laptops often switch it off on battery: plug in the charger, or set the GPU mode")
        print("   to Hybrid/Standard in the laptop's control app, then run:  python demo.py --part 4")
        print()
        print(f"   Recorded result from our measurements (RTX 4060, {N:,} stars, ms per step):")
        print("   exact on CPU 32,147  |  exact on GPU 263  |  shortcut on CPU 272  |  shortcut CPU + GPU 206")
        return
    c, _ = run(common + ["--force", "bh", "--leaf", 16])
    h, _ = run(common + ["--force", "bh-gpu", "--gpu-float", "--leaf", 256])
    gpu = next((l.split(":", 1)[1].strip() for l in out if l.startswith("GPU:")), "GPU")
    rows = [("Exact method, CPU only", 32147.0, "measured separately: too slow to run live"),
            ("Exact method, graphics card", float(g["t_step"]) * 1000, "live"),
            ("Barnes-Hut shortcut, CPU only", float(c["t_step"]) * 1000, "live"),
            ("Barnes-Hut shortcut, CPU + graphics card", float(h["t_step"]) * 1000, "live")]
    best = min(rows, key=lambda r: r[1])
    print(f"   Graphics card: {gpu}")
    print()
    print("   Method                                      time per step")
    for name, ms, how in rows:
        mark = "   <-- fastest" if (name, ms, how) == best else ""
        print(f"   {name:<42} {ms:>9,.0f} ms   ({how}){mark}")
    print()
    print(f"   Our CPU + graphics card shortcut is {rows[0][1] / rows[3][1]:,.0f}x faster than the exact CPU method,")
    print(f"   {rows[2][1] / rows[3][1]:.1f}x faster than the shortcut on the CPU alone, and "
          f"{rows[1][1] / rows[3][1]:.1f}x faster than brute force on the graphics card.")


# --------------------------------------------------------------------------------------------
def part5():
    banner(5, "Results", """
Opening the results page in your web browser (works without internet):
the animation, the main graphs, and what they mean.""")
    page = os.path.join(RESULTS, "presentation.html")
    if not os.path.isfile(page):
        print("   results/presentation.html is missing")
        return
    print("   " + page)
    if os.name == "nt":
        os.startfile(page)  # noqa: S606 - opens in the default browser
    else:
        import webbrowser
        webbrowser.open("file://" + page)


PARTS = [part1, part2, part3, part4, part5]


def main():
    if not os.path.isfile(EXE):
        sys.exit("bin/nbody.exe not found. Run `python build.py` first.")
    for i, p in enumerate(PARTS, 1):
        if ONLY and i != ONLY:
            continue
        p()
        if i < len(PARTS) and not ONLY:
            wait()
    print("\nDemo finished.")


if __name__ == "__main__":
    main()
