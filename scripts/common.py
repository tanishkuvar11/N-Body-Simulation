"""Shared helpers: locate the executable and mpiexec, run one simulation, parse RESULT."""
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.path.join(ROOT, "bin")
RESULTS = os.path.join(ROOT, "results")
EXE = os.path.join(BIN, "nbody.exe" if os.name == "nt" else "nbody")


def find_mpiexec():
    cands = [shutil.which("mpiexec")]
    for p in (os.environ.get("CONDA_PREFIX"), os.path.dirname(sys.executable)):
        if p:
            cands.append(os.path.join(p, "Library", "bin", "mpiexec.exe"))
    msmpi = os.environ.get("MSMPI_BIN")
    if msmpi:
        cands.append(os.path.join(msmpi, "mpiexec.exe"))
    cands.append(r"C:\Program Files\Microsoft MPI\Bin\mpiexec.exe")
    for c in cands:
        if c and os.path.isfile(c):
            return c
    return None


MPIEXEC = find_mpiexec()


def env_with_dlls():
    """PATH including bin/ (msmpi.dll, nvrtc DLLs are copied there by build.py)."""
    env = dict(os.environ)
    extra = [BIN]
    if MPIEXEC:
        extra.append(os.path.dirname(MPIEXEC))
    env["PATH"] = os.pathsep.join(extra + [env.get("PATH", "")])
    return env


def run_sim(args, ranks=1, timeout=3600):
    """Run the simulator; returns (dict_of_RESULT, full_stdout)."""
    if not os.path.isfile(EXE):
        raise SystemExit(f"{EXE} not found - run `python build.py` first")
    cmd = [EXE] + [str(a) for a in args]
    if ranks > 1:
        if not MPIEXEC:
            raise SystemExit("mpiexec not found (needed for ranks > 1)")
        cmd = [MPIEXEC, "-n", str(ranks)] + cmd
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env_with_dlls(), cwd=BIN)
    if p.returncode != 0:
        raise RuntimeError(f"command failed ({p.returncode}): {' '.join(cmd)}\n{p.stdout}\n{p.stderr}")
    res = None
    for line in p.stdout.splitlines():
        if line.startswith("RESULT "):
            res = dict(kv.split("=", 1) for kv in line[7:].split())
    if res is None:
        raise RuntimeError(f"no RESULT line from: {' '.join(cmd)}\n{p.stdout}\n{p.stderr}")
    return res, p.stdout
