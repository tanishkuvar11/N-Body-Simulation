#!/usr/bin/env python3
"""Build bin/nbody(.exe).

Windows: MinGW-w64 gcc + MS-MPI (SDK or conda 'msmpi' package).
Linux:   mpicc (Open MPI / MPICH).
CUDA:    only headers are needed at build time (NVRTC + driver are loaded at run time).
         Searched in third_party/cuda (pip wheels) and $CUDA_PATH; otherwise -DNO_CUDA.

Usage:  python build.py [--fetch-cuda] [--no-cuda] [--debug]

  --fetch-cuda   first download the CUDA headers + NVRTC (~80 MB, pip wheels from NVIDIA)
                 into third_party/cuda - needed once for the GPU modes; no CUDA toolkit required
"""
import glob
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")
BIN = os.path.join(ROOT, "bin")
BUILD = os.path.join(ROOT, "build")
WIN = os.name == "nt"


def run(cmd, **kw):
    print("  $", " ".join(f'"{c}"' if " " in c else c for c in cmd))
    subprocess.check_call(cmd, **kw)


def find_msmpi():
    """Return (include_dir, lib_dir_or_None, dll_path_or_None)."""
    cands = []
    if os.environ.get("MSMPI_INC"):
        cands.append((os.environ["MSMPI_INC"], os.environ.get("MSMPI_LIB64"), None))
    prefixes = [os.environ.get("CONDA_PREFIX"), os.path.dirname(sys.executable)]
    for p in prefixes:
        if p:
            inc = os.path.join(p, "Library", "include")
            cands.append((inc, os.path.join(p, "Library", "lib"), os.path.join(p, "Library", "bin", "msmpi.dll")))
    sdk = os.path.join(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"), "Microsoft SDKs", "MPI")
    cands.append((os.path.join(sdk, "Include"), os.path.join(sdk, "Lib", "x64"), None))
    for inc, lib, dll in cands:
        if inc and os.path.isfile(os.path.join(inc, "mpi.h")):
            if not (dll and os.path.isfile(dll)):
                sysdll = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "msmpi.dll")
                dll = sysdll if os.path.isfile(sysdll) else None
            return inc, lib, dll
    return None, None, None


def find_cuda_headers():
    tp = os.path.join(ROOT, "third_party", "cuda", "nvidia")
    a, b = os.path.join(tp, "cuda_runtime", "include"), os.path.join(tp, "cuda_nvrtc", "include")
    if os.path.isfile(os.path.join(a, "cuda.h")) and os.path.isfile(os.path.join(b, "nvrtc.h")):
        return [a, b]
    cp = os.environ.get("CUDA_PATH") or ("/usr/local/cuda" if not WIN else "")
    if cp and os.path.isfile(os.path.join(cp, "include", "cuda.h")):
        return [os.path.join(cp, "include")]
    return None


def fetch_cuda():
    target = os.path.join(ROOT, "third_party", "cuda")
    os.makedirs(target, exist_ok=True)
    run([sys.executable, "-m", "pip", "install", "--upgrade", "--target", target,
         "nvidia-cuda-nvrtc-cu12", "nvidia-cuda-runtime-cu12"])


def main():
    if "--fetch-cuda" in sys.argv:
        fetch_cuda()
    no_cuda = "--no-cuda" in sys.argv
    debug = "--debug" in sys.argv
    os.makedirs(BIN, exist_ok=True)
    os.makedirs(BUILD, exist_ok=True)
    sources = sorted(glob.glob(os.path.join(SRC, "*.c")))
    opt = ["-O0", "-g"] if debug else ["-O3", "-march=native"]
    cflags = ["-std=gnu11", "-fopenmp", "-Wall", "-Wextra", "-Wno-unused-parameter"] + opt
    libs = ["-lm"]

    cuda_inc = None if no_cuda else find_cuda_headers()
    if cuda_inc:
        cflags += [f"-I{d}" for d in cuda_inc]
        print("CUDA headers:", ", ".join(cuda_inc))
    else:
        cflags.append("-DNO_CUDA")
        print("CUDA: disabled (headers not found or --no-cuda); run `python build.py --fetch-cuda` for GPU modes")

    if WIN:
        cc = shutil.which("gcc")
        if not cc:
            sys.exit("gcc (MinGW-w64) not found on PATH")
        inc, libdir, dll = find_msmpi()
        if not inc:
            sys.exit("MS-MPI headers not found (install MS-MPI SDK or `conda install msmpi`)")
        print("MS-MPI include:", inc)
        cflags.append(f"-I{inc}")
        implib = os.path.join(BUILD, "libmsmpi.a")
        if not os.path.isfile(implib):
            if dll and shutil.which("gendef") and shutil.which("dlltool"):
                run(["gendef", dll], cwd=BUILD)
                run(["dlltool", "-d", "msmpi.def", "-l", implib, "-D", "msmpi.dll"], cwd=BUILD)
            elif libdir and os.path.isfile(os.path.join(libdir, "msmpi.lib")):
                implib = os.path.join(libdir, "msmpi.lib")
            else:
                sys.exit("cannot create an import library for msmpi.dll")
        libs = [implib] + libs
        out = os.path.join(BIN, "nbody.exe")
        run([cc] + cflags + sources + ["-o", out] + libs)
        # runtime DLLs next to the executable
        if dll:
            shutil.copy2(dll, BIN)
        if cuda_inc:
            for d in glob.glob(os.path.join(ROOT, "third_party", "cuda", "nvidia", "cuda_nvrtc", "bin", "*.dll")):
                shutil.copy2(d, BIN)
    else:
        cc = shutil.which("mpicc")
        if not cc:
            sys.exit("mpicc not found")
        out = os.path.join(BIN, "nbody")
        run([cc] + cflags + sources + ["-o", out] + libs + ["-ldl"])
    print("built", out)


if __name__ == "__main__":
    main()
