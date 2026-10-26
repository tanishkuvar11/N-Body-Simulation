#!/usr/bin/env python3
"""Interactive 3D viewer: runs a few scenarios with --snap and writes ONE self-contained HTML
page (three.js from a CDN, the data embedded) that you can open in any browser.

  * orbit / zoom / pan the cloud, play / pause / scrub through time
  * colour each star by the OpenMP thread that computed its force (shows the static Morton
    blocks, and how they mix once the adaptive controller switches to dynamic), or by speed
  * live per-thread loop-time bars, imbalance lambda and the active schedule

    python scripts/export_web.py                 # -> results/web/viewer.html
    python scripts/export_web.py --n 8192 --every 2
"""
import argparse
import base64
import json
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, RESULTS, run_sim  # noqa: E402

SCENARIOS = [
    ("collapse", "Cold collapse (adaptive)", ["--ic", "collapse"]),
    ("collide", "Colliding clusters (adaptive)", ["--ic", "collide"]),
    ("disk", "Rotating disk galaxy (adaptive)", ["--ic", "disk"]),
    ("spin", "Spinning collapse (adaptive)", ["--ic", "collapse", "--spin", 0.5]),
]
POLICY = {0: "static", 1: "dynamic", 2: "guided"}
LIM = 3.0                                   # positions are quantised to int16 over [-LIM, LIM]


def read_snap3d(path):
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
        frames.append((step, t, pol, lam, wt.copy(), xyz.T.copy(), own.copy()))
        off += fsize
    return n, w, frames


def encode(frames):
    pos = np.concatenate([f[5] for f in frames])                      # (F*n, 3)
    q = np.clip(np.round(pos / LIM * 32767), -32767, 32767).astype("<i2")
    own = np.concatenate([f[6] for f in frames]).astype("u1")
    meta = [{"step": int(f[0]), "t": round(float(f[1]), 4), "pol": POLICY.get(int(f[2]), "?"),
             "lam": round(float(f[3]), 4), "wt": [round(float(x) * 1e3, 4) for x in f[4]]} for f in frames]
    return base64.b64encode(q.tobytes()).decode(), base64.b64encode(own.tobytes()).decode(), meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4096)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--every", type=int, default=3, help="snapshot every K steps")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", default=os.path.join(RESULTS, "web", "viewer.html"))
    a = ap.parse_args()

    tmp = os.path.join(ROOT, "build", "web")
    os.makedirs(tmp, exist_ok=True)
    datasets = []
    for key, title, args in SCENARIOS:
        snap = os.path.join(tmp, f"{key}.snap")
        print(f"  running {title} ...", flush=True)
        res, _ = run_sim(args + ["--n", a.n, "--steps", a.steps, "--threads", a.threads, "--sched", "adaptive",
                                 "--snap", snap, "--snap-every", a.every, "--no-energy", "--quiet", "--print", 0])
        n, w, frames = read_snap3d(snap)
        p, o, meta = encode(frames)
        datasets.append({"key": key, "title": title, "n": n, "w": w, "lim": LIM, "pos": p, "own": o, "meta": meta,
                         "switches": int(res["switches"])})

    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "viewer_template.html"), encoding="utf-8") as f:
        html = f.read()
    html = html.replace("/*__DATA__*/null", json.dumps(datasets, separators=(",", ":")))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        f.write(html)
    print(f"wrote {a.out} ({os.path.getsize(a.out) / 1e6:.1f} MB) - open it in a browser")


if __name__ == "__main__":
    main()
