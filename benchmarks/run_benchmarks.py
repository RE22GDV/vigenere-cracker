# -*- coding: utf-8 -*-
"""Performance measurements.

Accuracy is *not* measured here - it lives in ``studies.py``, which uses
held-out corpora and separated key groups. This file answers only "how fast",
and it keeps measured numbers and extrapolated numbers strictly apart.

Usage:
  python benchmarks/run_benchmarks.py             full run
  python benchmarks/run_benchmarks.py --quick
  python benchmarks/run_benchmarks.py --only throughput,endtoend
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

from vigenere import attack, calibrate, core, langs             # noqa: E402

RESULTS = os.path.join(HERE, "results")
DOCS = os.path.join(REPO, "docs")

M_REF, N_REF = 32, 30          # alphabet size and text length for the kernels


def _synthetic():
    rng = np.random.default_rng(12345)
    tri = rng.random(M_REF ** 3, dtype=np.float32) * -10.0
    c32 = (np.arange(N_REF) % M_REF).astype(np.int32)
    return c32, tri


# --------------------------------------------------------------------------- #
#  1. Raw throughput and parallel scaling
# --------------------------------------------------------------------------- #
def bench_throughput(a):
    print("\n[1] Throughput (measured)")
    out = {"cpu_single": None, "gpu_by_len": [], "pool": [], "gpu": None}
    c32, tri = _synthetic()

    L, step = 4, 1 << 16
    pos = np.arange(N_REF, dtype=np.int64) % L
    t0, done = time.perf_counter(), 0
    while time.perf_counter() - t0 < 1.0:
        attack.scan_range_numpy(c32, tri, M_REF, L, 0, step, "vigenere", pos, batch=step)
        done += step
    out["cpu_single"] = done / (time.perf_counter() - t0)
    print("    CPU, 1 core : %12.0f keys/s" % out["cpu_single"])

    if attack.gpu_available():
        attack.gpu_warmup()
        for L in ((4, 5, 6) if not a.quick else (5,)):
            total = M_REF ** L
            t0 = time.perf_counter()
            attack.gpu_scan(c32, tri, M_REF, L, "vigenere", topk=64)
            dt = time.perf_counter() - t0
            out["gpu_by_len"].append({"L": L, "keys": total, "seconds": dt,
                                      "rate": total / dt})
            print("    GPU, L=%d    : %12.0f keys/s (%.2f s)" % (L, total / dt, dt))
        out["gpu"] = max(x["rate"] for x in out["gpu_by_len"])

    cores = os.cpu_count() or 8
    procs = sorted({p for p in [1, 2, 4, 8, 16, cores] if p <= cores})
    if a.quick:
        procs = sorted({1, cores})
    for p in procs:
        total, used = calibrate.measure_cpu(nproc=p, seconds=0.5)
        out["pool"].append({"procs": used, "rate": total})
        print("    pool x%-3d   : %12.0f keys/s" % (used, total))
    if out["gpu"] and out["pool"]:
        best = max(x["rate"] for x in out["pool"])
        out["speedup_gpu_vs_pool"] = out["gpu"] / best
        out["speedup_gpu_vs_core"] = out["gpu"] / out["cpu_single"]
        print("    GPU speed-up: %.1fx over the full CPU pool, %.0fx over one core"
              % (out["speedup_gpu_vs_pool"], out["speedup_gpu_vs_core"]))
    return out


# --------------------------------------------------------------------------- #
#  2. Cold start versus warm start
# --------------------------------------------------------------------------- #
def bench_startup(a):
    print("\n[2] Start-up costs (measured)")
    out = {}
    script = ("import time,sys;sys.path.insert(0,%r);"
              "t=time.perf_counter();from vigenere import core;"
              "m=core.get_model('en-26');print(time.perf_counter()-t)" % REPO)
    times = []
    for _ in range(1 if a.quick else 3):
        r = subprocess.run([sys.executable, "-c", script], capture_output=True,
                           text=True)
        try:
            times.append(float(r.stdout.strip().splitlines()[-1]))
        except (ValueError, IndexError):
            pass
    if times:
        out["cold_model_load_s"] = min(times)
        print("    cold process + model from cache : %6.2f s" % min(times))

    t0 = time.perf_counter()
    core.get_model("en-26")
    out["warm_model_load_s"] = time.perf_counter() - t0
    print("    warm model (already in memory)  : %6.2f s" % out["warm_model_load_s"])

    if attack.gpu_available():
        script = ("import time,sys;sys.path.insert(0,%r);"
                  "from vigenere import attack;t=time.perf_counter();"
                  "attack.gpu_warmup();print(time.perf_counter()-t)" % REPO)
        r = subprocess.run([sys.executable, "-c", script], capture_output=True,
                           text=True)
        try:
            out["cuda_context_s"] = float(r.stdout.strip().splitlines()[-1])
            print("    CUDA context creation           : %6.2f s"
                  % out["cuda_context_s"])
        except (ValueError, IndexError):
            pass

    t0 = time.perf_counter()
    calibrate.measure_cpu(nproc=min(8, os.cpu_count() or 4), seconds=0.05)
    out["pool_spawn_s"] = time.perf_counter() - t0
    print("    spawning a worker pool          : %6.2f s" % out["pool_spawn_s"])
    return out


# --------------------------------------------------------------------------- #
#  3. End-to-end time, broken down
# --------------------------------------------------------------------------- #
def bench_endtoend(a):
    print("\n[3] End to end, from ciphertext to answer (measured)")
    lengths = [30, 100, 400] if a.quick else [30, 60, 100, 200, 400]
    out = {"rows": []}
    models = {k: core.get_model(k) for k in langs.auto_keys("en")}
    alpha = core.get_alphabet("en-26")
    base = ("the quick brown fox jumps over the lazy dog while the other dogs "
            "watch the river and the trees near the old stone bridge ") * 12
    for n in lengths:
        plain = "".join(base[:n * 2])
        idx = alpha.to_indices(plain)[:n]
        plain = alpha.render(idx)
        ct = core.encrypt_text(plain, "silver", alpha)
        cfg = dict(text=ct, alpha_list=[("en-26", models["en-26"])],
                   variant="vigenere", mode="auto", min_len=1, max_len=20,
                   processes=min(8, os.cpu_count() or 4), restarts=500,
                   keylen_penalty=3.47, dict_words=[], dict_reversed=True,
                   freq_seeds=320, use_gpu=attack.gpu_available(), gpu_batch=None)
        q = queue.Queue()
        eng = attack.Engine(cfg, q)
        t0 = time.perf_counter()
        eng.start()
        while eng.is_alive() or not q.empty():
            try:
                q.get(timeout=0.3)
            except queue.Empty:
                continue
        eng.join()
        wall = time.perf_counter() - t0
        out["rows"].append({"letters": n, "wall_s": wall})
        print("    %4d letters -> %6.2f s wall (includes pool spawn)" % (n, wall))
    return out


# --------------------------------------------------------------------------- #
#  4. Video memory
# --------------------------------------------------------------------------- #
def bench_vram(a):
    print("\n[4] Video memory (measured)")
    if not attack.gpu_available():
        print("    no CUDA, skipped")
        return {"available": False}
    torch = attack._torch()
    out = {"available": True,
           "total_mb": attack.gpu_total_memory_mb(), "rows": []}
    c32, tri = _synthetic()
    alpha = core.get_alphabet("en-26")
    for n in (30, 100, 400):
        c = (np.arange(n) % M_REF).astype(np.int32)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        attack.gpu_scan(c, tri, M_REF, 5, "vigenere", topk=256)
        peak = torch.cuda.max_memory_allocated() / 2 ** 20
        batch = attack.gpu_batch_for(n)
        out["rows"].append({"letters": n, "peak_mb": peak, "batch": batch})
        print("    %4d letters: batch %8d keys, peak %6.0f MB" % (n, batch, peak))
    out["batch_by_free_mb"] = {str(f): attack.gpu_batch_for(30, free_mb=f)
                               for f in (1024, 2048, 4096, 8192, 24576)}
    print("    batch chosen for a card with 1 GB free: %d keys"
          % out["batch_by_free_mb"]["1024"])
    return out


# --------------------------------------------------------------------------- #
#  5. Brute-force cost - measured where feasible, extrapolated beyond
# --------------------------------------------------------------------------- #
def bench_bruteforce(a):
    print("\n[5] Brute-force cost (measured up to L=6, extrapolated above)")
    calib = calibrate.measure()
    cpu_rate = calibrate.rate(False, os.cpu_count() or 8, calib)
    gpu_rate = calibrate.rate(True, 1, calib) if attack.gpu_available() else None
    measured_upto = 6 if not a.quick else 5
    c32, tri = _synthetic()
    rows = []
    for L in range(1, 9):
        keys = M_REF ** L
        row = {"L": L, "keys": keys, "measured": False}
        if L <= measured_upto and gpu_rate:
            t0 = time.perf_counter()
            attack.gpu_scan(c32, tri, M_REF, L, "vigenere", topk=64)
            row["gpu_seconds"] = time.perf_counter() - t0
            row["measured"] = True
        elif gpu_rate:
            row["gpu_seconds"] = keys / gpu_rate
        row["cpu_seconds"] = keys / cpu_rate
        rows.append(row)
        print("    L=%d %18s keys  GPU %9.1f s %s" %
              (L, f"{keys:,}".replace(",", " "), row.get("gpu_seconds", float("nan")),
               "(measured)" if row["measured"] else "(extrapolated)"))
    return {"cpu_rate": cpu_rate, "gpu_rate": gpu_rate,
            "measured_upto_L": measured_upto, "rows": rows}


# --------------------------------------------------------------------------- #
#  Charts
# --------------------------------------------------------------------------- #
def make_charts(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(DOCS, exist_ok=True)
    plt.rcParams.update({"figure.dpi": 130, "font.size": 9,
                         "axes.grid": True, "grid.alpha": 0.3})

    if "throughput" in res:
        t = res["throughput"]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.6))
        labels, vals = ["CPU\n1 core"], [t["cpu_single"]]
        if t["pool"]:
            best = max(t["pool"], key=lambda p: p["rate"])
            labels.append("CPU pool\nx%d" % best["procs"]); vals.append(best["rate"])
        if t.get("gpu"):
            labels.append("GPU\n(CUDA)"); vals.append(t["gpu"])
        bars = ax1.bar(labels, vals, color=["#8c8c8c", "#1f77b4", "#2ca02c"][:len(vals)])
        ax1.set_yscale("log"); ax1.set_ylabel("keys / second")
        ax1.set_title("Brute-force throughput (measured)")
        for b, v in zip(bars, vals):
            ax1.text(b.get_x() + b.get_width() / 2, v * 1.15,
                     ("%.1e" % v).replace("e+0", "e"), ha="center", fontsize=8)
        ax1.set_ylim(top=max(vals) * 4)
        if t["pool"]:
            p = [x["procs"] for x in t["pool"]]
            r = [x["rate"] for x in t["pool"]]
            ax2.plot(p, [x / r[0] for x in r], "o-", color="#1f77b4", label="measured")
            ax2.plot(p, p, "--", color="#999", label="ideal linear")
            ax2.set_xlabel("worker processes"); ax2.set_ylabel("speed-up vs 1 process")
            ax2.set_title("Parallel scaling (CPU)"); ax2.legend()
        fig.tight_layout(); fig.savefig(os.path.join(DOCS, "throughput.png"))
        plt.close(fig)

    if "bruteforce" in res:
        b = res["bruteforce"]
        fig, ax = plt.subplots(figsize=(6.5, 3.8))
        meas = [(r["L"], r["gpu_seconds"]) for r in b["rows"]
                if r.get("measured") and r.get("gpu_seconds")]
        extr = [(r["L"], r["gpu_seconds"]) for r in b["rows"]
                if not r.get("measured") and r.get("gpu_seconds")]
        if meas:
            ax.semilogy([x for x, _ in meas], [y for _, y in meas], "o-",
                        color="#2ca02c", label="GPU, measured")
        if extr:
            ax.semilogy([x for x, _ in extr], [y for _, y in extr], "o--",
                        color="#2ca02c", alpha=0.5, label="GPU, extrapolated")
        ax.semilogy([r["L"] for r in b["rows"]], [r["cpu_seconds"] for r in b["rows"]],
                    "o--", color="#1f77b4", alpha=0.6, label="CPU pool, extrapolated")
        for sec, lab in ((60, "1 minute"), (3600, "1 hour"), (86400, "1 day")):
            ax.axhline(sec, color="#bbb", lw=0.8, ls=":")
            ax.text(1, sec * 1.3, lab, fontsize=7, color="#888")
        ax.set_xlabel("key length (32-letter alphabet)")
        ax.set_ylabel("time to enumerate the space, s")
        ax.set_title("Brute-force cost")
        ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(os.path.join(DOCS, "bruteforce_cost.png"))
        plt.close(fig)
    print("\nCharts written to %s" % DOCS)


EXPERIMENTS = {"throughput": bench_throughput, "startup": bench_startup,
               "endtoend": bench_endtoend, "vram": bench_vram,
               "bruteforce": bench_bruteforce}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", default="")
    ap.add_argument("--charts-only", action="store_true")
    a = ap.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, "performance.json")
    res = {}
    if os.path.exists(path):
        try:
            res = json.load(open(path, encoding="utf-8"))
        except Exception:                                       # noqa: BLE001
            res = {}
    if a.charts_only:
        make_charts(res)
        return 0

    res.setdefault("meta", {}).update({
        "date": time.strftime("%Y-%m-%d %H:%M"), "cpu_count": os.cpu_count(),
        "gpu": attack.gpu_name() if attack.gpu_available() else None,
        "gpu_memory_mb": attack.gpu_total_memory_mb(),
        "python": sys.version.split()[0], "numpy": np.__version__,
        "quick": a.quick})

    names = [w.strip() for w in a.only.split(",") if w.strip()] or list(EXPERIMENTS)
    t0 = time.time()
    for name in names:
        res[name] = EXPERIMENTS[name](a)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=1)
    print("\nTotal: %.1f min" % ((time.time() - t0) / 60))
    make_charts(res)
    return 0


if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    sys.exit(main())
