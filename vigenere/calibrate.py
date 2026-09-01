# -*- coding: utf-8 -*-
"""Machine-specific throughput calibration.

The estimate shown before a run ("key space ~ 00:11") needs to know how many
keys per second *this* machine sustains. Hard-coding numbers measured on one
GPU would make the estimate wrong everywhere else, so they are measured once
and cached in the data directory.

Two details make the measurement honest:

* the trigram table is filled with random values, because a table of zeros is
  unrealistically cache-friendly;
* CPU throughput is measured with a real worker pool rather than extrapolated
  from one core. The kernel is memory-bound, so N processes are far from N
  times faster - on the reference machine one core reaches 6.3M keys/s while
  32 processes together reach about 40M, not 200M.

Progress and ETA *during* a run always come from the observed rate, so a stale
calibration can only affect the up-front estimate.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import time

import numpy as np

from . import attack
from .paths import DATA_DIR, ensure_data_dir

CACHE = os.path.join(DATA_DIR, "calibration.json")

# Conservative values used until a measurement exists.
FALLBACK_CPU_PER_PROC = 4.0e5
FALLBACK_GPU = 2.0e7

_M, _N = 32, 30            # alphabet size and text length used by the kernels


def _kernel(seconds: float) -> float:
    """Run the brute-force kernel for a while; return keys per second."""
    rng = np.random.default_rng(12345)
    tri = rng.random(_M ** 3, dtype=np.float32) * -10.0
    c32 = (np.arange(_N) % _M).astype(np.int32)
    L = 4
    pos = np.arange(_N, dtype=np.int64) % L
    step = 1 << 16
    done, t0 = 0, time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        attack.scan_range_numpy(c32, tri, _M, L, 0, step, "vigenere", pos, batch=step)
        done += step
    return done / (time.perf_counter() - t0)


def _pool_worker(seconds):
    return _kernel(seconds)


def measure_cpu(nproc: int = None, seconds: float = 0.4):
    """Aggregate CPU throughput. Returns ``(keys/s total, processes used)``."""
    nproc = max(1, nproc or (os.cpu_count() or 4))
    if nproc == 1:
        return _kernel(seconds), 1
    try:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=nproc) as pool:
            rates = pool.map(_pool_worker, [seconds] * nproc)
        return float(sum(rates)), nproc
    except Exception:                                          # noqa: BLE001
        return _kernel(seconds) * nproc * 0.35, nproc          # rough fallback


def measure_gpu(step_budget: float = 0.6):
    """GPU throughput in keys per second, or None when CUDA is absent.

    The key length is stepped up while the previous step stayed cheap. Small
    runs are dominated by fixed setup (context, table upload) and would badly
    understate a fast card - a 1M-key run measured 6M keys/s on hardware that
    actually sustains 170M - so the best rate over the steps is taken.
    """
    if not attack.gpu_available():
        return None
    try:
        torch = attack._torch()
        attack.gpu_warmup()
        rng = np.random.default_rng(12345)
        tri = rng.random(_M ** 3, dtype=np.float32) * -10.0
        c32 = (np.arange(_N) % _M).astype(np.int32)
        best = 0.0
        for L in (4, 5, 6):
            total = _M ** L
            t0 = time.perf_counter()
            attack.gpu_scan(c32, tri, _M, L, "vigenere", topk=64)
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            best = max(best, total / max(1e-6, dt))
            if dt > step_budget:            # next step would be 32x longer
                break
        return best
    except Exception:                                          # noqa: BLE001
        return None


def load() -> dict:
    try:
        with open(CACHE, encoding="utf-8") as f:
            return json.load(f)
    except Exception:                                          # noqa: BLE001
        return {}


def save(data: dict):
    try:
        ensure_data_dir()
        with open(CACHE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception:                                          # noqa: BLE001
        pass


def measure(force: bool = False, want_gpu: bool = True, nproc: int = None) -> dict:
    """Return the cached figures, measuring whatever is missing."""
    data = load()
    changed = False
    cores = os.cpu_count() or 4

    if force or "cpu_total" not in data or data.get("cpu_cores") != cores:
        try:
            total, used = measure_cpu(nproc)
            data["cpu_total"] = total
            data["cpu_procs"] = used
            data["cpu_cores"] = cores
            data["cpu_name"] = "%d cores" % cores
            changed = True
        except Exception:                                      # noqa: BLE001
            data.setdefault("cpu_total", FALLBACK_CPU_PER_PROC * cores)
            data.setdefault("cpu_procs", cores)

    if want_gpu and attack.gpu_available():
        name = attack.gpu_name()
        if force or data.get("gpu_name") != name or "gpu" not in data:
            g = measure_gpu()
            if g:
                data["gpu"] = g
                data["gpu_name"] = name
                changed = True

    if changed:
        data["measured_at"] = time.strftime("%Y-%m-%d %H:%M")
        save(data)
    return data


def rate(use_gpu: bool, nproc: int, data: dict = None) -> float:
    """Estimated keys per second for the given configuration."""
    data = data if data is not None else load()
    if use_gpu:
        return float(data.get("gpu", FALLBACK_GPU))
    total = data.get("cpu_total")
    procs = data.get("cpu_procs") or 1
    if total:
        # Scale linearly around the measured operating point; the per-process
        # figure already includes memory-bandwidth contention.
        return float(total) * max(1, nproc) / max(1, procs)
    return FALLBACK_CPU_PER_PROC * max(1, nproc)
