# -*- coding: utf-8 -*-
"""Measurement suite: throughput, scaling and success rate.

Plaintexts are sampled from the downloaded corpora and keys from the word
lists, so the accuracy figures are averages over real material rather than
over a handful of hand-picked sentences.

Usage:
  python benchmarks/run_benchmarks.py            full run (~20 min)
  python benchmarks/run_benchmarks.py --quick    reduced trial counts
  python benchmarks/run_benchmarks.py --only throughput,accuracy_len

Results are written to benchmarks/results/*.json and charts to docs/*.png.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import queue
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vigenere import attack, calibrate, core, langs                # noqa: E402
from vigenere.paths import DATA_DIR                                # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
RESULTS = os.path.join(HERE, "results")
DOCS = os.path.join(REPO, "docs")

LANGS = ["en", "ru", "uk"]
SEED = 20260901


# --------------------------------------------------------------------------- #
#  Sampling material
# --------------------------------------------------------------------------- #
class Material:
    """Sentences and keys drawn from the real corpora."""

    def __init__(self, lang, rng):
        self.lang = lang
        self.rng = rng
        self.alpha = core.get_alphabet(langs.auto_keys(lang)[0])
        self.model = core.get_model(self.alpha.key)
        self.sentences = self._load_sentences()
        self.keys = self._load_keys()

    def _load_sentences(self):
        path = os.path.join(DATA_DIR, langs.get(self.lang)["files"][0])
        out = []
        if os.path.exists(path):
            with open(path, encoding="utf-8", errors="ignore") as f:
                for i, line in enumerate(f):
                    if i > 400000:
                        break
                    if len(line) > 40:
                        out.append(line.strip())
        if not out:
            out = [s for s in langs.builtin_corpus(self.lang).split("\n") if len(s) > 40]
        return out

    def _load_keys(self):
        words = core.load_frequent_words(self.lang) or langs.builtin_words(self.lang)
        by_len = {}
        for w in words:
            if all(c in self.alpha.index for c in w):
                by_len.setdefault(len(w), []).append(w)
        return by_len

    def text(self, n_letters):
        """Concatenate sentences until the requested number of letters is met."""
        buf = []
        have = 0
        while have < n_letters:
            s = self.sentences[self.rng.randrange(len(self.sentences))]
            buf.append(s)
            have += len(self.alpha.to_indices(s))
        joined = " ".join(buf)
        # trim to exactly n_letters letters, keeping the spacing
        kept, count = [], 0
        for ch in joined:
            if ch.lower() in self.alpha.index:
                if count >= n_letters:
                    break
                count += 1
            kept.append(ch)
        return "".join(kept).strip()

    def key(self, length):
        pool = self.keys.get(length)
        if pool:
            return pool[self.rng.randrange(len(pool))]
        letters = self.alpha.letters
        return "".join(letters[self.rng.randrange(len(letters))] for _ in range(length))


# --------------------------------------------------------------------------- #
#  A light in-process solver (no worker pool) so that the measurements reflect
#  the algorithms and not the fixed cost of starting 32 processes.
# --------------------------------------------------------------------------- #
def solve(ciphertext, lang, models, max_key_len=16, freq_seeds=320,
          brute_max=0, use_gpu=False, seed=0):
    """Return the ranked rows for a ciphertext using every alphabet variant."""
    rows = []
    for akey in langs.auto_keys(lang):
        alpha = core.get_alphabet(akey)
        model = models[akey]
        idx, _ = alpha.layout(ciphertext)
        c32 = idx.astype(np.int32)
        cands = []
        for L in range(1, max_key_len + 1):
            if len(idx) < 2 * L:
                break
            cands += attack.frequency_attack(c32, model.tri, alpha.M, "vigenere",
                                             model.letter_freq(), L,
                                             n_seeds=freq_seeds, seed=seed + L)
        for L in range(1, brute_max + 1):
            if use_gpu:
                res = attack.gpu_scan(c32, model.tri, alpha.M, L, "vigenere", topk=256)
            else:
                pos = np.arange(len(idx), dtype=np.int64) % L
                res = attack.scan_range_numpy(c32, model.tri, alpha.M, L, 0,
                                              alpha.M ** L, "vigenere", pos, topk=256)
            cands += [(s, tuple(int(v) for v in alpha.key_from_int(k, L)))
                      for s, k in res]
        if cands:
            rows += attack.final_rank(cands, ciphertext, alpha, model, "vigenere",
                                      limit=20, keylen_penalty=math.log(alpha.M),
                                      alpha_name=akey)
    rows.sort(key=lambda r: -r['final'])
    return rows[:20]


def _norm(text, alpha):
    return ''.join(c for c in alpha.fold(text.lower()) if c in alpha.index)


def solved(rows, plain, alpha):
    if not rows:
        return False
    a2 = core.get_alphabet(rows[0]['alpha'])
    return _norm(rows[0]['plain'], a2) == _norm(plain, alpha)


# --------------------------------------------------------------------------- #
#  Trials run in a worker pool: they are independent, and a few hundred of them
#  would take hours one at a time. Each worker loads the models of one language
#  once (the word sets are large, so the pool is deliberately small).
# --------------------------------------------------------------------------- #
_W = {}


def _init_lang(lang):
    _W['lang'] = lang
    _W['models'] = {k: core.get_model(k) for k in langs.auto_keys(lang)}
    _W['alpha'] = core.get_alphabet(langs.auto_keys(lang)[0])


def _trial(args):
    """One encrypt-and-break round. Returns (solved, seconds, score gap)."""
    plain, key, max_key_len, freq_seeds, seed = args
    alpha = _W['alpha']
    ct = core.encrypt_text(plain, key, alpha, "vigenere")
    t0 = time.perf_counter()
    rows = solve(ct, _W['lang'], _W['models'], max_key_len=max_key_len,
                 freq_seeds=freq_seeds, seed=seed)
    dt = time.perf_counter() - t0
    ok = solved(rows, plain, alpha)
    gap = None
    want = _norm(plain, alpha)
    truth = next((r for r in rows if _norm(r['plain'], core.get_alphabet(r['alpha']))
                  == want), None)
    other = next((r for r in rows if r is not truth), None)
    if truth is not None and other is not None:
        gap = truth['final'] - other['final']
    return ok, dt, gap


def run_trials(lang, jobs, cases):
    """cases: list of (plain, key, max_key_len, freq_seeds, seed)."""
    if jobs <= 1:
        _init_lang(lang)
        return [_trial(c) for c in cases]
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=jobs, initializer=_init_lang, initargs=(lang,)) as pool:
        return pool.map(_trial, cases, chunksize=1)


# --------------------------------------------------------------------------- #
#  Experiment 1: raw throughput
# --------------------------------------------------------------------------- #
def bench_throughput(quick=False):
    print("\n[1] Throughput")
    out = {"cpu_single": None, "gpu": None, "pool": [], "gpu_by_len": []}

    M, n = 32, 30
    tri = np.zeros(M ** 3, dtype=np.float32)
    c32 = (np.arange(n) % M).astype(np.int32)

    # single core
    L, step = 4, 1 << 16
    pos = np.arange(n, dtype=np.int64) % L
    t0, done = time.perf_counter(), 0
    while time.perf_counter() - t0 < 1.0:
        attack.scan_range_numpy(c32, tri, M, L, 0, step, "vigenere", pos, batch=step)
        done += step
    out["cpu_single"] = done / (time.perf_counter() - t0)
    print("    CPU, 1 core : %12.0f keys/s" % out["cpu_single"])

    # GPU, by key length
    if attack.gpu_available():
        attack.gpu_warmup()
        for L in (4, 5, 6) if not quick else (5,):
            total = M ** L
            t0 = time.perf_counter()
            attack.gpu_scan(c32, tri, M, L, "vigenere", topk=64)
            dt = time.perf_counter() - t0
            out["gpu_by_len"].append({"L": L, "keys": total, "seconds": dt,
                                      "rate": total / dt})
            print("    GPU, L=%d    : %12.0f keys/s (%.2f s)" % (L, total / dt, dt))
        out["gpu"] = max(x["rate"] for x in out["gpu_by_len"])

    # Process-pool scaling. The pool is measured directly rather than through
    # the Engine, because the Engine's reported rate includes the several
    # seconds it takes Windows to spawn the workers, which would understate
    # the steady-state throughput.
    cores = os.cpu_count() or 8
    procs = sorted({p for p in [1, 2, 4, 8, 16, cores] if p <= cores})
    if quick:
        procs = sorted({1, max(1, cores // 4), cores})
    for p in procs:
        total, used = calibrate.measure_cpu(nproc=p, seconds=0.5)
        out["pool"].append({"procs": used, "rate": total})
        print("    pool x%-3d   : %12.0f keys/s" % (used, total))
    return out


# --------------------------------------------------------------------------- #
#  Experiment 2: success rate vs text length
# --------------------------------------------------------------------------- #
def bench_accuracy_length(quick=False, jobs=6):
    print("\n[2] Success rate vs text length (key length 6)")
    lengths = [20, 30, 40, 60, 90, 130, 200, 300]
    trials = 12 if quick else 30
    out = {}
    for lang in LANGS:
        rng = random.Random(SEED)
        mat = Material(lang, rng)
        row = []
        for n in lengths:
            cases = [(mat.text(n), mat.key(6), 10, 200, t) for t in range(trials)]
            res = run_trials(lang, jobs, cases)
            hit = sum(r[0] for r in res)
            dt = float(np.mean([r[1] for r in res]))
            row.append({"letters": n, "rate": hit / trials, "seconds": dt})
            print("    %s n=%-4d %5.0f%%  (%.2f s/attack)"
                  % (lang, n, 100 * hit / trials, dt), flush=True)
        out[lang] = row
    return {"lengths": lengths, "trials": trials, "data": out}


# --------------------------------------------------------------------------- #
#  Experiment 3: success rate vs key length
# --------------------------------------------------------------------------- #
def bench_accuracy_keylen(quick=False, jobs=6):
    print("\n[3] Success rate vs key length (text 200 letters)")
    key_lens = [2, 3, 4, 6, 8, 10, 12, 16, 20]
    trials = 10 if quick else 25
    out = {}
    for lang in LANGS:
        rng = random.Random(SEED + 1)
        mat = Material(lang, rng)
        row = []
        for L in key_lens:
            cases = [(mat.text(200), mat.key(L), max(10, L + 2), 200, t)
                     for t in range(trials)]
            res = run_trials(lang, jobs, cases)
            hit = sum(r[0] for r in res)
            row.append({"keylen": L, "rate": hit / trials})
            print("    %s L=%-3d %5.0f%%" % (lang, L, 100 * hit / trials), flush=True)
        out[lang] = row
    return {"key_lens": key_lens, "trials": trials, "data": out}


# --------------------------------------------------------------------------- #
#  Experiment 4: score separation between the true key and the best wrong one
# --------------------------------------------------------------------------- #
def bench_separation(quick=False, jobs=6):
    print("\n[4] Score separation (true key vs best competitor)")
    trials = 20 if quick else 50
    out = {}
    for lang in LANGS:
        rng = random.Random(SEED + 2)
        mat = Material(lang, rng)
        cases, lengths = [], []
        for t in range(trials):
            n = rng.choice([40, 60, 100, 160])
            lengths.append(n)
            cases.append((mat.text(n), mat.key(rng.choice([4, 5, 6, 8])), 10, 200, t))
        res = run_trials(lang, jobs, cases)
        gaps = [{"letters": n, "gap": g, "first": bool(ok)}
                for (ok, _dt, g), n in zip(res, lengths) if g is not None]
        rate = sum(g["first"] for g in gaps) / max(1, len(gaps))
        print("    %s  median gap %.1f nats, first place %.0f%% (%d samples)"
              % (lang, float(np.median([g["gap"] for g in gaps])) if gaps else 0,
                 100 * rate, len(gaps)), flush=True)
        out[lang] = gaps
    return out


# --------------------------------------------------------------------------- #
#  Experiment 5: brute-force time vs key length (measured + extrapolated)
# --------------------------------------------------------------------------- #
def bench_bruteforce(quick=False):
    print("\n[5] Brute-force cost vs key length")
    M = 32
    calib = calibrate.measure()
    cpu_rate = calibrate.rate(False, os.cpu_count() or 8, calib)
    gpu_rate = calibrate.rate(True, 1, calib) if attack.gpu_available() else None
    rows = []
    for L in range(1, 9):
        keys = M ** L
        rows.append({"L": L, "keys": keys,
                     "cpu_seconds": keys / cpu_rate,
                     "gpu_seconds": (keys / gpu_rate) if gpu_rate else None})
        print("    L=%d  %18s keys   CPU %8.1f s   GPU %8s s"
              % (L, f"{keys:,}".replace(",", " "), keys / cpu_rate,
                 ("%.1f" % (keys / gpu_rate)) if gpu_rate else "-"))
    return {"cpu_rate": cpu_rate, "gpu_rate": gpu_rate, "rows": rows}


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
    colors = {"en": "#1f77b4", "ru": "#d62728", "uk": "#2ca02c"}
    names = {"en": "English", "ru": "Russian", "uk": "Ukrainian"}

    # --- 1. throughput -----------------------------------------------------
    if "throughput" in res:
        t = res["throughput"]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.6))
        labels, vals = ["CPU\n1 core"], [t["cpu_single"]]
        if t["pool"]:
            best = max(t["pool"], key=lambda p: p["rate"])
            labels.append("CPU pool\nx%d" % best["procs"])
            vals.append(best["rate"])
        if t.get("gpu"):
            labels.append("GPU\n(CUDA)")
            vals.append(t["gpu"])
        bars = ax1.bar(labels, vals, color=["#8c8c8c", "#1f77b4", "#2ca02c"][:len(vals)])
        ax1.set_yscale("log")
        ax1.set_ylabel("keys / second")
        ax1.set_title("Brute-force throughput")
        for b, v in zip(bars, vals):
            ax1.text(b.get_x() + b.get_width() / 2, v * 1.15,
                     ("%.1e" % v).replace("e+0", "e"), ha="center", fontsize=8)
        ax1.set_ylim(top=max(vals) * 4)

        if t["pool"]:
            p = [x["procs"] for x in t["pool"]]
            r = [x["rate"] for x in t["pool"]]
            base = r[0] if r else 1
            ax2.plot(p, [x / base for x in r], "o-", color="#1f77b4", label="measured")
            ax2.plot(p, p, "--", color="#999", label="ideal linear")
            ax2.set_xlabel("worker processes")
            ax2.set_ylabel("speed-up vs 1 process")
            ax2.set_title("Parallel scaling (CPU)")
            ax2.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(DOCS, "throughput.png"))
        plt.close(fig)

    # --- 2. accuracy vs text length ---------------------------------------
    if "accuracy_len" in res:
        a = res["accuracy_len"]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.6))
        for lang, rows in a["data"].items():
            ax1.plot([r["letters"] for r in rows], [100 * r["rate"] for r in rows],
                     "o-", color=colors[lang], label=names[lang])
            ax2.plot([r["letters"] for r in rows], [r["seconds"] for r in rows],
                     "o-", color=colors[lang], label=names[lang])
        ax1.set_xlabel("ciphertext length, letters")
        ax1.set_ylabel("solved on first place, %")
        ax1.set_title("Success rate vs text length (key = 6, n=%d)" % a["trials"])
        ax1.set_ylim(-3, 103)
        ax1.legend()
        ax2.set_xlabel("ciphertext length, letters")
        ax2.set_ylabel("seconds per attack")
        ax2.set_title("Time to solve")
        ax2.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(DOCS, "accuracy_vs_length.png"))
        plt.close(fig)

    # --- 3. accuracy vs key length ----------------------------------------
    if "accuracy_keylen" in res:
        a = res["accuracy_keylen"]
        fig, ax = plt.subplots(figsize=(6, 3.6))
        for lang, rows in a["data"].items():
            ax.plot([r["keylen"] for r in rows], [100 * r["rate"] for r in rows],
                    "o-", color=colors[lang], label=names[lang])
        ax.set_xlabel("key length, letters")
        ax.set_ylabel("solved on first place, %")
        ax.set_title("Success rate vs key length (200-letter text, n=%d)" % a["trials"])
        ax.set_ylim(-3, 103)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(DOCS, "accuracy_vs_keylen.png"))
        plt.close(fig)

    # --- 4. score separation ----------------------------------------------
    if "separation" in res:
        fig, ax = plt.subplots(figsize=(6, 3.6))
        data = [[g["gap"] for g in res["separation"][l]] for l in LANGS
                if res["separation"].get(l)]
        labels = [names[l] for l in LANGS if res["separation"].get(l)]
        if data:
            bp = ax.boxplot(data, labels=labels, showfliers=False, patch_artist=True)
            for patch, l in zip(bp['boxes'], [l for l in LANGS if res["separation"].get(l)]):
                patch.set_facecolor(colors[l])
                patch.set_alpha(0.5)
        ax.axhline(0, color="#d62728", lw=1, ls="--")
        ax.set_ylabel("score of true key minus best rival, nats")
        ax.set_title("Confidence margin of the winning candidate")
        fig.tight_layout()
        fig.savefig(os.path.join(DOCS, "score_separation.png"))
        plt.close(fig)

    # --- 5. brute-force cost ----------------------------------------------
    if "bruteforce" in res:
        b = res["bruteforce"]
        fig, ax = plt.subplots(figsize=(6, 3.6))
        L = [r["L"] for r in b["rows"]]
        ax.semilogy(L, [r["cpu_seconds"] for r in b["rows"]], "o-",
                    color="#1f77b4", label="CPU pool")
        if b.get("gpu_rate"):
            ax.semilogy(L, [r["gpu_seconds"] for r in b["rows"]], "o-",
                        color="#2ca02c", label="GPU (CUDA)")
        for sec, lab in ((60, "1 minute"), (3600, "1 hour"), (86400, "1 day")):
            ax.axhline(sec, color="#bbb", lw=0.8, ls=":")
            ax.text(L[0], sec * 1.3, lab, fontsize=7, color="#888")
        ax.set_xlabel("key length, letters (32-letter alphabet)")
        ax.set_ylabel("time to enumerate the whole space, s")
        ax.set_title("Brute-force cost grows as 32^L")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(DOCS, "bruteforce_cost.png"))
        plt.close(fig)

    print("\nCharts written to %s" % DOCS)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="fewer trials")
    ap.add_argument("--only", default="", help="comma-separated experiment names")
    ap.add_argument("--charts-only", action="store_true",
                    help="rebuild charts from the saved json")
    ap.add_argument("--jobs", type=int, default=min(6, os.cpu_count() or 2),
                    help="parallel trials; each worker holds a full word set, "
                         "so keep this modest on low-memory machines")
    a = ap.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, "benchmarks.json")

    if a.charts_only:
        with open(path, encoding="utf-8") as f:
            make_charts(json.load(f))
        return 0

    missing = core.missing_languages(LANGS)
    if missing:
        print("! No language data for: %s\n  Run: python -m vigenere.download_data %s"
              % (", ".join(missing), " ".join(missing)), file=sys.stderr)

    experiments = {
        "throughput": lambda q: bench_throughput(q),
        "accuracy_len": lambda q: bench_accuracy_length(q, a.jobs),
        "accuracy_keylen": lambda q: bench_accuracy_keylen(q, a.jobs),
        "separation": lambda q: bench_separation(q, a.jobs),
        "bruteforce": lambda q: bench_bruteforce(q),
    }
    wanted = [w.strip() for w in a.only.split(",") if w.strip()] or list(experiments)

    res = {}
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                res = json.load(f)
        except Exception:                                       # noqa: BLE001
            res = {}

    res.setdefault("meta", {})
    res["meta"].update({
        "date": time.strftime("%Y-%m-%d %H:%M"),
        "cpu_count": os.cpu_count(),
        "gpu": attack.gpu_name() if attack.gpu_available() else None,
        "gpu_memory_mb": attack.gpu_total_memory_mb(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "quick": a.quick,
    })

    t_all = time.time()
    for name in wanted:
        res[name] = experiments[name](a.quick)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=1)
    print("\nTotal time: %.1f min" % ((time.time() - t_all) / 60))

    make_charts(res)
    print("Raw results: %s" % path)
    return 0


if __name__ == "__main__":
    mp.freeze_support()
    sys.exit(main())
