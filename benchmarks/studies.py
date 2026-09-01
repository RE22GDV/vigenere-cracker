# -*- coding: utf-8 -*-
"""Research studies.

Data discipline used throughout:

    train       Tatoeba, conversational   builds the model - never evaluated on
    validation  Wikipedia, encyclopedic   tunes weights - never reported as a result
    test        news, journalistic        final numbers only, touched once

Key groups are always reported separately, because a key that is a dictionary
word is found by the dictionary attack and tells nothing about cryptanalysis.

Usage:
  python benchmarks/studies.py --list
  python benchmarks/studies.py --only matrix,keygroups --trials 30
  python benchmarks/studies.py --all --trials 100        (long)
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import random
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

import corpora                                                  # noqa: E402
import harness                                                  # noqa: E402
from vigenere import attack, core, langs                        # noqa: E402

RESULTS = os.path.join(HERE, "results")
DOCS = os.path.join(REPO, "docs")
LANGS = ["en", "ru", "uk"]
SEED = 20260901

# The scoring configuration the shipped program uses.
FULL_SCORING = {"use_quad": True, "w_word": attack.W_WORD, "penalty_scale": 1.0}
# The search configuration used wherever a study is not ablating the search.
STD_SEARCH = {"freq": True, "polish": True, "freq_seeds": 200, "max_key_len": 28}


# Operating points, given as (ciphertext letters, key length).
#
# The first round of these studies was run at a single easy point (120 letters,
# 6-letter key). The matrix study later showed that N/L >= 10 saturates at
# 100 % for every configuration, so those runs could not discriminate between
# anything. The hard set below sits in the 25-80 % band, where a component that
# helps or hurts actually shows up.
EASY_POINTS = [(120, 6)]
HARD_POINTS = [(30, 6), (40, 8), (60, 12), (100, 20)]      # N/L = 5

# The key-group study needs every group to have genuine keys of the length
# under test. Dictionary words of 16+ letters barely exist (0-31 per language),
# so those lengths would silently fall back to random keys and destroy the very
# comparison the study is for. Its points stop at 12 letters.
KEYGROUP_POINTS = [(30, 6), (40, 8), (60, 12)]


def points(args):
    return HARD_POINTS if getattr(args, "hard", False) else EASY_POINTS


def fmt_pct(x):
    return "%5.1f%%" % (100.0 * x)


def wilson(k, n, z=1.96):
    """95 % Wilson interval - a proportion without an honest interval is noise."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def summarise(results, scoring="full"):
    """Success rate with its confidence interval."""
    n = len(results)
    k = sum(1 for r in results if r["ranks"].get(scoring) == 1)
    lo, hi = wilson(k, n)
    return {"top1": k / max(1, n), "ci_lo": lo, "ci_hi": hi, "n": n,
            "top5": topk(results, 5, scoring),
            "seconds": float(np.mean([r["t_search"] for r in results])) if n else 0.0}


def fmt_ci(s):
    return "%5.1f%% [%4.1f-%4.1f]" % (100 * s["top1"], 100 * s["ci_lo"],
                                      100 * s["ci_hi"])


_POOL = {"lang": None, "pool": None, "workers": 0}


def _get_pool(lang, workers):
    """One pool per language, reused across every cell of a study.

    Each worker loads the language models on startup, which for Russian and
    Ukrainian means word sets of millions of entries. Creating a fresh pool per
    measurement point cost far more than the measurement itself.
    """
    if _POOL["pool"] is not None and _POOL["lang"] == lang \
            and _POOL["workers"] == workers:
        return _POOL["pool"]
    close_pool()
    ctx = mp.get_context("spawn")
    _POOL.update(lang=lang, workers=workers,
                 pool=ctx.Pool(processes=workers, initializer=harness.init_worker,
                               initargs=(lang,)))
    return _POOL["pool"]


def close_pool():
    if _POOL["pool"] is not None:
        try:
            _POOL["pool"].close()
            _POOL["pool"].join()
        except Exception:                                       # noqa: BLE001
            pass
    _POOL.update(lang=None, pool=None, workers=0)


def run_jobs(lang, jobs, workers):
    """Execute trial jobs for one language."""
    if workers <= 1:
        if _POOL["lang"] != lang:
            harness.init_worker(lang)
            _POOL["lang"] = lang
        return [harness.trial(j) for j in jobs]
    return _get_pool(lang, workers).map(harness.trial, jobs, chunksize=1)


def top1(results, scoring="full"):
    return sum(1 for r in results if r["ranks"].get(scoring) == 1) / max(1, len(results))


def topk(results, k, scoring="full"):
    return sum(1 for r in results
               if r["ranks"].get(scoring) is not None
               and r["ranks"][scoring] <= k) / max(1, len(results))


# --------------------------------------------------------------------------- #
#  Study 1: accuracy over the text-length x key-length matrix
# --------------------------------------------------------------------------- #
def study_matrix(args):
    print("\n=== Study 1: accuracy matrix, text length x key length ===")
    print("    split=test (news, held out), keys=random (hardest, no dictionary help)")
    lengths = [20, 30, 40, 60, 100, 200, 400]
    key_lens = [2, 4, 6, 8, 12, 16, 20, 25]
    out = {"lengths": lengths, "key_lens": key_lens, "trials": args.trials, "data": {}}
    for lang in LANGS:
        src = corpora.TextSource("test", lang, seed=SEED)
        km = harness.KeyMaker(lang, seed=SEED)
        grid = {}
        for n in lengths:
            for L in key_lens:
                if L * 2 > n:                       # fewer than 2 letters per column
                    grid["%d,%d" % (n, L)] = None
                    continue
                jobs = [{"plain": src.text(n), "key": km.make("random", L),
                         "search": dict(STD_SEARCH, max_key_len=max(10, L + 4)),
                         "scorings": {"full": FULL_SCORING}, "seed": t}
                        for t in range(args.trials)]
                res = run_jobs(lang, jobs, args.jobs)
                rate = top1(res)
                grid["%d,%d" % (n, L)] = {"top1": rate, "top5": topk(res, 5),
                                          "ratio": n / L,
                                          "seconds": float(np.mean([r["t_search"]
                                                                    for r in res]))}
                print("    %s n=%-4d L=%-3d N/L=%5.1f  top1=%s" %
                      (lang, n, L, n / L, fmt_pct(rate)), flush=True)
        out["data"][lang] = grid
    return out


# --------------------------------------------------------------------------- #
#  Study 2: does it recognise the language, or just its training corpus?
# --------------------------------------------------------------------------- #
def study_domain(args):
    pts = points(args)
    print("\n=== Study 2: domain shift (points %s) ===" % (pts,))
    splits = ["train", "validation", "test", "test_web"]
    out = {"splits": splits, "points": pts, "trials": args.trials, "data": {}}
    for lang in LANGS:
        km = harness.KeyMaker(lang, seed=SEED + 1)
        row = {}
        for split in splits:
            try:
                src = corpora.TextSource(split, lang, seed=SEED + 1)
            except Exception as exc:                            # noqa: BLE001
                print("    %s %s skipped (%s)" % (lang, split, exc))
                continue
            pooled = []
            for (n, L) in pts:
                jobs = [{"plain": src.text(n), "key": km.make("random", L),
                         "search": dict(STD_SEARCH, max_key_len=max(10, L + 4)),
                         "scorings": {"full": FULL_SCORING}, "seed": t}
                        for t in range(args.trials)]
                pooled += run_jobs(lang, jobs, args.jobs)
            row[split] = summarise(pooled)
            print("    %s %-11s (%-28s) %s" %
                  (lang, split, corpora.GENRE[split], fmt_ci(row[split])), flush=True)
        out["data"][lang] = row
    return out


# --------------------------------------------------------------------------- #
#  Study 3: dictionary keys versus random keys
# --------------------------------------------------------------------------- #
def study_keygroups(args):
    pts = KEYGROUP_POINTS if getattr(args, "hard", False) else EASY_POINTS
    print("\n=== Study 3: key groups (split=test, points %s) ===" % (pts,))
    out = {"points": pts, "groups": list(harness.KEY_GROUPS),
           "trials": args.trials, "data": {}, "skipped": []}
    for lang in LANGS:
        src = corpora.TextSource("test", lang, seed=SEED + 2)
        km = harness.KeyMaker(lang, seed=SEED + 2)
        words = core.load_frequent_words(lang) or langs.builtin_words(lang)
        row = {}
        for group in harness.KEY_GROUPS:
            pooled = []
            for (n, L) in pts:
                if not km.available(group, L):
                    # Never let a missing pool turn a dictionary key into a
                    # random one - that would silently fake the comparison.
                    out["skipped"].append([lang, group, n, L])
                    print("    %s %-7s L=%-3d skipped: no real keys of this length"
                          % (lang, group, L), flush=True)
                    continue
                jobs = [{"plain": src.text(n), "key": km.make(group, L),
                         "search": dict(STD_SEARCH, max_key_len=max(10, L + 4),
                                        dict_attack=True,
                                        dict_words=[w for w in words if len(w) == L]),
                         "scorings": {"full": FULL_SCORING}, "seed": t}
                        for t in range(args.trials)]
                res = run_jobs(lang, jobs, args.jobs)
                pooled += res
                row["%s,%d,%d" % (group, n, L)] = summarise(res)
            row[group] = summarise(pooled)
            print("    %s %-7s pooled %s" % (lang, group, fmt_ci(row[group])),
                  flush=True)
        out["data"][lang] = row
    return out


# --------------------------------------------------------------------------- #
#  Study 4: ablation
# --------------------------------------------------------------------------- #
ABLATIONS_SEARCH = {
    "freq_only": {"freq": True, "polish": False, "max_key_len": 16},
    "freq_hill": {"freq": True, "polish": True, "freq_seeds": 200, "max_key_len": 16},
}
ABLATIONS_SCORING = {
    "trigram_no_dict": {"use_quad": False, "w_word": 0.0, "penalty_scale": 1.0},
    "quadgram_no_dict": {"use_quad": True, "w_word": 0.0, "penalty_scale": 1.0},
    "quadgram_dict": {"use_quad": True, "w_word": attack.W_WORD, "penalty_scale": 1.0},
    "no_penalty": {"use_quad": True, "w_word": attack.W_WORD, "penalty_scale": 0.0},
}


def study_ablation(args):
    pts = points(args)
    print("\n=== Study 4: ablation (split=test, points %s) ===" % (pts,))
    out = {"points": pts, "trials": args.trials, "data": {}}
    for lang in LANGS:
        src = corpora.TextSource("test", lang, seed=SEED + 3)
        km = harness.KeyMaker(lang, seed=SEED + 3)
        row = {}
        for sname, scfg in ABLATIONS_SEARCH.items():
            pooled = []
            for (n, L) in pts:
                jobs = [{"plain": src.text(n), "key": km.make("random", L),
                         "search": dict(scfg, max_key_len=max(10, L + 4)),
                         "scorings": ABLATIONS_SCORING, "seed": t}
                        for t in range(args.trials)]
                pooled += run_jobs(lang, jobs, args.jobs)
            for rname in ABLATIONS_SCORING:
                ranks = [r["ranks"][rname] for r in pooled]
                gaps = [r["gaps"][rname] for r in pooled
                        if r["gaps"][rname] is not None]
                k = sum(1 for x in ranks if x == 1)
                lo, hi = wilson(k, len(ranks))
                key = "%s + %s" % (sname, rname)
                row[key] = {
                    "top1": k / len(ranks), "ci_lo": lo, "ci_hi": hi,
                    "n": len(ranks),
                    "top5": sum(1 for x in ranks
                                if x is not None and x <= 5) / len(ranks),
                    "median_rank": float(np.median([x if x else 99 for x in ranks])),
                    "median_gap": float(np.median(gaps)) if gaps else None,
                    "seconds": float(np.mean([r["t_search"] for r in pooled])),
                }
                print("    %s %-34s %s  %.2fs" %
                      (lang, key, fmt_ci(row[key]), row[key]["seconds"]), flush=True)
        out["data"][lang] = row
    return out


# --------------------------------------------------------------------------- #
#  Study 5: weight selection on validation, confirmed once on test
# --------------------------------------------------------------------------- #
def study_weights(args):
    print("\n=== Study 5: weight selection (tuned on validation, checked on test) ===")
    w_grid = [0.0, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 15.0]
    p_grid = [0.0, 0.5, 1.0, 1.5, 2.0]
    scorings = {}
    for w in w_grid:
        scorings["w=%g" % w] = {"use_quad": True, "w_word": w, "penalty_scale": 1.0}
    for p in p_grid:
        scorings["p=%g" % p] = {"use_quad": True, "w_word": attack.W_WORD,
                                "penalty_scale": p}
    out = {"w_grid": w_grid, "p_grid": p_grid, "trials": args.trials, "data": {}}
    for split in ("validation", "test"):
        per_lang = {}
        for lang in LANGS:
            src = corpora.TextSource(split, lang, seed=SEED + 4)
            km = harness.KeyMaker(lang, seed=SEED + 4)
            jobs = []
            for t in range(args.trials):
                n = random.Random(SEED + t).choice([40, 60, 100, 160])
                L = random.Random(SEED * 2 + t).choice([4, 6, 8, 12])
                jobs.append({"plain": src.text(n), "key": km.make("random", L),
                             "search": dict(STD_SEARCH, max_key_len=16),
                             "scorings": scorings, "seed": t})
            res = run_jobs(lang, jobs, args.jobs)
            per_lang[lang] = {name: top1(res, name) for name in scorings}
            print("    %s %-11s best w=%s" %
                  (split, lang,
                   max((v, k) for k, v in per_lang[lang].items()
                       if k.startswith("w="))[1]), flush=True)
        out["data"][split] = per_lang
    return out


# --------------------------------------------------------------------------- #
#  Study 6: comparison with baseline methods
# --------------------------------------------------------------------------- #
BASELINES = {
    "ic_chi2": {"freq": True, "polish": False, "max_key_len": 16},
    "hill_only": {"freq": False, "hill_restarts": 20, "max_key_len": 16},
    "brute_force": {"freq": False, "brute_max": 4, "max_key_len": 4},
    "full_system": dict(STD_SEARCH, max_key_len=16),
}


def study_baselines(args):
    pts = points(args)
    print("\n=== Study 6: baselines (split=test, points %s) ===" % (pts,))
    out = {"points": pts, "trials": args.trials, "data": {}}
    for lang in LANGS:
        src = corpora.TextSource("test", lang, seed=SEED + 5)
        km = harness.KeyMaker(lang, seed=SEED + 5)
        row = {}
        for name, cfg in BASELINES.items():
            pooled, skipped = [], []
            for (n, L) in pts:
                # Exhaustive search is only defined for short keys: 32^8 keys
                # is a trillion times too many. That limitation is the result.
                if name == "brute_force" and L > 4:
                    skipped.append((n, L))
                    continue
                jobs = [{"plain": src.text(n), "key": km.make("random", L),
                         "search": dict(cfg, max_key_len=max(10, L + 4)),
                         "scorings": {"full": FULL_SCORING}, "seed": t}
                        for t in range(args.trials)]
                pooled += run_jobs(lang, jobs, args.jobs)
            row[name] = summarise(pooled) if pooled else {
                "top1": 0.0, "ci_lo": 0.0, "ci_hi": 0.0, "n": 0, "top5": 0.0,
                "seconds": 0.0}
            row[name]["not_applicable"] = skipped
            print("    %s %-12s %s  %.2fs%s" %
                  (lang, name, fmt_ci(row[name]), row[name]["seconds"],
                   "  (n/a for %d of %d points)" % (len(skipped), len(pts))
                   if skipped else ""), flush=True)
        out["data"][lang] = row
    return out


# --------------------------------------------------------------------------- #
#  Study 7: robustness to messy input
# --------------------------------------------------------------------------- #
def _corrupt(text, kind, rng, alpha):
    letters = alpha.letters
    if kind == "clean":
        return text
    if kind == "no_spaces":
        return text.replace(" ", "")
    if kind == "punctuation":
        return "".join(ch + (rng.choice("!?,.;:-") if rng.random() < 0.15 else "")
                       for ch in text)
    if kind == "typos":
        out = list(text)
        for i, ch in enumerate(out):
            if ch.lower() in alpha.index and rng.random() < 0.05:
                out[i] = letters[rng.randrange(len(letters))]
        return "".join(out)
    if kind == "digits":
        return " ".join(w + (str(rng.randrange(1000)) if rng.random() < 0.2 else "")
                        for w in text.split())
    if kind == "latin_mix":
        words = text.split()
        for i in range(len(words)):
            if rng.random() < 0.12:
                words[i] = "".join(rng.choice("abcdefghijklmnopqrstuvwxyz")
                                   for _ in range(len(words[i])))
        return " ".join(words)
    return text


def study_dirty(args):
    pts = points(args)[:2]          # two points are enough for six corruptions
    print("\n=== Study 7: messy input (split=test, points %s) ===" % (pts,))
    kinds = ["clean", "no_spaces", "punctuation", "typos", "digits", "latin_mix"]
    out = {"kinds": kinds, "points": pts, "trials": args.trials, "data": {}}
    for lang in LANGS:
        src = corpora.TextSource("test", lang, seed=SEED + 6)
        km = harness.KeyMaker(lang, seed=SEED + 6)
        alpha = core.get_alphabet(langs.auto_keys(lang)[0])
        # The same plaintexts and keys for every corruption, so the only
        # difference between rows is the corruption itself.
        base = {n: [src.text(n) for _ in range(args.trials)] for n, _ in pts}
        keys = {L: [km.make("random", L) for _ in range(args.trials)] for _, L in pts}
        row = {}
        for kind in kinds:
            pooled = []
            for (n, L) in pts:
                rng = random.Random(SEED + 6)
                jobs = [{"plain": _corrupt(base[n][t], kind, rng, alpha),
                         "key": keys[L][t],
                         "search": dict(STD_SEARCH, max_key_len=max(10, L + 4)),
                         "scorings": {"full": FULL_SCORING}, "seed": t}
                        for t in range(args.trials)]
                pooled += run_jobs(lang, jobs, args.jobs)
            row[kind] = summarise(pooled)
            print("    %s %-12s %s" % (lang, kind, fmt_ci(row[kind])), flush=True)
        out["data"][lang] = row
    return out


# --------------------------------------------------------------------------- #
#  Study 8: the three cipher variants
# --------------------------------------------------------------------------- #
def study_variants(args):
    pts = points(args)
    print("\n=== Study 8: cipher variants (split=test, points %s) ===" % (pts,))
    out = {"variants": list(core.VARIANTS), "points": pts,
           "trials": args.trials, "data": {}}
    for lang in LANGS:
        src = corpora.TextSource("test", lang, seed=SEED + 7)
        km = harness.KeyMaker(lang, seed=SEED + 7)
        # Identical material across variants, so any difference is the cipher.
        base = {n: [src.text(n) for _ in range(args.trials)] for n, _ in pts}
        keys = {L: [km.make("random", L) for _ in range(args.trials)] for _, L in pts}
        row = {}
        for variant in core.VARIANTS:
            pooled = []
            for (n, L) in pts:
                jobs = [{"plain": base[n][t], "key": keys[L][t], "variant": variant,
                         "search": dict(STD_SEARCH, max_key_len=max(10, L + 4)),
                         "scorings": {"full": FULL_SCORING}, "seed": t}
                        for t in range(args.trials)]
                pooled += run_jobs(lang, jobs, args.jobs)
            row[variant] = summarise(pooled)
            print("    %s %-10s %s" % (lang, variant, fmt_ci(row[variant])), flush=True)
        out["data"][lang] = row
    return out


# --------------------------------------------------------------------------- #
#  Study 9: how well is the key length identified?
# --------------------------------------------------------------------------- #
def study_keylen_detection(args):
    print("\n=== Study 9: key-length identification (IC and Kasiski) ===")
    lengths = [60, 120, 250, 500]
    key_lens = [3, 4, 5, 6, 8, 12]
    trials = max(20, args.trials)
    out = {"lengths": lengths, "key_lens": key_lens, "trials": trials, "data": {}}
    for lang in LANGS:
        src = corpora.TextSource("test", lang, seed=SEED + 8)
        km = harness.KeyMaker(lang, seed=SEED + 8)
        alpha = core.get_alphabet(langs.auto_keys(lang)[0])
        row = {}
        for n in lengths:
            stats = {"ic_top1": 0, "ic_top3": 0, "kas_top1": 0, "kas_top3": 0,
                     "ic_multiple": 0, "total": 0}
            for L in key_lens:
                for t in range(trials):
                    plain = src.text(n)
                    key = km.make("random", L)
                    ct = core.encrypt_text(plain, key, alpha, "vigenere")
                    c = alpha.layout(ct)[0]
                    hints = core.key_length_hints(c, alpha.M, min(20, max(2, n // 2)))
                    order = [x[0] for x in sorted(hints, key=lambda h: -h[1])]
                    kas = [x[0] for x in core.kasiski(c, max_len=min(20, n // 3))]
                    stats["total"] += 1
                    if order[:1] == [L]:
                        stats["ic_top1"] += 1
                    if L in order[:3]:
                        stats["ic_top3"] += 1
                    if order and order[0] != L and order[0] % L == 0:
                        stats["ic_multiple"] += 1
                    if kas[:1] == [L]:
                        stats["kas_top1"] += 1
                    if L in kas[:3]:
                        stats["kas_top3"] += 1
            tot = max(1, stats["total"])
            row[str(n)] = {k: v / tot for k, v in stats.items() if k != "total"}
            row[str(n)]["total"] = stats["total"]
            print("    %s n=%-4d IC top1=%s top3=%s | Kasiski top1=%s top3=%s | "
                  "IC picked a multiple: %s" %
                  (lang, n, fmt_pct(row[str(n)]["ic_top1"]),
                   fmt_pct(row[str(n)]["ic_top3"]), fmt_pct(row[str(n)]["kas_top1"]),
                   fmt_pct(row[str(n)]["kas_top3"]),
                   fmt_pct(row[str(n)]["ic_multiple"])), flush=True)
        out["data"][lang] = row
    return out


STUDIES = {
    "matrix": study_matrix,
    "domain": study_domain,
    "keygroups": study_keygroups,
    "ablation": study_ablation,
    "weights": study_weights,
    "baselines": study_baselines,
    "dirty": study_dirty,
    "variants": study_variants,
    "keylen": study_keylen_detection,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="comma-separated study names")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--jobs", type=int, default=min(6, os.cpu_count() or 2))
    ap.add_argument("--hard", action="store_true",
                    help="use the hard operating points (N/L = 5) instead of one "
                         "easy point; without this the studies saturate at 100 %%")
    ap.add_argument("--suffix", default="",
                    help="store results under '<study><suffix>' so an easy and a "
                         "hard run can live side by side")
    a = ap.parse_args()

    if a.list:
        for k in STUDIES:
            print(" ", k)
        return 0

    names = [s.strip() for s in a.only.split(",") if s.strip()]
    if a.all or not names:
        names = list(STUDIES)

    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, "studies.json")
    res = {}
    if os.path.exists(path):
        try:
            res = json.load(open(path, encoding="utf-8"))
        except Exception:                                       # noqa: BLE001
            res = {}
    res.setdefault("meta", {}).update({
        "date": time.strftime("%Y-%m-%d %H:%M"), "trials": a.trials,
        "cpu_count": os.cpu_count(),
        "gpu": attack.gpu_name() if attack.gpu_available() else None,
        "splits": {k: corpora.GENRE[k] for k in corpora.GENRE},
    })

    t0 = time.time()
    for name in names:
        started = time.time()
        try:
            out = STUDIES[name](a)
        finally:
            close_pool()
        out["elapsed_min"] = (time.time() - started) / 60.0
        out["hard"] = bool(a.hard)
        res[name + a.suffix] = out
        # Each study is also written to its own file. One run finished a study
        # and still lost it from the shared file, and six minutes of compute is
        # not worth risking to a write race between concurrent invocations.
        with open(os.path.join(RESULTS, "study_%s%s.json" % (name, a.suffix)),
                  "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1, ensure_ascii=False)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=1, ensure_ascii=False)
        print("    [%s%s done in %.1f min]" % (name, a.suffix, out["elapsed_min"]),
              flush=True)
    print("\nTotal: %.1f min -> %s" % ((time.time() - t0) / 60.0, path))
    return 0


if __name__ == "__main__":
    mp.freeze_support()
    sys.exit(main())
