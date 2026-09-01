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


def fmt_pct(x):
    return "%5.1f%%" % (100.0 * x)


def run_jobs(lang, jobs, workers):
    """Execute trial jobs for one language, in a small pool."""
    if workers <= 1:
        harness.init_worker(lang)
        return [harness.trial(j) for j in jobs]
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=workers, initializer=harness.init_worker,
                  initargs=(lang,)) as pool:
        return pool.map(harness.trial, jobs, chunksize=1)


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
    print("\n=== Study 2: domain shift ===")
    splits = ["train", "validation", "test", "test_web"]
    out = {"splits": splits, "trials": args.trials, "data": {}}
    for lang in LANGS:
        km = harness.KeyMaker(lang, seed=SEED + 1)
        row = {}
        for split in splits:
            try:
                src = corpora.TextSource(split, lang, seed=SEED + 1)
            except Exception as exc:                            # noqa: BLE001
                print("    %s %s skipped (%s)" % (lang, split, exc))
                continue
            jobs = [{"plain": src.text(120), "key": km.make("random", 6),
                     "search": dict(STD_SEARCH, max_key_len=12),
                     "scorings": {"full": FULL_SCORING}, "seed": t}
                    for t in range(args.trials)]
            res = run_jobs(lang, jobs, args.jobs)
            row[split] = {"top1": top1(res), "top5": topk(res, 5)}
            print("    %s %-11s (%-28s) top1=%s" %
                  (lang, split, corpora.GENRE[split], fmt_pct(row[split]["top1"])),
                  flush=True)
        out["data"][lang] = row
    return out


# --------------------------------------------------------------------------- #
#  Study 3: dictionary keys versus random keys
# --------------------------------------------------------------------------- #
def study_keygroups(args):
    print("\n=== Study 3: key groups (text 120 letters, split=test) ===")
    key_lens = [4, 6, 8, 12]
    out = {"key_lens": key_lens, "groups": list(harness.KEY_GROUPS),
           "trials": args.trials, "data": {}}
    for lang in LANGS:
        src = corpora.TextSource("test", lang, seed=SEED + 2)
        km = harness.KeyMaker(lang, seed=SEED + 2)
        words = core.load_frequent_words(lang) or langs.builtin_words(lang)
        row = {}
        for group in harness.KEY_GROUPS:
            for L in key_lens:
                jobs = [{"plain": src.text(120), "key": km.make(group, L),
                         "search": dict(STD_SEARCH, max_key_len=max(10, L + 4),
                                        dict_attack=True,
                                        dict_words=[w for w in words if len(w) == L]),
                         "scorings": {"full": FULL_SCORING}, "seed": t}
                        for t in range(args.trials)]
                res = run_jobs(lang, jobs, args.jobs)
                row["%s,%d" % (group, L)] = {"top1": top1(res), "top5": topk(res, 5)}
                print("    %s %-7s L=%-3d top1=%s" %
                      (lang, group, L, fmt_pct(top1(res))), flush=True)
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
    print("\n=== Study 4: ablation (text 100 letters, key 6 random, split=test) ===")
    out = {"trials": args.trials, "data": {}}
    for lang in LANGS:
        src = corpora.TextSource("test", lang, seed=SEED + 3)
        km = harness.KeyMaker(lang, seed=SEED + 3)
        plains = [src.text(100) for _ in range(args.trials)]
        keys = [km.make("random", 6) for _ in range(args.trials)]
        row = {}
        for sname, scfg in ABLATIONS_SEARCH.items():
            jobs = [{"plain": plains[t], "key": keys[t], "search": scfg,
                     "scorings": ABLATIONS_SCORING, "seed": t}
                    for t in range(args.trials)]
            res = run_jobs(lang, jobs, args.jobs)
            for rname in ABLATIONS_SCORING:
                ranks = [r["ranks"][rname] for r in res]
                gaps = [r["gaps"][rname] for r in res if r["gaps"][rname] is not None]
                key = "%s + %s" % (sname, rname)
                row[key] = {
                    "top1": sum(1 for x in ranks if x == 1) / len(ranks),
                    "top5": sum(1 for x in ranks if x is not None and x <= 5) / len(ranks),
                    "median_rank": float(np.median([x if x else 99 for x in ranks])),
                    "median_gap": float(np.median(gaps)) if gaps else None,
                    "seconds": float(np.mean([r["t_search"] for r in res])),
                }
                print("    %s %-34s top1=%s top5=%s  %.2fs" %
                      (lang, key, fmt_pct(row[key]["top1"]),
                       fmt_pct(row[key]["top5"]), row[key]["seconds"]), flush=True)
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
    print("\n=== Study 6: baselines (text 120 letters, key 4 random, split=test) ===")
    out = {"trials": args.trials, "data": {}}
    for lang in LANGS:
        src = corpora.TextSource("test", lang, seed=SEED + 5)
        km = harness.KeyMaker(lang, seed=SEED + 5)
        plains = [src.text(120) for _ in range(args.trials)]
        keys = [km.make("random", 4) for _ in range(args.trials)]
        row = {}
        for name, cfg in BASELINES.items():
            jobs = [{"plain": plains[t], "key": keys[t], "search": cfg,
                     "scorings": {"full": FULL_SCORING}, "seed": t}
                    for t in range(args.trials)]
            res = run_jobs(lang, jobs, args.jobs)
            row[name] = {"top1": top1(res), "top5": topk(res, 5),
                         "seconds": float(np.mean([r["t_search"] for r in res]))}
            print("    %s %-12s top1=%s top5=%s  %.2fs" %
                  (lang, name, fmt_pct(row[name]["top1"]),
                   fmt_pct(row[name]["top5"]), row[name]["seconds"]), flush=True)
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
    print("\n=== Study 7: messy input (text 120 letters, key 6 random, split=test) ===")
    kinds = ["clean", "no_spaces", "punctuation", "typos", "digits", "latin_mix"]
    out = {"kinds": kinds, "trials": args.trials, "data": {}}
    for lang in LANGS:
        src = corpora.TextSource("test", lang, seed=SEED + 6)
        km = harness.KeyMaker(lang, seed=SEED + 6)
        alpha = core.get_alphabet(langs.auto_keys(lang)[0])
        base = [src.text(120) for _ in range(args.trials)]
        keys = [km.make("random", 6) for _ in range(args.trials)]
        row = {}
        for kind in kinds:
            rng = random.Random(SEED + 6)
            jobs = [{"plain": _corrupt(base[t], kind, rng, alpha), "key": keys[t],
                     "search": dict(STD_SEARCH, max_key_len=12),
                     "scorings": {"full": FULL_SCORING}, "seed": t}
                    for t in range(args.trials)]
            res = run_jobs(lang, jobs, args.jobs)
            row[kind] = {"top1": top1(res)}
            print("    %s %-12s top1=%s" % (lang, kind, fmt_pct(top1(res))), flush=True)
        out["data"][lang] = row
    return out


# --------------------------------------------------------------------------- #
#  Study 8: the three cipher variants
# --------------------------------------------------------------------------- #
def study_variants(args):
    print("\n=== Study 8: cipher variants (text 120 letters, key 6 random) ===")
    out = {"variants": list(core.VARIANTS), "trials": args.trials, "data": {}}
    for lang in LANGS:
        src = corpora.TextSource("test", lang, seed=SEED + 7)
        km = harness.KeyMaker(lang, seed=SEED + 7)
        plains = [src.text(120) for _ in range(args.trials)]
        keys = [km.make("random", 6) for _ in range(args.trials)]
        row = {}
        for variant in core.VARIANTS:
            jobs = [{"plain": plains[t], "key": keys[t], "variant": variant,
                     "search": dict(STD_SEARCH, max_key_len=12),
                     "scorings": {"full": FULL_SCORING}, "seed": t}
                    for t in range(args.trials)]
            res = run_jobs(lang, jobs, args.jobs)
            row[variant] = {"top1": top1(res), "top5": topk(res, 5)}
            print("    %s %-10s top1=%s" % (lang, variant, fmt_pct(top1(res))),
                  flush=True)
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
        res[name] = STUDIES[name](a)
        res[name]["elapsed_min"] = (time.time() - started) / 60.0
        with open(path, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=1, ensure_ascii=False)
        print("    [%s done in %.1f min]" % (name, res[name]["elapsed_min"]), flush=True)
    print("\nTotal: %.1f min -> %s" % ((time.time() - t0) / 60.0, path))
    return 0


if __name__ == "__main__":
    mp.freeze_support()
    sys.exit(main())
