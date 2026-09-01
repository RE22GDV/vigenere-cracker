# -*- coding: utf-8 -*-
"""Evaluation harness: key groups, attack variants, configurable scoring.

Two things matter for honest measurement:

* **Key groups are separated.** A key drawn from the word list that the program
  itself ships is found by the dictionary attack alone and says nothing about
  the cryptanalysis. Random keys say the opposite. They are never mixed into
  one number.
* **Search and ranking are decoupled.** A trial runs the search once and caches
  the candidate pool, so any number of scoring variants can be compared on
  exactly the same candidates. That makes the ablation study cheap and fair.
"""

from __future__ import annotations

import math
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vigenere import attack, core, langs                        # noqa: E402
import corpora                                                  # noqa: E402

KEY_GROUPS = ("dict", "oov", "random", "repeat")

KEY_GROUP_DESC = {
    "dict": "a word from the frequent-word list the program ships",
    "oov": "a real word absent from the program dictionary (names, rare forms)",
    "random": "uniformly random letters",
    "repeat": "structured, a short pattern padded out",
}


class KeyMaker:
    """Produces keys of a requested length from four disjoint groups."""

    def __init__(self, lang, seed=0, oov_source="test"):
        self.lang = lang
        self.rng = random.Random(seed)
        self.alpha = core.get_alphabet(langs.auto_keys(lang)[0])
        self.letters = self.alpha.letters

        known = core.load_frequent_words(lang) or langs.builtin_words(lang)
        self.dict_by_len = {}
        for w in known:
            if all(c in self.alpha.index for c in w):
                self.dict_by_len.setdefault(len(w), []).append(w)

        # Out-of-vocabulary: real tokens from a held-out corpus that the full
        # dictionary does not contain - names, rare inflections, foreign words.
        full = set(core.get_model(self.alpha.key).words)
        src = corpora.TextSource(oov_source, lang, seed=seed + 7)
        self.oov_by_len = {}
        for w in src.tokens():
            if self.alpha.fold(w) not in full:
                self.oov_by_len.setdefault(len(w), []).append(w)

    def make(self, group, length):
        if group == "dict":
            pool = self.dict_by_len.get(length)
            return self.rng.choice(pool) if pool else self._random(length)
        if group == "oov":
            pool = self.oov_by_len.get(length)
            return self.rng.choice(pool) if pool else self._random(length)
        if group == "repeat":
            # A repeating pattern with the tail perturbed, so the key is
            # structured but does not collapse to a shorter minimal period.
            unit = max(2, length // 3)
            base = self._random(unit)
            key = list((base * (length // unit + 1))[:length])
            if length > unit:
                key[-1] = self.letters[self.rng.randrange(len(self.letters))]
            return "".join(key)
        return self._random(length)

    def _random(self, length):
        return "".join(self.letters[self.rng.randrange(len(self.letters))]
                       for _ in range(length))

    def available(self, group, length):
        if group == "dict":
            return bool(self.dict_by_len.get(length))
        if group == "oov":
            return bool(self.oov_by_len.get(length))
        return True


def freq_no_polish(c32, model, alpha, L, variant="vigenere"):
    """Pure frequency analysis: best letter per column, no local search."""
    if len(c32) < 2 * L:
        return []
    exp = model.letter_freq()
    key = []
    for i in range(L):
        corr = core.column_correlation(c32[i::L], alpha.M, variant, exp)
        key.append(int(np.argmax(corr)))
    kt = core.minimal_period(tuple(key))
    kexp = np.array(kt, dtype=np.int64)[np.arange(len(c32)) % len(kt)]
    p = core.decrypt_indices(c32.astype(np.int64), kexp, alpha.M, variant)
    return [(model.score_text(p), kt)]


def hill_only(c32, model, alpha, L, restarts, seed, variant="vigenere"):
    """Hill climbing from random starts, with no frequency seeding."""
    rng = random.Random(seed)
    M, n = alpha.M, len(c32)
    if n < 3:
        return []
    pos = np.arange(n, dtype=np.int64) % L
    tri = model.tri
    out = []
    for _ in range(restarts):
        key = np.array([rng.randrange(M) for _ in range(L)], dtype=np.int32)
        best = float(attack._score_matrix(key[None, :], c32, tri, M, variant, pos)[0])
        improved = True
        while improved:
            improved = False
            for p in range(L):
                K = np.tile(key, (M, 1))
                K[:, p] = np.arange(M, dtype=np.int32)
                sc = attack._score_matrix(K, c32, tri, M, variant, pos)
                j = int(np.argmax(sc))
                if float(sc[j]) > best + 1e-9:
                    best, key, improved = float(sc[j]), K[j].copy(), True
        out.append((best, core.minimal_period(tuple(int(v) for v in key))))
    out.sort(reverse=True)
    return out[:20]


def search(ciphertext, lang, models, cfg, seed=0, variant="vigenere"):
    """Run the configured attacks; return {alphabet key: [(score, key), ...]}."""
    pools = {}
    for akey in langs.auto_keys(lang):
        alpha = core.get_alphabet(akey)
        model = models[akey]
        idx, _ = alpha.layout(ciphertext)
        c32 = idx.astype(np.int32)
        n = len(idx)
        cands = []
        maxL = min(cfg.get("max_key_len", 16), max(1, n // 2))

        if cfg.get("freq", True):
            for L in range(1, maxL + 1):
                if n < 2 * L:
                    break
                if cfg.get("polish", True):
                    cands += attack.frequency_attack(
                        c32, model.tri, alpha.M, variant, model.letter_freq(), L,
                        n_seeds=cfg.get("freq_seeds", 200), seed=seed + L)
                else:
                    cands += freq_no_polish(c32, model, alpha, L, variant)

        if cfg.get("hill_restarts", 0):
            for L in range(1, maxL + 1):
                cands += hill_only(c32, model, alpha, L,
                                   cfg["hill_restarts"], seed + 100 + L, variant)

        for L in range(1, cfg.get("brute_max", 0) + 1):
            pos = np.arange(n, dtype=np.int64) % L
            res = attack.scan_range_numpy(c32, model.tri, alpha.M, L, 0,
                                          alpha.M ** L, variant, pos, topk=128)
            cands += [(s, tuple(int(v) for v in alpha.key_from_int(k, L)))
                      for s, k in res]

        if cfg.get("dict_attack"):
            for w in cfg.get("dict_words") or []:
                k = tuple(int(v) for v in alpha.to_indices(w))
                if 1 <= len(k) <= maxL:
                    kexp = np.array(k, dtype=np.int64)[np.arange(n) % len(k)]
                    p = core.decrypt_indices(idx, kexp, alpha.M, variant)
                    cands.append((model.score_text(p), k))

        pools[akey] = cands
    return pools


def rank_pool(pools, ciphertext, models, variant="vigenere", use_quad=True,
              w_word=attack.W_WORD, penalty_scale=1.0, limit=10, cov_limit=400):
    """Rank a cached candidate pool under a chosen scoring configuration."""
    rows = []
    for akey, cands in pools.items():
        if not cands:
            continue
        alpha = core.get_alphabet(akey)
        model = models[akey]
        idx, tmpl = alpha.layout(ciphertext)
        n = max(1, len(idx))
        ar = np.arange(len(idx))
        pen = math.log(alpha.M) * penalty_scale
        top = sorted(cands, reverse=True)[:2000]

        prelim, seen = [], set()
        for _raw, key in top:
            if key in seen:
                continue
            seen.add(key)
            kexp = np.array(key, dtype=np.int64)[ar % len(key)]
            p = core.decrypt_indices(idx, kexp, alpha.M, variant)
            base = model.score_text4(p) if use_quad else model.score_text(p)
            prelim.append((base - pen * len(key), key, p))
        prelim.sort(key=lambda t: -t[0])

        for i, (base, key, p) in enumerate(prelim[:max(cov_limit, limit)]):
            cov = 0.0
            if w_word and i < cov_limit:
                cov, _ = model.word_coverage("".join(alpha.letters[j] for j in p))
            rows.append({"key": alpha.key_to_str(key), "alpha": akey,
                         "final": base + w_word * cov * n,
                         "plain": alpha.render(p, tmpl)})
    rows.sort(key=lambda r: -r["final"])
    return rows[:limit]


def norm(text, alpha):
    return "".join(c for c in alpha.fold(text.lower()) if c in alpha.index)


def rank_of_truth(rows, plain, alpha):
    want = norm(plain, alpha)
    for i, r in enumerate(rows, 1):
        if norm(r["plain"], core.get_alphabet(r["alpha"])) == want:
            return i
    return None


_W = {}


def init_worker(lang):
    _W["lang"] = lang
    _W["models"] = {k: core.get_model(k) for k in langs.auto_keys(lang)}
    _W["alpha"] = core.get_alphabet(langs.auto_keys(lang)[0])


def trial(job):
    """One encrypt-and-break round, scored under several configurations."""
    alpha = _W["alpha"]
    models = _W["models"]
    variant = job.get("variant", "vigenere")
    ct = core.encrypt_text(job["plain"], job["key"], alpha, variant)
    t0 = time.perf_counter()
    pools = search(ct, _W["lang"], models, job["search"], seed=job.get("seed", 0),
                   variant=variant)
    t_search = time.perf_counter() - t0

    out = {"t_search": t_search, "ranks": {}, "gaps": {}}
    want = norm(job["plain"], alpha)
    for name, sc in job["scorings"].items():
        rows = rank_pool(pools, ct, models, variant=variant, **sc)
        out["ranks"][name] = rank_of_truth(rows, job["plain"], alpha)
        truth = next((x for x in rows
                      if norm(x["plain"], core.get_alphabet(x["alpha"])) == want), None)
        other = next((x for x in rows if x is not truth), None)
        out["gaps"][name] = (truth["final"] - other["final"]
                             if truth is not None and other is not None else None)
    return out
