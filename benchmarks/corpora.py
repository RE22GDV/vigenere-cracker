# -*- coding: utf-8 -*-
"""Held-out corpora for evaluation.

The language model and the dictionary are built from Tatoeba (the *train*
split). Evaluating on Tatoeba sentences would therefore measure how well the
model memorised its own training material, not how well it recognises the
language. So evaluation uses two further splits, from a different provider and
different genres:

    train       Tatoeba            conversational   builds the n-gram model
    validation  Leipzig Wikipedia  encyclopedic     tunes weights, never reported
    test        Leipzig News       journalistic     final numbers only

Nothing from the test split is allowed to influence any parameter.
"""

from __future__ import annotations

import io
import os
import random
import re
import sys
import tarfile
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vigenere import langs                                      # noqa: E402
from vigenere.paths import DATA_DIR                             # noqa: E402

EVAL_DIR = os.path.join(DATA_DIR, "eval")
LEIPZIG = "https://downloads.wortschatz-leipzig.de/corpora/"

# split -> language -> Leipzig package name
PACKAGES = {
    "validation": {
        "en": "eng_wikipedia_2016_30K",
        "ru": "rus_wikipedia_2021_30K",
        "uk": "ukr_wikipedia_2021_30K",
    },
    "test": {
        "en": "eng_news_2023_30K",
        "ru": "rus_news_2023_30K",
        "uk": "ukr_news_2023_30K",
    },
    # a third genre, used only by the domain-shift study
    "test_web": {
        "en": "eng-com_web-public_2018_30K",
        "ru": "rus_news_2022_30K",
        "uk": "ukr_news_2022_30K",
    },
}

GENRE = {"train": "conversational (Tatoeba)",
         "validation": "encyclopedic (Wikipedia)",
         "test": "journalistic (news)",
         "test_web": "web / second news slice"}


def _package_path(name):
    return os.path.join(EVAL_DIR, name + ".tar.gz")


def _sentences_path(name):
    return os.path.join(EVAL_DIR, name + "-sentences.txt")


def download(split, lang, quiet=False):
    """Fetch and unpack one Leipzig package; returns the sentences file path."""
    name = PACKAGES[split][lang]
    out = _sentences_path(name)
    if os.path.exists(out):
        return out
    os.makedirs(EVAL_DIR, exist_ok=True)
    arch = _package_path(name)
    if not os.path.exists(arch):
        if not quiet:
            print("  downloading %s ..." % name, flush=True)
        req = urllib.request.Request(LEIPZIG + name + ".tar.gz",
                                     headers={"User-Agent": "vigenere-cracker"})
        with urllib.request.urlopen(req, timeout=300) as r, open(arch, "wb") as f:
            f.write(r.read())
    with tarfile.open(arch, "r:gz") as tf:
        member = next((m for m in tf.getmembers()
                       if m.name.endswith("-sentences.txt")), None)
        if member is None:
            raise RuntimeError("no sentences file inside %s" % name)
        data = tf.extractfile(member).read().decode("utf-8", errors="ignore")
    with io.open(out, "w", encoding="utf-8") as f:
        f.write(data)
    os.remove(arch)
    return out


def ensure_all(quiet=False):
    for split in PACKAGES:
        for lang in PACKAGES[split]:
            download(split, lang, quiet=quiet)


_CACHE = {}


def sentences(split, lang, min_letters=60):
    """Sentences of one split, long enough to cut evaluation texts from."""
    key = (split, lang)
    if key in _CACHE:
        return _CACHE[key]
    alpha = langs.get(lang)
    letters = set(alpha["variants"][alpha["auto"][0]]["letters"])
    out = []
    if split == "train":
        path = os.path.join(DATA_DIR, langs.get(lang)["files"][0])
        lines = io.open(path, encoding="utf-8", errors="ignore") if os.path.exists(path) else []
        for line in lines:
            s = line.strip()
            if sum(1 for c in s.lower() if c in letters) >= min_letters:
                out.append(s)
            if len(out) > 200000:
                break
    else:
        path = download(split, lang, quiet=True)
        with io.open(path, encoding="utf-8", errors="ignore") as f:
            for line in f:
                # Leipzig format: "<id>\t<sentence>"
                parts = line.rstrip("\n").split("\t", 1)
                if len(parts) != 2:
                    continue
                s = parts[1].strip()
                if sum(1 for c in s.lower() if c in letters) >= min_letters:
                    out.append(s)
    _CACHE[key] = out
    return out


class TextSource:
    """Cuts evaluation plaintexts of an exact letter count from one split."""

    def __init__(self, split, lang, seed=0):
        self.split, self.lang = split, lang
        self.rng = random.Random(seed)
        self.alpha = None
        from vigenere import core
        self.alpha = core.get_alphabet(langs.auto_keys(lang)[0])
        self.pool = sentences(split, lang)
        if not self.pool:
            raise RuntimeError("empty corpus for %s/%s" % (split, lang))

    def text(self, n_letters):
        buf, have = [], 0
        while have < n_letters:
            s = self.pool[self.rng.randrange(len(self.pool))]
            buf.append(s)
            have += len(self.alpha.to_indices(s))
        joined = " ".join(buf)
        kept, count = [], 0
        for ch in joined:
            if ch.lower() in self.alpha.index:
                if count >= n_letters:
                    break
                count += 1
            kept.append(ch)
        return "".join(kept).strip()

    def tokens(self, minlen=3, maxlen=24, limit=200000):
        """Word tokens of this split, for building out-of-vocabulary keys."""
        pat = re.compile(r"[^\W\d_]+", re.UNICODE)
        seen = set()
        for s in self.pool[:limit]:
            for w in pat.findall(s.lower()):
                if minlen <= len(w) <= maxlen and all(c in self.alpha.index for c in w):
                    seen.add(w)
        return seen


if __name__ == "__main__":
    print("Downloading held-out corpora into %s" % EVAL_DIR)
    ensure_all()
    for split in ("train", "validation", "test", "test_web"):
        for lang in ("en", "ru", "uk"):
            try:
                n = len(sentences(split, lang))
            except Exception as exc:                            # noqa: BLE001
                n = "error: %s" % exc
            print("  %-11s %s  %s sentences   [%s]" % (split, lang, n, GENRE[split]))
