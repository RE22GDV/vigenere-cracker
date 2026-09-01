# -*- coding: utf-8 -*-
"""Download the language data: word lists and text corpora.

Creates the following files in the data directory, per language:
  words_<lang>.txt           full word list (coverage scoring + dictionary attack)
  words_<lang>_frequent.txt  frequent-word list (fast modes)
  corpus_<lang>.txt          running text (tri/quadgram model)

Usage:
  python -m vigenere.download_data           all languages (en, ru, uk)
  python -m vigenere.download_data ru uk     only the listed ones

All sources are public data sets (GitHub, Tatoeba).
"""

from __future__ import annotations

import bz2
import io
import os
import re
import sys
import urllib.request

from .paths import DATA_DIR, ensure_data_dir

# Source kinds:
#   words    plain list, one word per line
#   dic      "word/flags" or "word tag" - keep the part before the separator
#   hunspell same, plus affix expansion when an .aff file is supplied
SOURCES = {
    "en": {
        "letters": "a-z",
        "words": [("https://raw.githubusercontent.com/dwyl/english-words/master/"
                   "words_alpha.txt", "utf-8", "words", "English word list (~370k)")],
        "frequent": ("https://raw.githubusercontent.com/first20hours/"
                     "google-10000-english/master/google-10000-english.txt",
                     "utf-8", "10000 most frequent words"),
        "corpus": ("https://downloads.tatoeba.org/exports/per_language/eng/"
                   "eng_sentences.tsv.bz2", "eng", "English sentence corpus"),
    },
    "ru": {
        "letters": "а-яё",
        "words": [("https://raw.githubusercontent.com/danakt/russian-words/master/"
                   "russian.txt", "cp1251", "words", "Russian word list (~1.5M)")],
        "frequent": ("https://raw.githubusercontent.com/hingston/russian/master/"
                     "10000-russian-words.txt", "utf-8", "10000 most frequent words"),
        "corpus": ("https://downloads.tatoeba.org/exports/per_language/rus/"
                   "rus_sentences.tsv.bz2", "rus", "Russian sentence corpus"),
    },
    "uk": {
        "letters": "а-щьюяєіїґ",
        "words": [("https://raw.githubusercontent.com/LibreOffice/dictionaries/master/"
                   "uk_UA/uk_UA.dic", "utf-8", "hunspell", "Ukrainian dictionary (hunspell)"),
                  ("https://raw.githubusercontent.com/brown-uk/dict_uk/master/"
                   "data/dict/base.lst", "utf-8", "dic", "Ukrainian dictionary (dict_uk)")],
        "aff": ("https://raw.githubusercontent.com/LibreOffice/dictionaries/master/"
                "uk_UA/uk_UA.aff", "inflection rules (affixes)"),
        "frequent": None,
        "corpus": ("https://downloads.tatoeba.org/exports/per_language/ukr/"
                   "ukr_sentences.tsv.bz2", "ukr", "Ukrainian sentence corpus"),
    },
}


def fetch(url, note):
    print("  downloading: %s\n    %s" % (note, url))
    req = urllib.request.Request(url, headers={"User-Agent": "vigenere-cracker"})
    with urllib.request.urlopen(req, timeout=300) as r:
        data = r.read()
    print("    got %.1f MB" % (len(data) / 1048576))
    return data


def fmt(n):
    return f"{n:,}".replace(",", " ")


def parse_aff(text):
    """Parse a hunspell affix file: flag -> [(strip, append, condition), ...].

    Only single-character flags and suffix rules are supported, which is all
    the Ukrainian dictionary needs.
    """
    rules = {}
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        p = lines[i].split()
        if len(p) >= 4 and p[0] == "SFX" and p[2] in ("Y", "N"):
            flag = p[1]
            try:
                count = int(p[3])
            except ValueError:
                i += 1
                continue
            lst = rules.setdefault(flag, [])
            for j in range(1, count + 1):
                if i + j >= len(lines):
                    break
                q = lines[i + j].split()
                if len(q) < 4 or q[0] != "SFX":
                    continue
                strip = "" if q[2] == "0" else q[2]
                affix = "" if q[3] == "0" else q[3].split("/")[0]
                cond = q[4] if len(q) > 4 else "."
                try:
                    lst.append((strip, affix, re.compile(cond + "$")))
                except re.error:
                    pass
            i += count + 1
            continue
        i += 1
    return rules


def parse_words(raw, kind, word_re, aff_rules=None):
    """Extract words from a source.

    For a hunspell dictionary the inflected forms are generated as well: the
    .dic file holds stems only, while real text contains declined forms. That
    single step took the Ukrainian list from 350k entries to 3.35M.
    """
    out = set()
    if kind in ("dic", "hunspell"):
        for line in raw.splitlines():
            line = line.strip()
            if not line or line.isdigit():
                continue
            head, _, flags = line.partition("/")
            w = re.split(r"[\s#]", head, 1)[0].lower()
            if not word_re.match(w):
                continue
            out.add(w)
            if kind == "hunspell" and aff_rules and flags:
                flags = re.split(r"[\s\t]", flags, 1)[0]
                for f in flags:
                    for strip, affix, cond in aff_rules.get(f, ()):
                        if strip and not w.endswith(strip):
                            continue
                        if not cond.search(w):
                            continue
                        form = (w[:len(w) - len(strip)] if strip else w) + affix
                        if word_re.match(form):
                            out.add(form)
    else:
        for w in raw.split():
            w = w.strip().lower()
            if word_re.match(w):
                out.add(w)
    return out


def do_lang(code):
    src = SOURCES[code]
    word_re = re.compile(r"^[%s]{2,24}$" % src["letters"])
    print("\n===== %s =====" % code.upper())
    ensure_data_dir()

    aff_rules = None
    if src.get("aff"):
        url, note = src["aff"]
        try:
            aff_rules = parse_aff(fetch(url, note).decode("utf-8", errors="ignore"))
            print("    rules parsed: %s (flags: %d)"
                  % (fmt(sum(len(v) for v in aff_rules.values())), len(aff_rules)))
        except Exception as exc:                                # noqa: BLE001
            print("    (affixes skipped: %s)" % exc)

    words = set()
    for url, enc, kind, note in src["words"]:
        try:
            raw = fetch(url, note).decode(enc, errors="ignore")
            got = parse_words(raw, kind, word_re, aff_rules)
            print("    words parsed: %s" % fmt(len(got)))
            words |= got
        except Exception as exc:                                # noqa: BLE001
            print("    (skipped: %s)" % exc)

    corpus_lines = []
    if src["corpus"]:
        url, tag, note = src["corpus"]
        try:
            text = bz2.decompress(fetch(url, note)).decode("utf-8", errors="ignore")
            for line in io.StringIO(text):
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 3 and parts[1] == tag:
                    corpus_lines.append(parts[2])
        except Exception as exc:                                # noqa: BLE001
            print("    (corpus skipped: %s)" % exc)

    # Words taken from the corpus are live inflected forms that stem-only
    # dictionaries miss.
    if corpus_lines:
        from_corpus = {w for w in re.findall(r"[^\W\d_]+",
                                             "\n".join(corpus_lines).lower(), re.UNICODE)
                       if word_re.match(w)}
        print("    words harvested from the corpus: %s" % fmt(len(from_corpus)))
        words |= from_corpus

    if not words:
        print("    ! empty word list, nothing written")
        return

    out = os.path.join(DATA_DIR, "words_%s.txt" % code)
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(sorted(words)))
    print("  -> words_%s.txt: %s words, %.1f MB"
          % (code, fmt(len(words)), os.path.getsize(out) / 1048576))

    freq = []
    if src["frequent"]:
        url, enc, note = src["frequent"]
        try:
            raw = fetch(url, note).decode(enc, errors="ignore")
            freq = [w.strip().lower() for w in raw.split()
                    if word_re.match(w.strip().lower())]
        except Exception as exc:                                # noqa: BLE001
            print("    (frequency list skipped: %s)" % exc)
    if not freq and corpus_lines:
        counts = {}
        for w in re.findall(r"[^\W\d_]+", "\n".join(corpus_lines).lower(), re.UNICODE):
            if word_re.match(w):
                counts[w] = counts.get(w, 0) + 1
        freq = [w for w, _ in sorted(counts.items(), key=lambda kv: -kv[1])[:12000]]
        print("    frequency list built from the corpus")
    if freq:
        out_f = os.path.join(DATA_DIR, "words_%s_frequent.txt" % code)
        with open(out_f, "w", encoding="utf-8") as f:
            f.write("\n".join(freq))
        print("  -> words_%s_frequent.txt: %s words" % (code, fmt(len(freq))))

    if corpus_lines:
        out = os.path.join(DATA_DIR, "corpus_%s.txt" % code)
        with open(out, "w", encoding="utf-8") as f:
            f.write("\n".join(corpus_lines))
        print("  -> corpus_%s.txt: %s sentences, %.1f MB"
              % (code, fmt(len(corpus_lines)), os.path.getsize(out) / 1048576))


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    codes = [c for c in argv if c in SOURCES] or list(SOURCES)
    print("Downloading language data into %s\nLanguages: %s"
          % (DATA_DIR, ", ".join(codes)))
    for code in codes:
        do_lang(code)
    print("\nDone. The files are picked up automatically on the next run.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as exc:                                    # noqa: BLE001
        print("ERROR: %s" % exc, file=sys.stderr)
        sys.exit(1)
