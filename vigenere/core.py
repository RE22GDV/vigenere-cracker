# -*- coding: utf-8 -*-
"""Alphabets, cipher variants and the statistical language model.

An alphabet is identified by a ``language-variant`` key such as ``ru-32``,
``uk-33`` or ``en-26``. The same plaintext enciphered under different variants
produces different ciphertext, which is why the tool can test them all at once.
"""

from __future__ import annotations

import os

import numpy as np

from . import langs
from .paths import DATA_DIR, ensure_data_dir

# Cipher variants. p = plaintext, c = ciphertext, k = key (letter indices).
VARIANTS = ("vigenere", "beaufort", "variant")

MAX_WORD = 16      # longest word considered by the dictionary-coverage search
MIN_WORD = 3       # shorter "words" are matched by chance far too often
                   # (tuned on a set of 18 short texts, see README)


class Alphabet:
    """One variant of one language: the ordered letters plus fold rules.

    A fold maps a letter outside the variant onto one inside it (``ё``→``е``
    for the 32-letter Russian alphabet, ``ґ``→``г`` for 32-letter Ukrainian).
    """

    def __init__(self, key: str):
        self.key = key
        self.lang, self.variant = langs.split_key(key)
        spec = langs.get(self.lang)
        v = spec["variants"][self.variant]
        self.letters = v["letters"]
        self.encoding = spec["encoding"]
        self.M = len(self.letters)
        self.index = {ch: i for i, ch in enumerate(self.letters)}
        self.folds = dict(v["folds"])
        for src, dst in self.folds.items():
            if dst in self.index:
                self.index[src] = self.index[dst]

    def fold(self, text: str) -> str:
        for src, dst in self.folds.items():
            text = text.replace(src, dst)
        return text

    def to_indices(self, text: str) -> np.ndarray:
        text = text.lower()
        return np.array([self.index[c] for c in text if c in self.index], dtype=np.int64)

    def layout(self, text: str):
        """Split a string into letter indices plus a template that restores
        the original spacing and punctuation when rendering the result back."""
        idx, tmpl = [], []
        for ch in text:
            low = ch.lower()
            if low in self.index:
                tmpl.append(None)          # placeholder for a letter
                idx.append(self.index[low])
            else:
                tmpl.append(ch)
        return np.array(idx, dtype=np.int64), tmpl

    def render(self, indices, tmpl=None) -> str:
        if tmpl is None:
            return ''.join(self.letters[i] for i in indices)
        out, k = [], 0
        for t in tmpl:
            if t is None:
                out.append(self.letters[indices[k]])
                k += 1
            else:
                out.append(t)
        return ''.join(out)

    def key_to_str(self, key_indices) -> str:
        return ''.join(self.letters[i] for i in key_indices)

    def key_from_int(self, n: int, length: int) -> np.ndarray:
        """Decode a key ordinal into its base-M digits (used by brute force)."""
        d = np.zeros(length, dtype=np.int64)
        for j in range(length - 1, -1, -1):
            d[j] = n % self.M
            n //= self.M
        return d


_ALPHA_CACHE = {}


def get_alphabet(key: str) -> Alphabet:
    a = _ALPHA_CACHE.get(key)
    if a is None:
        a = _ALPHA_CACHE[key] = Alphabet(key)
    return a


# --------------------------------------------------------------------------- #
#  Encryption / decryption
# --------------------------------------------------------------------------- #
def decrypt_indices(c: np.ndarray, k: np.ndarray, M: int, variant: str) -> np.ndarray:
    """``k`` must already be expanded to the length of ``c`` (or broadcastable)."""
    if variant == "vigenere":
        return (c - k) % M
    if variant == "beaufort":
        return (k - c) % M
    return (c + k) % M                      # variant Beaufort


def encrypt_indices(p: np.ndarray, k: np.ndarray, M: int, variant: str) -> np.ndarray:
    if variant == "vigenere":
        return (p + k) % M
    if variant == "beaufort":
        return (k - p) % M
    return (p - k) % M


def encrypt_text(text: str, key: str, alpha: Alphabet, variant: str = "vigenere") -> str:
    idx, tmpl = alpha.layout(text)
    kidx = alpha.to_indices(key)
    if len(kidx) == 0 or len(idx) == 0:
        return text
    kexp = kidx[np.arange(len(idx)) % len(kidx)]
    return alpha.render(encrypt_indices(idx, kexp, alpha.M, variant), tmpl)


def decrypt_text(text: str, key: str, alpha: Alphabet, variant: str = "vigenere") -> str:
    idx, tmpl = alpha.layout(text)
    kidx = alpha.to_indices(key)
    if len(kidx) == 0 or len(idx) == 0:
        return text
    kexp = kidx[np.arange(len(idx)) % len(kidx)]
    return alpha.render(decrypt_indices(idx, kexp, alpha.M, variant), tmpl)


# --------------------------------------------------------------------------- #
#  Language model
# --------------------------------------------------------------------------- #
class LangModel:
    """Trigrams (M^3) drive the search: the table is small enough to stay in
    cache and to be gathered from efficiently on the GPU. Quadgrams (M^4) and
    the word list are used only to re-rank the few thousand finalists."""

    def __init__(self, M, tri, bi, uni, words, quad=None, folds=None):
        self.M = M
        self.tri = tri            # float32, M^3 — log P(c | a, b)
        self.bi = bi              # float32, M^2
        self.uni = uni            # float32, M
        self.quad = quad          # float32, M^4 — log P(d | a, b, c)
        self.words = words        # folded word set
        self.folds = dict(folds or {})
        self.maxlen = min(MAX_WORD, max((len(w) for w in words), default=0))

    def letter_freq(self) -> np.ndarray:
        """Letter frequencies taken from the model itself — more accurate than
        a textbook table and works for any language without extra data."""
        p = np.exp(self.uni.astype(np.float64))
        s = p.sum()
        return p / s if s > 0 else np.full(self.M, 1.0 / self.M)

    def score_text(self, p: np.ndarray) -> float:
        if len(p) < 3:
            return float(self.uni[p].sum()) if len(p) else -1e9
        idx = (p[:-2] * self.M + p[1:-1]) * self.M + p[2:]
        return float(self.tri[idx].sum() + self.bi[p[0] * self.M + p[1]])

    def score_text4(self, p: np.ndarray) -> float:
        """Quadgram score, backing off to trigrams for the first characters."""
        if self.quad is None or len(p) < 4:
            return self.score_text(p)
        M = self.M
        idx = ((p[:-3] * M + p[1:-2]) * M + p[2:-1]) * M + p[3:]
        head = self.tri[(p[0] * M + p[1]) * M + p[2]] + self.bi[p[0] * M + p[1]]
        return float(self.quad[idx].sum() + head)

    def word_coverage(self, text: str):
        """Best split of the string into real words (dynamic programming).

        Returns ``(coverage in 0..1, number of words)``. Longer words weigh
        more (L^1.4); words shorter than MIN_WORD are ignored because random
        letter strings hit them constantly.
        """
        for src, dst in self.folds.items():
            text = text.replace(src, dst)
        n = len(text)
        if n == 0 or not self.words:
            return 0.0, 0
        words, maxlen = self.words, self.maxlen
        best = [0.0] * (n + 1)
        cnt = [0] * (n + 1)
        for i in range(1, n + 1):
            best[i], cnt[i] = best[i - 1], cnt[i - 1]          # skip this letter
            lim = min(maxlen, i)
            for L in range(MIN_WORD, lim + 1):
                if text[i - L:i] in words:
                    val = best[i - L] + L ** 1.4
                    if val > best[i]:
                        best[i], cnt[i] = val, cnt[i - L] + 1
        return best[n] / max(1.0, n ** 1.4), cnt[n]


# --------------------------------------------------------------------------- #
#  Fast n-gram counting: text -> index array through a 256-entry table
# --------------------------------------------------------------------------- #
def _lut_for(alpha: Alphabet) -> np.ndarray:
    """256-entry table mapping a single-byte code to a letter index, or M for
    "not a letter". cp1251 covers both Russian and Ukrainian, latin-1 covers
    English, so one table lookup replaces a per-character Python loop."""
    lut = np.full(256, alpha.M, dtype=np.uint8)
    for ch, i in alpha.index.items():
        try:
            lut[ch.encode(alpha.encoding)[0]] = i
        except (UnicodeEncodeError, IndexError):
            pass
    return lut


def _to_arr(text: str, lut: np.ndarray, encoding: str) -> np.ndarray:
    b = text.lower().encode(encoding, errors='replace')
    return lut[np.frombuffer(b, dtype=np.uint8)]


def _accumulate(arr, M, uni, bi, tri, quad, w):
    """Count n-grams over runs of consecutive letters (M marks a separator)."""
    ok = arr < M
    a64 = arr.astype(np.int64)
    v = a64[ok]
    if v.size:
        uni += np.bincount(v, minlength=M) * w
    m2 = ok[:-1] & ok[1:]
    if m2.any():
        bi += np.bincount(a64[:-1][m2] * M + a64[1:][m2], minlength=M * M) * w
    if arr.size > 2:
        m3 = ok[:-2] & ok[1:-1] & ok[2:]
        if m3.any():
            tri += np.bincount((a64[:-2][m3] * M + a64[1:-1][m3]) * M + a64[2:][m3],
                               minlength=M ** 3) * w
    if quad is not None and arr.size > 3:
        m4 = ok[:-3] & ok[1:-2] & ok[2:-1] & ok[3:]
        if m4.any():
            quad += np.bincount(
                ((a64[:-3][m4] * M + a64[1:-2][m4]) * M + a64[2:-1][m4]) * M + a64[3:][m4],
                minlength=M ** 4) * w


CHUNK = 4 << 20        # characters processed per pass


def build_model(alpha: Alphabet, extra_corpus: str = "", extra_words=None,
                progress=None) -> LangModel:
    M = alpha.M
    lut = _lut_for(alpha)

    uni = np.full(M, 1.0)
    bi = np.full(M * M, 0.1)
    tri = np.full(M ** 3, 0.01)
    quad = np.full(M ** 4, 0.002)

    def feed(text, w):
        for s in range(0, len(text), CHUNK):
            arr = _to_arr(text[s:s + CHUNK + 3], lut, alpha.encoding)
            _accumulate(arr, M, uni, bi, tri, quad, w)         # inside words
            letters = arr[arr < M]                             # and across word breaks:
            _accumulate(letters, M, uni, bi, tri, quad, w * 0.35)  # ciphertext has no spaces

    if progress:
        progress("corpus")
    feed(langs.builtin_corpus(alpha.lang), 1.0)
    if extra_corpus:
        feed(extra_corpus, 1.0)

    # The word list is good letter-statistics material in its own right.
    words = {w for w in langs.builtin_words(alpha.lang) if 2 <= len(w)}
    if extra_words:
        words |= {w for w in extra_words if 2 <= len(w) <= 24}
    words = {alpha.fold(w) for w in words}
    if extra_words:
        if progress:
            progress("dictionary")
        feed("\n".join(sorted(words)), 0.6)

    if progress:
        progress("probabilities")

    # Jelinek-Mercer interpolation with back-off to shorter contexts.
    p_uni = uni / uni.sum()
    p_bi = (bi.reshape(M, M) / bi.reshape(M, M).sum(axis=1, keepdims=True))
    tri_m = tri.reshape(M * M, M)
    p_tri = tri_m / tri_m.sum(axis=1, keepdims=True)

    l3, l2, l1, l0 = 0.85, 0.10, 0.04, 0.01
    tri_log = np.log(l3 * p_tri + l2 * np.tile(p_bi, (M, 1))
                     + l1 * p_uni[None, :] + l0 / M).astype(np.float32).ravel()
    bi_log = np.log(0.88 * p_bi + 0.11 * p_uni[None, :] + 0.01 / M).astype(np.float32).ravel()
    uni_log = np.log(p_uni).astype(np.float32)

    quad_m = quad.reshape(M ** 3, M)
    p_quad = quad_m / quad_m.sum(axis=1, keepdims=True)
    q4, q3, q2, q0 = 0.80, 0.14, 0.05, 0.01
    # back-off chain: P(d|a,b,c) -> P(d|b,c) -> P(d|c) -> uniform
    quad_log = np.log(q4 * p_quad
                      + q3 * np.tile(p_tri, (M, 1))
                      + q2 * np.tile(p_bi, (M * M, 1))
                      + q0 / M).astype(np.float32).ravel()
    del quad, quad_m, p_quad

    return LangModel(M, tri_log, bi_log, uni_log, words, quad_log, alpha.folds)


# --------------------------------------------------------------------------- #
#  Downloaded data and on-disk model cache
# --------------------------------------------------------------------------- #
def _paths(lang: str):
    corpus_f, words_f, freq_f = langs.get(lang)["files"]
    return (os.path.join(DATA_DIR, corpus_f),
            os.path.join(DATA_DIR, words_f),
            os.path.join(DATA_DIR, freq_f))


def data_signature(lang: str) -> str:
    """Fingerprint of the external files; the model cache is keyed on it."""
    parts = ["v4", lang]
    for p in _paths(lang)[:2]:
        if os.path.exists(p):
            st = os.stat(p)
            parts.append("%s:%d:%d" % (os.path.basename(p), st.st_size, int(st.st_mtime)))
    import hashlib
    return hashlib.md5("|".join(parts).encode()).hexdigest()[:12]


def has_data(lang: str) -> bool:
    c, w, _ = _paths(lang)
    return os.path.exists(c) or os.path.exists(w)


def missing_languages(codes=None):
    return [c for c in (codes or langs.lang_codes()) if not has_data(c)]


def load_user_data(lang: str):
    """Read the downloaded corpus and word list; both are optional."""
    corpus_p, words_p, _ = _paths(lang)
    corpus, words = "", []
    if os.path.exists(corpus_p):
        with open(corpus_p, encoding="utf-8", errors="ignore") as f:
            corpus = f.read()
    if os.path.exists(words_p):
        with open(words_p, encoding="utf-8", errors="ignore") as f:
            words = [w.strip().lower() for w in f if w.strip()]
    return corpus, words


def load_frequent_words(lang: str):
    """Compact candidate-key list for the fast modes. The full dictionary is
    only used by the explicit dictionary attack."""
    p = _paths(lang)[2]
    if os.path.exists(p):
        with open(p, encoding="utf-8", errors="ignore") as f:
            return [w.strip().lower() for w in f if w.strip()]
    return []


def get_model(alpha_key: str, progress=None) -> LangModel:
    """Build the model, or load it from the on-disk cache when possible."""
    alpha = get_alphabet(alpha_key)
    sig = data_signature(alpha.lang)
    prefix = "model_%s_%s_" % (alpha.lang, alpha.variant)
    cache = os.path.join(DATA_DIR, prefix + sig + ".npz")
    corpus, words = load_user_data(alpha.lang)

    if os.path.exists(cache):
        try:
            if progress:
                progress("cache")
            z = np.load(cache)
            wset = {w for w in words if 2 <= len(w) <= 24} | set(
                langs.builtin_words(alpha.lang))
            wset = {alpha.fold(w) for w in wset}
            return LangModel(alpha.M, z['tri'], z['bi'], z['uni'], wset, z['quad'],
                             alpha.folds)
        except Exception:                                       # noqa: BLE001
            pass

    m = build_model(alpha, corpus, words, progress=progress)
    try:
        ensure_data_dir()
        for old in os.listdir(DATA_DIR):
            if old.startswith(prefix) and old.endswith(".npz"):
                os.remove(os.path.join(DATA_DIR, old))
        np.savez(cache, tri=m.tri, bi=m.bi, uni=m.uni, quad=m.quad)
    except Exception:                                           # noqa: BLE001
        pass
    return m


# --------------------------------------------------------------------------- #
#  Key-length hints
# --------------------------------------------------------------------------- #
# Index of coincidence of natural text: ~0.055 for Russian and Ukrainian,
# ~0.067 for English, whose smaller alphabet makes coincidences more likely.
NATURAL_IC = {"ru": 0.0553, "uk": 0.0545, "en": 0.0667}


def natural_ic(lang: str) -> float:
    return NATURAL_IC.get(lang, 0.055)


def index_of_coincidence(seq: np.ndarray, M: int) -> float:
    n = len(seq)
    if n < 2:
        return 0.0
    counts = np.bincount(seq, minlength=M).astype(np.float64)
    return float((counts * (counts - 1)).sum() / (n * (n - 1)))


def key_length_hints(c: np.ndarray, M: int, max_len: int = 12):
    """Mean IC per column for each candidate key length."""
    out = []
    for L in range(1, max_len + 1):
        if len(c) < 2 * L:
            break
        ics = [index_of_coincidence(c[i::L], M) for i in range(L) if len(c[i::L]) >= 2]
        if ics:
            out.append((L, float(np.mean(ics))))
    return out


def kasiski(c: np.ndarray, max_len: int = 30, ngram: int = 3, top: int = 6):
    """Kasiski examination: distances between repeated n-grams are multiples of
    the key length. Returns ``[(length, votes), ...]`` best first."""
    n = len(c)
    if n < 2 * ngram + 2:
        return []
    seen = {}
    for i in range(n - ngram + 1):
        seen.setdefault(c[i:i + ngram].tobytes(), []).append(i)
    votes = np.zeros(max_len + 1)
    total = 0
    for positions in seen.values():
        if len(positions) < 2:
            continue
        for j in range(len(positions) - 1):
            d = positions[j + 1] - positions[j]
            if d < 2:
                continue
            total += 1
            for L in range(2, min(max_len, d) + 1):
                if d % L == 0:
                    votes[L] += 1
    if total == 0:
        return []
    order = np.argsort(-votes[2:]) + 2
    return [(int(L), int(votes[L])) for L in order[:top] if votes[L] > 0]


def column_correlation(col: np.ndarray, M: int, variant: str, exp_freq: np.ndarray):
    """For every possible key letter, how much the decrypted column looks like
    natural language (correlation of letter frequencies). Higher is better."""
    if len(col) == 0:
        return np.zeros(M)
    corr = np.empty(M)
    for k in range(M):
        p = decrypt_indices(col, np.int64(k), M, variant)
        counts = np.bincount(p, minlength=M).astype(np.float64)
        corr[k] = counts @ exp_freq
    return corr / max(1, len(col))


def minimal_period(key):
    """``abcabc`` -> ``abc``: a shorter key is fairer under the length penalty."""
    key = list(key)
    L = len(key)
    for p in range(1, L):
        if L % p == 0 and all(key[i] == key[i % p] for i in range(L)):
            return tuple(key[:p])
    return tuple(key)
