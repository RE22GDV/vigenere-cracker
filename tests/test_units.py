# -*- coding: utf-8 -*-
"""Unit tests: ciphers, alphabets, analysis primitives, edge cases.

Runs standalone (``python tests/test_units.py``) or under pytest.
Nothing here needs the downloaded language data except the two tests that say
so explicitly, which skip when it is missing.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import queue
import sys
import threading
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vigenere import attack, core, i18n, langs                  # noqa: E402


# --------------------------------------------------------------------------- #
#  Ciphers: encrypt then decrypt must be the identity
# --------------------------------------------------------------------------- #
SAMPLES = {
    "en-26": ("Meet me at midnight, by the bridge!", "silver"),
    "ru-32": ("Найди меня в старом лесу, у камня.", "ключ"),
    "ru-33": ("Ёжик в тумане: приходи ещё!", "роза"),
    "uk-33": ("Знайди мене у старому лісі, біля каменя.", "ключ"),
    "uk-32": ("Ґанок,єдність, їжак — і все це українська.", "шифр"),
}


def test_roundtrip_all_alphabets_and_variants():
    for akey, (text, key) in SAMPLES.items():
        alpha = core.get_alphabet(akey)
        for variant in core.VARIANTS:
            ct = core.encrypt_text(text, key, alpha, variant)
            back = core.decrypt_text(ct, key, alpha, variant)
            # The round trip returns the text lower-cased and with any letter
            # outside the variant folded onto its base, so compare against that.
            idx, tmpl = alpha.layout(text)
            expect = alpha.render(idx, tmpl)
            assert back == expect, "%s/%s: %r != %r" % (akey, variant, back, expect)


def test_beaufort_is_an_involution():
    """Beaufort encryption and decryption are the same operation."""
    alpha = core.get_alphabet("en-26")
    text, key = "attack at dawn", "lemon"
    ct = core.encrypt_text(text, key, alpha, "beaufort")
    assert core.encrypt_text(ct, key, alpha, "beaufort") == text


def test_variants_differ():
    alpha = core.get_alphabet("en-26")
    outs = {v: core.encrypt_text("attack at dawn", "lemon", alpha, v)
            for v in core.VARIANTS}
    assert len(set(outs.values())) == 3, outs


def test_known_vigenere_vector():
    """The textbook example: ATTACKATDAWN + LEMON -> LXFOPVEFRNHR."""
    alpha = core.get_alphabet("en-26")
    assert core.encrypt_text("attackatdawn", "lemon", alpha, "vigenere") == "lxfopvefrnhr"


def test_layout_preserves_punctuation_and_case_positions():
    alpha = core.get_alphabet("en-26")
    src = "Hello, World! 42"
    ct = core.encrypt_text(src, "key", alpha)
    assert len(ct) == len(src)
    for a, b in zip(src, ct):
        if a.lower() not in alpha.index:
            assert a == b, "non-letter %r changed to %r" % (a, b)


def test_alphabet_folds():
    """The reduced alphabets fold the extra letter onto its base."""
    ru32, ru33 = core.get_alphabet("ru-32"), core.get_alphabet("ru-33")
    assert ru32.index["ё"] == ru32.index["е"]
    assert ru33.index["ё"] != ru33.index["е"]
    uk32, uk33 = core.get_alphabet("uk-32"), core.get_alphabet("uk-33")
    assert uk32.index["ґ"] == uk32.index["г"]
    assert uk33.index["ґ"] != uk33.index["г"]
    # Ukrainian-specific letters exist in both variants
    for ch in "єії":
        assert ch in uk33.index and ch in uk32.index


def test_alphabet_variants_produce_different_ciphertext():
    text, key = "если бы", "код"
    a32, a33 = core.get_alphabet("ru-32"), core.get_alphabet("ru-33")
    assert core.encrypt_text(text, key, a32) != core.encrypt_text(text, key, a33)


# --------------------------------------------------------------------------- #
#  Analysis primitives
# --------------------------------------------------------------------------- #
def test_index_of_coincidence_known_values():
    """A constant string has IC 1.0; a uniform permutation is near 1/M."""
    M = 26
    same = np.zeros(200, dtype=np.int64)
    assert abs(core.index_of_coincidence(same, M) - 1.0) < 1e-9
    uniform = np.arange(200, dtype=np.int64) % M
    ic = core.index_of_coincidence(uniform, M)
    assert abs(ic - 1.0 / M) < 0.01, ic


def test_index_of_coincidence_detects_natural_text():
    """English text should sit near 0.067, its Vigenere ciphertext much lower."""
    alpha = core.get_alphabet("en-26")
    plain = ("the quick brown fox jumps over the lazy dog while the other dogs "
             "watch the river and the trees near the old stone bridge ") * 6
    p = alpha.to_indices(plain)
    ic_plain = core.index_of_coincidence(p, alpha.M)
    ct = core.encrypt_text(plain, "cryptography", alpha)
    ic_ct = core.index_of_coincidence(alpha.to_indices(ct), alpha.M)
    assert ic_plain > 0.06, ic_plain
    assert ic_ct < ic_plain, (ic_plain, ic_ct)


def test_key_length_hints_favour_the_true_length():
    alpha = core.get_alphabet("en-26")
    plain = ("it is a truth universally acknowledged that a single man in "
             "possession of a good fortune must be in want of a wife ") * 8
    ct = core.encrypt_text(plain, "lemon", alpha)
    hints = core.key_length_hints(alpha.to_indices(ct), alpha.M, 12)
    best = [L for L, _ic in sorted(hints, key=lambda h: -h[1])[:3]]
    assert 5 in best, best


def test_kasiski_finds_multiples_of_the_key_length():
    alpha = core.get_alphabet("en-26")
    plain = "the cat sat on the mat and the cat ran to the mat again " * 10
    ct = core.encrypt_text(plain, "abcd", alpha)
    votes = core.kasiski(alpha.to_indices(ct), max_len=16)
    assert votes, "Kasiski returned nothing"
    tops = [L for L, _v in votes[:4]]
    assert any(L % 4 == 0 for L in tops), tops


def test_minimal_period():
    assert core.minimal_period((1, 2, 3, 1, 2, 3)) == (1, 2, 3)
    assert core.minimal_period((5, 5, 5, 5)) == (5,)
    assert core.minimal_period((1, 2, 3, 4)) == (1, 2, 3, 4)
    assert core.minimal_period((1, 2, 3, 1, 2, 4)) == (1, 2, 3, 1, 2, 4)


def test_column_correlation_recovers_a_caesar_shift():
    """With a one-letter key the correlation must peak at that letter."""
    alpha = core.get_alphabet("en-26")
    model = _model_or_skip("en-26")
    if model is None:
        return
    plain = "the quick brown fox jumps over the lazy dog " * 8
    for shift_letter in "dqz":
        ct = core.encrypt_text(plain, shift_letter, alpha)
        c = alpha.to_indices(ct)
        corr = core.column_correlation(c, alpha.M, "vigenere", model.letter_freq())
        assert int(np.argmax(corr)) == alpha.index[shift_letter], shift_letter


# --------------------------------------------------------------------------- #
#  Edge cases
# --------------------------------------------------------------------------- #
def test_empty_text():
    alpha = core.get_alphabet("en-26")
    assert core.encrypt_text("", "key", alpha) == ""
    assert core.decrypt_text("", "key", alpha) == ""
    idx, tmpl = alpha.layout("")
    assert len(idx) == 0 and tmpl == []


def test_text_without_letters():
    alpha = core.get_alphabet("en-26")
    src = "123 456 !!! ***"
    assert core.encrypt_text(src, "key", alpha) == src
    assert core.index_of_coincidence(alpha.to_indices(src), alpha.M) == 0.0
    assert core.kasiski(alpha.to_indices(src)) == []


def test_empty_key_is_a_no_op():
    alpha = core.get_alphabet("en-26")
    assert core.encrypt_text("hello", "", alpha) == "hello"


def test_single_letter_key_is_a_caesar_shift():
    alpha = core.get_alphabet("en-26")
    assert core.encrypt_text("abc", "b", alpha) == "bcd"
    assert core.decrypt_text("bcd", "b", alpha) == "abc"


def test_key_longer_than_text():
    alpha = core.get_alphabet("en-26")
    key = "averyverylongkeyindeed"
    for text in ("hi", "a", "ab cd"):
        ct = core.encrypt_text(text, key, alpha)
        assert core.decrypt_text(ct, key, alpha) == text


def test_engine_refuses_texts_shorter_than_three_letters():
    model = _model_or_skip("en-26")
    if model is None:
        return
    cfg = _engine_cfg("ab", [("en-26", model)], mode="freq")
    q = queue.Queue()
    eng = attack.Engine(cfg, q)
    eng.start()
    eng.join(timeout=60)
    events = _drain(q)
    done = [e for e in events if e["type"] == "done"]
    assert done and done[-1]["ok"] is False


def test_user_stop_terminates_the_engine():
    model = _model_or_skip("en-26")
    if model is None:
        return
    alpha = core.get_alphabet("en-26")
    ct = core.encrypt_text("the quick brown fox jumps over the lazy dog " * 3,
                           "silver", alpha)
    cfg = _engine_cfg(ct, [("en-26", model)], mode="brute")
    cfg.update(min_len=5, max_len=5, processes=2, use_gpu=False)
    q = queue.Queue()
    eng = attack.Engine(cfg, q)
    eng.start()
    threading.Timer(1.5, eng.request_stop).start()
    eng.join(timeout=180)
    assert not eng.is_alive(), "engine did not stop"
    done = [e for e in _drain(q) if e["type"] == "done"]
    assert done, "engine produced no done event"


def test_runs_without_external_dictionaries():
    """A model built from the built-in sample only must still work."""
    alpha = core.get_alphabet("en-26")
    model = core.build_model(alpha)            # no extra corpus, no extra words
    assert model.tri.shape[0] == alpha.M ** 3
    assert model.quad.shape[0] == alpha.M ** 4
    assert len(model.words) > 0
    p = alpha.to_indices("the quick brown fox")
    assert np.isfinite(model.score_text(p)) and np.isfinite(model.score_text4(p))
    cov, n_words = model.word_coverage("thequickbrownfox")
    assert 0.0 <= cov <= 1.0 and n_words >= 0


def test_cpu_and_gpu_agree():
    """Same key space, same winner and same score, on both back-ends."""
    if not attack.gpu_available():
        print("      (skipped: no CUDA)")
        return
    model = _model_or_skip("en-26")
    if model is None:
        return
    alpha = core.get_alphabet("en-26")
    ct = core.encrypt_text("meet me at midnight by the bridge", "silv", alpha)
    c32 = alpha.layout(ct)[0].astype(np.int32)
    L = 4
    pos = np.arange(len(c32), dtype=np.int64) % L
    cpu = attack.scan_range_numpy(c32, model.tri, alpha.M, L, 0, alpha.M ** L,
                                  "vigenere", pos, topk=5)
    gpu = attack.gpu_scan(c32, model.tri, alpha.M, L, "vigenere", topk=5)
    assert cpu[0][1] == gpu[0][1], (cpu[0], gpu[0])
    assert abs(cpu[0][0] - gpu[0][0]) < 1e-2, (cpu[0][0], gpu[0][0])


def test_gpu_batch_size_scales_with_memory():
    small = attack.gpu_batch_for(30, free_mb=1024)
    large = attack.gpu_batch_for(30, free_mb=16384)
    assert small < large
    assert attack.gpu_batch_for(400, free_mb=1024) < small


# --------------------------------------------------------------------------- #
#  Localisation and language registry
# --------------------------------------------------------------------------- #
def test_every_ui_string_exists_in_every_language():
    missing = [(k, l) for k, entry in i18n.STRINGS.items()
               for l in i18n.UI_LANGS if not entry.get(l)]
    assert not missing, missing[:10]


def test_ui_format_placeholders_match_across_languages():
    import re
    pat = re.compile(r"%[-+ #0]*[\d.*]*[a-zA-Z%]")
    for key, entry in i18n.STRINGS.items():
        counts = {l: len([m for m in pat.findall(entry[l]) if m != "%%"])
                  for l in i18n.UI_LANGS}
        assert len(set(counts.values())) == 1, (key, counts)


def test_language_registry_is_consistent():
    for code in langs.lang_codes():
        spec = langs.get(code)
        for name, v in spec["variants"].items():
            letters = v["letters"]
            assert len(set(letters)) == len(letters), (code, name)
            for src, dst in v["folds"].items():
                assert src not in letters and dst in letters, (code, name, src)
            "".encode(spec["encoding"])
            for ch in letters:                      # encoding must cover the alphabet
                ch.encode(spec["encoding"])
        for key in langs.auto_keys(code):
            assert core.get_alphabet(key).key == key


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #
def _model_or_skip(akey):
    try:
        return core.get_model(akey)
    except Exception as exc:                                    # noqa: BLE001
        print("      (skipped: model unavailable: %s)" % exc)
        return None


def _engine_cfg(text, alpha_list, mode="auto"):
    return dict(text=text, alpha_list=alpha_list, variant="vigenere", mode=mode,
                min_len=1, max_len=6, processes=2, restarts=50, keylen_penalty=3.47,
                dict_words=[], dict_reversed=True, freq_seeds=8,
                use_gpu=False, gpu_batch=None)


def _drain(q):
    out = []
    while True:
        try:
            out.append(q.get_nowait())
        except queue.Empty:
            return out


def main():
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = []
    for name, fn in tests:
        t0 = time.perf_counter()
        try:
            fn()
            print("PASS  %-52s %5.2fs" % (name, time.perf_counter() - t0))
        except AssertionError as exc:
            failed.append((name, exc))
            print("FAIL  %-52s %s" % (name, exc))
        except Exception as exc:                                # noqa: BLE001
            failed.append((name, exc))
            print("ERROR %-52s %r" % (name, exc))
    print("\n%d of %d passed" % (len(tests) - len(failed), len(tests)))
    return 1 if failed else 0


if __name__ == "__main__":
    mp.freeze_support()
    sys.exit(main())
