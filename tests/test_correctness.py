# -*- coding: utf-8 -*-
"""Correctness suite: encrypt known phrases in all three languages, break them,
and check that the true plaintext comes out on top.

Run directly (``python tests/test_correctness.py``) or under pytest.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import queue
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vigenere import attack, core, langs                        # noqa: E402

# (plaintext, key, alphabet key)
CASES = [
    # English
    ("find me at the old stone in the forest", "rose", "en-26"),
    ("the red flower grows near the river", "cipher", "en-26"),
    ("science begins where a person stops believing the obvious and starts "
     "checking their guesses with precise measurements", "cryptography", "en-26"),
    # Russian
    ("найди меня в старом лесу у камня", "ключ", "ru-32"),
    ("карл у клары украл кораллы", "роза", "ru-33"),
    ("наука начинается там где человек перестает верить очевидному и начинает "
     "проверять свои догадки", "криптография", "ru-32"),
    # Ukrainian
    ("знайди мене у старому лісі біля каменя", "ключ", "uk-33"),
    ("червона квітка росте біля річки", "шифр", "uk-33"),
    ("наука починається там де людина перестає вірити очевидному і починає "
     "перевіряти свої здогади", "криптографія", "uk-33"),
]


def normalise(text, alpha):
    return ''.join(c for c in alpha.fold(text.lower()) if c in alpha.index)


def break_text(ciphertext, lang, models, mode="auto", use_gpu=None):
    """Run the engine and return the ranked candidate rows."""
    keys = langs.auto_keys(lang)
    cfg = dict(text=ciphertext, alpha_list=[(k, models[k]) for k in keys],
               variant="vigenere", mode=mode, min_len=1, max_len=20,
               processes=mp.cpu_count(), restarts=2000, keylen_penalty=3.47,
               dict_words=sorted(set(core.load_frequent_words(lang))
                                 | set(langs.builtin_words(lang))),
               dict_reversed=True, freq_seeds=320,
               use_gpu=attack.gpu_available() if use_gpu is None else use_gpu,
               gpu_batch=None)
    q = queue.Queue()
    eng = attack.Engine(cfg, q)
    eng.start()
    rows = None
    while eng.is_alive() or not q.empty():
        try:
            ev = q.get(timeout=0.5)
        except queue.Empty:
            continue
        if ev['type'] == 'results':
            rows = ev['rows']
    eng.join()
    return rows or []


def load_models():
    models = {}
    for lang in langs.lang_codes():
        for key in langs.auto_keys(lang):
            models[key] = core.get_model(key)
    return models


def rank_of_truth(rows, plain, alpha, limit=20):
    want = normalise(plain, alpha)
    for i, r in enumerate(rows[:limit], 1):
        if normalise(r['plain'], core.get_alphabet(r['alpha'])) == want:
            return i
    return None


def run_all(verbose=True):
    models = load_models()
    passed = 0
    for plain, key, akey in CASES:
        alpha = core.get_alphabet(akey)
        ct = core.encrypt_text(plain, key, alpha, "vigenere")
        t0 = time.time()
        rows = break_text(ct, alpha.lang, models)
        dt = time.time() - t0
        pos = rank_of_truth(rows, plain, alpha)
        ok = pos == 1
        passed += ok
        if verbose:
            print("%s %-8s key=%-14s %5.1f s  rank=%s"
                  % ("PASS" if ok else "FAIL", akey, key, dt, pos))
            if not ok and rows:
                print("      top: %s -> %s" % (rows[0]['key'], rows[0]['plain'][:60]))
    return passed, len(CASES)


def test_all_languages():
    """pytest entry point."""
    passed, total = run_all(verbose=False)
    assert passed == total, "%d of %d cases did not rank first" % (passed, total)


if __name__ == "__main__":
    mp.freeze_support()
    ok, total = run_all()
    print("\nRESULT: %d of %d ranked first" % (ok, total))
    sys.exit(0 if ok == total else 1)
