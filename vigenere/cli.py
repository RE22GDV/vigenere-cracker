# -*- coding: utf-8 -*-
"""Command-line interface, for scripting and batch work.

Examples:
  python -m vigenere.cli "wggmv rjbvz kv wfx" --lang en
  python -m vigenere.cli "..." --lang ru --mode brute --max-len 4 --gpu
  python -m vigenere.cli "..." --lang uk --mode freq
  python -m vigenere.cli --encrypt "find me" --key rose --lang en
"""

from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import queue
import sys
import time

from . import attack, calibrate, core, i18n, langs


def build_parser():
    ap = argparse.ArgumentParser(
        prog="vigenere", description="Vigenere cipher cracker (English / Russian / Ukrainian)")
    ap.add_argument("text", nargs="?", help="ciphertext to break")
    ap.add_argument("--lang", choices=langs.lang_codes(), default=langs.DEFAULT_LANG,
                    help="plaintext language (default: en)")
    ap.add_argument("--ui-lang", choices=i18n.UI_LANGS, default=i18n.DEFAULT_UI,
                    help="language of the messages")
    ap.add_argument("--alphabet", default="auto",
                    help="auto (test every variant) or a specific one: 26, 32, 33")
    ap.add_argument("--mode", choices=["auto", "freq", "brute", "dict", "hill"],
                    default="auto", help="attack strategy (default: auto)")
    ap.add_argument("--min-len", type=int, default=1, help="minimum key length")
    ap.add_argument("--max-len", type=int, default=20, help="maximum key length")
    ap.add_argument("--variant", choices=list(core.VARIANTS), default="vigenere")
    ap.add_argument("--gpu", action="store_true", help="use CUDA if available")
    ap.add_argument("--procs", type=int, default=mp.cpu_count(), help="worker processes")
    ap.add_argument("--restarts", type=int, default=4000, help="hill-climbing restarts")
    ap.add_argument("--penalty", type=float, default=None,
                    help="key-length penalty (default: ln M)")
    ap.add_argument("--top", type=int, default=10, help="how many candidates to print")
    ap.add_argument("--quiet", action="store_true", help="suppress progress output")
    ap.add_argument("--encrypt", help="encrypt this text (requires --key)")
    ap.add_argument("--decrypt", help="decrypt this text with --key")
    ap.add_argument("--key", help="key for --encrypt / --decrypt")
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    tr = i18n.Translator(a.ui_lang)
    keys = (langs.auto_keys(a.lang) if a.alphabet == "auto"
            else ["%s-%s" % (a.lang, a.alphabet)])
    try:
        alpha = core.get_alphabet(keys[0])
    except KeyError:
        print("unknown alphabet variant: %s" % a.alphabet, file=sys.stderr)
        return 2

    if a.encrypt:
        print(core.encrypt_text(a.encrypt, a.key or "", alpha, a.variant))
        return 0
    if a.decrypt:
        print(core.decrypt_text(a.decrypt, a.key or "", alpha, a.variant))
        return 0
    if not a.text:
        build_parser().error("a ciphertext is required")

    if not core.has_data(a.lang):
        print("! Language data for '%s' is not downloaded - quality will be much worse.\n"
              "  Run: python -m vigenere.download_data %s\n" % (a.lang, a.lang),
              file=sys.stderr)

    if not a.quiet:
        print(tr('log.buildmodel', ", ".join(keys)), file=sys.stderr)
    alpha_list = [(k, core.get_model(k)) for k in keys]

    dict_words = []
    if a.mode == "dict":
        dict_words = sorted(alpha_list[0][1].words | set(langs.builtin_words(a.lang)))
    elif a.mode == "auto":
        dict_words = sorted(set(core.load_frequent_words(a.lang))
                            | set(langs.builtin_words(a.lang)))

    use_gpu = a.gpu and attack.gpu_available()
    if a.gpu and not use_gpu:
        print("! CUDA is not available, falling back to the CPU", file=sys.stderr)

    cfg = dict(text=a.text, alpha_list=alpha_list, variant=a.variant, mode=a.mode,
               min_len=a.min_len, max_len=a.max_len, processes=a.procs,
               restarts=a.restarts,
               keylen_penalty=a.penalty if a.penalty is not None else math.log(alpha.M),
               dict_words=dict_words, dict_reversed=True, freq_seeds=320,
               use_gpu=use_gpu, gpu_batch=None, tr=tr)

    q = queue.Queue()
    eng = attack.Engine(cfg, q)
    eng.start()
    rows, last = None, 0.0
    while eng.is_alive() or not q.empty():
        try:
            ev = q.get(timeout=0.3)
        except queue.Empty:
            continue
        if ev['type'] == 'log' and not a.quiet:
            print("  " + ev['msg'], file=sys.stderr)
        elif ev['type'] == 'progress' and not a.quiet and time.time() - last > 0.5:
            last = time.time()
            pct = min(100.0, 100.0 * ev['done'] / max(1, ev['total']))
            bar = "#" * int(pct / 2.5) + "." * (40 - int(pct / 2.5))
            print("\r  [%s] %5.1f%%  %s k/s  ETA %s   " %
                  (bar, pct, f"{int(ev['rate']):,}".replace(",", " "),
                   time.strftime("%M:%S", time.gmtime(min(359999, ev['eta'])))),
                  end="", file=sys.stderr)
        elif ev['type'] == 'results':
            rows = ev['rows']
    eng.join()
    if not a.quiet:
        print("", file=sys.stderr)

    if not rows:
        print("nothing found")
        return 1
    print("\n%-3s %-16s %-7s %9s %5s  %s"
          % ("#", tr('col.key'), tr('col.alpha'), tr('col.score'), tr('col.words'),
             tr('col.plain')))
    print("-" * 110)
    for i, r in enumerate(rows[:a.top], 1):
        print("%-3d %-16s %-7s %9.1f %5d  %s"
              % (i, r['key'], r['alpha'], r['final'], r['words'], r['plain']))
    return 0


if __name__ == "__main__":
    mp.freeze_support()
    sys.exit(main())
