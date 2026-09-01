# -*- coding: utf-8 -*-
"""Charts for the research studies (reads benchmarks/results/studies.json)."""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                 # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
RESULTS = os.path.join(HERE, "results", "studies.json")
DOCS = os.path.join(REPO, "docs")

LANGS = ["en", "ru", "uk"]
NAMES = {"en": "English", "ru": "Russian", "uk": "Ukrainian"}
COLORS = {"en": "#1f77b4", "ru": "#d62728", "uk": "#2ca02c"}

plt.rcParams.update({"figure.dpi": 130, "font.size": 9,
                     "axes.grid": True, "grid.alpha": 0.3})


def save(fig, name):
    os.makedirs(DOCS, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(DOCS, name))
    plt.close(fig)
    print("  ->", name)


def chart_matrix(res):
    d = res["matrix"]
    lengths, key_lens = d["lengths"], d["key_lens"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, lang in zip(axes, LANGS):
        grid = d["data"].get(lang, {})
        Z = np.full((len(key_lens), len(lengths)), np.nan)
        for i, L in enumerate(key_lens):
            for j, n in enumerate(lengths):
                cell = grid.get("%d,%d" % (n, L))
                if cell:
                    Z[i, j] = 100.0 * cell["top1"]
        im = ax.imshow(Z, origin="lower", aspect="auto", cmap="RdYlGn",
                       vmin=0, vmax=100)
        ax.set_xticks(range(len(lengths)), [str(x) for x in lengths])
        ax.set_yticks(range(len(key_lens)), [str(x) for x in key_lens])
        ax.set_xlabel("ciphertext length N, letters")
        if lang == "en":
            ax.set_ylabel("key length L")
        ax.set_title("%s (n=%d per cell)" % (NAMES[lang], d["trials"]))
        ax.grid(False)
        for i in range(len(key_lens)):
            for j in range(len(lengths)):
                if not np.isnan(Z[i, j]):
                    ax.text(j, i, "%.0f" % Z[i, j], ha="center", va="center",
                            fontsize=7,
                            color="black" if 25 < Z[i, j] < 85 else "white")
    fig.colorbar(im, ax=axes, fraction=0.02, label="top-1, %")
    fig.suptitle("Success rate over the text-length x key-length matrix "
                 "(random keys, held-out news corpus)", y=1.04)
    save(fig, "study_matrix.png")


def chart_ratio(res):
    """All cells collapsed onto the single variable N/L."""
    d = res["matrix"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8))
    allpts = []
    for lang in LANGS:
        pts = [(c["ratio"], 100.0 * c["top1"])
               for c in d["data"].get(lang, {}).values() if c]
        pts.sort()
        allpts += [(r, v, lang) for r, v in pts]
        ax1.scatter([p[0] for p in pts], [p[1] for p in pts], s=16,
                    color=COLORS[lang], alpha=0.75, label=NAMES[lang])
    ax1.set_xscale("log")
    ax1.set_xlabel("N / L  (letters available per key position)")
    ax1.set_ylabel("top-1, %")
    ax1.set_title("Every cell of the matrix, plotted against N/L")
    ax1.legend()
    ax1.set_ylim(-3, 103)

    # binned mean with the 90 % threshold marked
    bins = [1.5, 2.5, 3.5, 5, 7, 10, 15, 25, 40, 70, 120, 250]
    centres, means = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        vals = [v for r, v, _ in allpts if lo <= r < hi]
        if vals:
            centres.append((lo * hi) ** 0.5)
            means.append(float(np.mean(vals)))
    ax2.plot(centres, means, "o-", color="#333")
    ax2.axhline(90, color="#d62728", ls="--", lw=1)
    ax2.text(centres[0], 91, "90 %", color="#d62728", fontsize=8)
    for thr in (50, 90):
        hit = next((c for c, m in zip(centres, means) if m >= thr), None)
        if hit:
            ax2.axvline(hit, color="#999", ls=":", lw=0.8)
    ax2.set_xscale("log")
    ax2.set_xlabel("N / L")
    ax2.set_ylabel("mean top-1, %")
    ax2.set_title("Averaged over all three languages")
    ax2.set_ylim(-3, 103)
    save(fig, "study_ratio.png")


def _grouped_bars(ax, categories, series, colors=None, ylabel="top-1, %"):
    x = np.arange(len(categories))
    w = 0.8 / max(1, len(series))
    for i, (name, vals) in enumerate(series.items()):
        ax.bar(x + i * w - 0.4 + w / 2, vals, width=w, label=name,
               color=(colors or {}).get(name))
    ax.set_xticks(x, categories)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8)


def chart_keygroups(res):
    d = res["keygroups"]
    key_lens, groups = d["key_lens"], d["groups"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
    for ax, lang in zip(axes, LANGS):
        grid = d["data"].get(lang, {})
        series = {g: [100.0 * grid.get("%s,%d" % (g, L), {}).get("top1", 0)
                      for L in key_lens] for g in groups}
        _grouped_bars(ax, [str(L) for L in key_lens], series)
        ax.set_xlabel("key length")
        ax.set_title(NAMES[lang])
    fig.suptitle("Key groups: dictionary words versus random keys "
                 "(120-letter texts, n=%d)" % d["trials"], y=1.03)
    save(fig, "study_keygroups.png")


def chart_ablation(res):
    d = res["ablation"]
    keys = sorted({k for lang in LANGS for k in d["data"].get(lang, {})})
    fig, ax = plt.subplots(figsize=(11, 4.2))
    series = {NAMES[l]: [100.0 * d["data"].get(l, {}).get(k, {}).get("top1", 0)
                         for k in keys] for l in LANGS}
    _grouped_bars(ax, [k.replace(" + ", "\n+ ") for k in keys], series,
                  {NAMES[l]: COLORS[l] for l in LANGS})
    ax.tick_params(axis="x", labelsize=7)
    ax.set_title("Ablation: contribution of each component "
                 "(100-letter texts, random 6-letter keys, n=%d)" % d["trials"])
    save(fig, "study_ablation.png")


def chart_baselines(res):
    d = res["baselines"]
    order = ["ic_chi2", "hill_only", "brute_force", "full_system"]
    label = {"ic_chi2": "frequency\nanalysis only", "hill_only": "hill climbing\nonly",
             "brute_force": "brute force\n(<=4 letters)", "full_system": "full system"}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8))
    series = {NAMES[l]: [100.0 * d["data"].get(l, {}).get(k, {}).get("top1", 0)
                         for k in order] for l in LANGS}
    _grouped_bars(ax1, [label[k] for k in order], series,
                  {NAMES[l]: COLORS[l] for l in LANGS})
    ax1.set_title("Baselines vs the full system")
    for l in LANGS:
        secs = [d["data"].get(l, {}).get(k, {}).get("seconds", 0) for k in order]
        ax2.plot(range(len(order)), secs, "o-", color=COLORS[l], label=NAMES[l])
    ax2.set_xticks(range(len(order)), [label[k] for k in order])
    ax2.set_yscale("log")
    ax2.set_ylabel("seconds per attack")
    ax2.set_title("Cost")
    ax2.legend(fontsize=8)
    save(fig, "study_baselines.png")


def chart_domain(res):
    d = res["domain"]
    splits = d["splits"]
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    series = {NAMES[l]: [100.0 * d["data"].get(l, {}).get(s, {}).get("top1", 0)
                         for s in splits] for l in LANGS}
    _grouped_bars(ax, ["train\n(Tatoeba)", "validation\n(Wikipedia)",
                       "test\n(news)", "test 2\n(web/news)"][:len(splits)],
                  series, {NAMES[l]: COLORS[l] for l in LANGS})
    ax.set_title("Domain shift: the model is trained on Tatoeba only "
                 "(120 letters, random 6-letter keys, n=%d)" % d["trials"])
    save(fig, "study_domain.png")


def chart_weights(res):
    d = res["weights"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8))
    for split, style in (("validation", "--"), ("test", "-")):
        for lang in LANGS:
            row = d["data"].get(split, {}).get(lang, {})
            ws = [(float(k[2:]), 100.0 * v) for k, v in row.items()
                  if k.startswith("w=")]
            ps = [(float(k[2:]), 100.0 * v) for k, v in row.items()
                  if k.startswith("p=")]
            ws.sort(); ps.sort()
            ax1.plot([x for x, _ in ws], [y for _, y in ws], style, marker="o",
                     ms=3, color=COLORS[lang],
                     label="%s %s" % (NAMES[lang], split))
            ax2.plot([x for x, _ in ps], [y for _, y in ps], style, marker="o",
                     ms=3, color=COLORS[lang])
    ax1.axvline(6.0, color="#666", ls=":", lw=1)
    ax1.text(6.1, 5, "shipped value", fontsize=7, color="#666")
    ax1.set_xlabel("dictionary weight w")
    ax1.set_ylabel("top-1, %")
    ax1.set_title("Dictionary weight (dashed = validation, solid = test)")
    ax1.legend(fontsize=6, ncol=2)
    ax2.axvline(1.0, color="#666", ls=":", lw=1)
    ax2.set_xlabel("key-length penalty scale")
    ax2.set_ylabel("top-1, %")
    ax2.set_title("Key-length penalty")
    save(fig, "study_weights.png")


def chart_keylen(res):
    d = res["keylen"]
    lengths = d["lengths"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8))
    for lang in LANGS:
        row = d["data"].get(lang, {})
        ax1.plot(lengths, [100.0 * row.get(str(n), {}).get("ic_top1", 0)
                           for n in lengths], "o-", color=COLORS[lang],
                 label="%s IC top-1" % NAMES[lang])
        ax1.plot(lengths, [100.0 * row.get(str(n), {}).get("ic_top3", 0)
                           for n in lengths], "o--", color=COLORS[lang], alpha=0.5,
                 label="%s IC top-3" % NAMES[lang])
        ax2.plot(lengths, [100.0 * row.get(str(n), {}).get("ic_multiple", 0)
                           for n in lengths], "o-", color=COLORS[lang],
                 label=NAMES[lang])
    ax1.set_xlabel("ciphertext length, letters")
    ax1.set_ylabel("correct key length identified, %")
    ax1.set_title("Index of coincidence as a key-length detector")
    ax1.legend(fontsize=6, ncol=2)
    ax2.set_xlabel("ciphertext length, letters")
    ax2.set_ylabel("% of cases")
    ax2.set_title("IC top choice is a multiple of the true length")
    ax2.legend(fontsize=8)
    save(fig, "study_keylen.png")


def chart_dirty(res):
    d = res["dirty"]
    kinds = d["kinds"]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    series = {NAMES[l]: [100.0 * d["data"].get(l, {}).get(k, {}).get("top1", 0)
                         for k in kinds] for l in LANGS}
    _grouped_bars(ax, [k.replace("_", "\n") for k in kinds], series,
                  {NAMES[l]: COLORS[l] for l in LANGS})
    ax.set_title("Robustness to messy input (120 letters, random 6-letter keys, "
                 "n=%d)" % d["trials"])
    save(fig, "study_dirty.png")


def chart_variants(res):
    d = res["variants"]
    vs = d["variants"]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    series = {NAMES[l]: [100.0 * d["data"].get(l, {}).get(v, {}).get("top1", 0)
                         for v in vs] for l in LANGS}
    _grouped_bars(ax, vs, series, {NAMES[l]: COLORS[l] for l in LANGS})
    ax.set_title("Cipher variants (120 letters, random 6-letter keys, n=%d)"
                 % d["trials"])
    save(fig, "study_variants.png")


BUILDERS = {"matrix": [chart_matrix, chart_ratio], "keygroups": [chart_keygroups],
            "ablation": [chart_ablation], "baselines": [chart_baselines],
            "domain": [chart_domain], "weights": [chart_weights],
            "keylen": [chart_keylen], "dirty": [chart_dirty],
            "variants": [chart_variants]}


def main():
    if not os.path.exists(RESULTS):
        print("no results yet: %s" % RESULTS)
        return 1
    res = json.load(open(RESULTS, encoding="utf-8"))
    print("studies present:", ", ".join(k for k in res if k != "meta"))
    for key, fns in BUILDERS.items():
        if key in res:
            for fn in fns:
                try:
                    fn(res)
                except Exception as exc:                        # noqa: BLE001
                    print("  !! %s failed: %r" % (fn.__name__, exc))
    return 0


if __name__ == "__main__":
    sys.exit(main())
