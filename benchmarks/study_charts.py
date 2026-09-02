# -*- coding: utf-8 -*-
"""Charts for the research studies (reads benchmarks/results/studies.json).

Where a study exists in both an easy and a hard variant, the hard one is used:
the easy operating point saturates at 100 % and shows nothing. Error bars are
95 % Wilson intervals.
"""

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


def save(fig, name, tight=True):
    os.makedirs(DOCS, exist_ok=True)
    if tight:
        fig.tight_layout()
    fig.savefig(os.path.join(DOCS, name), bbox_inches="tight")
    plt.close(fig)
    print("  ->", name)


def pick(res, name):
    """Prefer the hard-operating-point variant of a study."""
    return res.get(name + "_hard") or res.get(name)


def cell(d, key):
    """A summary dict, tolerant of the older shape without intervals."""
    v = d.get(key)
    if not isinstance(v, dict):
        return None
    return {"top1": v.get("top1", 0.0),
            "lo": v.get("ci_lo", v.get("top1", 0.0)),
            "hi": v.get("ci_hi", v.get("top1", 0.0)),
            "n": v.get("n", 0), "seconds": v.get("seconds", 0.0),
            "gap": v.get("median_gap")}


def bars_with_ci(ax, categories, series, colors=None, ylabel="top-1, %"):
    x = np.arange(len(categories))
    w = 0.8 / max(1, len(series))
    for i, (name, vals) in enumerate(series.items()):
        pos = x + i * w - 0.4 + w / 2
        heights = [100 * v["top1"] if v else 0 for v in vals]
        err = np.array([[100 * (v["top1"] - v["lo"]) if v else 0 for v in vals],
                        [100 * (v["hi"] - v["top1"]) if v else 0 for v in vals]])
        ax.bar(pos, heights, width=w, label=name, color=(colors or {}).get(name),
               yerr=err, capsize=2, error_kw={"lw": 0.8, "ecolor": "#444"})
    ax.set_xticks(x, categories)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 108)
    ax.legend(fontsize=8)


# --------------------------------------------------------------------------- #
def chart_matrix(res):
    d = res["matrix"]
    lengths, key_lens = d["lengths"], d["key_lens"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9), constrained_layout=True)
    im = None
    for ax, lang in zip(axes, LANGS):
        grid = d["data"].get(lang, {})
        Z = np.full((len(key_lens), len(lengths)), np.nan)
        for i, L in enumerate(key_lens):
            for j, n in enumerate(lengths):
                c = grid.get("%d,%d" % (n, L))
                if c:
                    Z[i, j] = 100.0 * c["top1"]
        im = ax.imshow(Z, origin="lower", aspect="auto", cmap="RdYlGn",
                       vmin=0, vmax=100)
        ax.set_xticks(range(len(lengths)), [str(x) for x in lengths])
        ax.set_yticks(range(len(key_lens)), [str(x) for x in key_lens])
        ax.set_xlabel("ciphertext length N")
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
                 "(random keys, held-out news corpus)")
    save(fig, "study_matrix.png", tight=False)


def chart_ratio(res):
    d = res["matrix"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8))
    allpts = []
    for lang in LANGS:
        pts = sorted((c["ratio"], 100.0 * c["top1"])
                     for c in d["data"].get(lang, {}).values() if c)
        allpts += [(r, v) for r, v in pts]
        ax1.scatter([p[0] for p in pts], [p[1] for p in pts], s=16,
                    color=COLORS[lang], alpha=0.75, label=NAMES[lang])
    ax1.set_xscale("log")
    ax1.set_xlabel("N / L  (letters per key position)")
    ax1.set_ylabel("top-1, %")
    ax1.set_title("Every matrix cell against N/L")
    ax1.legend(); ax1.set_ylim(-3, 103)

    bins = [1.5, 2.5, 3.5, 4.5, 6, 8, 11, 16, 25, 50, 300]
    centres, means = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        vals = [v for r, v in allpts if lo <= r < hi]
        if vals:
            centres.append((lo * hi) ** 0.5)
            means.append(float(np.mean(vals)))
    ax2.plot(centres, means, "o-", color="#333")
    for thr, col in ((50, "#d62728"), (90, "#1f77b4")):
        ax2.axhline(thr, color=col, ls="--", lw=1)
        ax2.text(centres[0], thr + 1.5, "%d %%" % thr, color=col, fontsize=8)
    ax2.set_xscale("log")
    ax2.set_xlabel("N / L")
    ax2.set_ylabel("mean top-1, %")
    ax2.set_title("Pooled over three languages")
    ax2.set_ylim(-3, 103)
    save(fig, "study_ratio.png")


def chart_keygroups(res):
    d = pick(res, "keygroups")
    if not d or "groups" not in d:
        return
    groups = d["groups"]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    series = {NAMES[l]: [cell(d["data"].get(l, {}), g) for g in groups]
              for l in LANGS}
    bars_with_ci(ax, [g + "\n" + {"dict": "(in dictionary)", "oov": "(real, unknown)",
                                  "random": "(random letters)",
                                  "repeat": "(patterned)"}[g] for g in groups],
                 series, {NAMES[l]: COLORS[l] for l in LANGS})
    ax.set_title("Key groups at N/L = 5 (points %s, n=%d per point)"
                 % (d.get("points"), d["trials"]))
    save(fig, "study_keygroups.png")


ABLATION_ORDER = [
    "freq_only + trigram_no_dict", "freq_only + quadgram_no_dict",
    "freq_only + quadgram_dict", "freq_only + no_penalty",
    "freq_hill + trigram_no_dict", "freq_hill + quadgram_no_dict",
    "freq_hill + quadgram_dict", "freq_hill + no_penalty",
]


def chart_ablation(res):
    d = pick(res, "ablation")
    if not d:
        return
    present = set(d["data"]["en"].keys())
    keys = [k for k in ABLATION_ORDER if k in present] or sorted(present)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.4),
                                   gridspec_kw={"width_ratios": [2, 1]})
    series = {NAMES[l]: [cell(d["data"].get(l, {}), k) for k in keys] for l in LANGS}
    bars_with_ci(ax1, [k.replace(" + ", "\n+ ") for k in keys], series,
                 {NAMES[l]: COLORS[l] for l in LANGS})
    ax1.tick_params(axis="x", labelsize=6.5)
    ax1.set_title("Ablation: top-1 accuracy")

    # The margin is only meaningful where the true key is actually found often
    # enough for a median to mean anything; below 5 % it is a handful of samples.
    for l in LANGS:
        gaps = [cell(d["data"].get(l, {}), k) for k in keys]
        xs = [i for i, g in enumerate(gaps) if g and g["gap"] and g["top1"] >= 0.05]
        ys = [gaps[i]["gap"] for i in xs]
        ax2.plot(xs, ys, "o-", color=COLORS[l], label=NAMES[l], ms=4)
    ax2.set_xticks(range(len(keys)), [str(i + 1) for i in range(len(keys))])
    ax2.set_xlabel("configuration (numbered as on the left)")
    ax2.set_ylabel("median score margin, nats")
    ax2.set_title("Confidence margin (only where top-1 >= 5 %)")
    ax2.legend(fontsize=8)
    save(fig, "study_ablation.png")


def chart_baselines(res):
    d = pick(res, "baselines")
    if not d:
        return
    order = ["ic_chi2", "hill_only", "brute_force", "full_system"]
    label = {"ic_chi2": "frequency\nanalysis only", "hill_only": "hill climbing\nonly",
             "brute_force": "brute force\n(L<=4 only)", "full_system": "full system"}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.9))
    series = {NAMES[l]: [cell(d["data"].get(l, {}), k) for k in order] for l in LANGS}
    bars_with_ci(ax1, [label[k] for k in order], series,
                 {NAMES[l]: COLORS[l] for l in LANGS})
    ax1.set_title("Baselines versus the full system")
    for l in LANGS:
        secs = [(cell(d["data"].get(l, {}), k) or {}).get("seconds", 0) for k in order]
        ax2.plot(range(len(order)), [max(s, 1e-3) for s in secs], "o-",
                 color=COLORS[l], label=NAMES[l])
    ax2.set_xticks(range(len(order)), [label[k] for k in order], fontsize=7)
    ax2.set_yscale("log"); ax2.set_ylabel("seconds per attack")
    ax2.set_title("Cost"); ax2.legend(fontsize=8)
    save(fig, "study_baselines.png")


def chart_domain(res):
    d = pick(res, "domain")
    if not d:
        return
    splits = [s for s in d["splits"] if s in d["data"].get("en", {})]
    label = {"train": "train\n(Tatoeba)", "validation": "validation\n(Wikipedia)",
             "test": "test\n(news)", "test_web": "test 2\n(web / news)"}
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    series = {NAMES[l]: [cell(d["data"].get(l, {}), s) for s in splits] for l in LANGS}
    bars_with_ci(ax, [label[s] for s in splits], series,
                 {NAMES[l]: COLORS[l] for l in LANGS})
    ax.set_title("Domain shift: the model is built from Tatoeba only "
                 "(N/L = 5, n=%d per point)" % d["trials"])
    save(fig, "study_domain.png")


def chart_variants(res):
    d = pick(res, "variants")
    if not d:
        return
    vs = d["variants"]
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    series = {NAMES[l]: [cell(d["data"].get(l, {}), v) for v in vs] for l in LANGS}
    bars_with_ci(ax, vs, series, {NAMES[l]: COLORS[l] for l in LANGS})
    ax.set_title("Cipher variants at N/L = 5 (n=%d per point)" % d["trials"])
    save(fig, "study_variants.png")


def chart_dirty(res):
    d = pick(res, "dirty")
    if not d:
        return
    kinds = d["kinds"]
    fig, ax = plt.subplots(figsize=(9.5, 3.8))
    series = {NAMES[l]: [cell(d["data"].get(l, {}), k) for k in kinds] for l in LANGS}
    bars_with_ci(ax, [k.replace("_", "\n") for k in kinds], series,
                 {NAMES[l]: COLORS[l] for l in LANGS})
    ax.set_title("Robustness to messy input (N/L = 5, n=%d per point)" % d["trials"])
    save(fig, "study_dirty.png")


def chart_weights(res):
    d = res.get("weights")
    if not d:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8))
    for split, style in (("validation", "--"), ("test", "-")):
        for lang in LANGS:
            row = d["data"].get(split, {}).get(lang, {})
            ws = sorted((float(k[2:]), 100.0 * v) for k, v in row.items()
                        if k.startswith("w="))
            ps = sorted((float(k[2:]), 100.0 * v) for k, v in row.items()
                        if k.startswith("p="))
            if ws:
                ax1.plot([x for x, _ in ws], [y for _, y in ws], style, marker="o",
                         ms=3, color=COLORS[lang],
                         label="%s %s" % (NAMES[lang], split))
            if ps:
                ax2.plot([x for x, _ in ps], [y for _, y in ps], style, marker="o",
                         ms=3, color=COLORS[lang])
    ax1.axvline(6.0, color="#666", ls=":", lw=1)
    ax1.text(6.1, 2, "shipped 6.0", fontsize=7, color="#666")
    ax1.set_xlabel("dictionary weight w"); ax1.set_ylabel("top-1, %")
    ax1.set_title("Dictionary weight (dashed = validation, solid = test)")
    ax1.legend(fontsize=6, ncol=2)
    ax2.axvline(1.0, color="#666", ls=":", lw=1)
    ax2.set_xlabel("key-length penalty scale"); ax2.set_ylabel("top-1, %")
    ax2.set_title("Key-length penalty")
    save(fig, "study_weights.png")


def chart_keylen(res):
    d = res.get("keylen")
    if not d:
        return
    lengths = d["lengths"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8))
    for lang in LANGS:
        row = d["data"].get(lang, {})
        ax1.plot(lengths, [100 * row.get(str(n), {}).get("ic_top1", 0)
                           for n in lengths], "o-", color=COLORS[lang],
                 label="%s, exact" % NAMES[lang])
        ax1.plot(lengths, [100 * row.get(str(n), {}).get("ic_top3", 0)
                           for n in lengths], "o--", color=COLORS[lang], alpha=0.5,
                 label="%s, in top 3" % NAMES[lang])
        ax2.plot(lengths, [100 * row.get(str(n), {}).get("ic_multiple", 0)
                           for n in lengths], "o-", color=COLORS[lang],
                 label=NAMES[lang])
    ax1.set_xlabel("ciphertext length, letters")
    ax1.set_ylabel("key length identified, %")
    ax1.set_title("Index of coincidence as a key-length detector")
    ax1.legend(fontsize=6, ncol=2); ax1.set_ylim(0, 100)
    ax2.set_xlabel("ciphertext length, letters")
    ax2.set_ylabel("% of cases")
    ax2.set_title("Top IC choice is a multiple of the true length")
    ax2.legend(fontsize=8); ax2.set_ylim(0, 100)
    save(fig, "study_keylen.png")


BUILDERS = [chart_matrix, chart_ratio, chart_keygroups, chart_ablation,
            chart_baselines, chart_domain, chart_variants, chart_dirty,
            chart_weights, chart_keylen]


def main():
    if not os.path.exists(RESULTS):
        print("no results yet: %s" % RESULTS)
        return 1
    res = json.load(open(RESULTS, encoding="utf-8"))
    print("studies present:", ", ".join(k for k in res if k != "meta"))
    for fn in BUILDERS:
        try:
            fn(res)
        except Exception as exc:                                # noqa: BLE001
            print("  !! %s: %r" % (fn.__name__, exc))
    return 0


if __name__ == "__main__":
    sys.exit(main())
