# -*- coding: utf-8 -*-
"""Graphical interface.

Plaintext languages: English, Russian, Ukrainian. The interface itself is
localised into the same three languages and defaults to English.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import queue
import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from . import attack, calibrate, core, i18n, langs
from .paths import DATA_DIR

MODES = ["auto", "freq", "brute", "dict", "hill"]
FREQ_SEEDS = 320        # frequency-analysis attempts per key length;
                        # a single hill climb costs a fraction of a millisecond


def fmt_int(n):
    return f"{int(n):,}".replace(",", " ")


def fmt_time(s):
    if s is None or s <= 0 or s != s or s == float('inf'):
        return "-"
    s = int(s)
    if s >= 3600:
        return "%d:%02d:%02d" % (s // 3600, (s % 3600) // 60, s % 60)
    return "%02d:%02d" % (s // 60, s % 60)


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.tr = i18n.Translator(i18n.DEFAULT_UI)
        self.evq = queue.Queue()
        self.engine = None
        self.models = {}                 # alphabet key -> model
        self.rows = []
        self.manual_alpha = None
        self.calib = calibrate.load()
        self._closing = False
        self._poll_id = None
        self._i18n = []                  # (widget, string key) for live retranslation
        self.has_gpu = attack.gpu_available()

        self.title(self.tr('app.title'))
        self.geometry("1200x880")
        self.minsize(1000, 720)

        self._build_ui()
        self._poll()
        self._prepare_model_async()
        threading.Thread(target=self._calibrate_async, daemon=True).start()

    # ------------------------------------------------------------------ UI --
    def _lab(self, parent, key, **kw):
        """A label that follows the interface language."""
        w = ttk.Label(parent, text=self.tr(key), **kw)
        self._i18n.append((w, key))
        return w

    def _build_ui(self):
        st = ttk.Style(self)
        for theme in ("vista", "clam", "default"):
            try:
                st.theme_use(theme)
                break
            except tk.TclError:
                continue
        st.configure("Head.TLabel", font=("Segoe UI", 10, "bold"))
        st.configure("Big.TLabel", font=("Consolas", 11, "bold"))

        root = ttk.Frame(self, padding=8)
        root.pack(fill="both", expand=True)
        root.columnconfigure(0, weight=1)
        root.rowconfigure(3, weight=3)
        root.rowconfigure(5, weight=1)

        # ---------- ciphertext ----------
        self.f_in = ttk.LabelFrame(root, text=self.tr('sec.cipher'), padding=6)
        self._i18n.append((self.f_in, 'sec.cipher'))
        self.f_in.grid(row=0, column=0, sticky="ew")
        self.f_in.columnconfigure(0, weight=1)

        self.txt_in = tk.Text(self.f_in, height=3, font=("Consolas", 12), wrap="word")
        self.txt_in.grid(row=0, column=0, sticky="ew")
        self.txt_in.bind("<KeyRelease>", lambda e: self._update_counts())

        side = ttk.Frame(self.f_in)
        side.grid(row=0, column=1, sticky="ns", padx=(8, 0))
        self.b_paste = ttk.Button(side, text=self.tr('btn.paste'), width=12,
                                  command=self._paste)
        self._i18n.append((self.b_paste, 'btn.paste'))
        self.b_paste.pack(fill="x")
        self.b_clear = ttk.Button(side, text=self.tr('btn.clear'), width=12,
                                  command=lambda: (self.txt_in.delete("1.0", "end"),
                                                   self._update_counts()))
        self._i18n.append((self.b_clear, 'btn.clear'))
        self.b_clear.pack(fill="x", pady=3)
        self.lbl_count = ttk.Label(side, text="")
        self.lbl_count.pack(anchor="w")

        # ---------- settings ----------
        self.f_p = ttk.LabelFrame(root, text=self.tr('sec.params'), padding=6)
        self._i18n.append((self.f_p, 'sec.params'))
        self.f_p.grid(row=1, column=0, sticky="ew", pady=6)
        self.f_p.columnconfigure(7, weight=1)

        self._lab(self.f_p, 'lbl.uilang').grid(row=0, column=0, sticky="w", padx=(0, 4))
        self.cb_ui = ttk.Combobox(self.f_p, state="readonly", width=13,
                                  values=[i18n.UI_NAMES[c] for c in i18n.UI_LANGS])
        self.cb_ui.current(i18n.UI_LANGS.index(i18n.DEFAULT_UI))
        self.cb_ui.grid(row=0, column=1, sticky="w")
        self.cb_ui.bind("<<ComboboxSelected>>", lambda e: self._ui_lang_changed())

        self._lab(self.f_p, 'lbl.textlang').grid(row=0, column=2, sticky="w", padx=(12, 4))
        self.cb_lang = ttk.Combobox(self.f_p, state="readonly", width=15)
        self.cb_lang.grid(row=0, column=3, sticky="w")
        self.cb_lang.bind("<<ComboboxSelected>>", lambda e: self._text_lang_changed())

        self._lab(self.f_p, 'lbl.alphabet').grid(row=0, column=4, sticky="w", padx=(12, 4))
        self.cb_alpha = ttk.Combobox(self.f_p, state="readonly", width=24)
        self.cb_alpha.grid(row=0, column=5, sticky="w")
        self.cb_alpha.bind("<<ComboboxSelected>>",
                           lambda e: (self._prepare_model_async(), self._update_counts()))

        self._lab(self.f_p, 'lbl.variant').grid(row=0, column=6, sticky="w", padx=(12, 4))
        self.cb_var = ttk.Combobox(self.f_p, state="readonly", width=26)
        self.cb_var.grid(row=0, column=7, sticky="w")

        self._lab(self.f_p, 'lbl.mode').grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.cb_mode = ttk.Combobox(self.f_p, state="readonly", width=34)
        self.cb_mode.grid(row=1, column=1, columnspan=2, sticky="w", pady=(6, 0))
        self.cb_mode.bind("<<ComboboxSelected>>", lambda e: self._mode_changed())

        kl = ttk.Frame(self.f_p)
        kl.grid(row=1, column=3, columnspan=2, sticky="w", pady=(6, 0), padx=(12, 0))
        self._lab(kl, 'lbl.keylen').pack(side="left")
        self.sp_min = tk.Spinbox(kl, from_=1, to=40, width=4, command=self._update_counts)
        self.sp_min.delete(0, "end"); self.sp_min.insert(0, "1")
        self.sp_min.pack(side="left", padx=4)
        self._lab(kl, 'lbl.to').pack(side="left")
        self.sp_max = tk.Spinbox(kl, from_=1, to=40, width=4, command=self._update_counts)
        self.sp_max.delete(0, "end"); self.sp_max.insert(0, "20")
        self.sp_max.pack(side="left", padx=4)

        dev = ttk.Frame(self.f_p)
        dev.grid(row=1, column=5, columnspan=2, sticky="w", pady=(6, 0), padx=(12, 0))
        self._lab(dev, 'lbl.device').pack(side="left", padx=(0, 4))
        self.var_dev = tk.StringVar(value="gpu" if self.has_gpu else "cpu")
        ttk.Radiobutton(dev, text="CPU", value="cpu", variable=self.var_dev,
                        command=self._update_counts).pack(side="left")
        gpu_txt = attack.gpu_name() if self.has_gpu else self.tr('lbl.nogpu')
        self.rb_gpu = ttk.Radiobutton(dev, text="GPU (%s)" % gpu_txt, value="gpu",
                                      variable=self.var_dev, command=self._update_counts)
        self.rb_gpu.pack(side="left", padx=(8, 0))
        if not self.has_gpu:
            self.rb_gpu.state(["disabled"])

        pr = ttk.Frame(self.f_p)
        pr.grid(row=1, column=7, sticky="w", pady=(6, 0), padx=(12, 0))
        self._lab(pr, 'lbl.procs').pack(side="left", padx=(0, 4))
        self.sp_proc = tk.Spinbox(pr, from_=1, to=256, width=5,
                                  command=self._update_counts)
        self.sp_proc.delete(0, "end"); self.sp_proc.insert(0, str(os.cpu_count() or 4))
        self.sp_proc.pack(side="left")

        pn = ttk.Frame(self.f_p)
        pn.grid(row=2, column=0, columnspan=4, sticky="w", pady=(6, 0))
        self._lab(pn, 'lbl.penalty').pack(side="left", padx=(0, 4))
        self.e_pen = ttk.Entry(pn, width=6)
        self.e_pen.insert(0, "3.47")
        self.e_pen.pack(side="left")
        self._lab(pn, 'lbl.penalty.hint', foreground="#666").pack(side="left", padx=(6, 0))

        rs = ttk.Frame(self.f_p)
        rs.grid(row=2, column=4, columnspan=2, sticky="w", pady=(6, 0), padx=(12, 0))
        self._lab(rs, 'lbl.restarts').pack(side="left", padx=(0, 4))
        self.sp_restart = tk.Spinbox(rs, from_=8, to=100000, width=7)
        self.sp_restart.delete(0, "end"); self.sp_restart.insert(0, "2000")
        self.sp_restart.pack(side="left")

        self.var_rev = tk.BooleanVar(value=True)
        self.ck_rev = ttk.Checkbutton(self.f_p, text=self.tr('lbl.reversed'),
                                      variable=self.var_rev)
        self._i18n.append((self.ck_rev, 'lbl.reversed'))
        self.ck_rev.grid(row=2, column=6, columnspan=2, sticky="w", pady=(6, 0),
                         padx=(12, 0))

        # ---------- run controls and progress ----------
        self.f_run = ttk.LabelFrame(root, text=self.tr('sec.run'), padding=6)
        self._i18n.append((self.f_run, 'sec.run'))
        self.f_run.grid(row=2, column=0, sticky="ew")
        self.f_run.columnconfigure(1, weight=1)

        btns = ttk.Frame(self.f_run)
        btns.grid(row=0, column=0, rowspan=2, sticky="w", padx=(0, 10))
        self.b_start = ttk.Button(btns, text=self.tr('btn.start'), width=14,
                                  command=self.start)
        self._i18n.append((self.b_start, 'btn.start'))
        self.b_start.pack(fill="x")
        self.b_stop = ttk.Button(btns, text=self.tr('btn.stop'), width=14,
                                 command=self.stop, state="disabled")
        self._i18n.append((self.b_stop, 'btn.stop'))
        self.b_stop.pack(fill="x", pady=3)

        self.pb = ttk.Progressbar(self.f_run, mode="determinate", maximum=1000)
        self.pb.grid(row=0, column=1, sticky="ew")
        self.lbl_stage = ttk.Label(self.f_run, text=self.tr('st.ready'),
                                   style="Head.TLabel")
        self.lbl_stage.grid(row=1, column=1, sticky="w", pady=(4, 0))

        stats = ttk.Frame(self.f_run)
        stats.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        self.lbl_stats = ttk.Label(stats, text="", font=("Consolas", 9))
        self.lbl_stats.pack(side="left")
        self.lbl_space = ttk.Label(stats, text="", foreground="#0a5", font=("Consolas", 9))
        self.lbl_space.pack(side="right")

        # ---------- results ----------
        self.f_res = ttk.LabelFrame(root, text=self.tr('sec.results'), padding=6)
        self._i18n.append((self.f_res, 'sec.results'))
        self.f_res.grid(row=3, column=0, sticky="nsew", pady=6)
        self.f_res.columnconfigure(0, weight=1)
        self.f_res.rowconfigure(0, weight=1)

        self.cols = ("n", "key", "alpha", "score", "ng", "words", "plain")
        self.tree = ttk.Treeview(self.f_res, columns=self.cols, show="headings", height=12)
        for c, w, a in (("n", 40, "e"), ("key", 150, "w"), ("alpha", 60, "e"),
                        ("score", 80, "e"), ("ng", 90, "e"), ("words", 50, "e"),
                        ("plain", 560, "w")):
            self.tree.column(c, width=w, anchor=a, stretch=(c == "plain"))
        self.tree.grid(row=0, column=0, sticky="nsew")
        sb = ttk.Scrollbar(self.f_res, orient="vertical", command=self.tree.yview)
        sb.grid(row=0, column=1, sticky="ns")
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.bind("<Double-1>", self._copy_row)
        self.tree.bind("<<TreeviewSelect>>", self._show_row)
        self.tree.tag_configure("top", background="#eaf7ea")

        # ---------- manual check ----------
        f_m = ttk.Frame(root)
        f_m.grid(row=4, column=0, sticky="ew")
        f_m.columnconfigure(3, weight=1)
        self._lab(f_m, 'lbl.manual').grid(row=0, column=0, sticky="w")
        self.e_key = ttk.Entry(f_m, width=24, font=("Consolas", 11))
        self.e_key.grid(row=0, column=1, sticky="w", padx=6)
        self.e_key.bind("<KeyRelease>", lambda e: self._manual())
        self.b_exp = ttk.Button(f_m, text=self.tr('btn.export'), command=self._export)
        self._i18n.append((self.b_exp, 'btn.export'))
        self.b_exp.grid(row=0, column=2, padx=(0, 8))
        self.lbl_manual = ttk.Label(f_m, text="", style="Big.TLabel", foreground="#036")
        self.lbl_manual.grid(row=0, column=3, sticky="w")

        # ---------- log ----------
        self.f_log = ttk.LabelFrame(root, text=self.tr('sec.log'), padding=4)
        self._i18n.append((self.f_log, 'sec.log'))
        self.f_log.grid(row=5, column=0, sticky="nsew", pady=(6, 0))
        self.f_log.columnconfigure(0, weight=1)
        self.f_log.rowconfigure(0, weight=1)
        self.txt_log = tk.Text(self.f_log, height=7, font=("Consolas", 9), wrap="word",
                               state="disabled", background="#fbfbfb")
        self.txt_log.grid(row=0, column=0, sticky="nsew")
        sb2 = ttk.Scrollbar(self.f_log, orient="vertical", command=self.txt_log.yview)
        sb2.grid(row=0, column=1, sticky="ns")
        self.txt_log.configure(yscrollcommand=sb2.set)

        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._fill_lang_combo()
        self._retranslate()

    # ------------------------------------------------ languages and text --
    def _fill_lang_combo(self):
        ui = self.tr.lang
        self.cb_lang['values'] = [langs.lang_name(c, ui) for c in langs.lang_codes()]
        if not self.cb_lang.get():
            self.cb_lang.current(langs.lang_codes().index(langs.DEFAULT_LANG))
        self._fill_alpha_combo()

    def _fill_alpha_combo(self):
        variants = list(langs.get(self.text_lang())["variants"])
        vals = []
        if len(variants) > 1:
            vals.append(self.tr('alpha.auto'))
        vals += [self.tr('alpha.one', v) for v in variants]
        keep = self.cb_alpha.current()
        self.cb_alpha['values'] = vals
        self.cb_alpha.current(0 if keep < 0 or keep >= len(vals) else keep)

    def _retranslate(self):
        tr = self.tr
        self.title(tr('app.title'))
        for w, key in self._i18n:
            try:
                w.configure(text=tr(key))
            except tk.TclError:
                pass
        for c, key in zip(self.cols, ('col.n', 'col.key', 'col.alpha', 'col.score',
                                      'col.ng', 'col.words', 'col.plain')):
            self.tree.heading(c, text=tr(key))
        gpu_txt = attack.gpu_name() if self.has_gpu else tr('lbl.nogpu')
        self.rb_gpu.configure(text="GPU (%s)" % gpu_txt)

        mi = max(0, self.cb_mode.current())
        self.cb_mode['values'] = [tr('mode.' + m) for m in MODES]
        self.cb_mode.current(mi)
        vi = max(0, self.cb_var.current())
        self.cb_var['values'] = [tr('cip.' + v) for v in core.VARIANTS]
        self.cb_var.current(vi)
        li = max(0, self.cb_lang.current())
        self._fill_lang_combo()
        self.cb_lang.current(li)
        self._fill_alpha_combo()
        if self.lbl_stage.cget("text") in [i18n.tr(l, 'st.ready') for l in i18n.UI_LANGS]:
            self.lbl_stage.configure(text=tr('st.ready'))
        self._update_counts()

    def _ui_lang_changed(self):
        self.tr.set(i18n.UI_LANGS[self.cb_ui.current()])
        self._retranslate()

    def _text_lang_changed(self):
        self._fill_alpha_combo()
        self.models = {}
        self.manual_alpha = None
        self._prepare_model_async()
        self._update_counts()

    def text_lang(self):
        return langs.lang_codes()[max(0, self.cb_lang.current())]

    def _alpha_keys(self):
        """Alphabet keys that will be tested."""
        lang = self.text_lang()
        variants = list(langs.get(lang)["variants"])
        i = max(0, self.cb_alpha.current())
        if len(variants) > 1:
            if i == 0:
                return langs.auto_keys(lang)
            return ["%s-%s" % (lang, variants[i - 1])]
        return ["%s-%s" % (lang, variants[i])]

    def _alpha(self):
        return core.get_alphabet(self._alpha_keys()[0])

    def _variant(self):
        return core.VARIANTS[max(0, self.cb_var.current())]

    def _mode(self):
        return MODES[max(0, self.cb_mode.current())]

    # -------------------------------------------------- background work --
    def _prepare_model_async(self, force=False):
        need = [k for k in self._alpha_keys() if force or k not in self.models]
        if not need:
            return
        self.log(self.tr('log.buildmodel', ", ".join(need)))

        def work():
            for key in need:
                t0 = time.time()
                m = core.get_model(key)
                self.evq.put({'type': 'model', 'name': key, 'model': m,
                              'secs': time.time() - t0})
        threading.Thread(target=work, daemon=True).start()

    def _calibrate_async(self):
        """Measure this machine's throughput so the time estimates are real."""
        if self.has_gpu:
            t0 = time.time()
            if attack.gpu_warmup():
                self.evq.put({'type': 'log',
                              'msg': self.tr('log.cuda', attack.gpu_name(),
                                             time.time() - t0)})
        data = calibrate.measure(want_gpu=self.has_gpu)
        self.evq.put({'type': 'calib', 'data': data})

    # ------------------------------------------------------------ helpers --
    def log(self, msg):
        self.txt_log.configure(state="normal")
        self.txt_log.insert("end", time.strftime("[%H:%M:%S] ") + msg + "\n")
        self.txt_log.see("end")
        self.txt_log.configure(state="disabled")

    def _paste(self):
        try:
            self.txt_in.insert("insert", self.clipboard_get())
            self._update_counts()
        except tk.TclError:
            pass

    def _text(self):
        return self.txt_in.get("1.0", "end-1c")

    def _rate(self):
        try:
            nproc = int(self.sp_proc.get())
        except (ValueError, tk.TclError):
            nproc = os.cpu_count() or 4
        return calibrate.rate(self.var_dev.get() == "gpu", nproc, self.calib)

    def _mode_changed(self):
        mode = self._mode()
        if mode in ('freq', 'hill', 'auto'):
            self.sp_max.delete(0, "end"); self.sp_max.insert(0, "20")
        elif mode == 'brute':
            self.sp_max.delete(0, "end"); self.sp_max.insert(0, "4")
        self._update_counts()

    def _update_counts(self):
        tr = self.tr
        n = len(self._alpha().to_indices(self._text()))
        self.lbl_count.configure(text=tr('lbl.letters', n))
        try:
            lo, hi = int(self.sp_min.get()), int(self.sp_max.get())
        except (ValueError, tk.TclError):
            return
        mode = self._mode()
        if mode == 'brute':
            total = sum(core.get_alphabet(k).M ** L
                        for k in self._alpha_keys()
                        for L in range(max(1, lo), max(lo, hi) + 1))
            rate = self._rate()
            self.lbl_space.configure(
                foreground="#0a5",
                text=tr('hint.space', fmt_int(total), fmt_time(total / rate + 5),
                        "GPU" if self.var_dev.get() == "gpu" else "CPU", fmt_int(rate)))
        elif mode == 'auto':
            if n == 0:
                self.lbl_space.configure(text="")
            elif n < attack.Engine.AUTO_SHORT:
                self.lbl_space.configure(
                    foreground="#a60", text=tr('hint.short', n, attack.Engine.AUTO_SHORT))
            else:
                self.lbl_space.configure(foreground="#0a5", text=tr('hint.long', n))
        elif mode == 'freq':
            self.lbl_space.configure(foreground="#0a5", text=tr('hint.freq'))
        else:
            self.lbl_space.configure(text="")

    def _manual(self, alpha_key=None):
        key = self.e_key.get().strip()
        if not key:
            self.lbl_manual.configure(text="")
            return
        k = alpha_key or self.manual_alpha or self._alpha_keys()[0]
        try:
            res = core.decrypt_text(self._text(), key, core.get_alphabet(k),
                                    self._variant())
        except Exception:                                        # noqa: BLE001
            res = ""
        self.lbl_manual.configure(text="(%s) %s" % (k, res[:100]))

    # -------------------------------------------------------------- start --
    def start(self):
        tr = self.tr
        if self.engine and self.engine.is_alive():
            return
        text = self._text().strip()
        lang = self.text_lang()
        keys = self._alpha_keys()
        if len(core.get_alphabet(keys[0]).to_indices(text)) < 3:
            messagebox.showwarning(tr('app.title'), tr('msg.needtext'))
            return
        if any(k not in self.models for k in keys):
            messagebox.showinfo(tr('app.title'), tr('msg.modelwait'))
            self._prepare_model_async()
            return
        if not core.has_data(lang) and not messagebox.askyesno(
                tr('msg.nodata.title'),
                tr('msg.nodata', langs.lang_name(lang, self.tr.lang), lang)):
            return
        alpha_list = [(k, self.models[k]) for k in keys]

        try:
            lo, hi = int(self.sp_min.get()), int(self.sp_max.get())
            pen = float(self.e_pen.get().replace(",", "."))
            nproc = int(self.sp_proc.get())
            restarts = int(self.sp_restart.get())
        except ValueError:
            messagebox.showwarning(tr('app.title'), tr('msg.numbers'))
            return
        if hi < lo:
            lo, hi = hi, lo
        mode = self._mode()

        if mode == 'brute':
            total = sum(core.get_alphabet(k).M ** L
                        for k in keys for L in range(lo, hi + 1))
            est = total / self._rate()
            if est > 300 and not messagebox.askyesno(
                    tr('app.title'), tr('msg.bigspace', fmt_int(total), fmt_time(est))):
                return

        dict_words = []
        if mode == 'dict':
            dict_words = sorted(self.models[keys[0]].words
                                | set(langs.builtin_words(lang)))
            self.log(tr('log.dictfull', fmt_int(len(dict_words))))
        elif mode == 'auto':
            dict_words = sorted(set(core.load_frequent_words(lang))
                                | set(langs.builtin_words(lang)))
            self.log(tr('log.dictfreq', fmt_int(len(dict_words))))

        cfg = dict(text=text, alpha_list=alpha_list, variant=self._variant(), mode=mode,
                   min_len=lo, max_len=hi, processes=nproc, restarts=restarts,
                   keylen_penalty=pen, dict_words=dict_words,
                   dict_reversed=self.var_rev.get(), freq_seeds=FREQ_SEEDS,
                   use_gpu=(self.var_dev.get() == "gpu" and self.has_gpu),
                   gpu_batch=None, tr=self.tr)

        self.tree.delete(*self.tree.get_children())
        self.rows = []
        self.pb['value'] = 0
        self.b_start.state(["disabled"])
        self.b_stop.state(["!disabled"])
        self.lbl_stage.configure(text=tr('st.starting'))
        self.log("=" * 70)
        self.log(tr('log.start', tr('mode.' + mode), lo, hi,
                    "GPU" if cfg['use_gpu'] else "CPU x%d" % nproc))

        self.engine = attack.Engine(cfg, self.evq)
        self.engine.start()

    def stop(self):
        if self.engine:
            self.lbl_stage.configure(text=self.tr('st.stopping'))
            self.engine.request_stop()
        self.b_stop.state(["disabled"])

    # ------------------------------------------------------------- events --
    def _poll(self):
        if self._closing:
            return
        try:
            while True:
                self._handle(self.evq.get_nowait())
        except queue.Empty:
            pass
        except tk.TclError:
            return
        self._poll_id = self.after(70, self._poll)

    def _handle(self, ev):
        t = ev.get('type')
        tr = self.tr
        if t == 'log':
            self.log(ev['msg'])
        elif t == 'model':
            self.models[ev['name']] = ev['model']
            self.log(tr('log.modelready', ev['name'], ev.get('secs', 0),
                        fmt_int(len(ev['model'].words))))
            self._update_counts()
        elif t == 'calib':
            self.calib = ev['data']
            dev = ev['data'].get('gpu_name') or ev['data'].get('cpu_name', 'CPU')
            self.log(tr('log.calib', fmt_int(self._rate()), dev))
            self._update_counts()
        elif t == 'progress':
            frac = min(1.0, ev['done'] / ev['total']) if ev['total'] else 0
            self.pb['value'] = max(0, min(1000, frac * 1000))
            self.lbl_stage.configure(text=ev['stage'])
            self.lbl_stats.configure(
                text=tr('st.stats', frac * 100, fmt_int(ev['done']), fmt_int(ev['total']),
                        fmt_int(ev['rate']), fmt_time(ev['elapsed']), fmt_time(ev['eta'])))
        elif t == 'interim':
            self._fill(ev['rows'], interim=True)
        elif t == 'results':
            self._fill(ev['rows'], interim=False)
        elif t == 'done':
            self.b_start.state(["!disabled"])
            self.b_stop.state(["disabled"])
            if ev.get('ok'):
                self.pb['value'] = 1000
                self.lbl_stage.configure(
                    text=(tr('st.stopped') if ev.get('stopped')
                          else tr('st.done', fmt_time(ev.get('elapsed', 0)))))
            else:
                self.lbl_stage.configure(text=tr('st.failed'))

    def _fill(self, rows, interim):
        self.rows = rows
        self.tree.delete(*self.tree.get_children())
        for i, r in enumerate(rows, 1):
            self.tree.insert("", "end", iid=str(i - 1),
                             values=(i, r['key'], r['alpha'], "%.1f" % r['final'],
                                     "%.3f" % r['ngpc'], r['words'],
                                     r['plain'].replace("\n", " ")),
                             tags=("top",) if i <= 3 else ())
        if rows and not interim:
            self.log(self.tr('log.best', rows[0]['key'], rows[0]['plain'][:80]))

    def _copy_row(self, _e):
        sel = self.tree.selection()
        if not sel:
            return
        r = self.rows[int(sel[0])]
        self.clipboard_clear()
        self.clipboard_append(r['plain'])
        self.log(self.tr('log.copied', r['plain'][:80]))

    def _show_row(self, _e):
        sel = self.tree.selection()
        if not sel:
            return
        r = self.rows[int(sel[0])]
        self.manual_alpha = r['alpha']
        self.e_key.delete(0, "end")
        self.e_key.insert(0, r['key'])
        self.lbl_manual.configure(text="(%s) %s" % (r['alpha'], r['plain'][:100]))

    def _export(self):
        tr = self.tr
        if not self.rows:
            messagebox.showinfo(tr('app.title'), tr('msg.noresults'))
            return
        p = filedialog.asksaveasfilename(defaultextension=".csv",
                                         filetypes=[("CSV", "*.csv")],
                                         initialfile="vigenere_results.csv")
        if not p:
            return
        import csv
        with open(p, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f, delimiter=";")
            w.writerow([tr('col.n'), tr('col.key'), tr('col.alpha'), "len",
                        tr('col.score'), tr('col.ng'), "coverage", tr('col.words'),
                        tr('col.plain')])
            for i, r in enumerate(self.rows, 1):
                w.writerow([i, r['key'], r['alpha'], r['keylen'], "%.2f" % r['final'],
                            "%.4f" % r['ngpc'], "%.3f" % r['cov'], r['words'], r['plain']])
        self.log(tr('log.exported', p))

    def _on_close(self):
        self._closing = True
        if self._poll_id is not None:
            try:
                self.after_cancel(self._poll_id)
            except tk.TclError:
                pass
        if self.engine and self.engine.is_alive():
            self.engine.request_stop()
            self.after(400, self.destroy)
        else:
            self.destroy()


def main():
    mp.freeze_support()
    App().mainloop()


if __name__ == "__main__":
    main()
