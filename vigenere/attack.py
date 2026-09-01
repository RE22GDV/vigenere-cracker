# -*- coding: utf-8 -*-
"""Attack engine: brute force (CPU pool / CUDA), frequency analysis,
dictionary attack and hill climbing."""

from __future__ import annotations

import heapq
import math
import multiprocessing as mp
import os
import random
import tempfile
import threading
import time

import numpy as np

from . import core, i18n

TOPK_PER_TASK = 60          # best keys returned by a single task
FINAL_POOL = 4000           # candidates promoted to the second scoring stage
COV_LIMIT = 600             # leaders that get the expensive dictionary search
KEYS_PER_TASK = 1 << 21     # brute-force work unit for the process pool
BATCH = 1 << 16             # vectorised batch inside a worker

W_WORD = 6.0                # weight of dictionary coverage (nats per letter).
                            # Tuned on 18 short texts: a stable 18/18 plateau
                            # spans 5..14 with MIN_WORD=3; the low end is used
                            # so letter statistics keep influence on rare words.


# --------------------------------------------------------------------------- #
#  Vectorised brute-force kernel (numpy)
# --------------------------------------------------------------------------- #
def _decrypt_batch(c32, kexp, M, variant):
    if variant == "vigenere":
        return (c32 - kexp) % M
    if variant == "beaufort":
        return (kexp - c32) % M
    return (c32 + kexp) % M


def scan_range_numpy(c32, tri, M, L, start, end, variant, pos,
                     topk=TOPK_PER_TASK, batch=BATCH,
                     on_batch=None, should_stop=None):
    """Test key ordinals ``[start, end)`` for length ``L``.
    Returns ``[(score, key_ordinal), ...]``."""
    n = c32.shape[0]
    heap = []
    s = start
    while s < end:
        if should_stop is not None and should_stop():
            break
        e = min(s + batch, end)
        nums = np.arange(s, e, dtype=np.int64)
        B = nums.shape[0]

        K = np.empty((B, L), dtype=np.int32)
        tmp = nums
        for j in range(L - 1, -1, -1):
            K[:, j] = (tmp % M).astype(np.int32)
            tmp = tmp // M

        kexp = K[:, pos]                                    # (B, n)
        P = _decrypt_batch(c32[None, :], kexp, M, variant)
        if n >= 3:
            idx = (P[:, :-2] * M + P[:, 1:-1]) * M + P[:, 2:]
            scores = tri[idx].sum(axis=1)
        else:
            scores = np.zeros(B, dtype=np.float32)

        k = min(topk, B)
        part = np.argpartition(-scores, k - 1)[:k]
        for i in part:
            item = (float(scores[i]), int(nums[i]))
            if len(heap) < topk:
                heapq.heappush(heap, item)
            elif item[0] > heap[0][0]:
                heapq.heapreplace(heap, item)

        if on_batch is not None:
            on_batch(B)
        s = e
    return sorted(heap, reverse=True)


# --------------------------------------------------------------------------- #
#  Process-pool workers
# --------------------------------------------------------------------------- #
_G = {}


def _init_worker(spec, variant, prog_q, stop_evt):
    """``spec``: {alphabet key: {'c', 'tri' (path to .npy), 'M', 'letters',
    'dict' (path to .npz), 'freq'}}.

    Heavy tables live in files so that a single pool can serve every alphabet
    and every phase: starting 32 processes costs several seconds on Windows and
    should be paid only once per run.
    """
    _G['spec'] = spec
    _G['ctx'] = {}
    _G['variant'] = variant
    _G['q'] = prog_q
    _G['stop'] = stop_evt
    _G['acc'] = 0
    _G['t'] = time.time()
    _G['dict_groups'] = {}


def _ctx(aname):
    """Per-alphabet context, loaded lazily and cached inside the worker."""
    c = _G['ctx'].get(aname)
    if c is None:
        s = _G['spec'][aname]
        c = {
            'c': np.array(s['c'], dtype=np.int32),
            'tri': np.load(s['tri']),
            'M': s['M'],
            'letters': s['letters'],
            'freq': np.array(s['freq'], dtype=np.float64),
        }
        _G['ctx'][aname] = c
    return c


def _dict_group(aname, L):
    """Matrix (count, L) of every dictionary key of length L, ready to score."""
    z = _G['dict_groups'].get(aname)
    if z is None:
        z = np.load(_G['spec'][aname]['dict'])
        _G['dict_groups'][aname] = z
    key = str(L)
    if key not in z.files:
        return np.zeros((0, max(1, L)), dtype=np.int32)
    return z[key]


def _score_matrix(K, c32, tri, M, variant, pos):
    """Score a batch of equal-length keys: (B, L) -> (B,)."""
    kexp = K[:, pos]
    P = _decrypt_batch(c32[None, :], kexp, M, variant)
    if c32.shape[0] < 3:
        return np.zeros(K.shape[0], dtype=np.float32)
    i3 = (P[:, :-2] * M + P[:, 1:-1]) * M + P[:, 2:]
    return tri[i3].sum(axis=1)


def _report(nkeys):
    _G['acc'] += nkeys
    now = time.time()
    if now - _G['t'] > 0.25:
        try:
            _G['q'].put_nowait(('prog', _G['acc']))
        except Exception:
            pass
        _G['acc'] = 0
        _G['t'] = now


def _flush():
    if _G['acc']:
        try:
            _G['q'].put_nowait(('prog', _G['acc']))
        except Exception:
            pass
        _G['acc'] = 0


def _stopped():
    return _G['stop'].is_set()


def task_brute(args):
    aname, L, start, end = args
    if _stopped():
        return aname, L, []
    x = _ctx(aname)
    pos = np.arange(x['c'].shape[0], dtype=np.int64) % L
    res = scan_range_numpy(x['c'], x['tri'], x['M'], L, start, end,
                           _G['variant'], pos, on_batch=_report, should_stop=_stopped)
    _flush()
    return aname, L, res


def task_keys(args):
    """Dictionary attack over keys ``[start, end)`` of one length group.
    Scored in batches; one key at a time would be an order of magnitude slower."""
    aname, L, start, end = args
    if _stopped():
        return aname, 0, []
    grp = _dict_group(aname, L)
    if grp.shape[0] == 0:
        return aname, 0, []
    x = _ctx(aname)
    c32, tri, M = x['c'], x['tri'], x['M']
    pos = np.arange(c32.shape[0], dtype=np.int64) % L
    heap = []
    s = start
    while s < min(end, grp.shape[0]):
        if _stopped():
            break
        e = min(s + BATCH, end, grp.shape[0])
        K = grp[s:e]
        scores = _score_matrix(K, c32, tri, M, _G['variant'], pos)
        k = min(TOPK_PER_TASK, K.shape[0])
        part = np.argpartition(-scores, k - 1)[:k]
        for i in part:
            item = (float(scores[i]), tuple(int(v) for v in K[i]))
            if len(heap) < TOPK_PER_TASK:
                heapq.heappush(heap, item)
            elif item[0] > heap[0][0]:
                heapq.heapreplace(heap, item)
        _G['acc'] += e - s
        _report(0)
        s = e
    _flush()
    return aname, 0, sorted(heap, reverse=True)


def frequency_attack(c, tri, M, variant, exp_freq, L, n_seeds=320, seed=0,
                     should_stop=None, on_progress=None):
    """Frequency analysis: the key is *computed*, not guessed.

    For key length L the text splits into L columns; inside a column the cipher
    is a plain shift, so the key letter follows from correlating letter
    frequencies with the language. The result is then polished by hill climbing
    on trigrams, because on short columns the frequency estimate is unreliable.

    Returns ``[(score, key_tuple), ...]``, best first.
    """
    rng = random.Random(seed)
    n = c.shape[0]
    if n < L * 2:
        return []
    stop = should_stop or (lambda: False)
    pos = np.arange(n, dtype=np.int64) % L

    # Step 1: rank the possible key letters for every column.
    ranked = []
    for i in range(L):
        corr = core.column_correlation(c[i::L], M, variant, exp_freq)
        ranked.append(np.argsort(-corr))

    def score_keys(Kmat):
        return _score_matrix(Kmat, c, tri, M, variant, pos)

    def climb(key):
        best = float(score_keys(key[None, :])[0])
        improved = True
        while improved and not stop():
            improved = False
            for p in range(L):
                Kmat = np.tile(key, (M, 1))
                Kmat[:, p] = np.arange(M, dtype=np.int32)
                sc = score_keys(Kmat)
                j = int(np.argmax(sc))
                if float(sc[j]) > best + 1e-9:
                    best, key, improved = float(sc[j]), Kmat[j].copy(), True
        return best, key

    # Step 2: iterated local search.
    # Plain hill climbing is not enough: with a long key each column holds only
    # 6-8 letters, the frequency signal is close to noise, and the search gets
    # stuck one or two letters away from the answer. So the best solution is
    # periodically perturbed and re-climbed.
    heap, seen = [], set()
    best_key, best_sc = None, -1e18

    def record(sc, key):
        nonlocal best_key, best_sc
        kt = core.minimal_period(tuple(int(v) for v in key))
        if kt not in seen:
            seen.add(kt)
            item = (sc, kt)
            if len(heap) < TOPK_PER_TASK:
                heapq.heappush(heap, item)
            elif sc > heap[0][0]:
                heapq.heapreplace(heap, item)
        if sc > best_sc:
            best_sc, best_key = sc, key.copy()

    record(*climb(np.array([r[0] for r in ranked], dtype=np.int32)))
    for _ in range(max(0, n_seeds - 1)):
        if stop():
            break
        if best_key is not None and rng.random() < 0.65:
            cand = best_key.copy()                       # shake the incumbent
            for _ in range(rng.randint(1, max(1, L // 3))):
                p = rng.randrange(L)
                cand[p] = ranked[p][rng.randrange(min(5, M))]
        else:
            cand = np.array([r[rng.randrange(min(3, M))] for r in ranked],
                            dtype=np.int32)              # fresh frequency guess
        record(*climb(cand))
        if on_progress is not None:
            on_progress(L * M * 4)
    return sorted(heap, reverse=True)


def task_freq(args):
    """Pool wrapper around :func:`frequency_attack`."""
    aname, L, n_seeds, seed = args
    if _stopped():
        return aname, L, []
    x = _ctx(aname)
    res = frequency_attack(x['c'], x['tri'], x['M'], _G['variant'], x['freq'],
                           L, n_seeds=n_seeds, seed=seed, should_stop=_stopped,
                           on_progress=lambda k: _G.__setitem__('acc', _G['acc'] + k))
    _flush()
    return aname, L, res


def task_hill(args):
    """Generic hill climbing with random restarts (any key length)."""
    aname, L, n_restarts, seed = args
    if _stopped():
        return aname, L, []
    rng = random.Random(seed)
    x = _ctx(aname)
    c, tri, M, variant = x['c'], x['tri'], x['M'], _G['variant']
    n = c.shape[0]
    pos = np.arange(n, dtype=np.int64) % L
    heap = []

    def score_keys(Kmat):
        return _score_matrix(Kmat, c, tri, M, variant, pos)

    done = 0
    for _ in range(n_restarts):
        if _stopped():
            break
        key = np.array([rng.randrange(M) for _ in range(L)], dtype=np.int32)
        best = float(score_keys(key[None, :])[0])
        improved = True
        while improved and not _stopped():
            improved = False
            order = list(range(L))
            rng.shuffle(order)
            for p in order:
                Kmat = np.tile(key, (M, 1))
                Kmat[:, p] = np.arange(M, dtype=np.int32)
                sc = score_keys(Kmat)
                j = int(np.argmax(sc))
                done += M
                if float(sc[j]) > best + 1e-9:
                    best, key, improved = float(sc[j]), Kmat[j].copy(), True
        item = (best, tuple(int(v) for v in key))
        if len(heap) < TOPK_PER_TASK:
            heapq.heappush(heap, item)
        elif item[0] > heap[0][0]:
            heapq.heapreplace(heap, item)
        _G['acc'] += done
        done = 0
        _report(0)
    _flush()
    return aname, L, sorted(heap, reverse=True)


# --------------------------------------------------------------------------- #
#  GPU back-end (CUDA through torch)
# --------------------------------------------------------------------------- #
def _torch():
    """Import torch without the noisy numpy 1.x/2.x ABI warnings.

    The numpy bridge is deliberately unused (tensors are built from Python
    lists), so a torch build compiled against a different numpy still works.
    """
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import torch
    return torch


def gpu_available() -> bool:
    try:
        return bool(_torch().cuda.is_available())
    except Exception:
        return False


def gpu_name() -> str:
    try:
        return _torch().cuda.get_device_name(0)
    except Exception:
        return "-"


def gpu_total_memory_mb() -> float:
    try:
        return _torch().cuda.get_device_properties(0).total_memory / 2 ** 20
    except Exception:
        return 0.0


def gpu_warmup() -> bool:
    """Create the CUDA context up front; the first call costs several seconds."""
    try:
        torch = _torch()
        if torch.cuda.is_available():
            torch.zeros(1024, device='cuda').sum().item()
            return True
    except Exception:
        pass
    return False


def _is_oom(exc) -> bool:
    return "out of memory" in str(exc).lower()


def gpu_batch_for(n_letters: int, free_mb: float = None) -> int:
    """Choose a batch size that fits the available video memory.

    Peak usage is dominated by a handful of (B, n) int64 tensors; measured at
    roughly 1.0-1.3 GB for the default batch. The estimate below is deliberately
    conservative and is additionally guarded by an out-of-memory retry.
    """
    n = max(8, n_letters)
    if free_mb is None:
        free_mb = gpu_total_memory_mb() or 2048.0
    budget_mb = max(192.0, min(1400.0, free_mb * 0.35))
    bytes_per_key = n * 40                       # ~5 int64 tensors of width n
    batch = int(budget_mb * 1024 * 1024 / max(1, bytes_per_key))
    return int(max(1 << 13, min(1 << 20, batch)))


def gpu_scan(c32, tri, M, L, variant, topk=4096, batch=None,
             on_batch=None, should_stop=None, on_note=None):
    """Brute force one key length on the GPU.

    Falls back to a smaller batch on CUDA out-of-memory, so the same code runs
    on a 2 GB card and on a 24 GB one.
    """
    torch = _torch()
    dev = torch.device('cuda')
    tri_t = torch.tensor(tri.tolist(), dtype=torch.float32, device=dev)
    c_t = torch.tensor([int(v) for v in c32], dtype=torch.int64, device=dev)
    n = c_t.numel()
    pos_t = torch.arange(n, device=dev, dtype=torch.int64) % L
    total = M ** L
    if batch is None:
        batch = gpu_batch_for(n)
    batch = int(max(1 << 12, min(batch, total)))

    # Leaders are kept in video memory: moving the top of every batch back to
    # Python costs more than the search itself (measured 81 s vs 11.6 s).
    per_batch = min(256, topk)
    best_v = torch.empty(0, dtype=torch.float32, device=dev)
    best_i = torch.empty(0, dtype=torch.int64, device=dev)

    s = 0
    while s < total:
        if should_stop is not None and should_stop():
            break
        e = min(s + batch, total)
        try:
            nums = torch.arange(s, e, device=dev, dtype=torch.int64)
            B = nums.numel()
            K = torch.empty((B, L), dtype=torch.int64, device=dev)
            tmp = nums
            for j in range(L - 1, -1, -1):
                K[:, j] = tmp % M
                tmp = torch.div(tmp, M, rounding_mode='floor')
            kexp = K[:, pos_t]
            if variant == "vigenere":
                P = (c_t.unsqueeze(0) - kexp) % M
            elif variant == "beaufort":
                P = (kexp - c_t.unsqueeze(0)) % M
            else:
                P = (c_t.unsqueeze(0) + kexp) % M
            if n >= 3:
                idx = (P[:, :-2] * M + P[:, 1:-1]) * M + P[:, 2:]
                sc = tri_t[idx].sum(dim=1)
            else:
                sc = torch.zeros(B, device=dev)
            vals, ind = torch.topk(sc, min(per_batch, B))
            best_v = torch.cat([best_v, vals.to(torch.float32)])
            best_i = torch.cat([best_i, ind.to(torch.int64) + s])
            if best_v.numel() > topk * 8:
                best_v, sel = torch.topk(best_v, topk)
                best_i = best_i[sel]
        except RuntimeError as exc:
            if not _is_oom(exc) or batch <= (1 << 12):
                raise
            torch.cuda.empty_cache()
            batch = max(1 << 12, batch // 2)
            if on_note is not None:
                on_note(batch)
            continue                                  # retry the same range
        if on_batch is not None:
            on_batch(B)
        s = e

    out = []
    if best_v.numel():
        k = min(topk, best_v.numel())
        vals, sel = torch.topk(best_v, k)
        out = list(zip(vals.tolist(), best_i[sel].tolist()))
    del tri_t, c_t, best_v, best_i
    torch.cuda.empty_cache()
    return out


# --------------------------------------------------------------------------- #
#  Final ranking
# --------------------------------------------------------------------------- #
def final_rank(cands, cipher_text, alpha, model, variant, limit=200, keylen_penalty=None,
               alpha_name="", cov_limit=COV_LIMIT):
    """Rank candidates given as ``[(search_score, key_tuple), ...]``.

    Two stages: every candidate gets the cheap quadgram score, and only the
    leaders get the dictionary search, whose cost grows with the text length
    and would otherwise dominate the whole run on long texts.
    """
    if keylen_penalty is None:
        keylen_penalty = math.log(alpha.M)
    idx, tmpl = alpha.layout(cipher_text)
    n = max(1, len(idx))
    ar = np.arange(len(idx))

    prelim, seen = [], set()
    for raw, key in cands:
        if key in seen:
            continue
        seen.add(key)
        kexp = np.array(key, dtype=np.int64)[ar % len(key)]
        p = core.decrypt_indices(idx, kexp, alpha.M, variant)
        raw4 = model.score_text4(p)
        prelim.append((raw4 - keylen_penalty * len(key), raw4, key, p))
    prelim.sort(key=lambda t: -t[0])

    rows = []
    for i, (base, raw4, key, p) in enumerate(prelim):
        if i < cov_limit:
            cov, nwords = model.word_coverage(''.join(alpha.letters[j] for j in p))
        else:
            cov, nwords = 0.0, 0
        rows.append({
            'key': alpha.key_to_str(key),
            'alpha': alpha_name or alpha.key,
            'keylen': len(key),
            'raw': raw4,
            'ngpc': raw4 / max(1, n - 3),
            'cov': cov,
            'words': nwords,
            'final': base + W_WORD * cov * n,
            'plain': alpha.render(p, tmpl),
        })
    rows.sort(key=lambda r: -r['final'])
    return rows[:limit]


# --------------------------------------------------------------------------- #
#  Dictionary preparation
# --------------------------------------------------------------------------- #
def write_dict_matrices(words, alpha, add_reversed=True, min_len=1, max_len=30):
    """Convert words into index matrices (one per key length) inside a .npz.

    Done once in the parent process and vectorised: letting the workers parse
    millions of words each takes longer than the attack itself (15 s -> 1.6 s).
    Returns ``(path, total keys, {length: count})``.
    """
    buckets = {}
    seen = set()
    for w in words:
        for cand in ((w, w[::-1]) if add_reversed else (w,)):
            if min_len <= len(cand) <= max_len and cand not in seen:
                seen.add(cand)
                buckets.setdefault(len(cand), []).append(cand)

    lut = core._lut_for(alpha)
    out, groups, total = {}, {}, 0
    for L, lst in buckets.items():
        blob = "".join(lst).encode(alpha.encoding, errors='replace')
        arr = lut[np.frombuffer(blob, dtype=np.uint8)]
        if arr.size != len(lst) * L:              # multi-byte characters: skip
            continue
        mat = arr.reshape(-1, L)
        mat = mat[(mat < alpha.M).all(axis=1)].astype(np.int32)
        if mat.shape[0]:
            out[str(L)] = mat
            groups[L] = mat.shape[0]
            total += mat.shape[0]

    fd, path = tempfile.mkstemp(prefix="vigenere_keys_", suffix=".npz")
    os.close(fd)
    np.savez(path, **out)
    return path, total, groups


def tag_of(alphas, aname):
    """Alphabet tag for the log, shown only when several alphabets are tested."""
    return ("[%s] " % aname) if len(alphas) > 1 else ""


# --------------------------------------------------------------------------- #
#  Engine: runs in its own thread, reports through a queue
# --------------------------------------------------------------------------- #
class Engine(threading.Thread):

    AUTO_SHORT = 60      # letters: below this brute force leads, above it analysis

    def __init__(self, cfg, out_queue):
        super().__init__(daemon=True)
        self.cfg = cfg
        self.tr = cfg.get('tr') or i18n.Translator(cfg.get('ui_lang', i18n.DEFAULT_UI))
        self.out = out_queue
        self._stop_flag = threading.Event()
        self.mgr = None
        self.stop_evt = None
        self.pool = None
        self._tmp_files = []
        self._dgroups = {}

    # ---- helpers ----
    def emit(self, **kw):
        self.out.put(kw)

    def log(self, msg):
        self.emit(type='log', msg=msg)

    def request_stop(self):
        self._stop_flag.set()
        if self.stop_evt is not None:
            try:
                self.stop_evt.set()
            except Exception:
                pass

    def stopped(self):
        return self._stop_flag.is_set()

    def run(self):
        try:
            self._run()
        except Exception as exc:                     # noqa: BLE001
            import traceback
            self.log(self.tr('eng.error', "%s\n%s" % (exc, traceback.format_exc())))
            self.emit(type='done', ok=False)
        finally:
            self._shutdown_pool()
            for p in self._tmp_files:
                try:
                    os.remove(p)
                except OSError:
                    pass
            self._tmp_files = []

    def _shutdown_pool(self):
        if self.pool is not None:
            try:
                self.pool.terminate()
                self.pool.join()
            except Exception:
                pass
            self.pool = None
        if self.mgr is not None:
            try:
                self.mgr.shutdown()
            except Exception:
                pass
            self.mgr = None

    # ---- main flow ----
    def _run(self):
        cfg = self.cfg
        tr = self.tr
        variant = cfg['variant']
        text = cfg['text']
        mode = cfg['mode']
        alphas = cfg['alpha_list']              # [(alphabet key, model), ...]

        a0 = core.get_alphabet(alphas[0][0])
        c0 = a0.layout(text)[0]
        n0 = len(c0)
        if n0 < 3:
            self.log(tr('eng.short'))
            self.emit(type='done', ok=False)
            return
        self.log(tr('eng.info', n0, variant, ", ".join(a for a, _ in alphas)))

        hints = core.key_length_hints(c0, a0.M, min(20, max(2, n0 // 2)))
        if hints:
            best_h = sorted(hints, key=lambda t: -t[1])[:4]
            self.log(tr('eng.ic', core.natural_ic(a0.lang))
                     + ", ".join("L=%d:%.3f" % h for h in best_h))
        kas = core.kasiski(c0, max_len=min(30, max(2, n0 // 3)))
        if kas:
            self.log(tr('eng.kasiski') + ", ".join("L=%d (%d)" % k for k in kas[:5]))
        if mode not in ('dict', 'freq', 'auto') and cfg['max_len'] >= n0:
            self.log(tr('eng.keytoolong'))

        phases = self._phases(mode, n0)
        self.log(tr('eng.strategy') + " -> ".join(tr('eng.ph.' + p) for p in phases))
        if cfg['use_gpu'] and 'brute' not in phases:
            self.log(tr('eng.gpuonly'))

        # --- work plan: [(alphabet, phase, key length, weight)] ---
        ctx_data, dict_paths, plan = {}, {}, []
        for aname, model in alphas:
            alpha = core.get_alphabet(aname)
            idx, _ = alpha.layout(text)
            dict_n = 0
            if 'dict' in phases:
                t_d = time.time()
                path, dict_n, groups = write_dict_matrices(
                    cfg['dict_words'], alpha, cfg['dict_reversed'])
                dict_paths[aname] = path
                self._dgroups[aname] = groups
                self._tmp_files.append(path)
                self.log(tr('eng.dictprep', tag_of(alphas, aname),
                            f"{dict_n:,}".replace(',', ' '), time.time() - t_d))
            ctx_data[aname] = (alpha, model, idx.astype(np.int32), dict_n)
            M = alpha.M
            for ph in phases:
                if ph == 'brute':
                    lo, hi = self._brute_range(mode)
                    for L in range(lo, hi + 1):
                        plan.append((aname, ph, L, M ** L))
                elif ph == 'dict':
                    plan.append((aname, ph, 0, dict_n))
                elif ph == 'freq':
                    lo, hi = self._freq_range(mode, n0)
                    for L in range(lo, hi + 1):
                        plan.append((aname, ph, L, cfg['freq_seeds'] * L * M * 4))
                else:
                    for L in range(cfg['min_len'], cfg['max_len'] + 1):
                        plan.append((aname, ph, L, cfg['restarts'] * L * M * 6))
        total = max(1, sum(w for _, _, _, w in plan))
        self.emit(type='total', total=total)

        t0 = time.time()
        done = [0]
        cands = {a: [] for a, _ in alphas}

        def push_progress(stage):
            el = time.time() - t0
            rate = done[0] / el if el > 0 else 0
            eta = (total - done[0]) / rate if rate > 0 else 0
            self.emit(type='progress', done=done[0], total=total, rate=rate,
                      elapsed=el, eta=eta, stage=stage)

        def tag(aname):
            return tag_of(alphas, aname)

        use_gpu = bool(cfg['use_gpu'])
        gpu_plan = [p for p in plan if use_gpu and p[1] == 'brute']
        cpu_plan = [p for p in plan if not (use_gpu and p[1] == 'brute')]
        # phase order matters: run first what is most likely to answer
        cpu_plan.sort(key=lambda p: (phases.index(p[1]), p[2]))

        # ---------------- GPU ----------------
        if gpu_plan:
            self.log(tr('eng.gpu', gpu_name()))
            for aname, ph, L, w in gpu_plan:
                if self.stopped():
                    break
                alpha, model, c32, _ = ctx_data[aname]
                stage = tr('eng.gpubrute', tag(aname), L,
                           f"{alpha.M ** L:,}".replace(',', ' '))
                self.log("-> " + stage)
                last = [time.time()]

                def on_batch(b, stage=stage):
                    done[0] += b
                    if time.time() - last[0] > 0.1:
                        push_progress(stage)
                        last[0] = time.time()

                def on_note(new_batch):
                    self.log(tr('eng.gpuoom', new_batch))

                try:
                    res = gpu_scan(c32, model.tri, alpha.M, L, variant,
                                   topk=min(4096, alpha.M ** L),
                                   batch=cfg.get('gpu_batch'),
                                   on_batch=on_batch, should_stop=self.stopped,
                                   on_note=on_note)
                except Exception as exc:                        # noqa: BLE001
                    # A GPU failure must never lose the run: fall back to CPU.
                    self.log(tr('eng.gpufail', exc))
                    cpu_plan.append((aname, ph, L, w))
                    cpu_plan.sort(key=lambda p: (phases.index(p[1]), p[2]))
                    continue
                cands[aname] += [(s, tuple(int(v) for v in alpha.key_from_int(k, L)))
                                 for s, k in res]
                push_progress(stage)
                self._interim(cands, ctx_data, text, variant)

        if not cpu_plan:
            self._finish(cands, ctx_data, text, variant, t0)
            return

        # ---------------- CPU: one pool for every alphabet and phase ----------
        nproc = max(1, int(cfg['processes']))
        spec = {}
        for aname, model in alphas:
            alpha, _, c32, _ = ctx_data[aname]
            tri_path = os.path.join(tempfile.gettempdir(),
                                    "vigenere_tri_%s_%d.npy" % (aname, os.getpid()))
            np.save(tri_path, model.tri)
            self._tmp_files.append(tri_path)
            spec[aname] = {'c': c32.tolist(), 'tri': tri_path, 'M': alpha.M,
                           'letters': alpha.letters,
                           'dict': dict_paths.get(aname),
                           'freq': model.letter_freq().tolist()}

        ctx = mp.get_context('spawn')
        self.mgr = ctx.Manager()
        prog_q = self.mgr.Queue()
        self.stop_evt = self.mgr.Event()
        if self.stopped():
            self.stop_evt.set()
        self.log(tr('eng.procs', nproc))
        self.pool = ctx.Pool(processes=nproc, initializer=_init_worker,
                             initargs=(spec, variant, prog_q, self.stop_evt))

        drain_stop = threading.Event()

        def drain():
            while not drain_stop.is_set():
                try:
                    kind, val = prog_q.get(timeout=0.2)
                except Exception:
                    continue
                if kind == 'prog':
                    done[0] += val
        threading.Thread(target=drain, daemon=True).start()

        try:
            for aname, ph, L, w in cpu_plan:
                if self.stopped():
                    break
                alpha, _, _, dkeys_n = ctx_data[aname]
                tasks = self._make_tasks(aname, ph, L, alpha.M, cfg, nproc, dkeys_n)
                if not tasks:
                    continue
                stage = tag(aname) + self._stage_name(ph, L, alpha.M, cfg, dkeys_n)
                self.log("-> " + stage)
                last = time.time()
                for a_out, L_out, res in self.pool.imap_unordered(
                        self._task_fn(ph), tasks, chunksize=1):
                    if ph == 'brute':
                        ao = ctx_data[a_out][0]
                        cands[a_out] += [
                            (s, tuple(int(v) for v in ao.key_from_int(k, L_out)))
                            for s, k in res]
                    else:
                        cands[a_out] += res
                    if time.time() - last > 0.15:
                        push_progress(stage)
                        self._interim(cands, ctx_data, text, variant)
                        last = time.time()
                    if self.stopped():
                        break
                push_progress(stage)
                self._interim(cands, ctx_data, text, variant)
        finally:
            drain_stop.set()
            self._shutdown_pool()

        self._finish(cands, ctx_data, text, variant, t0)

    # ---- strategy ----
    def _phases(self, mode, n):
        """Which methods to apply. In ``auto`` this depends on the text length."""
        if mode != 'auto':
            return [mode]
        if n < self.AUTO_SHORT:
            # Short text: frequency analysis has no statistics to work with,
            # but the space of short keys is fully enumerable.
            return ['brute', 'dict', 'freq']
        # Long text: the key follows from the text itself, brute force is moot.
        return ['freq', 'dict', 'brute']

    def _brute_range(self, mode):
        if mode != 'auto':
            return self.cfg['min_len'], self.cfg['max_len']
        return 1, (5 if self.cfg['use_gpu'] else 4)

    def _freq_range(self, mode, n):
        lo, hi = 1, self.cfg['max_len']
        if mode == 'auto':
            hi = max(8, min(28, n // 4))
        return lo, max(lo, min(hi, max(1, n // 2)))

    # ---- task construction ----
    @staticmethod
    def _task_fn(phase):
        return {'brute': task_brute, 'dict': task_keys,
                'hill': task_hill, 'freq': task_freq}[phase]

    def _stage_name(self, phase, L, M, cfg, dkeys_n):
        if phase == 'brute':
            return self.tr('eng.stage.brute', L, f"{M ** L:,}".replace(',', ' '))
        if phase == 'dict':
            return self.tr('eng.stage.dict', f"{dkeys_n:,}".replace(',', ' '))
        if phase == 'freq':
            return self.tr('eng.stage.freq', L)
        return self.tr('eng.stage.hill', L, cfg['restarts'])

    def _make_tasks(self, aname, phase, L, M, cfg, nproc, dkeys_n):
        if phase == 'brute':
            total = M ** L
            step = max(1, min(KEYS_PER_TASK, total // (nproc * 4) or total))
            return [(aname, L, s, min(s + step, total)) for s in range(0, total, step)]
        if phase == 'dict':
            tasks = []
            for KL, cnt in sorted(self._dgroups.get(aname, {}).items()):
                if KL < 1 or cnt < 1:
                    continue
                step = max(1, cnt // max(1, nproc // 2) or 1)
                tasks += [(aname, KL, s, min(s + step, cnt)) for s in range(0, cnt, step)]
            return tasks
        if phase == 'freq':
            seeds = max(1, int(cfg['freq_seeds']))
            per = max(1, seeds // nproc)
            k = max(1, min(nproc, seeds))
            return [(aname, L, per, random.randrange(1 << 30)) for _ in range(k)]
        per = max(1, cfg['restarts'] // nproc)
        return [(aname, L, per, random.randrange(1 << 30)) for _ in range(nproc)]

    # ---- merging results across alphabets ----
    def _merge(self, cands, ctx_data, text, variant, pool_size, limit, cov_limit=COV_LIMIT):
        rows = []
        for aname, lst in cands.items():
            if not lst:
                continue
            alpha, model, _, _ = ctx_data[aname]
            top = heapq.nlargest(pool_size, lst, key=lambda t: t[0])
            rows += final_rank(top, text, alpha, model, variant, limit=limit,
                               keylen_penalty=self.cfg['keylen_penalty'],
                               alpha_name=aname, cov_limit=cov_limit)
        rows.sort(key=lambda r: -r['final'])
        return rows[:limit]

    def _interim(self, cands, ctx_data, text, variant):
        rows = self._merge(cands, ctx_data, text, variant, 60, 25)
        if rows:
            self.emit(type='interim', rows=rows)

    def _finish(self, cands, ctx_data, text, variant, t0):
        self.log(self.tr('eng.rescore'))
        rows = self._merge(cands, ctx_data, text, variant, FINAL_POOL, 200)
        el = time.time() - t0
        self.log(self.tr('eng.finished', el, sum(len(v) for v in cands.values())))
        self.emit(type='results', rows=rows)
        self.emit(type='done', ok=True, stopped=self.stopped(), elapsed=el)
