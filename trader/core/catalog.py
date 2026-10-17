"""Indicator component catalog (Wave 1, PROTECTED core).

Faithful port of research/kronos_lab/overnight/worker.py (numba kernels, the 15 components, FILTERS,
rand_comp / rand_cfg / mutate) plus indicators/engine.py `base_signals` (legacy named signals).
Numbers must stay identical to the research code: do not "improve" any formula here.
See trader/CONTRACTS.md section 4. Python >= 3.9; numpy, pandas, numba only.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import math
from typing import Dict, List

import numpy as np
import pandas as pd
from numba import njit

SPEC_KEYS = ("tf", "mode", "comps", "comb", "k", "adx", "vt", "stop")
TFS = ("1d", "4h")
MODES = ("SPOT", "PERP")
BPY = {"1d": 365, "4h": 2190}
VT_CAP = {"SPOT": 1.0, "PERP": 2.0}


# ── numba kernels (verbatim worker.py) ───────────────────────────────────────
@njit(cache=True)
def nb_supertrend(h, l, c, a, mult):
    n = len(c)
    d = np.zeros(n)
    fu = np.empty(n); fl = np.empty(n)
    started = False
    for i in range(n):
        if np.isnan(a[i]):
            continue
        hl2 = (h[i] + l[i]) / 2
        ub, lb = hl2 + mult * a[i], hl2 - mult * a[i]
        if not started:
            fu[i], fl[i], d[i] = ub, lb, 1.0
            started = True
            continue
        fu[i] = ub if (ub < fu[i - 1] or c[i - 1] > fu[i - 1]) else fu[i - 1]
        fl[i] = lb if (lb > fl[i - 1] or c[i - 1] < fl[i - 1]) else fl[i - 1]
        if d[i - 1] == 1.0 and c[i] < fl[i]:
            d[i] = -1.0
        elif d[i - 1] == -1.0 and c[i] > fu[i]:
            d[i] = 1.0
        else:
            d[i] = d[i - 1]
    return d


@njit(cache=True)
def nb_hyst(li, lo, si, so):
    n = len(li)
    out = np.zeros(n)
    L = 0.0; S = 0.0
    for i in range(n):
        if li[i]:
            L = 1.0
        elif lo[i]:
            L = 0.0
        if si[i]:
            S = 1.0
        elif so[i]:
            S = 0.0
        out[i] = L - S
    return out


@njit(cache=True)
def nb_trailing(pos, c, a, k):
    """ATR trailing stop: once stopped, stay flat until the raw signal changes."""
    n = len(pos)
    out = pos.copy()
    ext = 0.0; stopped = 0.0; prev = 0.0
    for i in range(n):
        p = pos[i]
        if np.sign(p) != np.sign(prev):
            stopped = 0.0
            ext = c[i]
        prev = p
        if p == 0.0 or np.isnan(a[i]):
            continue
        if stopped != 0.0:
            out[i] = 0.0
            continue
        if p > 0:
            ext = max(ext, c[i])
            if c[i] < ext - k * a[i]:
                stopped = 1.0; out[i] = 0.0
        else:
            ext = min(ext, c[i])
            if c[i] > ext + k * a[i]:
                stopped = -1.0; out[i] = 0.0
    return out


# ── numpy indicator helpers (verbatim worker.py) ─────────────────────────────
def ema(x, n): return pd.Series(x).ewm(span=n, adjust=False).mean().values
def sma(x, n): return pd.Series(x).rolling(n).mean().values
def rmax(x, n): return pd.Series(x).rolling(n).max().shift(1).values
def rmin(x, n): return pd.Series(x).rolling(n).min().shift(1).values


def atr(d, n=14):
    c, h, l = d['c'], d['h'], d['l']
    pc = np.r_[np.nan, c[:-1]]
    tr = np.nanmax(np.vstack([h - l, np.abs(h - pc), np.abs(l - pc)]), axis=0)
    return pd.Series(tr).ewm(alpha=1 / n, adjust=False).mean().values


def rsi(c, n):
    d = np.diff(c, prepend=np.nan)
    up = pd.Series(np.clip(d, 0, None)).ewm(alpha=1 / n, adjust=False).mean().values
    dn = pd.Series(np.clip(-d, 0, None)).ewm(alpha=1 / n, adjust=False).mean().values
    with np.errstate(divide='ignore', invalid='ignore'):
        return 100 - 100 / (1 + up / dn)


def adx(d, n=14):
    h, l = d['h'], d['l']
    up, dn = np.diff(h, prepend=np.nan), -np.diff(l, prepend=np.nan)
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    mdm = np.where((dn > up) & (dn > 0), dn, 0.0)
    a = atr(d, n)
    with np.errstate(divide='ignore', invalid='ignore'):
        pdi = 100 * pd.Series(pdm).ewm(alpha=1 / n, adjust=False).mean().values / a
        mdi = 100 * pd.Series(mdm).ewm(alpha=1 / n, adjust=False).mean().values / a
        dx = 100 * np.abs(pdi - mdi) / (pdi + mdi)
    return pd.Series(dx).ewm(alpha=1 / n, adjust=False).mean().values


def sgn(x): return np.nan_to_num(np.sign(x))


# ── component catalog: name -> (fn(d, **p) -> state in [-1,1], param grid) ──
def c_ema_x(d, f, s): return sgn(ema(d['c'], f) - ema(d['c'], s))
def c_ma(d, kind, n): return sgn(d['c'] - (sma(d['c'], n) if kind == 'sma' else ema(d['c'], n)))
def c_donch(d, N, M):
    c = d['c']
    return nb_hyst(c > rmax(d['h'], N), c < rmin(d['l'], M), c < rmin(d['l'], N), c > rmax(d['h'], M))
def c_st(d, n, m): return nb_supertrend(d['h'], d['l'], d['c'], atr(d, n), float(m))
def c_macd(d, f, s, g):
    m = ema(d['c'], f) - ema(d['c'], s)
    return sgn(m - ema(m, g))
def c_rsi_reg(d, n, up, gap):
    r = rsi(d['c'], n)
    return nb_hyst(r > up, r < up - gap, r < 100 - up, r > 100 - up + gap)
def c_roc(d, n):
    c = d['c']
    return sgn(c / np.r_[np.full(n, np.nan), c[:-n]] - 1)
def c_bb_break(d, n, k):
    c = d['c']; mid = sma(c, n); sd = pd.Series(c).rolling(n).std().values
    return nb_hyst(c > mid + k * sd, c < mid, c < mid - k * sd, c > mid)
def c_kelt(d, n, m):
    c = d['c']; mid = ema(c, n); a = atr(d, n)
    return nb_hyst(c > mid + m * a, c < mid, c < mid - m * a, c > mid)
def c_aroon(d, n):
    h = pd.Series(d['h']).rolling(n + 1).apply(np.argmax, raw=True).values
    l = pd.Series(d['l']).rolling(n + 1).apply(np.argmin, raw=True).values
    return sgn(h - l)
def c_cci(d, n):
    tp = (d['h'] + d['l'] + d['c']) / 3
    m = sma(tp, n)
    md = pd.Series(tp).rolling(n).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True).values
    with np.errstate(divide='ignore', invalid='ignore'):
        cci = (tp - m) / (0.015 * md)
    return nb_hyst(cci > 100, cci < 0, cci < -100, cci > 0)
def c_ichi(d, t, k):
    h, l, c = pd.Series(d['h']), pd.Series(d['l']), d['c']
    ten = ((h.rolling(t).max() + l.rolling(t).min()) / 2).values
    kij = ((h.rolling(k).max() + l.rolling(k).min()) / 2).values
    sa = pd.Series((ten + kij) / 2).shift(k).values
    sb = ((h.rolling(2 * k).max() + l.rolling(2 * k).min()) / 2).shift(k).values
    top, bot = np.fmax(sa, sb), np.fmin(sa, sb)
    return np.where(c > top, 1.0, np.where(c < bot, -1.0, 0.0))
def c_obv(d, n):
    obv = np.cumsum(sgn(np.diff(d['c'], prepend=d['c'][0])) * d['v'])
    return sgn(obv - ema(obv, n))
def c_rsi_rev(d, n, lo, hi, tr):
    c = d['c']; r = rsi(c, n); t = sma(c, tr)
    return nb_hyst((r < lo) & (c > t), r > hi, (r > 100 - lo) & (c < t), r < 100 - hi)
def c_bb_rev(d, n, k, tr):
    c = d['c']; mid = sma(c, n); sd = pd.Series(c).rolling(n).std().values; t = sma(c, tr)
    return nb_hyst((c < mid - k * sd) & (c > t), c > mid, (c > mid + k * sd) & (c < t), c < mid)


CATALOG = {
    'ema_x': (c_ema_x, dict(f=[5, 10, 20, 30, 50], s=[50, 100, 150, 200])),
    'ma': (c_ma, dict(kind=['sma', 'ema'], n=[20, 50, 100, 150, 200])),
    'donch': (c_donch, dict(N=[20, 30, 50, 100], M=[10, 20, 50])),
    'st': (c_st, dict(n=[10, 14, 20], m=[2, 3, 4])),
    'macd': (c_macd, dict(f=[8, 12], s=[21, 26], g=[9])),
    'rsi_reg': (c_rsi_reg, dict(n=[7, 14, 21], up=[50, 55, 60], gap=[5, 10])),
    'roc': (c_roc, dict(n=[10, 20, 30, 60, 90, 120])),
    'bb_break': (c_bb_break, dict(n=[20, 30], k=[1.5, 2.0, 2.5])),
    'kelt': (c_kelt, dict(n=[20, 30], m=[1.5, 2.0])),
    'aroon': (c_aroon, dict(n=[25, 50])),
    'cci': (c_cci, dict(n=[20, 40])),
    'ichi': (c_ichi, dict(t=[9, 20], k=[26, 60])),
    'obv': (c_obv, dict(n=[20, 50])),
    'rsi_rev': (c_rsi_rev, dict(n=[2, 3], lo=[5, 10, 20], hi=[60, 70, 80], tr=[100, 200])),
    'bb_rev': (c_bb_rev, dict(n=[20], k=[2.0, 2.5], tr=[100, 200])),
}
# window-length parameters that must be integers (thresholds/multipliers may be off-grid floats)
INT_PARAMS = {('ema_x', 'f'), ('ema_x', 's'), ('ma', 'n'), ('donch', 'N'), ('donch', 'M'), ('st', 'n'),
              ('macd', 'f'), ('macd', 's'), ('macd', 'g'), ('rsi_reg', 'n'), ('roc', 'n'), ('bb_break', 'n'),
              ('kelt', 'n'), ('aroon', 'n'), ('cci', 'n'), ('ichi', 't'), ('ichi', 'k'), ('obv', 'n'),
              ('rsi_rev', 'n'), ('rsi_rev', 'tr'), ('bb_rev', 'n'), ('bb_rev', 'tr')}
FILTERS = dict(adx=[0, 15, 20, 25, 30], vt=[0, 0.3, 0.5, 0.8], stop=[0, 2, 3, 4])


# ── random configs (RNG call sequence identical to worker.py with default args) ─
def rand_comp(rng):
    t = rng.choice(list(CATALOG))
    p = {k: rng.choice(v) for k, v in CATALOG[t][1].items()}
    if t == 'ema_x' and p['f'] >= p['s']:
        p['f'] = min(CATALOG['ema_x'][1]['f'])
    if t == 'donch' and p['M'] >= p['N']:
        p['M'] = 10
    return dict(t=t, p=p)


def rand_cfg(rng, modes=("SPOT", "PERP"), tfs=("1d", "1d", "4h")):
    n = rng.choice([1, 1, 2, 2, 2, 3])
    comps = [rand_comp(rng) for _ in range(n)]
    comb = 'and' if n == 1 else rng.choice(['and', 'vote'])
    return dict(tf=rng.choice(list(tfs)), mode=rng.choice(list(modes)), comps=comps, comb=comb,
                k=n - 1 if comb == 'vote' and n == 3 else n,
                **{k: rng.choice(v) for k, v in FILTERS.items()})


def mutate(cfg, rng):
    c = json.loads(json.dumps(cfg))
    what = rng.choice(['param', 'param', 'filter', 'comp', 'tf'])
    if what == 'param':
        comp = rng.choice(c['comps'])
        k = rng.choice(list(CATALOG[comp['t']][1]))
        comp['p'][k] = rng.choice(CATALOG[comp['t']][1][k])
    elif what == 'filter':
        k = rng.choice(list(FILTERS))
        c[k] = rng.choice(FILTERS[k])
    elif what == 'comp':
        if len(c['comps']) < 3 and rng.random() < 0.5:
            c['comps'].append(rand_comp(rng))
        else:
            c['comps'][rng.randrange(len(c['comps']))] = rand_comp(rng)
        c['comb'] = 'and' if len(c['comps']) == 1 else c['comb']
        c['k'] = len(c['comps']) if c['comb'] == 'and' else max(1, len(c['comps']) - 1)
    else:
        c['tf'] = '4h' if c['tf'] == '1d' else '1d'
    return c


# ── spec helpers ─────────────────────────────────────────────────────────────
def core_spec(spec) -> dict:
    """The 8 core keys of a strategy spec (extra keys such as name/notes are dropped)."""
    return {k: spec[k] for k in SPEC_KEYS if k in spec}


def spec_hash(spec) -> str:
    """md5(json.dumps(core_spec, sort_keys=True)) — byte-identical to worker.py cfg_hash."""
    return hashlib.md5(json.dumps(core_spec(spec), sort_keys=True).encode()).hexdigest()


def _is_num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def validate_spec(spec) -> List[str]:
    """[] if the spec is valid, else a list of human-readable problems."""
    errs = []
    if not isinstance(spec, dict):
        return ["spec must be a JSON object"]
    for k in SPEC_KEYS:
        if k not in spec:
            errs.append("missing key %r" % k)
    if errs:
        return errs
    if spec['tf'] not in TFS:
        errs.append("tf must be one of %s, got %r" % (list(TFS), spec['tf']))
    if spec['mode'] not in MODES:
        errs.append("mode must be one of %s, got %r" % (list(MODES), spec['mode']))
    comps = spec['comps']
    if not isinstance(comps, list) or not 1 <= len(comps) <= 3:
        errs.append("comps must be a list of 1-3 components")
        comps = []
    for i, c in enumerate(comps):
        if not isinstance(c, dict) or 't' not in c or 'p' not in c or not isinstance(c.get('p'), dict):
            errs.append("comps[%d] must be {'t': name, 'p': {params}}" % i)
            continue
        if c['t'] not in CATALOG:
            errs.append("comps[%d].t %r not in catalog %s" % (i, c['t'], sorted(CATALOG)))
            continue
        fn, grid = CATALOG[c['t']]
        need = [p for p in inspect.signature(fn).parameters if p != 'd']
        missing = [p for p in need if p not in c['p']]
        extra = [p for p in c['p'] if p not in need]
        if missing:
            errs.append("comps[%d] (%s) missing params %s" % (i, c['t'], missing))
        if extra:
            errs.append("comps[%d] (%s) unknown params %s" % (i, c['t'], extra))
        for p, v in c['p'].items():
            if p in grid and isinstance(grid[p][0], str):
                if v not in grid[p]:
                    errs.append("comps[%d] (%s) %s must be one of %s" % (i, c['t'], p, grid[p]))
            elif p in grid and not (_is_num(v) and v > 0):
                errs.append("comps[%d] (%s) %s must be a positive number" % (i, c['t'], p))
            elif (c['t'], p) in INT_PARAMS and not isinstance(v, int):
                errs.append("comps[%d] (%s) %s must be an integer" % (i, c['t'], p))
    if spec['comb'] not in ('and', 'vote'):
        errs.append("comb must be 'and' or 'vote'")
    k = spec['k']
    if not (isinstance(k, int) and not isinstance(k, bool)) or (comps and not 1 <= k <= len(comps)):
        errs.append("k must be an integer in 1..len(comps)")
    elif spec['comb'] == 'and' and comps and k != len(comps):
        errs.append("comb 'and' requires k == len(comps)")
    if not (_is_num(spec['adx']) and spec['adx'] >= 0):
        errs.append("adx must be a number >= 0 (0 = off)")
    cap = VT_CAP.get(spec['mode'], 1.0)
    if not (_is_num(spec['vt']) and 0 <= spec['vt'] <= cap):
        errs.append("vt must be a number in [0, %s] (0 = off)" % cap)
    if not (_is_num(spec['stop']) and spec['stop'] >= 0):
        errs.append("stop must be a number >= 0 (0 = off)")
    return errs


def make_d(df) -> Dict[str, np.ndarray]:
    """{"c","h","l","v","fund","idx"} from a vault.load-shaped DataFrame (+ a private per-frame cache)."""
    fund = df['funding'].values.astype(np.float64) if 'funding' in df.columns else np.zeros(len(df))
    return {"c": df['close'].values.astype(np.float64), "h": df['high'].values.astype(np.float64),
            "l": df['low'].values.astype(np.float64), "v": df['volume'].values.astype(np.float64),
            "fund": np.nan_to_num(fund), "idx": df.index, "_cache": {}}


# ── engine.py legacy named signals (verbatim, pandas versions) ───────────────
def _e_ema(s, n): return s.ewm(span=n, adjust=False).mean()


def _e_rsi(c, n):
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def _e_atr(df, n):
    pc = df.close.shift(1)
    tr = pd.concat([df.high - df.low, (df.high - pc).abs(), (df.low - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def legacy_adx(df, n=14):
    """engine.py adx(df) (pandas Series) — the adx_s argument of backtest.legacy_bar_returns."""
    up, dn = df.high.diff(), -df.low.diff()
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    mdm = np.where((dn > up) & (dn > 0), dn, 0.0)
    a = _e_atr(df, n)
    pdi = 100 * pd.Series(pdm, df.index).ewm(alpha=1 / n, adjust=False).mean() / a
    mdi = 100 * pd.Series(mdm, df.index).ewm(alpha=1 / n, adjust=False).mean() / a
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False).mean()


def _e_supertrend_dir(df, n, mult):
    a = _e_atr(df, n).values
    hl2 = ((df.high + df.low) / 2).values
    c = df.close.values
    ub, lb = hl2 + mult * a, hl2 - mult * a
    fu, fl = ub.copy(), lb.copy()
    d = np.ones(len(c))
    for i in range(1, len(c)):
        fu[i] = ub[i] if (ub[i] < fu[i - 1] or c[i - 1] > fu[i - 1]) else fu[i - 1]
        fl[i] = lb[i] if (lb[i] > fl[i - 1] or c[i - 1] < fl[i - 1]) else fl[i - 1]
        if d[i - 1] == 1 and c[i] < fl[i]:
            d[i] = -1
        elif d[i - 1] == -1 and c[i] > fu[i]:
            d[i] = 1
        else:
            d[i] = d[i - 1]
    s = pd.Series(d, df.index)
    s[np.isnan(a)] = 0
    return s


def _e_state(long_in, long_out, short_in, short_out):
    """Hysteresis state machine via ffill: +1 long, -1 short, 0 flat."""
    L = pd.Series(np.where(long_in, 1.0, np.where(long_out, 0.0, np.nan)), long_in.index).ffill().fillna(0)
    S = pd.Series(np.where(short_in, 1.0, np.where(short_out, 0.0, np.nan)), long_in.index).ffill().fillna(0)
    return L - S


def base_signals(df) -> Dict[str, pd.Series]:
    """engine.py base_signals: legacy named signals in [-1, 1] (e.g. "rsi55_50", "donch100_50")."""
    c = df.close
    out = {}
    for f, s in [(10, 50), (10, 100), (10, 200), (20, 50), (20, 100), (20, 200), (50, 100), (50, 200)]:
        out[f'ema{f}_{s}'] = np.sign(_e_ema(c, f) - _e_ema(c, s))
    for n in (50, 100, 150, 200):
        out[f'sma{n}'] = np.sign(c - c.rolling(n).mean()).fillna(0)
    for N, M in [(20, 10), (50, 10), (50, 20), (100, 10), (100, 20), (100, 50)]:
        hiN, loN = df.high.rolling(N).max().shift(1), df.low.rolling(N).min().shift(1)
        hiM, loM = df.high.rolling(M).max().shift(1), df.low.rolling(M).min().shift(1)
        out[f'donch{N}_{M}'] = _e_state(c > hiN, c < loM, c < loN, c > hiM)
    for n in (10, 14):
        for m in (2, 3, 4):
            out[f'st{n}x{m}'] = _e_supertrend_dir(df, n, m)
    macd = _e_ema(c, 12) - _e_ema(c, 26)
    hist = macd - _e_ema(macd, 9)
    out['macd'] = np.sign(hist).fillna(0)
    sma200 = c.rolling(200).mean()
    out['macd_sma200'] = (np.sign(hist) * ((np.sign(hist) == np.sign(c - sma200)))).fillna(0)
    r14 = _e_rsi(c, 14)
    for up in (50, 55):
        for dn in (45, 50):
            out[f'rsi{up}_{dn}'] = _e_state(r14 > up, r14 < dn, r14 < 100 - up, r14 > 100 - dn)
    mid = c.rolling(20).mean()
    sd = c.rolling(20).std()
    out['bb_break'] = _e_state(c > mid + 2 * sd, c < mid, c < mid - 2 * sd, c > mid)
    r2 = _e_rsi(c, 2)
    out['rsi2_rev'] = _e_state((r2 < 10) & (c > sma200), r2 > 70, (r2 > 90) & (c < sma200), r2 < 30)
    out['bb_rev'] = _e_state((c < mid - 2 * sd) & (c > sma200), c > mid, (c > mid + 2 * sd) & (c < sma200), c < mid)
    for n in (30, 60, 90):
        out[f'roc{n}'] = np.sign(c / c.shift(n) - 1).fillna(0)
    vote5 = pd.concat([out['ema20_100'], out['sma100'], out['donch50_20'], out['st14x3'], out['macd']], axis=1)
    out['vote_mean'] = vote5.mean(axis=1)
    for k in (3, 4, 5):
        bull, bear = (vote5 > 0).sum(axis=1), (vote5 < 0).sum(axis=1)
        out[f'vote_k{k}'] = pd.Series(np.where(bull >= k, 1.0, np.where(bear >= k, -1.0, 0.0)), c.index)
    return {k: v.astype(float) for k, v in out.items()}
