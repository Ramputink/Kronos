"""Indicator-combination search engine (see PREREGISTRATION.md). numpy/pandas only."""
import math
import numpy as np
import pandas as pd

D = '/Users/matveypro/kronos_lab/indicators/data'
UNIV = ['BTC', 'ETH', 'SOL', 'XRP', 'BNB', 'DOGE', 'ADA', 'LINK', 'AVAX']
TF = {'1d': pd.Timedelta('1D'), '4h': pd.Timedelta('4h')}
COSTS = {'SPOT': dict(side=0.0010 + 0.0005, funding=False, lo=0.0, cap=1.0),
         'PERP': dict(side=0.00055 + 0.0005, funding=True, lo=-1.0, cap=2.0)}
TRAIN_END, VAL_END = pd.Timestamp('2024-01-01'), pd.Timestamp('2025-07-01')


# ── indicators ───────────────────────────────────────────────────────────────
def ema(s, n): return s.ewm(span=n, adjust=False).mean()


def rsi(c, n):
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def atr(df, n):
    pc = df.close.shift(1)
    tr = pd.concat([df.high - df.low, (df.high - pc).abs(), (df.low - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def adx(df, n=14):
    up, dn = df.high.diff(), -df.low.diff()
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    mdm = np.where((dn > up) & (dn > 0), dn, 0.0)
    a = atr(df, n)
    pdi = 100 * pd.Series(pdm, df.index).ewm(alpha=1 / n, adjust=False).mean() / a
    mdi = 100 * pd.Series(mdm, df.index).ewm(alpha=1 / n, adjust=False).mean() / a
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False).mean()


def supertrend_dir(df, n, mult):
    a = atr(df, n).values
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


def state(long_in, long_out, short_in, short_out):
    """Hysteresis state machine via ffill: +1 long, -1 short, 0 flat."""
    L = pd.Series(np.where(long_in, 1.0, np.where(long_out, 0.0, np.nan)), long_in.index).ffill().fillna(0)
    S = pd.Series(np.where(short_in, 1.0, np.where(short_out, 0.0, np.nan)), long_in.index).ffill().fillna(0)
    return L - S


# ── base signals in [-1, 1] ──────────────────────────────────────────────────
def base_signals(df):
    c = df.close
    out = {}
    for f, s in [(10, 50), (10, 100), (10, 200), (20, 50), (20, 100), (20, 200), (50, 100), (50, 200)]:
        out[f'ema{f}_{s}'] = np.sign(ema(c, f) - ema(c, s))
    for n in (50, 100, 150, 200):
        out[f'sma{n}'] = np.sign(c - c.rolling(n).mean()).fillna(0)
    for N, M in [(20, 10), (50, 10), (50, 20), (100, 10), (100, 20), (100, 50)]:
        hiN, loN = df.high.rolling(N).max().shift(1), df.low.rolling(N).min().shift(1)
        hiM, loM = df.high.rolling(M).max().shift(1), df.low.rolling(M).min().shift(1)
        out[f'donch{N}_{M}'] = state(c > hiN, c < loM, c < loN, c > hiM)
    for n in (10, 14):
        for m in (2, 3, 4):
            out[f'st{n}x{m}'] = supertrend_dir(df, n, m)
    macd = ema(c, 12) - ema(c, 26)
    hist = macd - ema(macd, 9)
    out['macd'] = np.sign(hist).fillna(0)
    sma200 = c.rolling(200).mean()
    out['macd_sma200'] = (np.sign(hist) * ((np.sign(hist) == np.sign(c - sma200)))).fillna(0)
    r14 = rsi(c, 14)
    for up in (50, 55):
        for dn in (45, 50):
            out[f'rsi{up}_{dn}'] = state(r14 > up, r14 < dn, r14 < 100 - up, r14 > 100 - dn)
    mid = c.rolling(20).mean()
    sd = c.rolling(20).std()
    out['bb_break'] = state(c > mid + 2 * sd, c < mid, c < mid - 2 * sd, c > mid)
    r2 = rsi(c, 2)
    out['rsi2_rev'] = state((r2 < 10) & (c > sma200), r2 > 70, (r2 > 90) & (c < sma200), r2 < 30)
    out['bb_rev'] = state((c < mid - 2 * sd) & (c > sma200), c > mid, (c > mid + 2 * sd) & (c < sma200), c < mid)
    for n in (30, 60, 90):
        out[f'roc{n}'] = np.sign(c / c.shift(n) - 1).fillna(0)
    vote5 = pd.concat([out['ema20_100'], out['sma100'], out['donch50_20'], out['st14x3'], out['macd']], axis=1)
    out['vote_mean'] = vote5.mean(axis=1)
    for k in (3, 4, 5):
        bull, bear = (vote5 > 0).sum(axis=1), (vote5 < 0).sum(axis=1)
        out[f'vote_k{k}'] = pd.Series(np.where(bull >= k, 1.0, np.where(bear >= k, -1.0, 0.0)), c.index)
    return {k: v.astype(float) for k, v in out.items()}


def load(sym, tf):
    df = pd.read_csv(f'{D}/{sym}_{tf}.csv', index_col=0, parse_dates=True)
    f = pd.read_csv(f'{D}/{sym}_funding.csv', index_col=0, parse_dates=True).iloc[:, 0]
    key = (f.index - pd.Timedelta(milliseconds=1)).floor(TF[tf])
    fund = f.groupby(key).sum().reindex(df.index).fillna(0.0)
    return df, fund


def bar_returns(df, fund, sig, cost, tf, adx_thr, vt, adx_s):
    """Per-bar strategy return series for one asset."""
    pos = sig.clip(lower=cost['lo'], upper=1.0)
    if adx_thr:
        pos = pos * (adx_s > adx_thr)
    ret = df.close.pct_change().fillna(0.0)
    if vt:
        bpy = 365 * (6 if tf == '4h' else 1)
        vol = ret.rolling(30).std() * math.sqrt(bpy)
        pos = pos * (0.5 / vol).clip(upper=cost['cap']).fillna(0.0)
    wx = pos.shift(1).fillna(0.0)
    turn = (pos - pos.shift(1).fillna(0.0)).abs()
    r = wx * ret - (turn * cost['side']).shift(1).fillna(0.0)
    if cost['funding']:
        r = r - wx * fund
    return r, wx
