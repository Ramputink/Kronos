"""Overnight strategy-search worker (see PREREGISTRATION.md).

Loads ONLY data before 2025-07-01 (TEST sealed). Samples random indicator-combination configs (or mutates
existing candidates), evaluates each on all coins, logs trial statistics and writes candidates.

usage: python worker.py <worker_id> <hours>
"""
import json, math, os, random, sys, time, hashlib
import numpy as np
import pandas as pd
from numba import njit

DATA = '/Users/matveypro/kronos_lab/indicators/data'
OUT = '/Users/matveypro/kronos_lab/overnight/out'
COINS = ["BTC", "ETH", "SOL", "XRP", "ADA", "AVAX", "LINK", "DOGE", "BNB", "LTC", "DOT", "ATOM", "NEAR", "APT",
         "BCH", "TRX", "ETC", "XLM", "DASH", "ZEC", "VET", "THETA", "ALGO", "COMP", "CRV", "RUNE", "SUSHI", "EGLD", "UNI"]
TRAIN_END, VAL_END = pd.Timestamp('2024-01-01'), pd.Timestamp('2025-07-01')
BPY = {'1d': 365, '4h': 2190}
COST = {'SPOT': 0.0015, 'PERP': 0.00105}
WARM_DAYS = 210


# ── numba kernels ────────────────────────────────────────────────────────────
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


# ── data ─────────────────────────────────────────────────────────────────────
def load_all():
    data = {}
    for tf, delta in (('1d', pd.Timedelta('1D')), ('4h', pd.Timedelta('4h'))):
        for coin in COINS:
            df = pd.read_csv(f'{DATA}/{coin}_{tf}.csv', index_col=0, parse_dates=True)
            df = df[df.index < VAL_END]  # TEST sealed
            f = pd.read_csv(f'{DATA}/{coin}_funding.csv', index_col=0, parse_dates=True).iloc[:, 0]
            fund = f.groupby((f.index - pd.Timedelta(milliseconds=1)).floor(delta)).sum().reindex(df.index).fillna(0.0)
            start = df.index[0] + pd.Timedelta(days=WARM_DAYS)
            data[(coin, tf)] = dict(df=df, fund=fund.values, idx=df.index,
                                    c=df.close.values, h=df.high.values, l=df.low.values, v=df.volume.values,
                                    m_eval=(df.index >= start), m_train=(df.index >= start) & (df.index < TRAIN_END),
                                    m_val=(df.index >= TRAIN_END))
    return data


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


# ── component catalog: name -> (param space, fn(d, **p) -> state in [-1,1]) ─
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
FILTERS = dict(adx=[0, 15, 20, 25, 30], vt=[0, 0.3, 0.5, 0.8], stop=[0, 2, 3, 4])


def rand_comp(rng):
    t = rng.choice(list(CATALOG))
    p = {k: rng.choice(v) for k, v in CATALOG[t][1].items()}
    if t == 'ema_x' and p['f'] >= p['s']:
        p['f'] = min(CATALOG['ema_x'][1]['f'])
    if t == 'donch' and p['M'] >= p['N']:
        p['M'] = 10
    return dict(t=t, p=p)


def rand_cfg(rng):
    n = rng.choice([1, 1, 2, 2, 2, 3])
    comps = [rand_comp(rng) for _ in range(n)]
    comb = 'and' if n == 1 else rng.choice(['and', 'vote'])
    return dict(tf=rng.choice(['1d', '1d', '4h']), mode=rng.choice(['SPOT', 'PERP']), comps=comps, comb=comb,
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


# ── evaluation ───────────────────────────────────────────────────────────────
CACHE = {}


def comp_state(d, key, comp):
    ck = (key, comp['t'], tuple(sorted(comp['p'].items())))
    if ck not in CACHE:
        if len(CACHE) > 4000:
            CACHE.clear()
        CACHE[ck] = CATALOG[comp['t']][0](d, **comp['p']).astype(np.float64)
    return CACHE[ck]


def aux(d, key, name):
    ck = (key, name)
    if ck not in CACHE:
        if name == 'adx':
            CACHE[ck] = adx(d)
        elif name == 'atr':
            CACHE[ck] = atr(d)
        elif name == 'vol':
            r = np.diff(d['c'], prepend=np.nan) / np.r_[np.nan, d['c'][:-1]]
            CACHE[ck] = pd.Series(r).rolling(30).std().values * math.sqrt(BPY[key[1]])
    return CACHE[ck]


def seg_stats(r, bpy):
    n = len(r)
    if n < 30:
        return None
    g = np.prod(1 + r)
    cagr = g ** (bpy / n) - 1 if g > 0 else -1.0
    sd = r.std()
    return cagr, (r.mean() / sd * math.sqrt(bpy)) if sd > 0 else 0.0


def evaluate(cfg, data):
    out = []
    for coin in COINS:
        key = (coin, cfg['tf'])
        d = data[key]
        S = np.vstack([comp_state(d, key, c) for c in cfg['comps']])
        bull, bear = (S > 0).sum(0), (S < 0).sum(0)
        pos = np.where(bull >= cfg['k'], 1.0, np.where(bear >= cfg['k'], -1.0, 0.0))
        if cfg['mode'] == 'SPOT':
            pos = np.clip(pos, 0, None)
        if cfg['adx']:
            pos = pos * (np.nan_to_num(aux(d, key, 'adx')) > cfg['adx'])
        if cfg['stop']:
            pos = nb_trailing(pos, d['c'], aux(d, key, 'atr'), float(cfg['stop']))
        if cfg['vt']:
            cap = 1.0 if cfg['mode'] == 'SPOT' else 2.0
            with np.errstate(divide='ignore', invalid='ignore'):
                pos = pos * np.nan_to_num(np.minimum(cap, cfg['vt'] / aux(d, key, 'vol')))
        c = d['c']
        ret = np.r_[0.0, c[1:] / c[:-1] - 1]
        wx = np.r_[0.0, pos[:-1]]
        turn = np.abs(np.diff(pos, prepend=0.0))
        r = wx * ret - np.r_[0.0, turn[:-1]] * COST[cfg['mode']]
        if cfg['mode'] == 'PERP':
            r = r - wx * d['fund']
        bpy = BPY[cfg['tf']]
        rt, rv, re = r[d['m_train']], r[d['m_val']], r[d['m_eval']]
        st, sv, se = seg_stats(rt, bpy), seg_stats(rv, bpy), seg_stats(re, bpy)
        if st is None or sv is None or len(rt) < 180 * bpy / 365:
            continue
        eq = np.cumprod(1 + re)
        mdd = float((eq / np.maximum.accumulate(eq) - 1).min())
        w = wx[d['m_eval']]
        entries = int(((w[1:] != 0) & (np.sign(w[1:]) != np.sign(w[:-1]))).sum())
        out.append(dict(coin=coin, cagr_tr=st[0], sh_tr=st[1], cagr_va=sv[0], sh_va=sv[1], sh_tv=se[1],
                        mdd=mdd, entries=entries, expo=float(np.abs(w).mean())))
    return out


def is_candidate(m):
    return (m['cagr_tr'] > 0.20 and m['cagr_va'] > 0.20 and m['sh_tr'] > 0.8 and m['sh_va'] > 0.8
            and m['entries'] >= 8 and m['mdd'] > -0.45)


def main():
    wid, hours = int(sys.argv[1]), float(sys.argv[2])
    deadline = time.time() + hours * 3600
    rng = random.Random(1000 + wid)
    os.makedirs(OUT, exist_ok=True)
    data = load_all()
    stats = {'trials': 0, 'configs': 0, 'candidates': 0, 'errors': 0,
             'sr_moments': {tf: [0, 0.0, 0.0] for tf in BPY}}  # n, sum, sumsq of annual Sharpe (train+val)
    seen = set()
    cand_f = open(f'{OUT}/candidates_w{wid}.jsonl', 'a')
    pool, last_pool, last_flush = [], 0.0, time.time()
    while time.time() < deadline:
        if time.time() - last_pool > 300:  # refresh mutation pool from all workers' candidates
            pool = []
            for fn in os.listdir(OUT):  # random ~256 KB chunk of each candidates file (bounded memory/IO)
                if fn.startswith('candidates_'):
                    path = f'{OUT}/{fn}'
                    size = os.path.getsize(path)
                    with open(path, 'rb') as fh:
                        fh.seek(rng.randrange(max(1, size - 262144)) if size > 262144 else 0)
                        for line in fh.read(262144).split(b'\n')[1:-1]:
                            try:
                                pool.append(json.loads(line)['cfg'])
                            except Exception:
                                pass
            last_pool = time.time()
        cfg = mutate(rng.choice(pool), rng) if pool and rng.random() < 0.4 else rand_cfg(rng)
        h = hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        try:
            res = evaluate(cfg, data)
        except Exception as e:
            stats['errors'] += 1
            continue
        stats['configs'] += 1
        mom = stats['sr_moments'][cfg['tf']]
        for m in res:
            stats['trials'] += 1
            mom[0] += 1; mom[1] += m['sh_tv']; mom[2] += m['sh_tv'] ** 2
            if is_candidate(m):
                stats['candidates'] += 1
                cand_f.write(json.dumps(dict(cfg=cfg, cfg_hash=h, **m)) + '\n')
        if time.time() - last_flush > 60:
            cand_f.flush()
            stats['updated'] = time.strftime('%Y-%m-%d %H:%M:%S')
            json.dump(stats, open(f'{OUT}/stats_w{wid}.json', 'w'))
            last_flush = time.time()
    cand_f.close()
    stats['finished'] = time.strftime('%Y-%m-%d %H:%M:%S')
    json.dump(stats, open(f'{OUT}/stats_w{wid}.json', 'w'))


if __name__ == '__main__':
    main()
