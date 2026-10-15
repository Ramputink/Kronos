"""MORNING ONLY — opens the sealed TEST once (PREREGISTRATION.md). Do not run during the search.

1. Dedupe candidates (cfg_hash, coin).
2. Evaluate each on TEST (2025-07-01 -> end) with identical rules/costs, all cores.
3. Baseline: random (cfg, coin) pairs that were NOT selected -> confirmation rate expected by chance.
4. Equal-weight portfolio of ALL candidates (what you'd trade without peeking), and per-coin/family tables.
usage: python morning_test.py [max_candidates]
"""
import glob, json, math, random, sys
from multiprocessing import Pool
import numpy as np
import pandas as pd
import worker as W

import os
DRY = os.environ.get('DRYRUN') == '1'
SAMPLE = float(os.environ.get('SAMPLE', '1'))   # mechanics check: keeps the seal, fake 'test' = 2025-01..2025-06 (inside VAL)
OUT = W.OUT
TEST_START = pd.Timestamp('2025-01-01') if DRY else W.VAL_END


def load_full():
    if not DRY:
        W.VAL_END = pd.Timestamp('2100-01-01')    # lift the seal for this script only
    data = W.load_all()
    for d in data.values():
        d['m_test'] = d['idx'] >= TEST_START
    return data


DATA = None


def init():
    global DATA
    DATA = load_full()


def strat_returns(cfg, coin):
    """Re-implements evaluate() for one coin and returns the per-bar strategy returns + exposure."""
    key = (coin, cfg['tf'])
    d = DATA[key]
    S = np.vstack([W.comp_state(d, key, c) for c in cfg['comps']])
    bull, bear = (S > 0).sum(0), (S < 0).sum(0)
    pos = np.where(bull >= cfg['k'], 1.0, np.where(bear >= cfg['k'], -1.0, 0.0))
    if cfg['mode'] == 'SPOT':
        pos = np.clip(pos, 0, None)
    if cfg['adx']:
        pos = pos * (np.nan_to_num(W.aux(d, key, 'adx')) > cfg['adx'])
    if cfg['stop']:
        pos = W.nb_trailing(pos, d['c'], W.aux(d, key, 'atr'), float(cfg['stop']))
    if cfg['vt']:
        cap = 1.0 if cfg['mode'] == 'SPOT' else 2.0
        with np.errstate(divide='ignore', invalid='ignore'):
            pos = pos * np.nan_to_num(np.minimum(cap, cfg['vt'] / W.aux(d, key, 'vol')))
    c = d['c']
    ret = np.r_[0.0, c[1:] / c[:-1] - 1]
    wx = np.r_[0.0, pos[:-1]]
    turn = np.abs(np.diff(pos, prepend=0.0))
    r = wx * ret - np.r_[0.0, turn[:-1]] * W.COST[cfg['mode']]
    if cfg['mode'] == 'PERP':
        r = r - wx * d['fund']
    return pd.Series(r, d['idx']), pd.Series(wx, d['idx'])


def one(item):
    cfg, coin = item
    r, wx = strat_returns(cfg, coin)
    t = r[r.index >= TEST_START]
    bpy = W.BPY[cfg['tf']]
    g = float(np.prod(1 + t.values))
    cagr = g ** (bpy / len(t)) - 1 if g > 0 else -1.0
    c = DATA[(coin, '1d')]['c'][DATA[(coin, '1d')]['m_test']]
    bh = (c[-1] / c[0]) ** (365 / (len(c) - 1)) - 1
    eq = np.cumprod(1 + t.values)
    daily = (1 + t).groupby(t.index.floor('D')).prod() - 1
    return dict(coin=coin, tf=cfg['tf'], mode=cfg['mode'], cagr_test=cagr, bh_test=bh,
                sh_test=float(t.mean() / t.std() * math.sqrt(bpy)) if t.std() > 0 else 0.0,
                mdd_test=float((eq / np.maximum.accumulate(eq) - 1).min()),
                expo_test=float(np.abs(wx[wx.index >= TEST_START]).mean()),
                family='+'.join(sorted(c_['t'] for c_ in cfg['comps'])), daily=daily)


META = ['coin', 'cagr_tr', 'cagr_va', 'sh_tr', 'sh_va', 'entries', 'mdd', 'cfg_hash']


def one_row(item):
    """item = (cfg, coin, meta_dict). Returns (flat result row, daily TEST returns aligned to DATES)."""
    cfg, coin, meta = item
    x = one((cfg, coin))
    daily = x.pop('daily').reindex(DATES).fillna(0.0).values.astype(np.float32)
    x.update({k: meta.get(k) for k in META if k != 'coin'})
    x['cfg'] = json.dumps(cfg, sort_keys=True)
    return x, daily


DATES = None


def init_full():
    global DATES
    init()
    d = DATA[('BTC', '1d')]
    DATES = d['idx'][d['m_test']]


def stream_candidates(cap):
    """Yield deduped candidates file by file, keeping only the (hash, coin) key set in memory."""
    seen, n = set(), 0
    for f in sorted(glob.glob(f'{OUT}/candidates_*.jsonl')):
        with open(f) as fh:
            for line in fh:
                try:
                    m = json.loads(line)
                except Exception:
                    continue
                k = m['cfg_hash'] + m['coin']
                if SAMPLE < 1 and int(m['cfg_hash'][:8], 16) % 100000 >= SAMPLE * 100000:
                    continue  # deterministic random sample by config hash
                if k in seen:
                    continue
                seen.add(k)
                n += 1
                if n > cap:
                    return
                yield (m['cfg'], m['coin'], {k2: m.get(k2) for k2 in META})


def summarise(path, label):
    T = pd.read_csv(path)
    T['confirmed'] = (T.cagr_test > 0.20) & (T.cagr_test > T.bh_test)
    print(f'{label}: confirmados (CAGR TEST > 20 % y > buy&hold) {T.confirmed.sum():,} de {len(T):,} = {T.confirmed.mean():.2%}')
    return T


def main():
    cap = int(sys.argv[1]) if len(sys.argv) > 1 else 10**12
    init_full()
    dates = DATES
    port_sum, n_port = np.zeros(len(dates)), 0
    pref = 'DRY_' if DRY else ''
    res_path, base_path = f'{OUT}/{pref}test_results.csv', f'{OUT}/{pref}test_baseline_random.csv'
    rng = random.Random(1)
    base = [(W.rand_cfg(rng), rng.choice(W.COINS), {}) for _ in range(min(20000, cap))]
    with Pool(10, initializer=init_full) as pool:
        for items, path, is_cand in ((stream_candidates(cap), res_path, True), (iter(base), base_path, False)):
            header = True
            buf = []
            for row, daily in pool.imap(one_row, items, chunksize=500):
                if is_cand:
                    port_sum += daily; n_port += 1
                buf.append(row)
                if len(buf) >= 50000:
                    pd.DataFrame(buf).to_csv(path, mode='w' if header else 'a', header=header, index=False)
                    header, buf = False, []
                    print(f'  {path.split("/")[-1]}: {n_port:,} evaluados', flush=True)
            if buf:
                pd.DataFrame(buf).to_csv(path, mode='w' if header else 'a', header=header, index=False)
    T = summarise(res_path, 'CANDIDATOS')
    B = summarise(base_path, 'AZAR (configs aleatorias)')
    port = pd.Series(port_sum / max(n_port, 1), dates)
    eq = (1 + port).cumprod()
    print(f'cartera equiponderada de TODOS los candidatos ({n_port:,}): CAGR {(eq.iloc[-1] ** (365 / len(port)) - 1) * 100:.1f}% '
          f'Sharpe {port.mean() / port.std() * math.sqrt(365):.2f} maxDD {((eq / eq.cummax()) - 1).min() * 100:.1f}%')
    for g in ['coin', 'tf', 'mode', 'family']:
        print(T.groupby(g).agg(n=('cagr_test', 'size'), conf_rate=('confirmed', 'mean'),
                               cagr_med=('cagr_test', 'median')).sort_values('conf_rate', ascending=False).head(25).round(3).to_string())
    port.to_csv(f'{OUT}/{pref}test_portfolio_all_candidates.csv')


if __name__ == '__main__':
    main()
