import numpy as np, pandas as pd, os, sys, json
C = '/Users/matveypro/kronos_lab/cache'
FEE = 0.0011
rng = np.random.default_rng(0)

def rsi(close, n=14):
    d = np.diff(close, prepend=close[0])
    up = pd.Series(np.clip(d, 0, None)).ewm(alpha=1/n, adjust=False).mean()
    dn = pd.Series(np.clip(-d, 0, None)).ewm(alpha=1/n, adjust=False).mean()
    return (100 - 100/(1 + up/dn.replace(0, 1e-12))).values

def divergence(close, r, o, lb=48, half=24):
    # bullish: price low in recent half < low in prior half, but rsi min recent > rsi min prior (and rsi recent min < 40)
    p1 = close[o-lb+1:o-half+1]; p2 = close[o-half+1:o+1]
    r1 = r[o-lb+1:o-half+1]; r2 = r[o-half+1:o+1]
    if p2.min() < p1.min() and r2.min() > r1.min() and r2.min() < 40: return 1
    if p2.max() > p1.max() and r2.max() < r1.max() and r2.max() > 60: return -1
    return 0

def stats(sig, ret):
    m = sig != 0
    n = int(m.sum())
    if n == 0: return dict(n=0)
    g = sig[m]*ret[m]
    return dict(n=n, gross_mean=float(g.mean()*100), net_mean=float((g-FEE).mean()*100),
                hit=float((g > 0).mean()), net_sum=float((g-FEE).sum()*100))

def rand_filter_pval(base_sig, filt_sig, ret, reps=5000):
    # among base trades, choose random subset of the same size as filtered; p = P(rand mean >= kronos-filtered mean)
    idx = np.where(base_sig != 0)[0]; k = int((filt_sig != 0).sum())
    if k == 0 or k >= len(idx): return None
    g = base_sig[idx]*ret[idx]
    obs = (filt_sig[filt_sig != 0]*ret[filt_sig != 0]).mean()
    sims = np.array([g[rng.choice(len(idx), k, replace=False)].mean() for _ in range(reps)])
    return float((sims >= obs).mean())

out = {}
for sym in ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']:
    f = f'{C}/{sym}_60_paths.npz'
    if not os.path.exists(f): continue
    df = pd.read_csv(f'{C}/{sym}_60_ohlcv.csv'); close = df.close.values
    z = np.load(f); O = z['origins']; P = z['paths']
    ok = O + 24 < len(close); O = O[ok]; P = P[ok]
    c0 = close[O]; ret = close[O+24]/c0 - 1
    pc = P[:, :, 23, 3]; kr = pc/c0[:, None] - 1
    cons = (kr > 0).mean(1)          # fraction of 6 paths up
    kmean = kr.mean(1)
    ksig = np.where(cons >= 4/6, 1, np.where(cons <= 2/6, -1, 0))
    res = {'n_windows': len(O), 'bh_24h_mean': float(ret.mean()*100),
           'bh_period': float((close[O[-1]+24]/close[O[0]]-1)*100),
           'corr_kmean_ret': float(np.corrcoef(kmean, ret)[0, 1])}
    rows = {}
    for n_ in [14, 7, 21]:
        r = rsi(close, n_)[O]
        for lo, hi in [(30, 70), (35, 65), (40, 60)]:
            s = np.where(r < lo, 1, np.where(r > hi, -1, 0))
            tag = f'RSI{n_}_{lo}/{hi}'
            rows[tag + ' alone(MR)'] = stats(s, ret)
            rows[tag + ' momentum(inv)'] = stats(-s, ret)
            fa = np.where(s == ksig, s, 0)          # Kronos agrees
            fd = np.where((s != 0) & (ksig == -s), s, 0)  # Kronos disagrees
            rows[tag + ' +Kronos agree'] = {**stats(fa, ret), 'p_vs_randsubset': rand_filter_pval(s, fa, ret)}
            rows[tag + ' +Kronos disagree'] = stats(fd, ret)
    # divergence
    r14 = rsi(close, 14)
    dv = np.array([divergence(close, r14, o) for o in O])
    rows['Div14 alone'] = stats(dv, ret)
    fa = np.where(dv == ksig, dv, 0)
    rows['Div14 +Kronos agree'] = {**stats(fa, ret), 'p_vs_randsubset': rand_filter_pval(dv, fa, ret)}
    rows['Kronos consensus alone'] = stats(ksig, ret)
    rows['Kronos consensus inverted'] = stats(-ksig, ret)
    rs = np.array([stats(rng.choice([-1, 1], len(ret)), ret)['net_mean'] for _ in range(2000)])
    rows['Random long/short (all windows) net_mean pct5/50/95'] = [float(x) for x in np.percentile(rs, [5, 50, 95])]
    rows['Always long'] = stats(np.ones(len(ret)), ret)
    # holdout: last third
    h = int(len(O)*2/3)
    r = rsi(close, 14)[O]; s = np.where(r < 35, 1, np.where(r > 65, -1, 0)); fa = np.where(s == ksig, s, 0)
    rows['HOLDOUT last1/3 RSI14_35/65 alone'] = stats(s[h:], ret[h:])
    rows['HOLDOUT last1/3 RSI14_35/65 +Kronos agree'] = stats(fa[h:], ret[h:])
    res['rows'] = rows
    out[sym] = res
    print(f'\n==== {sym}  windows={len(O)}  B&H 24h mean {res["bh_24h_mean"]:.3f}%  period {res["bh_period"]:.1f}%  corr(kmean,ret)={res["corr_kmean_ret"]:.3f}')
    for k, v in rows.items():
        if isinstance(v, dict) and v.get('n'):
            print(f'{k:42s} n={v["n"]:4d} gross={v["gross_mean"]:+.3f}% net={v["net_mean"]:+.3f}% hit={v["hit"]:.2f} sumNet={v["net_sum"]:+.1f}%' + (f' p={v["p_vs_randsubset"]:.2f}' if v.get('p_vs_randsubset') is not None else ''))
        elif isinstance(v, dict): print(f'{k:42s} n=0')
        else: print(k, v)
json.dump(out, open('/Users/matveypro/kronos_lab/agents/rsi/results.json', 'w'), indent=1)
