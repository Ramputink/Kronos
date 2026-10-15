import numpy as np, pandas as pd, os, sys, json
C = '/Users/matveypro/kronos_lab/cache'
FEE = 0.0011
rng = np.random.default_rng(0)

def adx(df, n=14):
    h, l, c = df.high.values, df.low.values, df.close.values
    pc = np.r_[c[0], c[:-1]]; ph = np.r_[h[0], h[:-1]]; pl = np.r_[l[0], l[:-1]]
    tr = np.maximum(h - l, np.maximum(abs(h - pc), abs(l - pc)))
    up = h - ph; dn = pl - l
    pdm = np.where((up > dn) & (up > 0), up, 0.); mdm = np.where((dn > up) & (dn > 0), dn, 0.)
    a = 1 / n
    s = lambda x: pd.Series(x).ewm(alpha=a, adjust=False).mean().values
    atr = s(tr); pdi = 100 * s(pdm) / atr; mdi = 100 * s(mdm) / atr
    dx = 100 * abs(pdi - mdi) / (pdi + mdi + 1e-12)
    return s(dx)

def eff_ratio(c, n):
    c = pd.Series(c)
    return (abs(c - c.shift(n)) / c.diff().abs().rolling(n).sum()).values

def var_ratio(c, k=8, n=168):
    r = np.log(pd.Series(c)).diff()
    rk = np.log(pd.Series(c)).diff(k)
    return (rk.rolling(n).var() / (k * r.rolling(n).var())).values  # >1 trending, <1 mean-reverting

def run(sym):
    df = pd.read_csv(f'{C}/{sym}_60_ohlcv.csv')
    z = np.load(f'{C}/{sym}_60_paths.npz')
    O, P = z['origins'], z['paths']
    c = df.close.values
    ok = O + 24 < len(df)
    O, P = O[ok], P[ok]
    feat = pd.DataFrame({
        'adx': adx(df), 'er48': eff_ratio(c, 48), 'vr': var_ratio(c),
        'vol24': np.log(pd.Series(c)).diff().rolling(24).std().values,
        'mom24': pd.Series(c).pct_change(24).values,
    })
    ts = pd.to_datetime(df.timestamps)
    rows = []
    for o, p in zip(O, P):
        c0 = c[o]
        pr = p[:, -1, 3] / c0 - 1
        rows.append(dict(o=o, pred=pr.mean(), agree=np.mean(np.sign(pr) == np.sign(pr.mean())),
                         spread=np.std(pr), real=c[o + 24] / c0 - 1,
                         lo=p[:, :, 3].min(), hi=p[:, :, 3].max(), rc=c[o + 24],
                         wkend=int(ts[o].dayofweek >= 5 or (ts[o].dayofweek == 4 and ts[o].hour >= 12)),
                         **feat.iloc[o].to_dict()))
    d = pd.DataFrame(rows)
    d['cover'] = ((d.rc >= d.lo) & (d.rc <= d.hi)).astype(int)
    # vol regime relative to trailing ~30d of vol at origins (expanding median, no lookahead)
    v = feat.vol24
    d['volrel'] = [feat.vol24.iloc[o] / feat.vol24.iloc[max(0, o - 720):o + 1].median() for o in d.o]
    d['sym'] = sym
    return d

def stats(g):
    s = np.sign(g.pred); r = g.real
    gross = s * r; net = gross - FEE
    mom = np.sign(g.mom24) * r - FEE
    return pd.Series(dict(n=len(g), hit=np.mean(s == np.sign(r)), corr=np.corrcoef(g.pred, r)[0, 1] if len(g) > 3 else np.nan,
                          k_gross=gross.mean() * 100, k_net=net.mean() * 100, mom_net=mom.mean() * 100,
                          bh=r.mean() * 100, cover=g.cover.mean(),
                          volcorr=np.corrcoef(g.spread, abs(r))[0, 1] if len(g) > 3 else np.nan))

syms = [s for s in ['BTCUSDT', 'ETHUSDT', 'SOLUSDT'] if os.path.exists(f'{C}/{s}_60_paths.npz')]
D = pd.concat([run(s) for s in syms], ignore_index=True)
D.to_csv('/Users/matveypro/kronos_lab/agents/regime/windows.csv', index=False)
pd.set_option('display.width', 200); pd.set_option('display.float_format', lambda x: f'{x:.3f}')
out = []
def show(title, key):
    t = D.groupby(['sym', key]).apply(stats)
    a = D.groupby(key).apply(stats); a.index = pd.MultiIndex.from_product([['ALL'], a.index])
    t = pd.concat([t, a]); print('\n==', title); print(t); out.append(f'\n== {title}\n{t.to_string()}')

print('ALL'); a = D.groupby('sym').apply(stats); print(a); out.append(a.to_string())
# fixed, a-priori thresholds (no tuning)
D['trend_adx'] = np.where(D.adx > 25, 'trend(ADX>25)', 'range(ADX<=25)')
D['trend_er'] = np.where(D.er48 > 0.2, 'ER>0.2', 'ER<=0.2')
D['vr_reg'] = np.where(D.vr > 1, 'VR>1 persist', 'VR<=1 revert')
D['vol_reg'] = np.where(D.volrel > 1, 'hi vol', 'lo vol')
D['wk'] = np.where(D.wkend == 1, 'weekend', 'weekday')
D['agree_reg'] = np.where(D.agree >= 5 / 6, 'paths agree>=5/6', 'paths mixed')
for k in ['trend_adx', 'trend_er', 'vr_reg', 'vol_reg', 'wk', 'agree_reg']:
    show(k, k)

# Shuffle baseline for best-looking cell: permutation p-value of hit rate per regime cell (pooled)
print('\n== permutation p-values (pooled, hit rate of sign(pred) vs sign(real) within cell, 2000 shuffles)')
res = []
for k in ['trend_adx', 'trend_er', 'vr_reg', 'vol_reg', 'wk', 'agree_reg']:
    for lev, g in D.groupby(k):
        s = np.sign(g.pred.values); r = np.sign(g.real.values)
        obs = np.mean(s == r)
        null = [np.mean(rng.permutation(s) == r) for _ in range(2000)]
        p = np.mean(np.array(null) >= obs); p2 = np.mean(np.array(null) <= obs)
        res.append((k, lev, len(g), obs, p, p2))
        print(f'{k:10s} {lev:20s} n={len(g):4d} hit={obs:.3f} p(>=)={p:.3f} p(<=)={p2:.3f}')
out.append('\n== perm\n' + '\n'.join(f'{a} {b} n={c} hit={d:.3f} p_hi={e:.3f} p_lo={f:.3f}' for a, b, c, d, e, f in res))

# Holdout: within each symbol, first 2/3 vs last 1/3 — does regime-conditional sign hold?
D['half'] = D.groupby('sym').o.transform(lambda x: np.where(x.rank(pct=True) <= 2 / 3, 'train', 'test'))
for k in ['trend_adx', 'vol_reg', 'vr_reg', 'agree_reg']:
    t = D.groupby([k, 'half']).apply(stats)[['n', 'hit', 'corr', 'k_net', 'mom_net', 'bh']]
    print('\n== holdout', k); print(t); out.append(f'\n== holdout {k}\n{t.to_string()}')
open('/Users/matveypro/kronos_lab/agents/regime/results.txt', 'w').write('\n'.join(out))
