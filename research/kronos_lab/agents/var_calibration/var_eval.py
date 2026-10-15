import numpy as np, pandas as pd, os
from math import log, erf, sqrt
C = '/Users/matveypro/kronos_lab/cache/'
OUT = '/Users/matveypro/kronos_lab/agents/var_calibration/'
Z95 = 1.6448536


def kupiec(x, n, p):
    ph = x / n
    l0 = (n - x) * np.log(1 - p) + x * np.log(p)
    l1 = ((n - x) * np.log(1 - ph) if ph < 1 else 0) + (x * np.log(ph) if x > 0 else 0)
    LR = max(-2 * (l0 - l1), 0)
    return 1 - erf(sqrt(LR / 2))  # chi2(1) p-value


def pinball(y, q, a):
    d = y - q
    return np.mean(np.maximum(a * d, (a - 1) * d))


def winkler(y, lo, hi, a=0.10):
    return np.mean((hi - lo) + (2 / a) * (lo - y) * (y < lo) + (2 / a) * (y - hi) * (y > hi))


rows = []
for sym in ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']:
    f = C + f'{sym}_60_paths.npz'
    if not os.path.exists(f):
        continue
    df = pd.read_csv(C + f'{sym}_60_ohlcv.csv')
    cl, hi, lo = df.close.values, df.high.values, df.low.values
    z = np.load(f)
    O, P = z['origins'], z['paths']
    lr1 = np.diff(np.log(cl), prepend=np.nan)
    w = 0.94 ** np.arange(720)[::-1]
    recs = []
    for i, o in enumerate(O):
        if o + 24 >= len(cl) or o < 800:
            continue
        c0 = cl[o]
        r = log(cl[o + 24] / c0)
        mx = log(hi[o + 1:o + 25].max() / c0)
        mn = log(lo[o + 1:o + 25].min() / c0)
        p = P[i]
        kr = np.log(p[:, 23, 3] / c0)
        kmx = np.log(p[:, :, 1].max(1) / c0)
        kmn = np.log(p[:, :, 2].min(1) / c0)
        h = np.log(cl[o - 720 + 24:o + 1] / cl[o - 720:o + 1 - 24])  # past 30d overlapping 24h returns, ends at o
        ts = range(o - 720, o - 23, 6)
        hmx = np.array([log(hi[t + 1:t + 25].max() / cl[t]) for t in ts])
        hmn = np.array([log(lo[t + 1:t + 25].min() / cl[t]) for t in ts])
        s = np.sqrt(np.sum(w * lr1[o - 719:o + 1] ** 2) / w.sum() * 24)
        recs.append(dict(o=o, r=r, mx=mx, mn=mn,
                         k_mu=kr.mean(), k_sd=kr.std(ddof=1), k_min=kr.min(), k_max=kr.max(),
                         k_mx_med=np.median(kmx), k_mx_max=kmx.max(), k_mn_med=np.median(kmn), k_mn_min=kmn.min(),
                         h05=np.quantile(h, .05), h95=np.quantile(h, .95), h_sd=h.std(),
                         hmx95=np.quantile(hmx, .95), hmn05=np.quantile(hmn, .05), ew_sd=s))
    d = pd.DataFrame(recs)
    n = len(d)
    d.to_csv(OUT + f'{sym}_windows.csv', index=False)
    cut = int(n * 2 / 3)
    tr, te = d.iloc[:cut], d.iloc[cut:]
    k_scale = np.sqrt(np.mean(tr.r ** 2 / tr.k_sd ** 2))
    e_scale = np.sqrt(np.mean(tr.r ** 2 / tr.ew_sd ** 2))
    # excursion scale for Kronos: multiply max-of-6/min-of-6 to hit 5% on train
    kx_scale = np.quantile(tr.mx / tr.k_mx_max.clip(1e-5), .95)
    kn_scale = np.quantile(tr.mn / tr.k_mn_min.clip(None, -1e-5), .95)

    def models(x):
        return {
            'Kronos raw N(mu,sd of 6)': (x.k_mu - Z95 * x.k_sd, x.k_mu + Z95 * x.k_sd),
            'Kronos raw min/max of 6 (nominal ~14/86%)': (x.k_min, x.k_max),
            'Kronos rescaled N(0,c*sd)': (-Z95 * k_scale * x.k_sd, Z95 * k_scale * x.k_sd),
            'HistSim 30d': (x.h05, x.h95),
            'EWMA N(0,sd)': (-Z95 * x.ew_sd, Z95 * x.ew_sd),
            'EWMA rescaled': (-Z95 * e_scale * x.ew_sd, Z95 * e_scale * x.ew_sd),
        }

    for part, x in [('all', d), ('test_last1/3', te)]:
        for name, (q05, q95) in models(x).items():
            q05, q95 = np.asarray(q05), np.asarray(q95)
            y = x.r.values
            m = len(x)
            nb = int((y[::2] < q05[::2]).sum())
            rows.append(dict(sym=sym, part=part, model=name, n=m,
                             lo_breach=np.mean(y < q05), hi_breach=np.mean(y > q95),
                             kupiec_p_lo=kupiec(nb, len(y[::2]), .05),
                             kupiec_p_hi=kupiec(int((y[::2] > q95[::2]).sum()), len(y[::2]), .05),
                             pinball05_bp=1e4 * pinball(y, q05, .05), pinball95_bp=1e4 * pinball(y, q95, .95),
                             winkler90_bp=1e4 * winkler(y, q05, q95), width_bp=1e4 * np.mean(q95 - q05)))
    rng = d.mx - d.mn
    print(f'\n=== {sym} n={n} train={cut} test={n-cut}  k_scale={k_scale:.2f} e_scale={e_scale:.2f} kx={kx_scale:.2f} kn={kn_scale:.2f}')
    print(f'  median Kronos 24h sd(6 paths)={d.k_sd.median()*1e4:.0f}bp  EWMA sd={d.ew_sd.median()*1e4:.0f}bp  realised |r| median={d.r.abs().median()*1e4:.0f}bp')
    for c in ['k_sd', 'h_sd', 'ew_sd']:
        print(f'  spearman({c},|r|)={d[c].rank().corr((d.r.abs()).rank()):.3f}  spearman({c},range)={d[c].rank().corr((rng).rank()):.3f}')
    ksp = d.k_mx_med - d.k_mn_med
    print(f'  spearman(Kronos median predicted range, realised range)={ksp.rank().corr((rng).rank()):.3f}')
    X = np.c_[np.ones(n), np.log(d.ew_sd), np.log(d.k_sd), np.log(ksp.clip(1e-5))]
    y = np.log(rng.values)
    for cols, lab in [([0, 1], 'EWMA only'), ([0, 2], 'Kronos sd only'), ([0, 3], 'Kronos range only'), ([0, 1, 2, 3], 'EWMA+Kronos')]:
        b = np.linalg.lstsq(X[:cut][:, cols], y[:cut], rcond=None)[0]
        e = y[cut:] - X[cut:][:, cols] @ b
        print(f'  OOS log(range) {lab:18s} MSE={np.mean(e**2):.4f} R2={1-np.mean(e**2)/np.var(y[cut:]):.3f}  coefs={np.round(b,2)}')
    for part, x in [('all', d), ('test', te)]:
        print(f'  [{part}] excursion breach (target 5%):'
              f' Kronos max/min-of-6 up={np.mean(x.mx>x.k_mx_max):.3f} dn={np.mean(x.mn<x.k_mn_min):.3f} |'
              f' Kronos median-path up={np.mean(x.mx>x.k_mx_med):.3f} dn={np.mean(x.mn<x.k_mn_med):.3f} |'
              f' HistSim up={np.mean(x.mx>x.hmx95):.3f} dn={np.mean(x.mn<x.hmn05):.3f}')
    x = te
    print(f'  [test] Kronos rescaled-of-6 excursion (scale fit on train) up={np.mean(x.mx>kx_scale*x.k_mx_max):.3f} dn={np.mean(x.mn<kn_scale*x.k_mn_min):.3f}'
          f' mean width up={1e4*np.mean(kx_scale*x.k_mx_max):.0f}bp vs HistSim {1e4*np.mean(x.hmx95):.0f}bp; dn={1e4*np.mean(kn_scale*x.k_mn_min):.0f}bp vs {1e4*np.mean(x.hmn05):.0f}bp')

R = pd.DataFrame(rows)
pd.set_option('display.width', 250)
pd.set_option('display.max_columns', 20)
print(R.round(3).to_string())
R.to_csv(OUT + 'var_results.csv', index=False)
