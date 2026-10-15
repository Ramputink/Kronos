import numpy as np, pandas as pd, sys, os, json
C = '/Users/matveypro/kronos_lab/cache/'
OUT = '/Users/matveypro/kronos_lab/agents/vol_forecast/'
FEE = 0.00055


def garch_fit(r):
    r = r - r.mean(); v0 = r.var(); best = (None, 1e18)
    for a in np.arange(0.01, 0.30, 0.01):
        for b in np.arange(0.60, 0.995, 0.005):
            if a + b >= 0.999: continue
            w = v0 * (1 - a - b); h = np.empty_like(r); h[0] = v0
            for t in range(1, len(r)): h[t] = w + a * r[t - 1] ** 2 + b * h[t - 1]
            ll = np.sum(np.log(h) + r ** 2 / h)
            if ll < best[1]: best = ((w, a, b), ll)
    return best[0]


def run(sym):
    df = pd.read_csv(C + f'{sym}_60_ohlcv.csv'); z = np.load(C + f'{sym}_60_paths.npz')
    O = z['origins']; P = z['paths']
    c = df.close.values; h = df.high.values; l = df.low.values
    lr = np.r_[np.nan, np.diff(np.log(c))]
    pc = np.r_[c[0], c[:-1]]
    tr = np.maximum(h - l, np.maximum(abs(h - pc), abs(l - pc))) / c
    atr = pd.Series(tr).rolling(24).mean().values
    ewma = pd.Series(np.nan_to_num(lr) ** 2).ewm(alpha=0.06, adjust=False).mean().values  # RiskMetrics lambda=.94
    # GARCH(1,1) fit by grid MLE on pre-sample data only (rows before first origin)
    pre = lr[1:O[0] + 1][-2000:]
    m = pre.mean()
    w, a, b = garch_fit(pre)
    hh = np.full(len(lr), np.nan); hv = pre.var()
    for t in range(1, len(lr)):
        hh[t] = hv  # conditional var of return t, known at t-1
        hv = w + a * (lr[t] - m) ** 2 + b * hv
    rows = []
    for i, o in enumerate(O):
        if o + 24 >= len(c): continue
        fut = lr[o + 1:o + 25]
        rv = np.sqrt(np.sum(fut ** 2))
        rng = np.log(h[o + 1:o + 25].max() / l[o + 1:o + 25].min())
        ret24 = np.log(c[o + 24] / c[o])
        p = P[i]; pcl = p[:, :, 3]
        plr = np.diff(np.log(np.c_[np.full(p.shape[0], c[o]), pcl]), axis=1)
        k_rv = np.mean(np.sqrt(np.sum(plr ** 2, 1)))
        k_disp = np.std(np.log(pcl[:, -1] / c[o]), ddof=1)
        k_hl = np.mean(np.log(p[:, :, 1] / p[:, :, 2]))
        k_rng = np.mean(np.log(p[:, :, 1].max(1) / p[:, :, 2].min(1)))
        tr24 = np.sqrt(np.sum(lr[o - 23:o + 1] ** 2))
        tr168 = np.sqrt(np.sum(lr[o - 167:o + 1] ** 2) * 24 / 168)
        hv0 = hh[o + 1]; uv = w / (1 - a - b)
        s = sum(uv + (a + b) ** k * (hv0 - uv) for k in range(24))
        rows.append(dict(o=o, rv=rv, rng=rng, ret24=ret24, k_rv=k_rv, k_disp=k_disp, k_hl=k_hl, k_rng=k_rng,
                         trail24=tr24, trail168=tr168, atr=atr[o], ewma=np.sqrt(24 * ewma[o]), garch=np.sqrt(s)))
    return pd.DataFrame(rows), (w, a, b)


def ols(X, y):
    X = np.c_[np.ones(len(X)), X]; return np.linalg.lstsq(X, y, rcond=None)[0]


F = ['k_rv', 'k_disp', 'k_hl', 'k_rng', 'trail24', 'trail168', 'atr', 'ewma', 'garch']
out = {}
for sym in sys.argv[1:]:
    if not os.path.exists(C + f'{sym}_60_paths.npz'): print('missing', sym); continue
    d, g = run(sym); n = len(d); cut = n // 3
    tr, te = d.iloc[:cut], d.iloc[cut:]
    res = {'n': n, 'n_test': len(te), 'garch_w_a_b': [float(x) for x in g]}
    for target in ['rv', 'rng']:
        for f in F:
            x = np.log(d[f]); y = np.log(d[target])
            sp = d[f].rank().corr(d[target].rank())
            pr = np.corrcoef(x, y)[0, 1]
            b = ols(np.log(tr[f]).values[:, None], np.log(tr[target]).values)
            pred = b[0] + b[1] * np.log(te[f]).values; yt = np.log(te[target]).values
            r2 = 1 - np.sum((yt - pred) ** 2) / np.sum((yt - yt.mean()) ** 2)
            fv = np.exp(2 * pred); rv2 = te[target].values ** 2
            ql = np.mean(rv2 / fv - np.log(rv2 / fv) - 1)
            res[f'{target}|{f}'] = dict(pearson_log_all=round(pr, 3), spearman_all=round(sp, 3),
                                        oos_r2=round(r2, 3), oos_qlike=round(ql, 3))
        for base in ['ewma', 'garch', 'trail24', 'atr']:
            for kf in ['k_rv', 'k_rng']:
                b = ols(np.log(tr[[base, kf]]).values, np.log(tr[target]).values)
                pred = b[0] + np.log(te[[base, kf]]).values @ b[1:]; yt = np.log(te[target]).values
                res[f'{target}|{base}+{kf}'] = dict(oos_r2=round(1 - np.sum((yt - pred) ** 2) / np.sum((yt - yt.mean()) ** 2), 3),
                                                   coef=[round(x, 3) for x in b])
    # vol targeting: non-overlapping 24h holds (every 2nd origin), test 2/3 only
    vt = {}
    te2 = te.iloc[::2]
    simple = np.exp(te2.ret24.values) - 1
    tgt = np.median(tr.rv)

    def st(x):
        eq = np.cumprod(1 + x); dd = np.max(1 - eq / np.maximum.accumulate(eq))
        return dict(tot=round(float(eq[-1] - 1), 4), sharpe_ann=round(float(x.mean() / x.std() * np.sqrt(365)), 3),
                    vol_d=round(float(x.std()), 4), maxdd=round(float(dd), 4))

    def perf(pos, label):
        g_ = pos * simple; nt = g_ - FEE * np.abs(np.diff(np.r_[0, pos]))
        vt[label] = dict(gross=st(g_), net=st(nt), avg_pos=round(float(pos.mean()), 3),
                         vol_of_ret_std=round(float(np.std(g_)), 4))
    perf(np.ones(len(te2)), 'buy_hold')
    for f in F + ['ewma+k_rv']:
        if '+' in f:
            cols = ['ewma', 'k_rv']
        else:
            cols = [f]
        b = ols(np.log(tr[cols]).values, np.log(tr.rv).values)
        fc = np.exp(b[0] + np.log(te2[cols]).values @ b[1:])
        perf(np.clip(tgt / fc, 0, 2), 'vt_' + f)
    rng_ = np.random.default_rng(0)
    b = ols(np.log(tr[['k_rv']]).values, np.log(tr.rv).values)
    pos = np.clip(tgt / np.exp(b[0] + b[1] * np.log(te2['k_rv']).values), 0, 2)
    sh = []
    for s in range(500):
        pp = rng_.permutation(pos); x = pp * simple - FEE * np.abs(np.diff(np.r_[0, pp]))
        sh.append(x.mean() / x.std() * np.sqrt(365))
    vt['shuffled_k_rv_sizing_net_sharpe'] = dict(mean=round(float(np.mean(sh)), 3), p5=round(float(np.percentile(sh, 5)), 3),
                                                 p95=round(float(np.percentile(sh, 95)), 3))
    vt['n_days'] = len(te2)
    res['voltarget'] = vt
    out[sym] = res
    d.to_csv(OUT + f'{sym}_forecasts.csv', index=False)
json.dump(out, open(OUT + 'results.json', 'w'), indent=1)
for sym, r in out.items():
    print('=====', sym, 'n', r['n'], 'n_test', r['n_test'], 'garch', r['garch_w_a_b'])
    for t in ['rv', 'rng']:
        print(' target', t)
        for k, v in r.items():
            if k.startswith(t + '|'): print('  ', k[len(t) + 1:], v)
    for k, v in r['voltarget'].items(): print('  VT', k, v)
