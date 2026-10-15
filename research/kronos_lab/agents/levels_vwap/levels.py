"""Levels (pivots, VWAP, Donchian S/R, Ichimoku) x Kronos paths.
Part A: probability skill. For each origin and level, events within next 24h:
  touch_R: max(high) >= R ; break_R: close[t+24] > R  (same for S on the downside)
  Kronos prob = fraction of 6 paths that do it. Baseline = logistic on distance-to-level/ATR
  (fit on first 2/3, evaluated on last 1/3), plus combined model. Metric: AUC + log loss on test.
Part B: trading. 24h hold from each origin, fee 0.11% round trip.
  Indicator-only signals, Kronos-filtered versions, Kronos-only, random baseline, buy&hold.
No lookahead: levels use rows <= o only.
"""
import numpy as np, pandas as pd, json
C = '/Users/matveypro/kronos_lab/cache/'
FEE = 0.0011
rng = np.random.default_rng(0)

def auc(y, s):
    y = np.asarray(y, bool); s = np.asarray(s, float)
    if y.all() or (~y).all(): return np.nan
    r = pd.Series(s).rank().values
    n1 = y.sum(); n0 = len(y) - n1
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

def logit_fit(X, y, l2=1e-2, it=500):
    X = np.c_[np.ones(len(X)), X]; w = np.zeros(X.shape[1])
    for _ in range(it):
        p = 1 / (1 + np.exp(-X @ w)); g = X.T @ (p - y) / len(y) + l2 * np.r_[0, w[1:]]
        H = (X.T * (p * (1 - p))) @ X / len(y) + l2 * np.eye(len(w)); w -= np.linalg.solve(H, g)
    return w
def logit_pred(w, X): return 1 / (1 + np.exp(-np.c_[np.ones(len(X)), X] @ w))
def ll(y, p): p = np.clip(p, 1e-3, 1 - 1e-3); return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

def levels(df, o):
    h, l, c, v = df.high.values, df.low.values, df.close.values, df.volume.values
    tp = (h + l + c) / 3
    H24, L24, C24 = h[o-23:o+1].max(), l[o-23:o+1].min(), c[o]
    P = (H24 + L24 + C24) / 3
    R1, S1 = 2 * P - L24, 2 * P - H24
    vwap24 = (tp[o-23:o+1] * v[o-23:o+1]).sum() / v[o-23:o+1].sum()
    vwap168 = (tp[o-167:o+1] * v[o-167:o+1]).sum() / v[o-167:o+1].sum()
    dH, dL = h[o-47:o+1].max(), l[o-47:o+1].min()
    tr = np.maximum(h[o-13:o+1] - l[o-13:o+1], np.abs(h[o-13:o+1] - c[o-14:o]))
    atr = tr.mean()
    # Ichimoku (9/26/52), cloud at time o is spans computed 26 bars earlier
    def mid(a, b, n, k): return (a[k-n+1:k+1].max() + b[k-n+1:k+1].min()) / 2
    k = o - 26
    spanA = (mid(h, l, 9, k) + mid(h, l, 26, k)) / 2; spanB = mid(h, l, 52, k)
    return dict(c=c[o], P=P, R1=R1, S1=S1, vwap24=vwap24, vwap168=vwap168, dH=dH, dL=dL,
                atr=atr, ctop=max(spanA, spanB), cbot=min(spanA, spanB))

def run(sym):
    df = pd.read_csv(C + f'{sym}_60_ohlcv.csv'); z = np.load(C + f'{sym}_60_paths.npz')
    O, PA = z['origins'], z['paths']
    rows = []
    for i, o in enumerate(O):
        if o + 24 >= len(df) or o < 200: continue
        L = levels(df, o); fut = df.iloc[o+1:o+25]
        ph, pl, pc = PA[i, :, :, 1], PA[i, :, :, 2], PA[i, :, -1, 3]
        r = dict(o=o, ret=fut.close.values[-1] / L['c'] - 1, k_ret=np.mean(pc) / L['c'] - 1, **L)
        # resistance set: R1, Donchian high, cloud top (if above), vwap (as level above/below)
        for name, lev in [('R1', L['R1']), ('dH', L['dH'])]:
            r[f'{name}_dist'] = (lev - L['c']) / L['atr']
            r[f'{name}_touch'] = float(fut.high.max() >= lev); r[f'{name}_break'] = float(fut.close.values[-1] > lev)
            r[f'{name}_ktouch'] = np.mean(ph.max(1) >= lev); r[f'{name}_kbreak'] = np.mean(pc > lev)
        for name, lev in [('S1', L['S1']), ('dL', L['dL'])]:
            r[f'{name}_dist'] = (L['c'] - lev) / L['atr']
            r[f'{name}_touch'] = float(fut.low.min() <= lev); r[f'{name}_break'] = float(fut.close.values[-1] < lev)
            r[f'{name}_ktouch'] = np.mean(pl.min(1) <= lev); r[f'{name}_kbreak'] = np.mean(pc < lev)
        # VWAP: will price be on the other side of vwap24 at +24h (cross)? and touch?
        for name in ['vwap24', 'vwap168']:
            lev = L[name]; side = np.sign(L['c'] - lev)
            r[f'{name}_dist'] = abs(L['c'] - lev) / L['atr']
            r[f'{name}_touch'] = float(fut.low.min() <= lev <= fut.high.max())
            r[f'{name}_break'] = float(np.sign(fut.close.values[-1] - lev) == -side)
            r[f'{name}_ktouch'] = np.mean((pl.min(1) <= lev) & (ph.max(1) >= lev))
            r[f'{name}_kbreak'] = np.mean(np.sign(pc - lev) == -side)
        rows.append(r)
    return pd.DataFrame(rows)

def prob_eval(d):
    n = len(d); cut = int(n * 2 / 3); tr, te = d.iloc[:cut], d.iloc[cut:]
    out = []
    for lev in ['R1', 'S1', 'dH', 'dL', 'vwap24', 'vwap168']:
        for ev in ['touch', 'break']:
            y, yt = tr[f'{lev}_{ev}'].values, te[f'{lev}_{ev}'].values
            if yt.min() == yt.max() or y.min() == y.max(): continue
            kf = lambda x: np.c_[x[f'{lev}_k{ev}'].values]
            df_ = lambda x: np.c_[x[f'{lev}_dist'].values]
            both = lambda x: np.c_[x[f'{lev}_dist'].values, x[f'{lev}_k{ev}'].values]
            res = dict(level=lev, event=ev, base_rate_test=round(yt.mean(), 3), n_test=len(yt),
                       auc_full_kronos=round(auc(d[f'{lev}_{ev}'], d[f'{lev}_k{ev}']), 3),
                       auc_full_dist=round(auc(d[f'{lev}_{ev}'], -d[f'{lev}_dist']), 3))
            res['ll_const'] = round(ll(yt, np.full(len(yt), y.mean())), 4)
            for nm, f in [('dist', df_), ('kronos', kf), ('both', both)]:
                w = logit_fit(f(tr), y); p = logit_pred(w, f(te))
                res[f'll_{nm}'] = round(ll(yt, p), 4); res[f'auc_te_{nm}'] = round(auc(yt, p), 3)
            out.append(res)
    return pd.DataFrame(out)

def stats(pos, ret):
    m = pos != 0; g = pos * ret; nt = g - FEE * np.abs(pos)
    return dict(n=int(m.sum()), gross_mean_bp=round(1e4 * g[m].mean(), 1) if m.any() else 0,
                net_mean_bp=round(1e4 * nt[m].mean(), 1) if m.any() else 0,
                hit=round((g[m] > 0).mean(), 3) if m.any() else np.nan,
                net_total_pct=round(100 * nt.sum() / 2, 2))  # /2: origins every 12h, 24h holds overlap

def trade_eval(d):
    ret = d.ret.values; kr = np.sign(d.k_ret.values); c = d.c.values
    sig = {
        'vwap24_revert': -np.sign(c - d.vwap24.values),
        'vwap168_trend': np.sign(c - d.vwap168.values),
        'ichimoku_trend': np.where(c > d.ctop, 1, np.where(c < d.cbot, -1, 0)),
        'pivot_revert': np.where(c > d.R1, -1, np.where(c < d.S1, 1, 0)),  # fade outside R1/S1
        'pivot_side': np.sign(c - d.P.values),
        'donchian_fade': np.where(d.dH_dist < 0.5, -1, np.where(d.dL_dist < 0.5, 1, 0)),
    }
    out = []
    bh = stats(np.ones(len(ret)), ret); out.append(dict(strategy='buy_hold', variant='-', **bh))
    out.append(dict(strategy='kronos_only', variant='-', **stats(kr, ret)))
    for k, s in sig.items():
        s = s.astype(float)
        out.append(dict(strategy=k, variant='indicator', **stats(s, ret)))
        f = np.where(s == kr, s, 0); out.append(dict(strategy=k, variant='+kronos_agree', **stats(f, ret)))
        f2 = np.where((s != 0) & (s != kr), s, 0); out.append(dict(strategy=k, variant='kronos_disagree', **stats(f2, ret)))
        # random filter keeping same fraction of trades as Kronos filter (1000 draws)
        keep = (f != 0).sum(); idx = np.flatnonzero(s != 0); rs = []
        for _ in range(1000):
            if keep == 0 or len(idx) == 0: break
            pick = rng.choice(idx, keep, replace=False); pr = np.zeros_like(s); pr[pick] = s[pick]
            rs.append(stats(pr, ret)['net_mean_bp'])
        kn = stats(f, ret)['net_mean_bp']
        out.append(dict(strategy=k, variant='random_filter_same_n', n=keep,
                        net_mean_bp=round(np.mean(rs), 1) if rs else np.nan,
                        pctile_of_kronos=round((np.array(rs) < kn).mean(), 3) if rs else np.nan))
    return pd.DataFrame(out)

if __name__ == '__main__':
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
    allp, allt = [], []
    for sym in ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']:
        d = run(sym); d.to_csv(f'/Users/matveypro/kronos_lab/agents/levels_vwap/rows_{sym}.csv', index=False)
        p = prob_eval(d); p.insert(0, 'sym', sym); allp.append(p)
        t = trade_eval(d); t.insert(0, 'sym', sym); allt.append(t)
    P = pd.concat(allp); T = pd.concat(allt)
    P.to_csv('/Users/matveypro/kronos_lab/agents/levels_vwap/prob_eval.csv', index=False)
    T.to_csv('/Users/matveypro/kronos_lab/agents/levels_vwap/trade_eval.csv', index=False)
    print(P.to_string(index=False)); print(); print(T.to_string(index=False))
