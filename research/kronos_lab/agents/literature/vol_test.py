"""Test the one claim the literature + user reports agree on: Kronos sample-path
dispersion forecasts realised volatility. Compare to HAR-style trailing RV.
No lookahead: features use csv rows <= o and paths[o]; target = rows o+1..o+24."""
import numpy as np, pandas as pd
C = '/Users/matveypro/kronos_lab/cache/'
OUT = '/Users/matveypro/kronos_lab/agents/literature/'


def spearman(a, b):
    return pd.Series(np.asarray(a)).rank().corr(pd.Series(np.asarray(b)).rank())


rows = []
for s in ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']:
    df = pd.read_csv(C + f'{s}_60_ohlcv.csv')
    z = np.load(C + f'{s}_60_paths.npz')
    o, P = z['origins'], z['paths']
    c = df.close.values
    lr = np.r_[np.nan, np.diff(np.log(c))]
    rec = []
    for i, oi in enumerate(o):
        if oi + 24 >= len(c) or oi < 200:
            continue
        fut = lr[oi + 1:oi + 25]
        rv = np.sqrt((fut ** 2).sum())
        rv24 = np.sqrt((lr[oi - 23:oi + 1] ** 2).sum())
        rv168 = np.sqrt((lr[oi - 167:oi + 1] ** 2).sum() / 7)
        pc = P[i, :, :, 3]
        last = c[oi]
        plr = np.diff(np.log(np.concatenate([np.full((pc.shape[0], 1), last), pc], 1)), axis=1)
        k_path = np.sqrt((plr ** 2).sum(1)).mean()
        k_disp = np.std(np.log(pc[:, -1] / last))
        k_rng = np.log(P[i, :, :, 1].max(1) / P[i, :, :, 2].min(1)).mean()
        r = np.log(c[oi + 24] / last)
        lo, hi = np.log(pc[:, -1].min() / last), np.log(pc[:, -1].max() / last)
        rec.append(dict(rv=rv, rv24=rv24, rv168=rv168, k_path=k_path, k_disp=k_disp,
                        k_rng=k_rng, absret=abs(r), cov=(lo <= r <= hi)))
    d = pd.DataFrame(rec)
    n = len(d)
    out = {'sym': s, 'n': n}
    for f in ['rv24', 'rv168', 'k_path', 'k_disp', 'k_rng']:
        out['rho_' + f + '_vs_RV'] = round(spearman(d[f], d.rv), 3)
    y = np.log(d.rv.values)
    cut = int(n * 2 / 3)

    def oos(cols):
        X = np.column_stack([np.ones(n)] + [np.log(d[c_].values) for c_ in cols])
        b = np.linalg.lstsq(X[:cut], y[:cut], rcond=None)[0]
        e = y[cut:] - X[cut:] @ b
        return 1 - (e ** 2).sum() / ((y[cut:] - y[:cut].mean()) ** 2).sum()

    out['oosR2_HAR'] = round(oos(['rv24', 'rv168']), 3)
    out['oosR2_Konly'] = round(oos(['k_path']), 3)
    out['oosR2_HAR+K'] = round(oos(['rv24', 'rv168', 'k_path']), 3)
    out['oosR2_HAR+Krng'] = round(oos(['rv24', 'rv168', 'k_rng']), 3)
    out['cov_6path_minmax(nominal~0.71)'] = round(d['cov'].mean(), 3)
    out['rho_kdisp_vs_absret'] = round(spearman(d.k_disp, d.absret), 3)
    out['rho_rv24_vs_absret'] = round(spearman(d.rv24, d.absret), 3)
    out['median_kpath/RV'] = round((d.k_path / d.rv).median(), 3)
    rows.append(out)

r = pd.DataFrame(rows).set_index('sym')
print(r.T.to_string())
r.to_csv(OUT + 'vol_test.csv')


# Partial Spearman of Kronos vol on realised vol, controlling for HAR terms (no fitting to target)
def resid(yv, Xm):
    X = np.column_stack([np.ones(len(yv))] + Xm)
    return yv - X @ np.linalg.lstsq(X, yv, rcond=None)[0]

print('\npartial rank-corr(k_path, RV | rv24, rv168), and second-half only')
for s in ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']:
    df = pd.read_csv(C + f'{s}_60_ohlcv.csv'); z = np.load(C + f'{s}_60_paths.npz')
    c = df.close.values; lr = np.r_[np.nan, np.diff(np.log(c))]; R = []
    for i, oi in enumerate(z['origins']):
        if oi + 24 >= len(c) or oi < 200: continue
        pc = z['paths'][i, :, :, 3]
        plr = np.diff(np.log(np.concatenate([np.full((6, 1), c[oi]), pc], 1)), axis=1)
        R.append([np.sqrt((lr[oi+1:oi+25]**2).sum()), np.sqrt((lr[oi-23:oi+1]**2).sum()),
                  np.sqrt((lr[oi-167:oi+1]**2).sum()), np.sqrt((plr**2).sum(1)).mean()])
    R = pd.DataFrame(R, columns=['rv', 'rv24', 'rv168', 'k']).rank().values
    for name, sl in [('all', slice(None)), ('2nd half', slice(len(R)//2, None))]:
        Q = R[sl]
        a = resid(Q[:, 3], [Q[:, 1], Q[:, 2]]); b = resid(Q[:, 0], [Q[:, 1], Q[:, 2]])
        print(s, name, 'n=', len(Q), 'partial rho=', round(np.corrcoef(a, b)[0, 1], 3))
