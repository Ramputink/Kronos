import numpy as np, pandas as pd
C = '/Users/matveypro/kronos_lab/cache/'
def rank(x): return pd.Series(x).rank().values
def sp(a, b): return np.corrcoef(rank(a), rank(b))[0, 1]
def partial(a, b, c):  # spearman partial corr of a,b given c (rank residualisation)
    ra, rb, rc = rank(a), rank(b), rank(c)
    ea = ra - np.polyval(np.polyfit(rc, ra, 1), rc); eb = rb - np.polyval(np.polyfit(rc, rb, 1), rc)
    return np.corrcoef(ea, eb)[0, 1]
for sym in ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']:
    df = pd.read_csv(C + f'{sym}_60_ohlcv.csv'); z = np.load(C + f'{sym}_60_paths.npz')
    o = z['origins']; P = z['paths']; cl = df.close.values
    keep = o + 24 < len(cl); o = o[keep]; P = P[keep]; c0 = cl[o]
    lr = np.log(cl); ret1 = np.diff(lr, prepend=lr[0])
    for h in [1, 2]:
        k = np.median(P[:, :, h - 1, 3], 1) / c0 - 1
        r = cl[o + h] / c0 - 1
        rev1 = -(cl[o] / cl[o - 1] - 1); rev4 = -(cl[o] / cl[o - 4] - 1)
        print(sym, f'h={h} IC kronos={sp(k,r):.3f} IC rev1h={sp(rev1,r):.3f} IC rev4h={sp(rev4,r):.3f} '
              f'corr(kronos,rev1)={sp(k,rev1):.2f} partial IC kronos|rev1={partial(k,r,rev1):.3f}')
    # volatility: Kronos path spread / median abs path move vs trailing realised vol
    kvol = np.median(np.abs(P[:, :, 23, 3] / c0[:, None] - 1), 1)
    kspread = P[:, :, 23, 3].std(1) / c0
    kpathvol = np.median(np.std(np.diff(np.log(P[:, :, :, 3]), axis=2), axis=2), 1)
    trail = np.array([ret1[i - 23:i + 1].std() for i in o])
    trail72 = np.array([ret1[i - 71:i + 1].std() for i in o])
    rabs = np.abs(cl[o + 24] / c0 - 1)
    rvol = np.array([ret1[i + 1:i + 25].std() for i in o])
    print(sym, f'vol: IC(|r24|) kronos_absmove={sp(kvol,rabs):.3f} spread={sp(kspread,rabs):.3f} trail24={sp(trail,rabs):.3f} | '
          f'IC(realised hourly vol next24) kronos_pathvol={sp(kpathvol,rvol):.3f} trail24={sp(trail,rvol):.3f} trail72={sp(trail72,rvol):.3f} '
          f'partial kronos|trail24={partial(kpathvol,rvol,trail):.3f}')
