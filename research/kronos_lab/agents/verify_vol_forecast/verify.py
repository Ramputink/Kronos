import numpy as np, pandas as pd
C = '/Users/matveypro/kronos_lab/cache/'; V = '/Users/matveypro/kronos_lab/agents/vol_forecast/'


def lstsq(X, y): return np.linalg.lstsq(X, y, rcond=None)[0]


rng = np.random.default_rng(1)
for sym in ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']:
    df = pd.read_csv(C + f'{sym}_60_ohlcv.csv'); d = pd.read_csv(V + f'{sym}_forecasts.csv')
    c = df.close.values; h = df.high.values; l = df.low.values
    lr = np.r_[np.nan, np.diff(np.log(c))]
    pk = (np.log(h / l)) ** 2 / (4 * np.log(2))  # Parkinson per-bar variance
    ts = pd.to_datetime(df.timestamps)
    rv2 = np.array([np.sqrt(np.sum(lr[o + 1:o + 25] ** 2)) for o in d.o]); assert np.allclose(rv2, d.rv)

    def trail(x, o, n): return np.sqrt(np.sum(x[o - n + 1:o + 1]) * 24 / n)
    F = pd.DataFrame({
        'rv24': [trail(lr ** 2, o, 24) for o in d.o], 'rv168': [trail(lr ** 2, o, 168) for o in d.o],
        'rv256': [trail(lr ** 2, o, 256) for o in d.o], 'pk24': [trail(pk, o, 24) for o in d.o],
        'pk168': [trail(pk, o, 168) for o in d.o]})
    L = np.log(F)
    wk = np.array([(ts.iloc[o + 1:o + 25].dt.dayofweek >= 5).mean() for o in d.o])
    hr = np.array([ts.iloc[o].hour for o in d.o]); hrd = (hr == hr[0]).astype(float)
    y = np.log(d.rv.values); k = np.log(d.k_rv.values); n = len(d)
    sets = {'orig(atr,ewma,cal)': np.c_[np.log(d.atr), np.log(d.ewma), wk, hrd],
            'HAR(rv24,rv168,rv256)+cal': np.c_[L.rv24, L.rv168, L.rv256, wk, hrd],
            'park(pk24,pk168)+cal': np.c_[L.pk24, L.pk168, wk, hrd],
            'HAR+park+atr+cal': np.c_[L.rv24, L.rv168, L.rv256, L.pk24, L.pk168, np.log(d.atr), wk, hrd]}
    print('=====', sym, 'n', n, 'corr(log k_rv, pk24/pk168/rv256): %.2f %.2f %.2f' % (
        np.corrcoef(k, L.pk24)[0, 1], np.corrcoef(k, L.pk168)[0, 1], np.corrcoef(k, L.rv256)[0, 1]))

    def pred(X, tr, te):
        X1 = np.c_[np.ones(n), X]; b = lstsq(X1[tr], y[tr]); return X1[te] @ b
    cut = n // 3; idx = np.arange(n)
    splits = {'calib1/3->test2/3': (idx[:cut], idx[cut:]),
              'calib first2/3->test last1/3': (idx[:2 * cut - 1], idx[2 * cut:]),
              'reverse calib last1/3->test first2/3': (idx[2 * cut + 1:], idx[:2 * cut])}
    for name, X in sets.items():
        print(' ', name)
        for sn, (tr, te) in splits.items():
            pb = pred(X, tr, te); pkk = pred(np.c_[X, k], tr, te); yt = y[te]; ss = np.sum((yt - yt.mean()) ** 2)
            e0 = (yt - pb) ** 2; e1 = (yt - pkk) ** 2; m = len(te); bs = []
            for _ in range(1000):
                st = rng.integers(0, m - 10, m // 10); ii = np.concatenate([np.arange(s, s + 10) for s in st])
                bs.append(np.mean(e0[ii] - e1[ii]))
            ci = np.percentile(bs, [5, 95])
            print('     %-38s R2 %.3f -> %.3f  dMSE(base-k) 90%%CI [%.4f, %.4f]' % (sn, 1 - e0.sum() / ss, 1 - e1.sum() / ss, *ci))
        pb = []; pk2 = []
        for t in range(100, n):
            tr = np.arange(0, t - 1)  # drop t-1 whose target overlaps beyond origin t
            pb.append(pred(X, tr, [t])[0]); pk2.append(pred(np.c_[X, k], tr, [t])[0])
        yt = y[100:]; ss = np.sum((yt - yt.mean()) ** 2)
        print('     %-38s R2 %.3f -> %.3f' % ('walk-forward expanding (t>=100)', 1 - np.sum((yt - np.array(pb)) ** 2) / ss,
                                             1 - np.sum((yt - np.array(pk2)) ** 2) / ss))
