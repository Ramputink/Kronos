import numpy as np, pandas as pd, sys
OUT = '/Users/matveypro/kronos_lab/agents/vol_forecast/'
for sym in sys.argv[1:]:
    d = pd.read_csv(OUT + f'{sym}_forecasts.csv')
    L = np.log(d[['rv', 'k_rv', 'k_rng', 'atr', 'ewma', 'garch', 'trail24']])
    print('==', sym, 'n', len(d))
    print(' raw scale: mean k_rv/rv=%.2f  ewma/rv=%.2f garch/rv=%.2f' % ((d.k_rv / d.rv).mean(), (d.ewma / d.rv).mean(), (d.garch / d.rv).mean()))
    print(' corr(log k_rv, log atr)=%.2f, corr(log k_rv, log ewma)=%.2f' % (L.k_rv.corr(L.atr), L.k_rv.corr(L.ewma)))
    # partial corr of k_rv with rv controlling for each baseline
    for base in ['atr', 'ewma', 'garch']:
        def res(a, b):
            X = np.c_[np.ones(len(b)), b]; return a - X @ np.linalg.lstsq(X, a, rcond=None)[0]
        pc = np.corrcoef(res(L.rv.values, L[base].values), res(L.k_rv.values, L[base].values))[0, 1]
        # block bootstrap (blocks of 10 windows) CI of partial corr
        rng = np.random.default_rng(0); n = len(d); bs = []
        for _ in range(1000):
            st = rng.integers(0, n - 10, n // 10); idx = np.concatenate([np.arange(s, s + 10) for s in st])
            a = L.iloc[idx]
            bs.append(np.corrcoef(res(a.rv.values, a[base].values), res(a.k_rv.values, a[base].values))[0, 1])
        print(' partial corr(k_rv, rv | %s) = %.3f  block-boot 90%% CI [%.3f, %.3f]' % (base, pc, *np.percentile(bs, [5, 95])))
    # direction: does k_rv vs ewma spread predict |ret| or sign? (sanity)
    print(' corr(k_rv - ewma (log), log rv - log ewma) = %.3f' % np.corrcoef(L.k_rv - L.ewma, L.rv - L.ewma)[0, 1])
    # calendar control: weekend share of the next 24h + origin hour (known at o)
    raw = pd.read_csv('/Users/matveypro/kronos_lab/cache/%s_60_ohlcv.csv' % sym)
    ts = pd.to_datetime(raw.timestamps)
    wk = np.array([(ts.iloc[o + 1:o + 25].dt.dayofweek >= 5).mean() for o in d.o])
    hr = np.array([ts.iloc[o].hour for o in d.o]); hrd = (hr == hr[0]).astype(float)
    X = np.c_[np.ones(len(d)), L.atr, L.ewma, wk, hrd]
    r1 = L.rv.values - X @ np.linalg.lstsq(X, L.rv.values, rcond=None)[0]
    r2 = L.k_rv.values - X @ np.linalg.lstsq(X, L.k_rv.values, rcond=None)[0]
    print(' partial corr(k_rv, rv | atr, ewma, weekend share, origin hour) = %.3f' % np.corrcoef(r1, r2)[0, 1])
    print(' corr(k_rv, weekend share)=%.2f corr(rv, weekend share)=%.2f' % (np.corrcoef(L.k_rv, wk)[0, 1], np.corrcoef(L.rv, wk)[0, 1]))
    cut = len(d) // 3
    def oos(cols):
        X = np.c_[np.ones(len(d))] if not cols else np.c_[np.ones(len(d)), np.column_stack(cols)]
        b = np.linalg.lstsq(X[:cut], L.rv.values[:cut], rcond=None)[0]
        p = X[cut:] @ b; y = L.rv.values[cut:]
        return 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)
    print(' OOS R2 (calib first 1/3): atr+ewma+cal=%.3f  +k_rv=%.3f  | cal only=%.3f  k_rv+cal=%.3f' % (
        oos([L.atr, L.ewma, wk, hrd]), oos([L.atr, L.ewma, wk, hrd, L.k_rv]), oos([wk, hrd]), oos([wk, hrd, L.k_rv])))
    # vol targeting with calendar-aware models, non-overlapping test days, net of 0.055%/side on position changes
    te = np.arange(cut, len(d), 2); simple = np.exp(d.ret24.values[te]) - 1; tgt = np.median(d.rv.values[:cut])
    def vt(cols):
        X = np.c_[np.ones(len(d)), np.column_stack(cols)]
        b = np.linalg.lstsq(X[:cut], L.rv.values[:cut], rcond=None)[0]
        pos = np.clip(tgt / np.exp(X[te] @ b), 0, 2)
        x = pos * simple - 0.00055 * np.abs(np.diff(np.r_[0, pos]))
        eq = np.cumprod(1 + x)
        return 'sharpe=%.2f tot=%.3f maxdd=%.3f std=%.4f' % (x.mean() / x.std() * np.sqrt(365), eq[-1] - 1, np.max(1 - eq / np.maximum.accumulate(eq)), x.std())
    eq = np.cumprod(1 + simple)
    print(' VT net n=%d | B&H sharpe=%.2f tot=%.3f maxdd=%.3f std=%.4f' % (len(te), simple.mean() / simple.std() * np.sqrt(365), eq[-1] - 1, np.max(1 - eq / np.maximum.accumulate(eq)), simple.std()))
    print('   atr+ewma+cal       ', vt([L.atr, L.ewma, wk, hrd]))
    print('   atr+ewma+cal+k_rv  ', vt([L.atr, L.ewma, wk, hrd, L.k_rv]))
    print('   k_rv only          ', vt([L.k_rv]))
