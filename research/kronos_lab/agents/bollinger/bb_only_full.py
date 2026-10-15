import numpy as np, pandas as pd, glob, os
FEE = 0.0011; N = 20; K = 2.0; H = 24
for f in sorted(glob.glob('/Users/matveypro/kronos_lab/cache/*_60_ohlcv.csv')):
    c = pd.read_csv(f).close.values
    mid = pd.Series(c).rolling(N).mean().values; sd = pd.Series(c).rolling(N).std(ddof=0).values
    pb = (c - (mid - K * sd)) / (2 * K * sd)
    # every 12h origin, full history, same rule as cached test
    idx = np.arange(N, len(c) - H, 12)
    fut = c[idx + H] / c[idx] - 1
    print(os.path.basename(f), 'origins', len(idx), 'BH mean24h %.3f%%' % (fut.mean() * 100))
    for name, lt, ht in [('strict', 0, 1), ('relaxed', .2, .8)]:
        side = np.where(pb[idx] < lt, 1, np.where(pb[idx] > ht, -1, 0))
        m = side != 0
        for mode, sg in [('MR', 1), ('BO', -1)]:
            r = sg * side[m] * fut[m]
            print('  %-7s %s n=%3d gross=%.3f%% net=%.3f%% hit=%.0f%%' % (name, mode, m.sum(), r.mean() * 100, (r - FEE).mean() * 100, (r > 0).mean() * 100))
