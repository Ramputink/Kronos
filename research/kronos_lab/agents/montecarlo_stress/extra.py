import numpy as np, pandas as pd
C = '/Users/matveypro/kronos_lab/cache/'
W = '/Users/matveypro/kronos_lab/agents/montecarlo_stress/'
for s in ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']:
    d = np.load(C + f'{s}_60_paths.npz'); P = d['paths']; O = d['origins']
    cl = pd.read_csv(C + f'{s}_60_ohlcv.csv').close.values
    kc = np.concatenate([cl[O][:, None, None].repeat(6, 1), P[:, :, :, 3]], 2); L = np.diff(np.log(kc), axis=2)
    k = lambda x: ((x - x.mean())**4).mean() / x.var()**2 - 3
    print(s, 'kurt all', round(k(L.ravel()), 1), 'excl step0', round(k(L[:, :, 1:].ravel()), 1),
          'max|r|', np.abs(L).max().round(3), 'step0std/rest', (L[:, :, 0].std() / L[:, :, 1:].std()).round(2))
    R = pd.read_csv(W + f'{s}_windows.csv')
    tr = L.sum(2)
    print('  kronos cross-path std of 24h logret', tr.std(1).mean().round(4), 'mean path RV',
          np.sqrt((L**2).sum(2)).mean().round(4), 'realised 24h logret rms', np.sqrt((np.log1p(R.real_ret)**2).mean()).round(4))
    y = np.log(R.real_rv)
    X1 = np.c_[np.ones(len(R)), np.log(R.ctx_rv24), np.log(R.ctx_rv256)]; X2 = np.c_[X1, np.log(R.kronos_rv)]
    n = len(R) * 2 // 3
    for nm, X in [('ctx', X1), ('ctx+kronos', X2)]:
        b = np.linalg.lstsq(X[:n], y[:n], rcond=None)[0]; e = y[n:] - X[n:] @ b
        print('  holdout logRV', nm, 'MSE', (e**2).mean().round(4), 'R2', (1 - (e**2).mean() / y[n:].var()).round(3))
