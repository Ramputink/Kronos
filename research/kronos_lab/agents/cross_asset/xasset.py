import numpy as np, pandas as pd
C = '/Users/matveypro/kronos_lab/cache/'
OUT = '/Users/matveypro/kronos_lab/agents/cross_asset/'
S = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
FEE = 0.0011  # round trip per leg (0.055% x2)
D = {s: pd.read_csv(C + s + '_60_ohlcv.csv') for s in S}
assert all((D[s].timestamps.values == D['BTCUSDT'].timestamps.values).all() for s in S)
Z = {s: np.load(C + s + '_60_paths.npz') for s in S}
o = Z['BTCUSDT']['origins']
assert all((Z[s]['origins'] == o).all() for s in S)
pred, real, mom = {}, {}, {}
for s in S:
    c = D[s].close.values
    p = Z[s]['paths'][:, :, -1, 3]  # (W,6) final predicted close at +24h
    pred[s] = p.mean(1) / c[o] - 1   # uses only paths for o and c[o]
    real[s] = c[o + 24] / c[o] - 1
    mom[s] = c[o] / c[o - 24] - 1    # past 24h, rows <= o
W = len(o)
hold = np.arange(W) >= 2 * W // 3
rng = np.random.default_rng(0)


def stats(r, legs):
    g = r.mean(); n = g - legs * FEE
    sd = r.std(ddof=1)
    t = g / (sd / np.sqrt(len(r) / 2)) if sd > 0 else 0.0  # effective n halved: 24h trades every 12h overlap
    return dict(n=len(r), hit=(r > 0).mean(), gross_bp=g * 1e4, net_bp=n * 1e4, t_eff=t,
                cumG=0.5 * r.sum(), cumN=0.5 * (r - legs * FEE).sum())  # half-capital per trade book


rows = []


def add(name, r, legs, mask_all=None):
    idx = np.arange(len(r))
    for lab, m in [('all', np.ones(len(r), bool)), ('hold', hold if mask_all is None else mask_all)]:
        d = stats(r[m], legs); d.update(strategy=name, sample=lab); rows.append(d)


def ic(a, b):
    return np.corrcoef(a, b)[0, 1]


lines = []
pr = lambda *a: (print(*a), lines.append(' '.join(str(x) for x in a)))
pairs = [('ETHUSDT', 'BTCUSDT'), ('SOLUSDT', 'BTCUSDT'), ('SOLUSDT', 'ETHUSDT')]
pr('Pair IC (pearson): predicted spread vs realised spread; momentum spread vs realised')
for a, b in pairs:
    ps = pred[a] - pred[b]; rs = real[a] - real[b]; ms = mom[a] - mom[b]
    tag = f'{a[:3]}/{b[:3]}'
    pr(f'{tag}: IC_kronos={ic(ps, rs):+.3f} (holdout {ic(ps[hold], rs[hold]):+.3f})  IC_mom={ic(ms, rs):+.3f}  corr(kronos,mom)={ic(ps, ms):+.3f}')
    add(tag + ' kronos long stronger fcst', np.sign(ps) * rs, 2)
    add(tag + ' kronos INVERTED', -np.sign(ps) * rs, 2)
    add(tag + ' momentum 24h', np.sign(ms) * rs, 2)
    add(tag + ' reversal 24h', -np.sign(ms) * rs, 2)
    add(tag + ' B&H spread (long alt/short base)', rs, 2)
    ag = np.sign(ps) == np.sign(ms)
    add(tag + ' kronos&mom agree (trades only)', (np.sign(ps) * rs)[ag], 2, mask_all=hold[ag])
    k = np.mean(np.sign(ps) * rs)
    rnd = np.array([(rng.permutation(np.sign(ps)) * rs).mean() for _ in range(2000)])
    pr(f'   kronos pair gross mean {k*1e4:+.1f}bp; percentile vs 2000 shuffled signals = {(rnd < k).mean():.2f}')
pr('\nCross-asset direction:')
for t in ['ETHUSDT', 'SOLUSDT']:
    pr(f'{t}: IC(BTC kronos fcst, {t} realised)={ic(pred["BTCUSDT"], real[t]):+.3f}  IC(own fcst)={ic(pred[t], real[t]):+.3f}  IC(BTC 24h mom)={ic(mom["BTCUSDT"], real[t]):+.3f}')
    add(f'{t[:3]} dir from BTC kronos', np.sign(pred['BTCUSDT']) * real[t], 1)
    add(f'{t[:3]} dir from own kronos', np.sign(pred[t]) * real[t], 1)
    add(f'{t[:3]} dir from BTC 24h mom', np.sign(mom['BTCUSDT']) * real[t], 1)
    add(f'{t[:3]} buy&hold', real[t], 1)
P = np.column_stack([pred[s] for s in S]); R = np.column_stack([real[s] for s in S]); M = np.column_stack([mom[s] for s in S])
ix = np.arange(W)
add('XS3 kronos long best/short worst', R[ix, P.argmax(1)] - R[ix, P.argmin(1)], 2)
add('XS3 momentum long best/short worst', R[ix, M.argmax(1)] - R[ix, M.argmin(1)], 2)
add('XS3 reversal', R[ix, M.argmin(1)] - R[ix, M.argmax(1)], 2)
add('XS3 equal-weight B&H', R.mean(1), 1)

X, y = [], []
for a, b in pairs:
    ps = pred[a] - pred[b]; ms = mom[a] - mom[b]; rs = real[a] - real[b]
    X.append(np.c_[(ps - ps.mean()) / ps.std(), (ms - ms.mean()) / ms.std()]); y.append(rs)
X = np.vstack(X); y = np.concatenate(y); X1 = np.c_[np.ones(len(y)), X]
b = np.linalg.lstsq(X1, y, rcond=None)[0]; e = y - X1 @ b
se = np.sqrt(np.diag(np.linalg.inv(X1.T @ X1)) * e.var() * 2)  # x2 overlap; pairs also correlated -> still optimistic
pr('\nPooled OLS realised spread ~ z(kronos spread) + z(mom spread): kronos b=%.5f t=%.2f | mom b=%.5f t=%.2f' % (b[1], b[1] / se[1], b[2], b[2] / se[2]))

df = pd.DataFrame(rows)[['strategy', 'sample', 'n', 'hit', 'gross_bp', 'net_bp', 't_eff', 'cumG', 'cumN']]
pd.set_option('display.width', 220); pd.set_option('display.max_rows', 300)
print(df.round(3).to_string(index=False))
df.to_csv(OUT + 'results.csv', index=False)
open(OUT + 'summary.txt', 'w').write('\n'.join(lines) + '\n\n' + df.round(3).to_string(index=False))
