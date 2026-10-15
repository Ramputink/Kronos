import numpy as np, pandas as pd, json, os
from statistics import NormalDist
N = NormalDist(); rng = np.random.default_rng(0)
C = '/Users/matveypro/kronos_lab/cache/'
OUT = '/Users/matveypro/kronos_lab/agents/stats_baseline/'
FEE = 0.0011; BLOCK = 10; B = 1500

def rank(x): return pd.Series(x).rank().values
def spear(a, b):
    if len(a) < 5: return np.nan
    return np.corrcoef(rank(a), rank(b))[0, 1]
def mbb_idx(n, L):
    k = int(np.ceil(n / L)); st = rng.integers(0, n - L + 1, k)
    return np.concatenate([np.arange(s, s + L) for s in st])[:n]
def ci(fn, n, L=BLOCK):
    v = np.array([fn(mbb_idx(n, L)) for _ in range(B)]); return np.nanpercentile(v, [2.5, 97.5])

out = {}; rows = []; pooled = []
for sym in ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']:
    df = pd.read_csv(C + f'{sym}_60_ohlcv.csv'); z = np.load(C + f'{sym}_60_paths.npz')
    o = z['origins']; P = z['paths']; cl = df.close.values
    keep = o + 24 < len(cl); o = o[keep]; P = P[keep]
    c0 = cl[o]
    pred = np.median(P[:, :, :, 3], axis=1) / c0[:, None] - 1
    pmean = P[:, :, :, 3].mean(1) / c0[:, None] - 1
    agree = np.sign(P[:, :, 23, 3] / c0[:, None] - 1).mean(1)
    real = np.stack([cl[o + h] / c0 - 1 for h in range(1, 25)], 1)
    n = len(o)
    for h in [1, 2, 4, 6, 12, 18, 24]:
        p, r = pred[:, h - 1], real[:, h - 1]
        ic = spear(p, r); hit = np.mean(np.sign(p) == np.sign(r))
        icci = ci(lambda i: spear(p[i], r[i]), n)
        hitci = ci(lambda i: np.mean(np.sign(p[i]) == np.sign(r[i])), n)
        icno = spear(p[::2], r[::2]) if h > 12 else ic
        null = np.array([spear(np.roll(p, s), r) for s in range(20, n - 20)])
        pval = np.mean(np.abs(null) >= abs(ic))
        rows.append(dict(sym=sym, h=h, n=n, IC=ic, IC_lo=icci[0], IC_hi=icci[1], IC_nonovl=icno,
                         hit=hit, hit_lo=hitci[0], hit_hi=hitci[1], up_frac=np.mean(r > 0), p_circ=pval))
    s = np.sign(pred[::2, 23]); r24 = real[::2, 23]
    g = s * r24; net = g - FEE
    conf = np.abs(agree[::2]) >= 0.67
    rsh = np.array([np.mean(rng.choice([-1, 1], len(r24)) * r24 - FEE) for _ in range(2000)])
    out[sym] = dict(start=str(df.timestamps.iloc[o[0]]), end=str(df.timestamps.iloc[o[-1]]),
                    n_trades=len(g), gross_mean=g.mean(), net_mean=net.mean(), net_sr_trade=net.mean() / net.std(),
                    bh_mean=r24.mean(), bh_total=cl[o[-1] + 24] / c0[0] - 1,
                    random_net_mean_p50=np.median(rsh), random_net_p_beats=np.mean(rsh >= net.mean()),
                    conf_n=int(conf.sum()), conf_net_mean=float(net[conf].mean()) if conf.any() else None,
                    ic24_pathmean=spear(pmean[:, 23], real[:, 23]), ic24_agree=spear(agree, real[:, 23]),
                    ic24_spread_vs_absret=spear(P[:, :, 23, 3].std(1) / c0, np.abs(real[:, 23])))
    pooled.append((pred[:, 23], real[:, 23]))

res = pd.DataFrame(rows)
pd.set_option('display.width', 220); pd.set_option('display.precision', 3)
print(res.to_string(index=False))
for k, v in out.items(): print(k, {a: (round(b, 4) if isinstance(b, float) else b) for a, b in v.items()})
pp = np.concatenate([rank(a) / len(a) for a, _ in pooled]); rr = np.concatenate([rank(b) / len(b) for _, b in pooled])
print('pooled IC24 (within-sym ranks):', round(np.corrcoef(pp, rr)[0, 1], 4), 'n=', len(pp))

def emax_z(Ntr):
    g = 0.5772156649
    return (1 - g) * N.inv_cdf(1 - 1 / Ntr) + g * N.inv_cdf(1 - 1 / (Ntr * np.e))
print('\nMultiple-testing thresholds:')
for neff in [75, 150, 450]:
    for Ntr in [1, 20, 30, 100, 300]:
        em = 1.96 if Ntr == 1 else emax_z(Ntr)
        zb = N.inv_cdf(1 - 0.025 / Ntr)
        print(f' n_eff={neff:3d} trials={Ntr:4d}: E[max null z]={em:.2f} Bonf z={zb:.2f} -> |IC|>{zb/np.sqrt(neff):.3f}, '
              f'per-trade SR>{zb/np.sqrt(neff):.3f} (ann. ~{zb/np.sqrt(neff)*np.sqrt(365):.2f}), hit>{0.5+zb*0.5/np.sqrt(neff):.3f}')
res.to_csv(OUT + 'ic_table.csv', index=False)
json.dump(out, open(OUT + 'strategy24.json', 'w'), indent=1, default=float)
