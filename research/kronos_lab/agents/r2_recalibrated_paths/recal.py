import numpy as np, pandas as pd, json
from math import erf
C = '/Users/matveypro/kronos_lab/cache/'
FEE = 0.0011
rng = np.random.default_rng(0)
SQ2 = np.sqrt(2)

def ncdf(x): return 0.5 * (1 + np.vectorize(erf)(x / SQ2))
def npdf(x): return np.exp(-0.5 * x * x) / np.sqrt(2 * np.pi)
def crps_gauss(mu, sd, y):
    z = (y - mu) / sd
    return sd * (z * (2 * ncdf(z) - 1) + 2 * npdf(z) - 1 / np.sqrt(np.pi))
def crps_fair(ens, y):  # ens (n,m)
    m = ens.shape[1]
    t1 = np.abs(ens - y[:, None]).mean(1)
    t2 = np.abs(ens[:, :, None] - ens[:, None, :]).sum((1, 2)) / (2 * m * (m - 1))
    return t1 - t2
def pinball(q, y, tau):
    u = y - q
    return np.maximum(tau * u, (tau - 1) * u)
Z95 = 1.6448536

def spearman(a, b):
    return np.corrcoef(pd.Series(a).rank(), pd.Series(b).rank())[0, 1]

def block_boot(fn, n, B=2000, bl=10):
    nb = int(np.ceil(n / bl)); out = []
    for _ in range(B):
        st = rng.integers(0, n - bl + 1, nb)
        idx = np.concatenate([np.arange(s, s + bl) for s in st])[:n]
        out.append(fn(idx))
    out = np.array(out)
    return np.nanpercentile(out, 2.5), np.nanpercentile(out, 97.5)

def ols(X, y):
    X1 = np.column_stack([np.ones(len(X)), X])
    b, *_ = np.linalg.lstsq(X1, y, rcond=None)
    return b
def pred(b, X): return np.column_stack([np.ones(len(X)), X]) @ b

res = {}
for sym in ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']:
    df = pd.read_csv(C + f'{sym}_60_ohlcv.csv')
    d = np.load(C + f'{sym}_60_paths.npz')
    O = d['origins']; P = d['paths']
    keep = O + 24 < len(df)
    O = O[keep]; P = P[keep]
    c = df.close.values; h = df.high.values; l = df.low.values
    lc = np.log(c); ts = pd.to_datetime(df.timestamps)
    n = len(O)
    y = lc[O + 24] - lc[O]
    ens = np.log(P[:, :, -1, 3]) - lc[O][:, None]           # (n,6) path 24h log returns
    kmean = ens.mean(1); ksd = ens.std(1, ddof=1)
    # path range coverage diagnostic via path highs/lows of close
    z = np.array([(lc[o] - lc[o - 255:o + 1].mean()) / lc[o - 255:o + 1].std() for o in O])
    r24 = lc[O] - lc[O - 24]; r4 = lc[O] - lc[O - 4]
    hr = np.diff(lc)
    vol7 = np.array([hr[o - 168:o].std() for o in O]) * np.sqrt(24)
    park = np.array([np.sqrt(np.mean(np.log(h[o - 23:o + 1] / l[o - 23:o + 1]) ** 2) / (4 * np.log(2))) for o in O]) * np.sqrt(24)
    # weekend fraction of the forecast window
    wk = np.array([np.mean(ts.iloc[o + 1:o + 25].dt.dayofweek.values >= 5) for o in O])
    ntr = int(round(n * 2 / 3)); tr = np.arange(ntr); te = np.arange(ntr, n)

    # bias diagnostics (train)
    slope_k = ols(z[tr], kmean[tr])[1]; slope_y = ols(z[tr], y[tr])[1]

    # drift regression with and without Kronos
    Xk = np.column_stack([kmean, z, r24]); Xn = np.column_stack([z, r24])
    bk = ols(Xk[tr], y[tr]); bn = ols(Xn[tr], y[tr])
    mu = pred(bk, Xk); mu_n = pred(bn, Xn)
    # spread scale: recentred ensemble members = mu + s*(ens - kmean); s s.t. min-max coverage on train = 5/7
    dev = ens - kmean[:, None]
    def cov_minmax(s, idx):
        e = mu[idx, None] + s * dev[idx]
        return np.mean((y[idx] >= e.min(1)) & (y[idx] <= e.max(1)))
    grid = np.linspace(0.5, 8, 301)
    covs = np.array([cov_minmax(s, tr) for s in grid])
    s_hat = grid[np.argmin(np.abs(covs - 5 / 7))]
    ens_r = mu[:, None] + s_hat * dev
    sd_r = s_hat * ksd
    raw_cov_minmax = np.mean((y[tr] >= ens[tr].min(1)) & (y[tr] <= ens[tr].max(1)))
    # weekend-adjusted Parkinson: variance multiplier fitted on train for wk>0 vs wk==0
    ratio = (y / park) ** 2
    fwe = ratio[tr][wk[tr] > 0].mean(); fwd = ratio[tr][wk[tr] == 0].mean()
    park_adj = park * np.sqrt(np.where(wk > 0, fwe, fwd))
    # also a train-calibrated vol7 (single scale) for fairness
    f7 = np.sqrt(((y / vol7) ** 2)[tr].mean())

    T = te; yt = y[T]
    models = {
        'kronos_raw_ens': dict(crps=crps_fair(ens[T], yt), mu=kmean[T], sd=ksd[T]),
        'kronos_recal_ens': dict(crps=crps_fair(ens_r[T], yt), mu=mu[T], sd=sd_r[T]),
        'kronos_recal_gauss': dict(crps=crps_gauss(mu[T], sd_r[T], yt), mu=mu[T], sd=sd_r[T]),
        'gauss_vol7': dict(crps=crps_gauss(0, vol7[T], yt), mu=np.zeros(len(T)), sd=vol7[T]),
        'gauss_vol7_cal': dict(crps=crps_gauss(0, f7 * vol7[T], yt), mu=np.zeros(len(T)), sd=f7 * vol7[T]),
        'gauss_park_wkend': dict(crps=crps_gauss(0, park_adj[T], yt), mu=np.zeros(len(T)), sd=park_adj[T]),
        'gauss_noKronosReg_park': dict(crps=crps_gauss(mu_n[T], park_adj[T], yt), mu=mu_n[T], sd=park_adj[T]),
    }
    dens = {}
    for k, m in models.items():
        lo = m['mu'] - Z95 * m['sd']; hi = m['mu'] + Z95 * m['sd']
        pb = 0.5 * (pinball(lo, yt, 0.05) + pinball(hi, yt, 0.95))
        m['pin'] = pb
        dens[k] = dict(crps=float(m['crps'].mean() * 1e4), cov90=float(np.mean((yt >= lo) & (yt <= hi))),
                       pinball_bp=float(pb.mean() * 1e4))
    dens['kronos_raw_minmax_cov_test'] = float(np.mean((yt >= ens[T].min(1)) & (yt <= ens[T].max(1))))
    dens['kronos_recal_minmax_cov_test'] = float(np.mean((yt >= ens_r[T].min(1)) & (yt <= ens_r[T].max(1))))
    # CRPS diff CIs (recal ens vs best Gaussian baselines), negative = Kronos better
    a = models['kronos_recal_ens']['crps']
    for b in ['gauss_vol7', 'gauss_park_wkend', 'gauss_vol7_cal']:
        bb = models[b]['crps']
        dens[f'crpsdiff_recal_minus_{b}_bp'] = [float((a - bb).mean() * 1e4)] + [float(v * 1e4) for v in block_boot(lambda i: (a[i] - bb[i]).mean(), len(T))]
    # IC
    sigs = {'kronos_debiased_mu': mu[T], 'kronos_raw_mean': kmean[T], 'noKronos_reg_mu': mu_n[T],
            'momentum_24h': r24[T], 'reversal_4h': -r4[T]}
    ic = {}
    for k, s in sigs.items():
        lo_, hi_ = block_boot(lambda i: spearman(s[i], yt[i]), len(T), B=1000)
        ic[k] = [float(spearman(s, yt)), float(lo_), float(hi_)]
    # trades
    tr_res = {}
    def trade(sign):
        g = sign * yt; return g, g - FEE
    for k, s in [('kronos_debiased', np.sign(mu[T])), ('noKronos_reg', np.sign(mu_n[T])),
                 ('momentum_24h', np.sign(r24[T])), ('reversal_4h', np.sign(-r4[T])), ('buyhold', np.ones(len(T)))]:
        g, nt = trade(s)
        lo_, hi_ = block_boot(lambda i: nt[i].mean(), len(T))
        tr_res[k] = dict(gross_bp=float(g.mean() * 1e4), net_bp=float(nt.mean() * 1e4), net_CI_bp=[float(lo_ * 1e4), float(hi_ * 1e4)],
                         hit=float(np.mean(g > 0)), frac_long=float(np.mean(s > 0)))
    sk = np.sign(mu[T])
    sh = np.array([(rng.permutation(sk) * yt).mean() - FEE for _ in range(2000)])
    tr_res['shuffled_sign_net_bp'] = dict(mean=float(sh.mean() * 1e4), p95=float(np.percentile(sh, 95) * 1e4),
                                          pval_kronos=float(np.mean(sh >= tr_res['kronos_debiased']['net_bp'] / 1e4)))
    res[sym] = dict(n=int(n), n_train=int(ntr), n_test=int(len(T)),
                    test_period=[str(ts.iloc[O[ntr]]), str(ts.iloc[O[-1]])],
                    bias_train=dict(slope_kmean_on_z=float(slope_k), slope_y_on_z=float(slope_y),
                                    mean_kmean_bp=float(kmean[tr].mean() * 1e4), mean_y_bp=float(y[tr].mean() * 1e4),
                                    raw_minmax_cov=float(raw_cov_minmax)),
                    coef_withK=dict(zip(['const', 'kmean', 'z', 'r24'], map(float, bk))),
                    coef_noK=dict(zip(['const', 'z', 'r24'], map(float, bn))),
                    spread_scale=float(s_hat), park_wk_var_mult=[float(fwd), float(fwe)], vol7_scale=float(f7),
                    density=dens, ic=ic, trades=tr_res)
json.dump(res, open('/Users/matveypro/kronos_lab/agents/r2_recalibrated_paths/results.json', 'w'), indent=1)
print(json.dumps(res, indent=1))
