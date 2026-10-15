import numpy as np, pandas as pd, os, json
C = '/Users/matveypro/kronos_lab/cache'; FEE = 0.0011; H = 24
rng = np.random.default_rng(0)
def ema(x, n): return x.ewm(span=n, adjust=False).mean()
out = {}
for sym in ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']:
    p = f'{C}/{sym}_60_paths.npz'
    if not os.path.exists(p): continue
    df = pd.read_csv(f'{C}/{sym}_60_ohlcv.csv'); z = np.load(p)
    org = z['origins']; paths = z['paths']
    ok = org + H < len(df); org = org[ok]; paths = paths[ok]
    c = df.close; hi = df.high; lo = df.low
    # causal indicators (row i uses rows <= i)
    ema_sig = np.sign(ema(c, 12) - ema(c, 48)).values
    macd = ema(c, 12) - ema(c, 26); macd_sig = np.sign(macd - ema(macd, 9)).values
    up = hi.rolling(48).max().shift(1); dn = lo.rolling(48).min().shift(1)
    st = np.zeros(len(c)); s = 0; cv = c.values
    for i in range(len(c)):
        if cv[i] > up.iloc[i]: s = 1
        elif cv[i] < dn.iloc[i]: s = -1
        st[i] = s
    c0 = cv[org]; real = cv[org + H] / c0 - 1
    pc = paths[:, :, -1, 3]                       # (W,6) predicted close at h24
    kr = np.median(pc, 1) / c0 - 1
    kagree = np.mean(np.sign(pc / c0[:, None] - 1), 1)
    T = len(org); split = 2 * T // 3
    def stats(pos):
        tr = pos != 0; g = pos * real; n = g - FEE * tr
        return dict(trades=int(tr.sum()),
                    gross_mean_bp=float(1e4 * g[tr].mean()) if tr.any() else 0.0,
                    net_mean_bp=float(1e4 * n[tr].mean()) if tr.any() else 0.0,
                    net_sum=float(n.sum() / 2),  # two interleaved half-capital books
                    gross_sum=float(g.sum() / 2),
                    hit=float((g[tr] > 0).mean()) if tr.any() else float('nan'))
    res = dict(n=T, buyhold_total=float(cv[org[-1] + H] / c0[0] - 1),
               buyhold_sum=float(real.sum() / 2), kronos_alone=stats(np.sign(kr)))
    ho = slice(split, T)
    def ns(pos):
        tr = pos[ho] != 0; return float(((pos[ho] * real[ho]) - FEE * tr).sum() / 2)
    for name, sig in [('ema12_48', ema_sig), ('macd', macd_sig), ('donchian48', st)]:
        t = sig[org]
        filt = np.where(np.sign(kr) == t, t, 0)
        strong = np.where(kagree * t >= 4 / 6, t, 0)
        fn = stats(filt)['net_sum']
        k = int((filt != 0).sum()); idx = np.where(t != 0)[0]; nets = []
        for _ in range(2000):
            m = np.zeros(T, bool); m[rng.choice(idx, k, replace=False)] = True
            nets.append(((t * real - FEE) * m).sum() / 2)
        nets = np.array(nets)
        ag = (np.sign(kr) == t) & (t != 0); dg = (np.sign(kr) != t) & (t != 0)
        res[name] = dict(alone=stats(t), kronos_filter=stats(filt), kronos_strong=stats(strong),
                         random_filter_net_sum_mean=float(nets.mean()),
                         random_filter_pctile_of_kronos=float((nets < fn).mean()),
                         holdout_net_sum=dict(alone=ns(t), kfilt=ns(filt), kstrong=ns(strong),
                                              bh=float(real[ho].sum() / 2)),
                         agree_bp=float(1e4 * (t * real)[ag].mean()), n_agree=int(ag.sum()),
                         disagree_bp=float(1e4 * (t * real)[dg].mean()), n_disagree=int(dg.sum()))
    out[sym] = res
json.dump(out, open('/Users/matveypro/kronos_lab/agents/trend_ma/results.json', 'w'), indent=1)
for s, r in out.items():
    print(f"\n== {s} n={r['n']} BH total {r['buyhold_total']:.3f} BH sum(half-books) {r['buyhold_sum']:.3f}")
    print('  Kronos alone', {k: round(v, 3) for k, v in r['kronos_alone'].items()})
    for nm in ['ema12_48', 'macd', 'donchian48']:
        x = r[nm]
        for lab in ['alone', 'kronos_filter', 'kronos_strong']:
            print(f"  {nm:10s} {lab:13s}", {k: round(v, 3) for k, v in x[lab].items()})
        print(f"  {nm:10s} randfilt mean net_sum {x['random_filter_net_sum_mean']:.3f}  Kfilt pctile {x['random_filter_pctile_of_kronos']:.2f}"
              f"  agree {x['agree_bp']:.1f}bp (n{x['n_agree']}) disagree {x['disagree_bp']:.1f}bp (n{x['n_disagree']})  HO {{{', '.join(f'{k}:{v:.3f}' for k,v in x['holdout_net_sum'].items())}}}")
