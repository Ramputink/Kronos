import numpy as np, pandas as pd, os, json
C = '/Users/matveypro/kronos_lab/cache/'
FEE = 0.0011; N = 20; K = 2.0; H = 24
rng = np.random.default_rng(0)


def stats(r):
    r = np.asarray(r)
    if len(r) == 0:
        return dict(n=0)
    return dict(n=int(len(r)), gross_mean=float(r.mean() * 100), net_mean=float((r - FEE).mean() * 100),
                net_sum=float((r - FEE).sum() * 100), hit=float((r > 0).mean() * 100))


out = {}
for sym in ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']:
    p = C + f'{sym}_60_paths.npz'
    if not os.path.exists(p):
        continue
    df = pd.read_csv(C + f'{sym}_60_ohlcv.csv'); c = df.close.values
    mid = pd.Series(c).rolling(N).mean().values; sd = pd.Series(c).rolling(N).std(ddof=0).values
    up = mid + K * sd; lo = mid - K * sd
    z = np.load(p); O = z['origins']; P = z['paths']
    O_ok = [(i, int(o)) for i, o in enumerate(O) if o + H < len(c) and o >= N]
    rows = []
    for i, o in O_ok:
        fut = c[o + H] / c[o] - 1
        pb = (c[o] - lo[o]) / (up[o] - lo[o])
        med = np.median(P[i, :, :, 3], axis=0)
        kdir = np.sign(med[-1] - c[o])
        frac_up = (P[i, :, -1, 3] > c[o]).mean()
        mh = np.median(P[i, :, :, 1], axis=0).max(); ml = np.median(P[i, :, :, 2], axis=0).min()
        rows.append(dict(o=o, fut=fut, pb=pb, kdir=kdir, frac_up=frac_up,
                         reach_mid_up=mh >= mid[o], reach_mid_dn=ml <= mid[o],
                         reach_up=mh >= up[o], reach_lo=ml <= lo[o]))
    d = pd.DataFrame(rows); n = len(d)
    res = {'n_windows': n, 'buyhold_mean24h_pct': float(d.fut.mean() * 100),
           'buyhold_total_pct': float((c[O_ok[-1][1] + H] / c[O_ok[0][1]] - 1) * 100),
           'kronos_alone_sign': stats(d.kdir * d.fut)}
    for thr_name, (lt, ht) in {'strict(%B<0,>1)': (0, 1), 'relaxed(%B<.2,>.8)': (0.2, 0.8)}.items():
        L = (d.pb < lt).values; S = (d.pb > ht).values
        for mode in ['MR', 'BO']:
            side = np.where(L, 1, np.where(S, -1, 0)) * (1 if mode == 'MR' else -1)
            m = side != 0
            base = side[m] * d.fut.values[m]
            agree_end = (d.kdir.values == side)
            if mode == 'MR':
                agree_path = np.where(side == 1, d.reach_mid_up, d.reach_mid_dn)
            else:
                agree_path = np.where(side == 1, d.reach_up, d.reach_lo)
            agree_prob = np.where(side == 1, d.frac_up >= 0.5, d.frac_up <= 0.5)
            r = {'alone': stats(base)}
            for fn, f in [('k_end', agree_end), ('k_path', agree_path), ('k_prob', agree_prob)]:
                f = f.astype(bool)
                fm = m & f
                kept = side[fm] * d.fut.values[fm]
                rej = side[m & ~f] * d.fut.values[m & ~f]
                s = stats(kept); s['rejected'] = stats(rej)
                k = len(kept)
                if 0 < k < len(base):
                    sims = np.array([rng.choice(base, k, replace=False).mean() for _ in range(5000)])
                    s['rand_pctile_of_kept_mean'] = float((sims < kept.mean()).mean() * 100)
                r[fn] = s
            res[f'{mode} {thr_name}'] = r
    out[sym] = res

json.dump(out, open('/Users/matveypro/kronos_lab/agents/bollinger/results.json', 'w'), indent=1)
for s, r in out.items():
    ka = r['kronos_alone_sign']
    print('=====', s, 'windows', r['n_windows'], 'BH mean24h %.3f%% total %.1f%%' % (r['buyhold_mean24h_pct'], r['buyhold_total_pct']),
          '| Kronos-alone sign: net %.3f%% hit %.0f%%' % (ka['net_mean'], ka['hit']))
    for k, v in r.items():
        if not isinstance(v, dict) or 'alone' not in v:
            continue
        print(' ', k)
        for fn, st in v.items():
            if st.get('n', 0) == 0:
                print('   ', fn, 'n=0'); continue
            extra = ''
            if 'rejected' in st and st['rejected'].get('n', 0):
                extra = ' | rej n=%d net=%.3f%%' % (st['rejected']['n'], st['rejected']['net_mean'])
            if 'rand_pctile_of_kept_mean' in st:
                extra += ' | randPct=%.0f' % st['rand_pctile_of_kept_mean']
            print('    %-6s n=%3d gross=%.3f%% net=%.3f%% hit=%.0f%% sumNet=%.1f%%' % (fn, st['n'], st['gross_mean'], st['net_mean'], st['hit'], st['net_sum']) + extra)
