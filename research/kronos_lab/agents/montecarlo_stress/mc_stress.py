"""Kronos as Monte-Carlo scenario generator: realism + strategy stress test.
No lookahead: everything at origin o uses csv rows <= o and Kronos paths for o.
"""
import numpy as np, pandas as pd, json, sys
C = '/Users/matveypro/kronos_lab/cache/'
FEE = 0.00055
H = 24
rng = np.random.default_rng(0)
NB = 6  # baseline paths per origin (same count as Kronos for fairness)
NBIG = 200  # bigger baseline MC

def kurt(x):
    x = x - x.mean(); return (x**4).mean() / (x**2).mean()**2 - 3

def acf_abs(R):  # R (n_paths, H) log returns -> mean lag1 corr of |r| within path, pooled demeaned per path
    A = np.abs(R); A = A - A.mean(1, keepdims=True)
    num = (A[:, 1:] * A[:, :-1]).sum(); den = (A**2).sum()
    return num / den

def acf_r(R):
    A = R - R.mean(1, keepdims=True)
    return (A[:, 1:] * A[:, :-1]).sum() / (A**2).sum()

def boll_strategy(closes, start):
    """closes: 1d array, start: index of origin close. Decisions at closes[start..start+H-1].
    returns gross, net, n_changes over window [start, start+H]."""
    pos = 0; gross = 0.0; changes = 0.0
    for t in range(start, start + H):
        w = closes[t-19:t+1]; m = w.mean(); s = w.std()
        c = closes[t]; new = pos
        if pos == 0:
            if c < m - 2*s: new = 1
            elif c > m + 2*s: new = -1
        elif pos == 1 and c >= m: new = 0
        elif pos == -1 and c <= m: new = 0
        changes += abs(new - pos); pos = new
        gross += pos * (closes[t+1] / c - 1)
    changes += abs(pos)  # forced close at end
    return gross, gross - FEE * changes, changes

def gen_baselines(ctx_lr, n, kind):
    if kind == 'gbm':
        return rng.normal(0, ctx_lr.std(), (n, H))
    if kind == 'boot':
        return rng.choice(ctx_lr, (n, H))
    if kind == 'block':  # stationary-ish block bootstrap, block 6
        out = np.empty((n, H))
        for i in range(n):
            seq = []
            while len(seq) < H:
                s = rng.integers(0, len(ctx_lr) - 6); seq.extend(ctx_lr[s:s+6])
            out[i] = seq[:H]
        return out

res_all = {}
for sym in ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']:
    df = pd.read_csv(C + f'{sym}_60_ohlcv.csv'); cl = df.close.values
    d = np.load(C + f'{sym}_60_paths.npz'); O = d['origins']; P = d['paths']
    O_ok = [i for i, o in enumerate(O) if o + H < len(cl)]
    rows = []
    pooled = {k: [] for k in ['real', 'kronos', 'gbm', 'boot', 'block']}
    for i in O_ok:
        o = O[i]; c0 = cl[o]
        ctx = cl[o-255:o+1]; ctx_lr = np.diff(np.log(ctx))
        real = cl[o:o+H+1]; real_lr = np.diff(np.log(real))
        kc = np.concatenate([np.full((6, 1), c0), P[i, :, :, 3]], 1); k_lr = np.diff(np.log(kc), axis=1)
        paths_lr = {'kronos': k_lr}
        for kind in ['gbm', 'boot', 'block']:
            paths_lr[kind] = gen_baselines(ctx_lr, NB, kind)
        pooled['real'].append(real_lr[None])
        for k in paths_lr: pooled[k].append(paths_lr[k])
        r = dict(i=i, o=o, real_ret=real[-1]/c0-1, real_rv=np.sqrt((real_lr**2).sum()),
                 ctx_rv24=np.sqrt((ctx_lr[-24:]**2).sum()), ctx_rv256=ctx_lr.std()*np.sqrt(24))
        # Strategy realised
        g, n_, ch = boll_strategy(cl[:o+H+1], o); r.update(strat_real_g=g, strat_real_n=n_, strat_ch=ch)
        hist = cl[o-40:o+1]
        for k, L in paths_lr.items():
            rets = np.exp(L.sum(1)) - 1
            r[f'{k}_rv'] = np.sqrt((L**2).sum(1)).mean()
            r[f'{k}_rank'] = int((rets < r['real_ret']).sum())  # 0..6
            r[f'{k}_lo'] = rets.min(); r[f'{k}_hi'] = rets.max()
            sp = []
            for j in range(L.shape[0]):
                pc = np.concatenate([hist, c0 * np.exp(np.cumsum(L[j]))])
                sp.append(boll_strategy(pc, len(hist)-1)[1])
            sp = np.array(sp); r[f'{k}_sp_mean'] = sp.mean(); r[f'{k}_sp_std'] = sp.std(); r[f'{k}_sp_min'] = sp.min()
            r[f'{k}_sp_ptrade'] = None
        # big GBM / block MC for strategy risk (VaR of strategy over window)
        for kind in ['block']:
            L = gen_baselines(ctx_lr, NBIG, kind)
            sp = np.array([boll_strategy(np.concatenate([hist, c0*np.exp(np.cumsum(L[j]))]), len(hist)-1)[1] for j in range(NBIG)])
            r['blockbig_sp_mean'] = sp.mean(); r['blockbig_sp_std'] = sp.std()
        rows.append(r)
    R = pd.DataFrame(rows)
    out = {'n_windows': len(R)}
    # ---- stylised facts
    sf = {}
    for k, v in pooled.items():
        A = np.concatenate(v)
        sf[k] = dict(std_h=float(A.std()), kurt=float(kurt(A.ravel())), acf_abs1=float(acf_abs(A)), acf_r1=float(acf_r(A)))
    out['stylised'] = sf
    # ---- risk / calibration
    cal = {}
    for k in ['kronos', 'gbm', 'boot', 'block']:
        cov = ((R.real_ret >= R[f'{k}_lo']) & (R.real_ret <= R[f'{k}_hi'])).mean()
        ranks = np.bincount(R[f'{k}_rank'], minlength=7) / len(R)
        cal[k] = dict(cover_minmax=float(cov), expected=5/7, rank_hist=[round(x, 3) for x in ranks],
                      corr_logRV=float(np.corrcoef(np.log(R[f'{k}_rv']), np.log(R.real_rv))[0, 1]),
                      ratio_predRV_realRV=float((R[f'{k}_rv'] / R.real_rv).median()))
    for k in ['ctx_rv24', 'ctx_rv256']:
        cal[k] = dict(corr_logRV=float(np.corrcoef(np.log(R[k]), np.log(R.real_rv))[0, 1]),
                      ratio_predRV_realRV=float((R[k] / R.real_rv).median()))
    out['calibration'] = cal
    # ---- strategy stress test
    st = {}
    tr = R[R.strat_ch > 0]  # windows where indicator trades at all
    st['n_windows_trading_real'] = int(len(tr))
    st['real_mean_net_per_window'] = float(R.strat_real_n.mean()); st['real_mean_gross_per_window'] = float(R.strat_real_g.mean())
    for k in ['kronos', 'gbm', 'boot', 'block', 'blockbig']:
        st[f'{k}_pred_mean'] = float(R[f'{k}_sp_mean'].mean())
        st[f'{k}_corr_pred_mean_vs_real'] = float(np.corrcoef(R[f'{k}_sp_mean'], R.strat_real_n)[0, 1])
        st[f'{k}_corr_pred_std_vs_real_abs'] = float(np.corrcoef(R[f'{k}_sp_std'], R.strat_real_n.abs())[0, 1])
        st[f'{k}_pred_std_median'] = float(R[f'{k}_sp_std'].median())
    st['real_abs_pnl_median'] = float(R.strat_real_n.abs().median())
    # Filter: trade window only if Kronos-simulated strategy mean > 0. Non-overlapping: even-indexed origins (24h apart).
    NO = R.iloc[::2]
    def summ(x):
        return dict(n=int((x != 0).sum()), total=float(x.sum()), mean=float(x.mean()), sharpe_win=float(x.mean()/x.std()) if x.std() > 0 else 0)
    st['nonoverlap_always'] = summ(NO.strat_real_n)
    st['nonoverlap_always_gross'] = summ(NO.strat_real_g)
    for k in ['kronos', 'block', 'blockbig']:
        st[f'nonoverlap_filter_{k}'] = summ(NO.strat_real_n * (NO[f'{k}_sp_mean'] > 0))
    # random filter same keep-fraction as kronos
    frac = (NO.kronos_sp_mean > 0).mean(); rs = []
    for _ in range(2000):
        m = rng.random(len(NO)) < frac; rs.append((NO.strat_real_n * m).sum())
    rs = np.array(rs); kt = st['nonoverlap_filter_kronos']['total']
    st['random_filter_total_mean'] = float(rs.mean()); st['kronos_filter_pctile_vs_random'] = float((rs < kt).mean())
    # vol-target sizing using predicted RV (kronos vs ctx_rv24) -> scaled PnL sharpe
    for k, col in [('kronos', 'kronos_rv'), ('ctx24', 'ctx_rv24'), ('block', 'block_rv')]:
        w = (NO[col].median() / NO[col]).clip(0, 3); x = NO.strat_real_n * w
        st[f'voltarget_{k}'] = summ(x)
    # buy & hold over same non-overlapping windows
    st['bh_nonoverlap'] = summ(NO.real_ret - 2*FEE)
    st['bh_full_period'] = float(cl[O[O_ok[-1]] + H] / cl[O[0]] - 1)
    out['strategy'] = st
    res_all[sym] = out
    R.to_csv(f'/Users/matveypro/kronos_lab/agents/montecarlo_stress/{sym}_windows.csv', index=False)
    print(sym, json.dumps(out, indent=1, default=float)); sys.stdout.flush()
json.dump(res_all, open('/Users/matveypro/kronos_lab/agents/montecarlo_stress/results.json', 'w'), indent=1, default=float)
