"""Contrarian test: is fading Kronos a robust edge, and is it just window mean-reversion?
No lookahead: features at origin o use csv rows <= o and the Kronos paths for o.
"""
import numpy as np, pandas as pd, os, json
C = '/Users/matveypro/kronos_lab/cache'
FEE = 0.0011  # round trip
HS = [1, 4, 12, 24]
rng = np.random.default_rng(0)


def block_boot_mean(x, B=2000, block=6):
    x = np.asarray(x); n = len(x)
    if n < 10: return (np.nan, np.nan)
    nb = int(np.ceil(n / block)); out = np.empty(B)
    for b in range(B):
        st = rng.integers(0, n - block + 1, nb)
        idx = (st[:, None] + np.arange(block)).ravel()[:n]
        out[b] = x[idx].mean()
    return tuple(np.percentile(out, [2.5, 97.5]))


def block_boot_corr(a, b, B=2000, block=6):
    n = len(a); nb = int(np.ceil(n / block)); out = np.empty(B)
    for k in range(B):
        st = rng.integers(0, n - block + 1, nb)
        idx = (st[:, None] + np.arange(block)).ravel()[:n]
        out[k] = np.corrcoef(a[idx], b[idx])[0, 1]
    return tuple(np.percentile(out, [2.5, 97.5]))


def strat_stats(pos, real):
    g = pos * real
    n = g - FEE * (pos != 0)
    lo, hi = block_boot_mean(n)
    return dict(n=int((pos != 0).sum()), gross_bp=g.mean() * 1e4, net_bp=n.mean() * 1e4,
                net_ci_bp=(lo * 1e4, hi * 1e4), hit=float((g > 0).mean()))


rows = []; report = {}
for sym in ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']:
    f = f'{C}/{sym}_60_paths.npz'
    if not os.path.exists(f):
        print('missing', sym); continue
    df = pd.read_csv(f'{C}/{sym}_60_ohlcv.csv'); cl = df.close.values
    d = np.load(f); O = d['origins']; P = d['paths']
    O_ok = O[O + 24 < len(cl)]; P = P[:len(O_ok)]; O = O_ok
    last = cl[O]
    ctx_mean = np.array([cl[o - 255:o + 1].mean() for o in O])
    ctx_std = np.array([cl[o - 255:o + 1].std() for o in O])
    z = (last - ctx_mean) / ctx_std                     # position of last close in the context window
    past24 = last / cl[O - 24] - 1
    # RSI14 (Wilder) as an independent mean-reversion indicator
    dlt = np.diff(cl, prepend=cl[0]); up = pd.Series(np.clip(dlt, 0, None)).ewm(alpha=1/14, adjust=False).mean()
    dn = pd.Series(np.clip(-dlt, 0, None)).ewm(alpha=1/14, adjust=False).mean(); rsi = (100 - 100 / (1 + up / dn)).values[O]
    # Kronos reconstruction bias proxy: predicted h=1 open vs actual last close (open of next bar ~= last close)
    p_open1 = np.median(P[:, :, 0, 0], axis=1)
    open_gap = p_open1 / last - 1
    sym_rep = {}
    ntr = len(O); split = int(ntr * 2 / 3)
    for h in HS:
        pred = np.median(P[:, :, h - 1, 3], axis=1) / last - 1
        real = cl[O + h] / last - 1
        c_all = np.corrcoef(pred, real)[0, 1]
        ci = block_boot_corr(pred, real)
        # does Kronos pred just encode window z / recent return?
        c_pz = np.corrcoef(pred, z)[0, 1]; c_pp = np.corrcoef(pred, past24)[0, 1]
        # partial: residual of pred after regressing on z and past24
        X = np.c_[np.ones(ntr), z, past24]
        beta = np.linalg.lstsq(X, pred, rcond=None)[0]; res = pred - X @ beta
        c_res = np.corrcoef(res, real)[0, 1]; c_res_ci = block_boot_corr(res, real)
        c_z = np.corrcoef(-z, real)[0, 1]
        r = dict(sym=sym, h=h, n=ntr, corr_pred_real=c_all, corr_ci=ci, corr_pred_z=c_pz, corr_pred_past24=c_pp,
                 corr_resid_real=c_res, corr_resid_ci=c_res_ci, corr_negz_real=c_z,
                 corr_first=np.corrcoef(pred[:split], real[:split])[0, 1],
                 corr_hold=np.corrcoef(pred[split:], real[split:])[0, 1])
        fade = -np.sign(pred); zrev = -np.sign(z); rsirev = np.where(rsi > 50, -1., 1.)
        both = np.where(fade == zrev, fade, 0.)  # fade only when agrees with z mean-reversion
        rnd = np.array([strat_stats(rng.permutation(fade), real)['net_bp'] for _ in range(500)])
        for name, pos in [('fade_kronos', fade), ('follow_kronos', -fade), ('zscore_revert', zrev),
                          ('rsi_revert', rsirev), ('fade_AND_z', both), ('buyhold', np.ones(ntr))]:
            s_all = strat_stats(pos, real); s_ho = strat_stats(pos[split:], real[split:])
            r[name] = s_all; r[name + '_holdout'] = s_ho
        r['random_fade_net_bp_p5_p50_p95'] = list(np.percentile(rnd, [5, 50, 95]))
        rows.append(r)
    sym_rep['corr_opengap_z'] = float(np.corrcoef(open_gap, z)[0, 1])
    sym_rep['mean_opengap_bp'] = float(open_gap.mean() * 1e4)
    sym_rep['corr_opengap_real24'] = float(np.corrcoef(open_gap, cl[O + 24] / last - 1)[0, 1])
    # shrinkage: mean predicted 24h price position vs context mean (in z units)
    pz24 = (np.median(P[:, :, 23, 3], axis=1) - ctx_mean) / ctx_std
    sym_rep['slope_predz24_on_z'] = float(np.polyfit(z, pz24, 1)[0])
    rz24 = (cl[O + 24] - ctx_mean) / ctx_std
    sym_rep['slope_realz24_on_z'] = float(np.polyfit(z, rz24, 1)[0])
    report[sym] = sym_rep

pd.set_option('display.width', 250)
for r in rows:
    print(f"\n== {r['sym']} h={r['h']} n={r['n']}  corr(pred,real)={r['corr_pred_real']:+.3f} CI[{r['corr_ci'][0]:+.2f},{r['corr_ci'][1]:+.2f}]"
          f" first2/3={r['corr_first']:+.3f} hold={r['corr_hold']:+.3f} | corr(pred,z)={r['corr_pred_z']:+.2f} corr(pred,past24)={r['corr_pred_past24']:+.2f}"
          f" | resid corr={r['corr_resid_real']:+.3f} CI[{r['corr_resid_ci'][0]:+.2f},{r['corr_resid_ci'][1]:+.2f}] corr(-z,real)={r['corr_negz_real']:+.3f}")
    for k in ['fade_kronos', 'follow_kronos', 'zscore_revert', 'rsi_revert', 'fade_AND_z', 'buyhold']:
        a, b = r[k], r[k + '_holdout']
        print(f"   {k:14s} n={a['n']:3d} gross={a['gross_bp']:+7.1f}bp net={a['net_bp']:+7.1f}bp CI[{a['net_ci_bp'][0]:+6.1f},{a['net_ci_bp'][1]:+6.1f}] hit={a['hit']:.2f}"
              f" | holdout n={b['n']} net={b['net_bp']:+7.1f}bp CI[{b['net_ci_bp'][0]:+6.1f},{b['net_ci_bp'][1]:+6.1f}]")
    print('   random-shuffled fade net bp p5/p50/p95:', np.round(r['random_fade_net_bp_p5_p50_p95'], 1))
print('\nmechanism:', json.dumps(report, indent=1))
json.dump(dict(rows=rows, mech=report), open('/Users/matveypro/kronos_lab/agents/contrarian/results.json', 'w'), default=float, indent=1)
