"""Walk-forward meta-labeling / ablation: do Kronos path features add OOS value over indicators?"""
import numpy as np, pandas as pd, json, os; os.chdir(os.path.dirname(os.path.abspath(__file__)))
C = '/Users/matveypro/kronos_lab/cache/'
SYMS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
FEE = 0.0011
H = 24
rng = np.random.default_rng(0)

def rsi(c, n=14):
    d = c.diff(); up = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    return 100 - 100/(1+up/dn)

def build(sym):
    df = pd.read_csv(C+f'{sym}_60_ohlcv.csv')
    z = np.load(C+f'{sym}_60_paths.npz'); O = z['origins']; P = z['paths']
    c, h, l, v = df.close, df.high, df.low, df.volume
    ind = pd.DataFrame({
        'rsi': rsi(c)/100 - 0.5,
        'bbb': (c - c.rolling(20).mean())/(2*c.rolling(20).std()),   # %b centred
        'atr': (pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)).rolling(14).mean()/c,
        'r24': np.log(c/c.shift(24)), 'r72': np.log(c/c.shift(72)),
        'rv24': np.log(c).diff().rolling(24).std(),
    })
    rows = []
    for i, o in enumerate(O):
        c0 = c.iloc[o]
        pc = P[i, :, :, 3]; ph = P[i, :, :, 1]; pl = P[i, :, :, 2]; pv = P[i, :, :, 4]
        fr = np.log(pc[:, -1]/c0)
        k = dict(
            k_med=np.median(fr), k_mean=fr.mean(), k_disp=fr.std(), k_cons=(fr > 0).mean()-0.5,
            k_range=np.log(ph.max(1)/pl.min(1)).mean(),
            k_vol=np.log(pv.mean()/v.iloc[o-23:o+1].mean()),
            k_pathvol=np.log(pc).std(axis=1).mean() if False else np.diff(np.log(np.c_[np.full(6, c0), pc]), axis=1).std(axis=1).mean(),
        )
        fut = df.iloc[o+1:o+1+H]
        y = np.log(fut.close.iloc[-1]/c0)
        realrange = np.log(fut.high.max()/fut.low.min())
        rows.append(dict(sym=sym, o=o, t=i, **ind.iloc[o].to_dict(), **k, y=y, absy=abs(y), rr=realrange))
    return pd.DataFrame(rows)

IND = ['rsi', 'bbb', 'atr', 'r24', 'r72', 'rv24']
KRO = ['k_med', 'k_mean', 'k_disp', 'k_cons', 'k_range', 'k_vol', 'k_pathvol']

def logreg(X, y, lam=1.0, it=300):
    w = np.zeros(X.shape[1]); b = 0.0
    for _ in range(it):  # newton-ish via gradient descent w/ fixed lr
        p = 1/(1+np.exp(-(X@w+b)))
        gw = X.T@(p-y)/len(y) + lam*w/len(y); gb = (p-y).mean()
        w -= 0.5*gw; b -= 0.5*gb
    return w, b

def ridge(X, y, lam=10.0):
    Xa = np.c_[X, np.ones(len(X))]; I = np.eye(Xa.shape[1]); I[-1, -1] = 0
    return np.linalg.solve(Xa.T@Xa + lam*I, Xa.T@y)

def walk(d, feats, target='dir', min_train=100):
    """Expanding window. At step t (origin index), train only on samples whose label is
    already realised: origin index <= t-2 (12h spacing, 24h horizon). Pooled across symbols by time index."""
    preds = np.full(len(d), np.nan)
    T = d.t.values
    for t in range(min_train, T.max()+1):
        tr = d[T <= t-2]; te = (T == t)
        if te.sum() == 0: continue
        X = tr[feats].values; mu, sd = X.mean(0), X.std(0)+1e-12
        Xs = (X-mu)/sd; Xt = (d.loc[te, feats].values-mu)/sd
        if target == 'dir':
            w, b = logreg(Xs, (tr.y.values > 0).astype(float))
            preds[te] = 1/(1+np.exp(-(Xt@w+b)))
        else:
            beta = ridge(Xs, tr[target].values)
            preds[te] = np.c_[Xt, np.ones(te.sum())]@beta
    return preds

def spearman(a, b):
    return pd.Series(a).rank().corr(pd.Series(b).rank())

def trade_stats(p, y, thr=0.0):
    pos = np.where(p > 0.5+thr, 1, np.where(p < 0.5-thr, -1, 0))
    g = pos*(np.exp(y)-1); n = (pos != 0)
    net = g - FEE*n
    return dict(trades=int(n.sum()), gross=g.sum(), net=net.sum(), mean_net_per_trade=net[n].mean() if n.any() else np.nan)

if __name__ == '__main__':
    d = pd.concat([build(s) for s in SYMS], ignore_index=True).dropna()
    d.to_csv('features.csv', index=False)
    # univariate in-sample ICs (descriptive only)
    uni = {f: dict(ic_ret=spearman(d[f], d.y), ic_absret=spearman(d[f], d.absy), ic_range=spearman(d[f], d.rr)) for f in IND+KRO}
    out = {'univariate_spearman_all': uni, 'n_total': len(d)}
    sets = {'ind': IND, 'kro': KRO, 'ind+kro': IND+KRO}
    res = {}
    for name, F in sets.items():
        p = walk(d, F, 'dir'); m = ~np.isnan(p)
        dd = d[m]; pp = p[m]
        r = dict(n=int(m.sum()), acc=float(((pp > 0.5) == (dd.y > 0)).mean()), ic=float(spearman(pp, dd.y)),
                 trade=trade_stats(pp, dd.y.values), trade_thr05=trade_stats(pp, dd.y.values, 0.05))
        r['per_sym'] = {s: dict(n=int((dd.sym == s).sum()), acc=float(((pp[dd.sym == s] > 0.5) == (dd.y[dd.sym == s] > 0)).mean()),
                                ic=float(spearman(pp[(dd.sym == s).values], dd.y[dd.sym == s].values)),
                                **trade_stats(pp[(dd.sym == s).values], dd.y[dd.sym == s].values)) for s in SYMS}
        # volatility target: predict realised 24h range
        pv = walk(d, F, 'rr'); mv = ~np.isnan(pv)
        r['range_ic'] = float(spearman(pv[mv], d.rr[mv])); r['range_corr'] = float(np.corrcoef(pv[mv], d.rr[mv])[0, 1])
        pa = walk(d, F, 'absy'); r['absret_ic'] = float(spearman(pa[mv], d.absy[mv]))
        res[name] = r
    oos = d[~np.isnan(walk(d, IND, 'dir'))]
    yo = oos.y.values
    res['baselines'] = dict(
        buy_hold_sum_of_24h_trades_net=float(((np.exp(yo)-1)-FEE).sum()),
        buy_hold_gross=float((np.exp(yo)-1).sum()),
        always_long_acc=float((yo > 0).mean()),
        random_sign_net_mean_over_200=float(np.mean([trade_stats(rng.random(len(yo)), yo)['net'] for _ in range(200)])),
        random_sign_net_p95=float(np.percentile([trade_stats(rng.random(len(yo)), yo)['net'] for _ in range(500)], 95)),
        random_acc_sd=float(np.sqrt(0.25/len(yo))),
    )
    out['walkforward'] = res
    json.dump(out, open('results.json', 'w'), indent=1, default=float)
    print(json.dumps(out, indent=1, default=lambda x: round(float(x), 4)))
