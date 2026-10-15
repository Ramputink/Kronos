"""Funding / OI extremes vs Kronos filter. No lookahead: decision at close of origin candle o.
Candle timestamps are open times -> decision time T = ts[o] + 1h. Funding with settle ts <= T, OI snapshot ts <= ts[o].
Thresholds are trailing (past-only) percentiles."""
import numpy as np, pandas as pd, os, json
D = os.path.dirname(os.path.abspath(__file__)); C = "/Users/matveypro/kronos_lab/cache"
FEE = 0.0011; H = 24
rng = np.random.default_rng(0)
out = {}
def stats(r):
    r = np.asarray(r, float); n = len(r)
    if n == 0: return dict(n=0)
    return dict(n=n, gross=float(r.mean()*100), net=float((r-FEE).mean()*100), hit=float((r > 0).mean()),
                t_net=float((r-FEE).mean()/(r.std(ddof=1)+1e-12)*np.sqrt(n/2)) if n > 2 else np.nan)  # /2 overlap adj
def tpct(series_vals, i, x, look):  # trailing percentile of x vs values[i-look:i]
    w = series_vals[max(0, i-look):i]
    return np.nan if len(w) < 20 else (w < x).mean()

allrows = []
for sym in ["BTCUSDT", "ETHUSDT", "SOLUSDT"]:
    df = pd.read_csv(f"{C}/{sym}_60_ohlcv.csv"); df["ts"] = pd.to_datetime(df.timestamps)
    z = np.load(f"{C}/{sym}_60_paths.npz"); org = z["origins"]; P = z["paths"]
    fu = pd.read_csv(f"{D}/{sym}_funding.csv", parse_dates=["ts"]); oi = pd.read_csv(f"{D}/{sym}_oi.csv", parse_dates=["ts"])
    fts, fv = fu.ts.values, fu.funding.values; ots, ov = oi.ts.values, oi.oi.values
    cl = df.close.values
    for k, o in enumerate(org):
        if o + H >= len(df): continue
        T = df.ts.iloc[o] + pd.Timedelta(hours=1)
        fi = np.searchsorted(fts, np.datetime64(T), side="right") - 1
        oi_i = np.searchsorted(ots, np.datetime64(df.ts.iloc[o]), side="right") - 1
        if fi < 100 or oi_i < 24 * 31: continue
        f_now = fv[fi-2:fi+1].mean()  # last 24h avg funding
        f_hist = pd.Series(fv[:fi+1]).rolling(3).mean().values
        fpct = tpct(f_hist, fi, f_now, 180)  # ~60 days trailing
        oich = ov[oi_i]/ov[oi_i-24]-1
        oich_hist = ov[24:oi_i+1]/ov[:oi_i+1-24]-1
        oipct = tpct(oich_hist, len(oich_hist)-1, oich, 24*30)
        pr24 = cl[o]/cl[o-24]-1
        kret = P[k, :, -1, 3]/cl[o]-1
        allrows.append(dict(sym=sym, o=o, ts=df.ts.iloc[o], fpct=fpct, f_now=f_now, oipct=oipct, oich=oich, pr24=pr24,
                            kmean=kret.mean(), kagree=(np.sign(kret) == np.sign(kret.mean())).mean(),
                            fwd=cl[o+H]/cl[o]-1))
R = pd.DataFrame(allrows).dropna(); R.to_csv(f"{D}/features.csv", index=False)
print(R.groupby("sym").size())

def run(R, name, sig):
    """sig: Series of -1/0/1. Returns per-trade returns."""
    m = sig != 0; return (sig[m]*R.fwd[m]).values, m

res = []
for sym, G in list(R.groupby("sym")) + [("ALL", R)]:
    G = G.reset_index(drop=True)
    ks = np.sign(G.kmean)
    sigs = {
        "funding_contra": np.where(G.fpct > .8, -1, np.where(G.fpct < .2, 1, 0)),
        "oi_fade": np.where(G.oipct > .8, -np.sign(G.pr24), 0),
        "combo_fund_or_oi": np.where(G.fpct > .8, -1, np.where(G.fpct < .2, 1, np.where(G.oipct > .8, -np.sign(G.pr24), 0))),
    }
    bh = stats(G.fwd.values); res.append(dict(sym=sym, strat="buy_hold(24h long every origin)", **bh))
    res.append(dict(sym=sym, strat="kronos_sign_alone", **stats((ks*G.fwd).values)))
    for nm, s in sigs.items():
        s = pd.Series(s, dtype=float)
        r, m = run(G, nm, s); base = stats(r); res.append(dict(sym=sym, strat=nm, **base))
        agree = m & (np.sign(s) == ks); rk = (s[agree]*G.fwd[agree]).values
        st = stats(rk)
        # random-filter baseline: random subset of same size from the indicator trades
        idx = np.where(m)[0]; nk = int(agree.sum()); rnd = []
        for _ in range(2000):
            if nk == 0 or nk > len(idx): break
            pick = rng.choice(idx, nk, replace=False); rnd.append((s[pick]*G.fwd.iloc[pick]).mean())
        rnd = np.array(rnd)
        st["rand_filter_gross_mean"] = float(rnd.mean()*100) if len(rnd) else np.nan
        st["p_rand>=kronos"] = float((rnd >= rk.mean()).mean()) if len(rnd) else np.nan
        res.append(dict(sym=sym, strat=nm+" + kronos_agree", **st))
        dis = m & (np.sign(s) != ks); res.append(dict(sym=sym, strat=nm+" + kronos_DISagree", **stats((s[dis]*G.fwd[dis]).values)))
        # random shuffled signal baseline
        sh = [ (rng.permutation(s.values)*G.fwd.values)[rng.permutation(s.values)!=0].mean() for _ in range(0) ]
    # correlations
    for c in ["fpct", "oipct", "kmean"]:
        res.append(dict(sym=sym, strat=f"spearman({c},fwd)", n=len(G), gross=float(G[c].rank().corr(G.fwd.rank()))))
    ext = G[(G.fpct > .8) | (G.fpct < .2)]
    res.append(dict(sym=sym, strat="spearman(kmean,fwd | funding extreme)", n=len(ext), gross=float(ext.kmean.rank().corr(ext.fwd.rank()))))
T_ = pd.DataFrame(res); pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
print(T_.round(3).to_string()); T_.to_csv(f"{D}/results.csv", index=False)
