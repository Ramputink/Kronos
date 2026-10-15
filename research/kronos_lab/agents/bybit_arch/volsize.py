"""Is Kronos useful live as a VOL forecaster for position sizing?
Compare predictors of realised next-24h vol (std of hourly log returns, rows o+1..o+24):
 - kronos: mean over 6 paths of std of hourly log close returns (path starts from close[o])
 - kronos_range: mean over paths of mean (high-low)/close
 - ewma: EWMA(lambda=0.94) hourly vol at o (no lookahead)
 - hist24: std of last 24 hourly returns
Report Spearman/Pearson corr (log-vol) and combined OLS (fit on first 2/3, test last 1/3).
Then vol-targeted buy&hold sizing using each forecast: net of 0.11% per rebalance turnover.
"""
import numpy as np, pandas as pd
C = "/Users/matveypro/kronos_lab/cache/"
FEE = 0.00055
def rank(x): return pd.Series(x).rank().values
out = []
for sym in ["BTCUSDT", "ETHUSDT", "SOLUSDT"]:
    df = pd.read_csv(C + f"{sym}_60_ohlcv.csv"); z = np.load(C + f"{sym}_60_paths.npz")
    O, P = z["origins"], z["paths"]
    c = df.close.values; lr = np.diff(np.log(c), prepend=np.nan)
    ew = pd.Series(lr).pow(2).ewm(alpha=0.06, adjust=False).mean().pow(.5).values
    h24 = pd.Series(lr).rolling(24).std().values
    rows = []
    for k, o in enumerate(O):
        if o + 24 >= len(c): continue
        pc = P[k][:, :, 3]
        pl = np.diff(np.log(np.concatenate([np.full((pc.shape[0], 1), c[o]), pc], 1)), axis=1)
        kv = pl.std(1).mean()
        kr = ((P[k][:, :, 1] - P[k][:, :, 2]) / P[k][:, :, 3]).mean()
        real = lr[o + 1:o + 25].std()
        ret24 = c[o + 24] / c[o] - 1
        rows.append((o, kv, kr, ew[o], h24[o], real, ret24))
    d = pd.DataFrame(rows, columns="o kron kron_rng ewma hist24 real ret24".split()).dropna()
    n = len(d); cut = int(n * 2 / 3)
    res = dict(sym=sym, n=n)
    for f in ["kron", "kron_rng", "ewma", "hist24"]:
        res["sp_" + f] = np.corrcoef(rank(d[f]), rank(d.real))[0, 1]
    # OLS on logs, train first 2/3, test last 1/3: ewma alone vs ewma+kron
    y = np.log(d.real.values)
    def fit(cols):
        X = np.column_stack([np.ones(n)] + [np.log(d[cc].values) for cc in cols])
        b = np.linalg.lstsq(X[:cut], y[:cut], rcond=None)[0]
        e = y[cut:] - X[cut:] @ b
        return (e ** 2).mean()
    res["oos_mse_ewma"] = fit(["ewma"]); res["oos_mse_ewma+kron"] = fit(["ewma", "kron"]); res["oos_mse_kron"] = fit(["kron"])
    # vol-targeted long over each 12h-spaced window (use only non-overlapping: every other origin)
    dd = d.iloc[::2]
    tgt = dd.real.median()
    for f in ["kron", "ewma", "const"]:
        w = np.ones(len(dd)) if f == "const" else np.clip(tgt / dd[f].values * (dd.ewma.median() / tgt if f == "ewma" else dd.kron.median() / tgt), 0, 3)
        w = w / w.mean()
        g = w * dd.ret24.values; net = g - 2 * FEE * w
        sh = net.mean() / net.std() * np.sqrt(365)
        res["sharpe_" + f] = sh; res["ret_" + f] = net.sum()
    out.append(res)
r = pd.DataFrame(out).set_index("sym").T
pd.set_option("display.float_format", lambda v: "%.4f" % v)
print(r)
r.to_csv("/Users/matveypro/kronos_lab/agents/bybit_arch/volsize.csv")
