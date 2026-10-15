"""Open the sealed TEST once for the pre-registered finalists; DSR on TRAIN+VAL with N = all configs."""
import math, pickle, sys
import numpy as np
import pandas as pd
sys.path.insert(0, '/Users/matveypro/kronos_lab/goal15')
from scipy_free_stats import norm_cdf, norm_ppf
import engine as E

OUT = '/Users/matveypro/kronos_lab/indicators'
store = pickle.load(open(f'{OUT}/store.pkl', 'rb'))
FINAL = [('1d', 'SPOT', 'donch100_50', 25, True),
         ('1d', 'SPOT', 'ema10_50', 25, True),
         ('1d', 'SPOT', 'rsi55_50', 0, True)]


def m(d):
    d = d.dropna()
    eq = (1 + d).cumprod()
    yrs = len(d) / 365
    return dict(cagr=(eq.iloc[-1] ** (1 / yrs) - 1) * 100, total=(eq.iloc[-1] - 1) * 100,
                sharpe=d.mean() / d.std() * math.sqrt(365), maxdd=(eq / eq.cummax() - 1).min() * 100,
                days=len(d))


tv = lambda s: s[s.index < E.VAL_END].dropna()
srs = np.array([tv(p).mean() / tv(p).std() for p, _ in store.values() if tv(p).std() > 0])
N, V, em = len(store), srs.var(ddof=1), 0.5772156649
sr0 = math.sqrt(V) * ((1 - em) * norm_ppf(1 - 1 / N) + em * norm_ppf(1 - 1 / (N * math.e)))
print(f'N trials = {N}, daily SR std across trials = {math.sqrt(V):.4f}, expected max SR0 = {sr0:.4f} '
      f'(annualised {sr0 * math.sqrt(365):.2f})')

test_lo = E.VAL_END
for key in FINAL:
    port, A = store[key]
    p = tv(port)
    sr = p.mean() / p.std()
    z = (sr - sr0) * math.sqrt(len(p) - 1) / math.sqrt(1 - p.skew() * sr + (p.kurt() + 2) / 4 * sr ** 2)
    dsr = norm_cdf(z)
    t = port[port.index >= test_lo]
    mt = m(t)
    eth, sol = m(A.loc[A.index >= test_lo, 'ETH']), m(A.loc[A.index >= test_lo, 'SOL'])
    ok = mt['cagr'] > 15 and (eth['cagr'] > 15 or sol['cagr'] > 15) and dsr > 0.5
    print(f'\n{key}\n  DSR(train+val) = {dsr:.3f}  (annual SR train+val {sr * math.sqrt(365):.2f})')
    print(f"  TEST {t.index[0].date()}->{t.index[-1].date()}: cartera CAGR {mt['cagr']:.1f}% "
          f"total {mt['total']:.1f}% Sharpe {mt['sharpe']:.2f} maxDD {mt['maxdd']:.1f}%")
    print(f"        ETH CAGR {eth['cagr']:.1f}% (DD {eth['maxdd']:.1f}%)  SOL CAGR {sol['cagr']:.1f}% (DD {sol['maxdd']:.1f}%)")
    per = {s: round(m(A.loc[A.index >= test_lo, s])['cagr'], 1) for s in A}
    print('        por activo CAGR:', per)
    print('  CRITERIO:', 'CUMPLE' if ok else 'no cumple')

bh = {}
for s in E.UNIV:
    c = pd.read_csv(f'{E.D}/{s}_1d.csv', index_col=0, parse_dates=True).close
    c = c[c.index >= test_lo - pd.Timedelta('1D')]
    bh[s] = round((c.iloc[-1] / c.iloc[0]) ** (365 / (len(c) - 1)) * 100 - 100, 1)
print('\nBuy&hold CAGR en TEST:', bh)
