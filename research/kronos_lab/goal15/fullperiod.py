"""Full-history and per-year view (context only; the pre-registered verdict is in evaluate.py)."""
import math, sys
import pandas as pd
sys.argv = ['x']
import evaluate as E  # reuses identical rules/costs

idx = E.close.index[E.close.index >= pd.Timestamp('2021-01-01')]  # once most alts + EMA200 warm-up exist
out = {}
for name in ['C1_breakout', 'C4_breakout_btc_gate', 'C6_btc_eth_trend']:
    p, _ = E.run(E.SPOT_W[name](E.COSTS['SPOT']), E.COSTS['SPOT'], idx)
    out[name] = p
parts = [E.run(f(E.close, E.fund), E.COSTS['SPOT'], idx)[0] for f in (E.s1_trend, E.s5_breakout, E.s4_capitulation)]
out['C3_ensemble3'] = sum(parts) / 3
r = E.close.reindex(idx).pct_change(fill_method=None)
out['BH_BTC'] = r['BTC'].fillna(0.0)
out['BH_basket_EW'] = r.mean(axis=1).fillna(0.0)
D = pd.DataFrame(out).iloc[1:]
yr = (1 + D).groupby(D.index.year).prod() - 1
print('Retorno por año natural (SPOT costs), %:')
print((yr * 100).round(1).to_string())
eq = (1 + D).cumprod()
yrs = len(D) / 365
print(f'\nPeriodo completo {D.index[0].date()} -> {D.index[-1].date()} ({yrs:.1f} años):')
for c in D:
    s = D[c]
    print(f'{c:22} CAGR {(eq[c].iloc[-1] ** (1 / yrs) - 1) * 100:6.1f}%  Sharpe {s.mean() / s.std() * math.sqrt(365):5.2f}  '
          f'maxDD {((eq[c] / eq[c].cummax()) - 1).min() * 100:6.1f}%')
