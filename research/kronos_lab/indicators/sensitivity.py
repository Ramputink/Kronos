"""Context only (not selection): RSI-regime finalist under Bybit EU costs, full history, per year, trades."""
import math
import numpy as np
import pandas as pd
import engine as E

COSTS = {'Global spot 0.10%+5bp': 0.0015, 'EU spot 0.25%+5bp': 0.0030}
idx_start = {}
for label, side in COSTS.items():
    cost = dict(E.COSTS['SPOT'], side=side)
    per_asset, trades = {}, {}
    for s in E.UNIV:
        df, fund = E.load(s, '1d')
        sig = E.base_signals(df)['rsi55_50']
        r, wx = E.bar_returns(df, fund, sig, cost, '1d', 0, True, E.adx(df))
        r = r[r.index >= df.index[0] + pd.Timedelta(days=210)]
        per_asset[s] = r
        trades[s] = int(((sig.clip(lower=0) > 0).astype(int).diff() == 1).sum())
    A = pd.DataFrame(per_asset)
    A['PORT'] = A[E.UNIV].mean(axis=1, skipna=True)
    print(f'\n=== {label} ===   entradas totales por activo: {trades}')
    for col in ['PORT', 'ETH', 'SOL', 'BTC']:
        d = A[col].dropna()
        eq = (1 + d).cumprod()
        yrs = len(d) / 365
        yr = ((1 + d).groupby(d.index.year).prod() - 1) * 100
        test = d[d.index >= E.VAL_END]
        teq = (1 + test).prod() ** (365 / len(test)) - 1
        print(f'{col:4} completo {d.index[0].date()}->: CAGR {(eq.iloc[-1] ** (1 / yrs) - 1) * 100:6.1f}%  '
              f'Sharpe {d.mean() / d.std() * math.sqrt(365):4.2f}  maxDD {((eq / eq.cummax()) - 1).min() * 100:6.1f}%  '
              f'| TEST CAGR {teq * 100:6.1f}%  | por año {yr.round(0).to_dict()}')
