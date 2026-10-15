"""EUR outcome on the TEST period (2025-07-01 -> 2026-10-13) of simple, previously pre-registered rules."""
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, '/Users/matveypro/kronos_lab/indicators')
import engine as E

START = pd.Timestamp('2025-07-01')
FEES = {'Bybit EU spot 0.25%+5bp': 0.0030, 'Bybit Global spot 0.10%+5bp': 0.0015}


def rsi_regime(coin, side_cost):
    df, fund = E.load(coin, '1d')
    cost = dict(E.COSTS['SPOT'], side=side_cost)
    r, wx = E.bar_returns(df, fund, E.base_signals(df)['rsi55_50'], cost, '1d', 0, True, E.adx(df))
    return r[r.index >= START], wx[wx.index >= START]


def hold(coin):
    df, _ = E.load(coin, '1d')
    return df.close.pct_change()[df.index >= START].fillna(0)


options = {
    'A. RSI(14) régimen + vol-target, solo ETH': ['ETH'],
    'B. RSI régimen, ETH+BNB': ['ETH', 'BNB'],
    'C. RSI régimen, BTC+ETH+SOL+BNB': ['BTC', 'ETH', 'SOL', 'BNB'],
    'D. RSI régimen, 9 large caps': E.UNIV,
}
rows = []
for label, side in FEES.items():
    for name, coins in options.items():
        sl = [rsi_regime(c, side) for c in coins]
        eq = pd.concat([(1 + s[0]).cumprod() for s in sl], axis=1).mean(axis=1)
        trades = sum(int(((s[1] > 0).astype(int).diff() == 1).sum()) for s in sl)
        expo = np.mean([s[1].mean() for s in sl]) * 100
        rows.append(dict(costes=label, opcion=name, total=(eq.iloc[-1] - 1) * 100,
                         maxdd=((eq / eq.cummax()) - 1).min() * 100, entradas=trades, exposicion_media=expo,
                         eur_1000=1000 * eq.iloc[-1], dd_eur_1000=1000 * (eq - eq.cummax()).min(),
                         eur_2000=2000 * eq.iloc[-1]))
for name, coins in {'E. Buy&hold BTC': ['BTC'], 'F. Buy&hold ETH': ['ETH'], 'G. Buy&hold 50/50 BTC-ETH': ['BTC', 'ETH']}.items():
    eq = pd.concat([(1 + hold(c)).cumprod() for c in coins], axis=1).mean(axis=1) * (1 - 0.003)
    rows.append(dict(costes='una compra', opcion=name, total=(eq.iloc[-1] - 1) * 100,
                     maxdd=((eq / eq.cummax()) - 1).min() * 100, entradas=1, exposicion_media=100,
                     eur_1000=1000 * eq.iloc[-1], dd_eur_1000=1000 * (eq - eq.cummax()).min(), eur_2000=2000 * eq.iloc[-1]))
pd.set_option('display.width', 250)
T = pd.DataFrame(rows)
print(T.round(1).to_string(index=False))
T.to_csv('/Users/matveypro/kronos_lab/overnight/out/budget_options.csv', index=False)
