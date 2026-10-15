"""Open the sealed TEST once for the frozen per-coin selections (selections.json)."""
import json, math, sys
import pandas as pd
sys.path.insert(0, '/Users/matveypro/kronos_lab/indicators')
import engine as E

OUT = '/Users/matveypro/kronos_lab/top10'
sel = json.load(open(f'{OUT}/selections.json'))
rows = []
for coin, s in sel.items():
    df, fund = E.load(coin, '1d')
    sig = E.base_signals(df)[s['signal']]
    r, wx = E.bar_returns(df, fund, sig, E.COSTS['SPOT'], '1d', s['adx'], s['voltarget'], E.adx(df))
    t = r[r.index >= E.VAL_END]
    eq = (1 + t).cumprod()
    yrs = len(t) / 365
    c = df.close[df.index >= E.VAL_END - pd.Timedelta('1D')]
    bh = (c.iloc[-1] / c.iloc[0]) ** (1 / yrs) - 1
    bh_eq = c / c.iloc[0]
    entries = int(((wx[wx.index >= E.VAL_END] > 0).astype(int).diff() == 1).sum())
    cagr = eq.iloc[-1] ** (1 / yrs) - 1
    rows.append(dict(coin=coin, strategy=f"{s['signal']}" + (f"+ADX>{s['adx']}" if s['adx'] else '') +
                     ('+voltarget' if s['voltarget'] else ''),
                     cagr=cagr * 100, buyhold=bh * 100, edge=(cagr - bh) * 100,
                     sharpe=t.mean() / t.std() * math.sqrt(365) if t.std() > 0 else 0.0,
                     maxdd=(eq / eq.cummax() - 1).min() * 100, bh_maxdd=(bh_eq / bh_eq.cummax() - 1).min() * 100,
                     expo=wx[wx.index >= E.VAL_END].mean() * 100, entries=entries, dsr=s['dsr'],
                     superior=bool(cagr > 0 and cagr > bh), over15=bool(cagr > 0.15 and cagr > bh)))
R = pd.DataFrame(rows).sort_values('cagr', ascending=False)
pd.set_option('display.width', 250)
print(f'TEST {E.VAL_END.date()} -> {df.index[-1].date()}  ({yrs:.2f} años), costes SPOT Bybit')
print(R.round(1).to_string(index=False))
print(f"\nSuperiores (CAGR>0 y > buy&hold): {R.superior.sum()} de {len(R)}")
print(f"De ellas con CAGR > 15 %: {R.over15.sum()}")
print(f"Media CAGR de todas las selecciones: {R.cagr.mean():.1f}%  vs media buy&hold {R.buyhold.mean():.1f}%")
R.to_csv(f'{OUT}/test_results.csv', index=False)
