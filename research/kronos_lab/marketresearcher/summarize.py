import json
import pandas as pd

D = '/Users/matveypro/kronos_lab/marketresearcher'
r = pd.DataFrame([json.loads(l) for l in open(f'{D}/results.jsonl')])
r = r.drop_duplicates(['strategy', 'symbol', 'variant'], keep='last')
r['sym'] = r.symbol.str[:3]

ret = r.pivot_table(index='strategy', columns=['variant', 'sym'], values='ret_pct')
port = r.groupby(['strategy', 'variant']).ret_pct.mean().unstack()  # 1/3 capital per symbol
trades = r.groupby(['strategy', 'variant']).trades.sum().unstack()
dd = r.groupby(['strategy', 'variant']).max_dd_pct.min().unstack()
out = pd.concat({'ret_cartera_%': port, 'trades': trades, 'peor_dd_%': dd}, axis=1)
out = out.sort_values(('ret_cartera_%', 'base'), ascending=False)
pd.set_option('display.width', 250)
print(out.round(2).to_string())
print('\nPor par (base / confirm / veto), % sobre capital:')
for v in ['base', 'confirm', 'veto']:
    print(v, ret[v].round(2).to_string().replace('\n', '\n  '))

print('\nResumen: estrategias positivas por variante (cartera 3 pares)')
print((port > 0).sum().to_string())
print('media de todas las estrategias:', port.mean().round(2).to_dict())
print('mejor estrategia por variante:', port.idxmax().to_dict(), port.max().round(2).to_dict())
print('confirm mejor que base en', int((port.confirm > port.base).sum()), 'de', len(port))
print('veto mejor que base en', int((port.veto > port.base).sum()), 'de', len(port))
print('comisiones totales base (USD sobre 10k x3):', round(r[r.variant == 'base'].fees.sum(), 1),
      ' pnl neto base:', round(r[r.variant == 'base'].pnl_net.sum(), 1))
print('buy&hold:', json.load(open(f'{D}/buy_hold.json')))
