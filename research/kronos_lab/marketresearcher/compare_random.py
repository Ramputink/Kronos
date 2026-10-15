import json
import numpy as np
import pandas as pd

D = '/Users/matveypro/kronos_lab/marketresearcher'
r = pd.DataFrame([json.loads(l) for l in open(f'{D}/results.jsonl')]).drop_duplicates(
    ['strategy', 'symbol', 'variant'], keep='last').set_index(['strategy', 'symbol', 'variant']).ret_pct
rows = []
for s in ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']:
    for l in open(f'{D}/random_{s}.jsonl'):
        x = json.loads(l)
        rets = np.array(x['rets'])
        c = r[(x['strategy'], s, 'confirm')]
        rows.append({'strategy': x['strategy'], 'sym': s[:3], 'confirm': c,
                     'random_median': np.median(rets), 'pctl': (rets < c).mean() * 100})
t = pd.DataFrame(rows)
port = t.groupby('strategy')[['confirm', 'random_median']].mean()
print(port.round(2).sort_values('confirm', ascending=False).to_string())
print('\nconfirm vs random median, cartera: mean diff %.2f pts; confirm > random in %d/15 strategies'
      % ((port.confirm - port.random_median).mean(), (port.confirm > port.random_median).sum()))
print('per strategy x symbol: confirm percentile vs random, median %.0f; >=95th in %d/45, <=5th in %d/45'
      % (t.pctl.median(), (t.pctl >= 95).sum(), (t.pctl <= 5).sum()))
print('mean over all strategies: confirm %.2f  random %.2f' % (port.confirm.mean(), port.random_median.mean()))
