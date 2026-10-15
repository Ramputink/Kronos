"""Run the full pre-registered grid; TRAIN/VAL metrics only. TEST stays sealed (not computed here)."""
import itertools, math, pickle
import numpy as np
import pandas as pd
import engine as E

OUT = '/Users/matveypro/kronos_lab/indicators'
WARM = pd.Timedelta(days=210)  # indicator warm-up after each asset's listing


def daily(r):
    return (1 + r).groupby(r.index.floor('D')).prod() - 1


def metrics(d):
    d = d.dropna()
    if len(d) < 30 or d.std() == 0:
        return dict(cagr=np.nan, sharpe=np.nan, maxdd=np.nan)
    eq = (1 + d).cumprod()
    yrs = len(d) / 365
    return dict(cagr=(eq.iloc[-1] ** (1 / yrs) - 1) * 100 if eq.iloc[-1] > 0 else -100.0,
                sharpe=d.mean() / d.std() * math.sqrt(365),
                maxdd=(eq / eq.cummax() - 1).min() * 100)


rows, store = [], {}
for tf in E.TF:
    data = {}
    for s in E.UNIV:
        df, fund = E.load(s, tf)
        data[s] = (df, fund, E.base_signals(df), E.adx(df))
        print('signals', s, tf, flush=True)
    names = list(next(iter(data.values()))[2])
    for cname, cost in E.COSTS.items():
        for name, adx_thr, vt in itertools.product(names, (0, 20, 25), (False, True)):
            per_asset = {}
            for s, (df, fund, sigs, adx_s) in data.items():
                r, _ = E.bar_returns(df, fund, sigs[name], cost, tf, adx_thr, vt, adx_s)
                r = r[r.index >= df.index[0] + WARM]
                per_asset[s] = daily(r)
            A = pd.DataFrame(per_asset)                       # daily returns, NaN before listing+warmup
            port = A.mean(axis=1, skipna=True)                # equal weight over available assets
            key = (tf, cname, name, adx_thr, vt)
            store[key] = (port, A)
            row = dict(tf=tf, costs=cname, signal=name, adx=adx_thr, voltarget=vt)
            for per, (a, b) in {'train': (None, E.TRAIN_END), 'val': (E.TRAIN_END, E.VAL_END)}.items():
                m = (port.index < b) if a is None else ((port.index >= a) & (port.index < b))
                pm = metrics(port[m])
                row.update({f'{per}_{k}': v for k, v in pm.items()})
                sh = [metrics(A.loc[m, s])['sharpe'] for s in A]
                row[f'{per}_median_asset_sharpe'] = float(np.nanmedian(sh))
                for s in ('ETH', 'SOL'):
                    row[f'{per}_{s}_cagr'] = metrics(A.loc[m, s])['cagr']
            rows.append(row)
        print(tf, cname, 'done', len(rows), flush=True)

res = pd.DataFrame(rows)
res.to_csv(f'{OUT}/search_train_val.csv', index=False)
pickle.dump(store, open(f'{OUT}/store.pkl', 'wb'))
print('total configurations:', len(res))
