"""TEST evaluation of the ex-ante portfolios (ADDENDUM_PREREG.md) and 1.000 / 2.000 EUR budget simulation."""
import json, math
import numpy as np
import pandas as pd
import morning_test as M

OUT = M.OUT
M.init_full()
DATES = M.DATES
sel = json.load(open(f'{OUT}/selections_exante.json'))


def sleeve(cfg, coin):
    r, wx = M.strat_returns(cfg, coin)
    t, w = r[r.index >= M.TEST_START], wx[wx.index >= M.TEST_START]
    daily = ((1 + t).groupby(t.index.floor('D')).prod() - 1).reindex(DATES).fillna(0.0)
    entries = int(((w.values[1:] != 0) & (np.sign(w.values[1:]) != np.sign(w.values[:-1]))).sum())
    nz = np.abs(w.values[w.values != 0])
    return daily, entries, (nz.min() if len(nz) else np.nan), float(np.abs(w).mean())


def bh(coin):
    d = M.DATA[(coin, '1d')]
    c = pd.Series(d['c'], d['idx'])
    return c.pct_change().reindex(DATES).fillna(0.0)


def evaluate(name, items, budgets=(1000, 2000)):
    sl = [sleeve(x['cfg'], x['coin']) for x in items]
    R = pd.concat([s[0] for s in sl], axis=1)
    eq_each = (1 + R).cumprod()                       # no rebalancing between sleeves
    eq = eq_each.mean(axis=1)
    yrs = len(DATES) / 365
    coins = sorted({x['coin'] for x in items})
    bh_eq = (1 + pd.concat([bh(c) for c in coins], axis=1)).cumprod().mean(axis=1)
    out = dict(rule=name, n_strategies=len(items), coins=len(coins),
               cagr=(eq.iloc[-1] ** (1 / yrs) - 1) * 100, total=(eq.iloc[-1] - 1) * 100,
               maxdd=((eq / eq.cummax()) - 1).min() * 100,
               sharpe=eq.pct_change().mean() / eq.pct_change().std() * math.sqrt(365),
               bh_total=(bh_eq.iloc[-1] - 1) * 100, bh_maxdd=((bh_eq / bh_eq.cummax()) - 1).min() * 100,
               entries=sum(s[1] for s in sl), avg_exposure=np.mean([s[3] for s in sl]) * 100,
               positive_sleeves=int((eq_each.iloc[-1] > 1).sum()))
    for b in budgets:
        per = b / len(items)
        euros = eq * b
        out[f'final_{b}'] = euros.iloc[-1]
        out[f'worst_dd_eur_{b}'] = (euros - euros.cummax()).min()
        out[f'min_order_ok_{b}'] = bool(np.nanmin([per * s[2] for s in sl]) >= 5 * 0.86)  # 5 USDT ≈ 4.3 EUR
    detail = pd.DataFrame([dict(coin=x['coin'], tf=x['cfg']['tf'], mode=x['cfg']['mode'],
                                comps=' & '.join(f"{c['t']}{tuple(c['p'].values())}" for c in x['cfg']['comps']),
                                adx=x['cfg']['adx'], vt=x['cfg']['vt'], stop=x['cfg']['stop'],
                                score_trval=x.get('score'), test_total=(eq_each.iloc[-1, i] - 1) * 100,
                                entries=sl[i][1]) for i, x in enumerate(items)])
    return out, detail, eq


rows, details, curves = [], {}, {}
for name in ['R4_top10_SPOT', 'R2_best_per_coin_SPOT', 'R3_cross_coin_SPOT', 'R4_top10_PERP_ref',
             'R2_best_per_coin_PERP_ref']:
    o, d, e = evaluate(name, sel[name])
    rows.append(o); details[name] = d; curves[name] = e
    print(name, 'done', flush=True)

pd.set_option('display.width', 250)
S = pd.DataFrame(rows).set_index('rule')
print(f'\nTEST {DATES[0].date()} -> {DATES[-1].date()} ({len(DATES)} días)\n')
print(S.round(1).T.to_string())
for name in ['R4_top10_SPOT', 'R2_best_per_coin_SPOT']:
    print(f'\n--- {name} ---')
    print(details[name].round(1).to_string(index=False))
S.to_csv(f'{OUT}/portfolio_test_summary.csv')
pd.DataFrame(curves).to_csv(f'{OUT}/portfolio_test_curves.csv')
for k, d in details.items():
    d.to_csv(f'{OUT}/portfolio_test_{k}.csv', index=False)
