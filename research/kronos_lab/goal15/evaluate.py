"""Evaluate the 6 pre-registered candidates (PREREGISTRATION.md) on the Bybit panel."""
import json, math, sys
import numpy as np
import pandas as pd
from scipy_free_stats import norm_cdf, norm_ppf  # noqa: F401  (local helper, numpy-only)

sys.path.insert(0, '/Users/matveypro/MarketReasercher')
from phase3_five_strategies import s1_trend, s4_capitulation, s5_breakout, _ema  # unchanged originals

G = '/Users/matveypro/kronos_lab/goal15'
close = pd.read_csv(f'{G}/close.csv', index_col=0, parse_dates=True)
fund = pd.read_csv(f'{G}/funding.csv', index_col=0, parse_dates=True).reindex(close.index).fillna(0.0)
fund = fund.reindex(columns=close.columns).fillna(0.0)
CUT_A, CUT_B = pd.Timestamp('2024-06-07'), pd.Timestamp('2026-06-07')
COSTS = {'SPOT': dict(side=0.0010 + 0.0005, funding=False),
         'PERP': dict(side=0.00055 + 0.0005, funding=True)}


def run(W, cost, idx):
    """Weights decided at close t earn close t->t+1. Costs on turnover; perps pay funding."""
    W = W.reindex(idx).fillna(0.0)
    c = close.reindex(idx)
    ret = c.pct_change(fill_method=None).fillna(0.0)
    wx = W.shift(1).fillna(0.0)
    gross = (wx * ret).sum(axis=1)
    turn = (W - W.shift(1).fillna(0.0)).abs().sum(axis=1)
    port = gross - (turn * cost['side']).shift(1).fillna(0.0)
    if cost['funding']:
        port = port - (wx * fund.reindex(idx)).sum(axis=1)
    return port, wx


def vol_target(W_base, cost, target=0.30, cap=2.0):
    p, _ = run(W_base, cost, close.index)
    vol = p.rolling(30).std() * math.sqrt(365)          # uses returns up to t only
    scale = (target / vol).clip(upper=cap).fillna(0.0)
    return W_base.mul(scale, axis=0)


def c6_btc_eth():
    W = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    for a in ['BTC', 'ETH']:
        W[a] = 0.5 * (close[a] > close[a].rolling(100).mean())
    return W


def stats(p, wx):
    p = p.iloc[1:]
    eq = (1 + p).cumprod()
    yrs = len(p) / 365
    cagr = eq.iloc[-1] ** (1 / yrs) - 1 if eq.iloc[-1] > 0 else -1.0
    sd = p.std()
    return dict(cagr=cagr * 100, total=(eq.iloc[-1] - 1) * 100,
                sharpe=p.mean() / sd * math.sqrt(365) if sd > 0 else 0.0,
                maxdd=(eq / eq.cummax() - 1).min() * 100,
                expo=wx.sum(axis=1).mean() * 100, days=len(p),
                skew=float(p.skew()), kurt=float(p.kurt() + 3), sr_daily=p.mean() / sd if sd > 0 else 0.0)


def deflated_sharpe(sr, n_obs, skew, kurt, sr_trials):
    """Bailey & Lopez de Prado DSR on daily Sharpe; sr_trials = daily Sharpes of all N trials."""
    N = len(sr_trials)
    v = np.var(sr_trials, ddof=1)
    em = 0.5772156649
    sr0 = math.sqrt(v) * ((1 - em) * norm_ppf(1 - 1 / N) + em * norm_ppf(1 - 1 / (N * math.e)))
    z = (sr - sr0) * math.sqrt(n_obs - 1) / math.sqrt(1 - skew * sr + (kurt - 1) / 4 * sr ** 2)
    return norm_cdf(z)


SPOT_W = {
    'C1_breakout': lambda c: s5_breakout(close, fund),
    'C2_trend_ts': lambda c: s1_trend(close, fund),
    'C4_breakout_btc_gate': lambda c: s5_breakout(close, fund).mul(close['BTC'] > _ema(close['BTC'], 200), axis=0),
    'C5_breakout_voltarget': lambda c: vol_target(s5_breakout(close, fund), c),
    'C6_btc_eth_trend': lambda c: c6_btc_eth(),
}
periods = {'IS': close.index[close.index < CUT_A],
           'OOS_A': close.index[(close.index >= CUT_A) & (close.index < CUT_B)],
           'OOS_B': close.index[close.index >= CUT_B]}

rows = []
for cname, cost in COSTS.items():
    for name, mk in SPOT_W.items():
        if name == 'C5_breakout_voltarget' and cname == 'SPOT':
            continue  # leverage only possible on perps (pre-registered)
        W = mk(cost)
        for per, idx in periods.items():
            p, wx = run(W, cost, idx)
            rows.append(dict(cand=name, costs=cname, period=per, **stats(p, wx), _p=p))
    # C3 ensemble = mean of the three sleeve returns (as in the repo)
    for per, idx in periods.items():
        parts = [run(f(close, fund), cost, idx) for f in (s1_trend, s5_breakout, s4_capitulation)]
        p = sum(x[0] for x in parts) / 3
        wx = sum(x[1] for x in parts) / 3
        rows.append(dict(cand='C3_ensemble3', costs=cname, period=per, **stats(p, wx), _p=p))
    for per, idx in periods.items():  # benchmarks
        r = close.reindex(idx).pct_change(fill_method=None)
        for bname, p in {'BH_BTC': r['BTC'].fillna(0.0), 'BH_basket_EW': r.mean(axis=1).fillna(0.0)}.items():
            rows.append(dict(cand=bname, costs=cname, period=per,
                             **stats(p, pd.DataFrame({'x': np.ones(len(p))}, index=p.index)), _p=p))

df = pd.DataFrame(rows)
cands = df[~df.cand.str.startswith('BH') & (df.period == 'OOS_A')]
n_trials = len(cands)
sr_trials = cands.sr_daily.values
df['dsr'] = np.nan
for i in cands.index:
    r = df.loc[i]
    df.loc[i, 'dsr'] = deflated_sharpe(r['sr_daily'], r['days'], r['skew'], r['kurt'], sr_trials)

pd.set_option('display.width', 220)
show = df.drop(columns=['_p', 'skew', 'kurt', 'sr_daily'])
for per in ['IS', 'OOS_A', 'OOS_B']:
    print(f'\n=== {per}  {periods[per][0].date()} -> {periods[per][-1].date()} ===')
    print(show[show.period == per].drop(columns='period').round(2).to_string(index=False))

bh_b = df[(df.cand == 'BH_BTC') & (df.period == 'OOS_B') & (df.costs == 'SPOT')].total.iloc[0]
print(f'\nN trials (OOS_A) = {n_trials}')
print('\nPRE-REGISTERED CRITERION:')
for (c, k), g in df[~df.cand.str.startswith('BH')].groupby(['cand', 'costs']):
    a = g[g.period == 'OOS_A'].iloc[0]
    b = g[g.period == 'OOS_B'].iloc[0]
    ok1 = a.cagr > 15 and a.sharpe > 0.8
    ok2 = b.total >= bh_b
    ok3 = a.dsr > 0.5
    print(f'{c:24} {k:4}  CAGR_A {a.cagr:6.1f}% Sh_A {a.sharpe:5.2f} DSR {a.dsr:4.2f} | '
          f'B total {b.total:6.1f}% (BTC {bh_b:.1f}%) -> {"CUMPLE" if ok1 and ok2 and ok3 else "no"}')
df.drop(columns=['_p']).to_csv(f'{G}/results.csv', index=False)
pd.DataFrame({f'{a}|{b}|{c}': q for a, b, c, q in zip(df.cand, df.costs, df.period, df._p)}).to_csv(f'{G}/daily_returns.csv')
