"""Per-coin selection on TRAIN+VAL only (TEST is truncated away before any metric is computed)."""
import itertools, json, math, sys
import numpy as np
import pandas as pd
sys.path.insert(0, '/Users/matveypro/kronos_lab/indicators')
sys.path.insert(0, '/Users/matveypro/kronos_lab/goal15')
import engine as E
from scipy_free_stats import norm_cdf, norm_ppf

OUT = '/Users/matveypro/kronos_lab/top10'
UNIV = ["BTC", "ETH", "SOL", "XRP", "ADA", "AVAX", "LINK", "DOGE", "BNB", "LTC", "DOT", "POL", "ATOM", "NEAR", "APT",
        "BCH", "TRX", "ETC", "XLM", "DASH", "ZEC", "VET", "THETA", "ALGO", "COMP", "CRV", "RUNE", "SUSHI", "EGLD", "UNI"]
COST = E.COSTS['SPOT']


def sr(d):
    return d.mean() / d.std() if len(d) > 30 and d.std() > 0 else np.nan


sel = {}
for coin in UNIV:
    df, fund = E.load(coin, '1d')
    if df.index[0] > pd.Timestamp('2022-12-31'):
        print(coin, 'excluded (history starts', df.index[0].date(), ')')
        continue
    sigs, adx_s = E.base_signals(df), E.adx(df)
    rows = []
    for name, a, vt in itertools.product(sigs, (0, 20, 25), (False, True)):
        r, _ = E.bar_returns(df, fund, sigs[name], COST, '1d', a, vt, adx_s)
        r = r[(r.index >= df.index[0] + pd.Timedelta(days=210)) & (r.index < E.VAL_END)]  # TEST cut off
        tr, va = r[r.index < E.TRAIN_END], r[r.index >= E.TRAIN_END]
        rows.append(dict(signal=name, adx=a, voltarget=vt, sr_train=sr(tr), sr_val=sr(va), sr_tv=sr(r),
                         n=len(r), skew=float(r.skew()), kurt=float(r.kurt() + 3)))
    R = pd.DataFrame(rows)
    ok = R[(R.sr_train > 0) & (R.sr_val > 0)].sort_values('sr_tv', ascending=False)
    if ok.empty:
        print(coin, 'no candidate')
        continue
    b = ok.iloc[0]
    v = np.nanvar(R.sr_tv, ddof=1)
    N, em = len(R), 0.5772156649
    sr0 = math.sqrt(v) * ((1 - em) * norm_ppf(1 - 1 / N) + em * norm_ppf(1 - 1 / (N * math.e)))
    s = b.sr_tv
    dsr = norm_cdf((s - sr0) * math.sqrt(b['n'] - 1) / math.sqrt(1 - b['skew'] * s + (b['kurt'] - 1) / 4 * s ** 2))
    sel[coin] = dict(signal=b.signal, adx=int(b.adx), voltarget=bool(b.voltarget),
                     sharpe_train=round(b.sr_train * math.sqrt(365), 2), sharpe_val=round(b.sr_val * math.sqrt(365), 2),
                     dsr=round(dsr, 3))
    print(coin, sel[coin], flush=True)

json.dump(sel, open(f'{OUT}/selections.json', 'w'), indent=1)
