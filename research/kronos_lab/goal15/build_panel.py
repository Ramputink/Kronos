"""Bybit linear-perp daily panel (close, open) + daily funding sum for the fixed 30-coin universe."""
import json, time
import pandas as pd
import requests

OUT = '/Users/matveypro/kronos_lab/goal15'
UNIVERSE = ["BTC", "ETH", "SOL", "XRP", "ADA", "AVAX", "LINK", "DOGE", "BNB", "LTC",
            "DOT", "POL", "ATOM", "NEAR", "APT", "BCH", "TRX", "ETC", "XLM", "DASH",
            "ZEC", "VET", "THETA", "ALGO", "COMP", "CRV", "RUNE", "SUSHI", "EGLD", "UNI"]
API = 'https://api.bybit.com/v5/market'


def get(path, **params):
    for attempt in range(5):
        try:
            r = requests.get(f'{API}/{path}', params=params, timeout=15).json()
            if r.get('retCode') == 0:
                return r['result']
            raise RuntimeError(r)
        except Exception as e:
            if attempt == 4:
                raise
            time.sleep(1 + attempt)


def klines(sym):
    rows, end = [], None
    while True:
        p = dict(category='linear', symbol=sym, interval='D', limit=1000)
        if end:
            p['end'] = end
        b = get('kline', **p)['list']
        if not b:
            break
        rows += b
        end = int(b[-1][0]) - 1
        if len(b) < 1000:
            break
    df = pd.DataFrame(rows, columns=['ts', 'open', 'high', 'low', 'close', 'volume', 'turnover']).astype(float)
    df['date'] = pd.to_datetime(df.ts.astype('int64'), unit='ms')
    return df.drop_duplicates('date').set_index('date').sort_index()


def funding(sym):
    rows, end = [], None
    while True:
        p = dict(category='linear', symbol=sym, limit=200)
        if end:
            p['endTime'] = end
        b = get('funding/history', **p)['list']
        if not b:
            break
        rows += b
        end = int(b[-1]['fundingRateTimestamp']) - 1
        if len(b) < 200:
            break
        time.sleep(0.05)
    if not rows:
        return pd.Series(dtype=float)
    f = pd.DataFrame(rows)
    f['t'] = pd.to_datetime(f.fundingRateTimestamp.astype('int64'), unit='ms')
    s = f.drop_duplicates('t').set_index('t').fundingRate.astype(float).sort_index()
    # funding paid at t belongs to the day that ends at t -> label by (t - 1ms).floor('D')
    return s.groupby((s.index - pd.Timedelta(milliseconds=1)).floor('D')).sum()


close, opn, fund, meta = {}, {}, {}, {}
for base in UNIVERSE:
    sym = f'{base}USDT'
    try:
        k = klines(sym)
    except Exception as e:
        print(base, 'kline FAILED', e, flush=True)
        continue
    k = k.iloc[:-1]  # drop the still-forming daily candle
    close[base], opn[base] = k.close, k.open
    try:
        fund[base] = funding(sym)
    except Exception as e:
        print(base, 'funding FAILED', e, flush=True)
    meta[base] = [str(k.index[0].date()), str(k.index[-1].date()), len(k)]
    print(base, meta[base], 'funding days', len(fund.get(base, [])), flush=True)

pd.DataFrame(close).sort_index().to_parquet(f'{OUT}/close.parquet') if False else None
pd.DataFrame(close).sort_index().to_csv(f'{OUT}/close.csv')
pd.DataFrame(opn).sort_index().to_csv(f'{OUT}/open.csv')
pd.DataFrame(fund).sort_index().to_csv(f'{OUT}/funding.csv')
json.dump(meta, open(f'{OUT}/panel_meta.json', 'w'), indent=1)
