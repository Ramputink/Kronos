"""Bybit linear-perp OHLCV (1d, 4h) + raw funding events for the fixed large-cap universe."""
import os, time
import pandas as pd
import requests

OUT = '/Users/matveypro/kronos_lab/indicators/data'
os.makedirs(OUT, exist_ok=True)
import sys
UNIV = sys.argv[1].split(',') if len(sys.argv) > 1 else ['BTC', 'ETH', 'SOL', 'XRP', 'BNB', 'DOGE', 'ADA', 'LINK', 'AVAX']
TFS = [('240', '4h')] if len(sys.argv) > 2 else [('D', '1d')] if len(sys.argv) > 1 else [('D', '1d'), ('240', '4h')]
API = 'https://api.bybit.com/v5/market'


def get(path, **params):
    for a in range(6):
        try:
            r = requests.get(f'{API}/{path}', params=params, timeout=15).json()
            if r.get('retCode') == 0:
                return r['result']
            raise RuntimeError(r)
        except Exception:
            if a == 5:
                raise
            time.sleep(1 + a)


def klines(sym, interval):
    rows, end = [], None
    while True:
        p = dict(category='linear', symbol=sym, interval=interval, limit=1000)
        if end:
            p['end'] = end
        b = get('kline', **p)['list']
        if not b:
            break
        rows += b
        end = int(b[-1][0]) - 1
        if len(b) < 1000:
            break
        time.sleep(0.05)
    df = pd.DataFrame(rows, columns=['ts', 'open', 'high', 'low', 'close', 'volume', 'turnover']).astype(float)
    df['time'] = pd.to_datetime(df.ts.astype('int64'), unit='ms')
    df = df.drop_duplicates('time').set_index('time').sort_index().drop(columns='ts')
    return df.iloc[:-1]  # drop forming candle


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
    f = pd.DataFrame(rows)
    f['time'] = pd.to_datetime(f.fundingRateTimestamp.astype('int64'), unit='ms')
    return f.drop_duplicates('time').set_index('time').fundingRate.astype(float).sort_index()


for base in UNIV:
    sym = f'{base}USDT'
    for iv, tag in TFS:
        k = klines(sym, iv)
        k.to_csv(f'{OUT}/{base}_{tag}.csv')
        print(base, tag, k.index[0], '->', k.index[-1], len(k), flush=True)
    funding(sym).to_csv(f'{OUT}/{base}_funding.csv')
