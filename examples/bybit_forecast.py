"""
Kronos + Bybit proof of concept.

Downloads K-lines from Bybit's public v5 market API (no API key needed),
produces a live forecast and runs a small walk-forward evaluation that
measures directional accuracy of the forecast vs. what actually happened.

Usage (from repo root):
    python examples/bybit_forecast.py --symbol BTCUSDT --interval 60 --pred-len 24
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import requests

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from model import Kronos, KronosTokenizer, KronosPredictor

BYBIT_URL = "https://api.bybit.com/v5/market/kline"
INTERVAL_MINUTES = {"1": 1, "3": 3, "5": 5, "15": 15, "30": 30, "60": 60, "120": 120,
                    "240": 240, "360": 360, "720": 720, "D": 1440}


def fetch_bybit_klines(symbol, interval, total, category="linear"):
    """Page backwards through Bybit klines (max 1000 per request). Returns oldest-first."""
    rows, end = [], None
    while len(rows) < total:
        params = {"category": category, "symbol": symbol, "interval": interval,
                  "limit": min(1000, total - len(rows))}
        if end is not None:
            params["end"] = end
        resp = requests.get(BYBIT_URL, params=params, timeout=10).json()
        if resp.get("retCode") != 0:
            raise RuntimeError(f"Bybit error: {resp}")
        batch = resp["result"]["list"]  # newest first
        if not batch:
            break
        rows.extend(batch)
        end = int(batch[-1][0]) - 1
        time.sleep(0.1)

    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume", "amount"])
    df = df.astype(float).drop_duplicates("ts").sort_values("ts")
    df["timestamps"] = pd.to_datetime(df["ts"].astype("int64"), unit="ms")
    # The newest candle is still forming; drop it.
    return df.iloc[:-1].reset_index(drop=True)


def future_timestamps(last_ts, interval, n):
    step = pd.Timedelta(minutes=INTERVAL_MINUTES[interval])
    return pd.Series([last_ts + step * (i + 1) for i in range(n)])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--category", default="linear", help="spot | linear | inverse")
    p.add_argument("--interval", default="60", choices=list(INTERVAL_MINUTES))
    p.add_argument("--model", default="NeoQuasar/Kronos-small")
    p.add_argument("--lookback", type=int, default=400)
    p.add_argument("--pred-len", type=int, default=24)
    p.add_argument("--samples", type=int, default=5, help="paths averaged per forecast")
    p.add_argument("--eval-windows", type=int, default=30)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained(args.model)
    predictor = KronosPredictor(model, tokenizer, device=args.device, max_context=512)
    print(f"device: {predictor.device}")

    stride = args.pred_len
    need = args.lookback + args.pred_len + stride * (args.eval_windows - 1)
    df = fetch_bybit_klines(args.symbol, args.interval, need + 1, args.category)
    print(f"{len(df)} candles {df.timestamps.iloc[0]} -> {df.timestamps.iloc[-1]} UTC")
    cols = ["open", "high", "low", "close", "volume", "amount"]

    # --- Walk-forward evaluation -------------------------------------------
    hits, pred_rets, real_rets = [], [], []
    t0 = time.time()
    for w in range(args.eval_windows):
        start = len(df) - args.lookback - args.pred_len - w * stride
        if start < 0:
            break
        hist = df.iloc[start:start + args.lookback]
        fut = df.iloc[start + args.lookback:start + args.lookback + args.pred_len]
        pred = predictor.predict(hist[cols], hist.timestamps, fut.timestamps.reset_index(drop=True),
                                 pred_len=args.pred_len, T=1.0, top_p=0.9,
                                 sample_count=args.samples, verbose=False)
        last = hist.close.iloc[-1]
        pr = pred.close.iloc[-1] / last - 1
        rr = fut.close.iloc[-1] / last - 1
        pred_rets.append(pr)
        real_rets.append(rr)
        hits.append(np.sign(pr) == np.sign(rr))
    pred_rets, real_rets = np.array(pred_rets), np.array(real_rets)
    n = len(hits)
    strat = np.sign(pred_rets) * real_rets
    print(f"\nWalk-forward: {n} windows, horizon {args.pred_len} x {args.interval}m "
          f"({time.time() - t0:.0f}s)")
    print(f"  directional hit rate : {np.mean(hits):.1%}")
    print(f"  always-long hit rate : {np.mean(real_rets > 0):.1%}")
    print(f"  corr(pred, real)     : {np.corrcoef(pred_rets, real_rets)[0, 1]:+.3f}")
    print(f"  long/short sum ret   : {strat.sum():+.2%} (no fees) vs buy&hold {real_rets.sum():+.2%}")

    # --- Live forecast -----------------------------------------------------
    hist = df.iloc[-args.lookback:]
    y_ts = future_timestamps(hist.timestamps.iloc[-1], args.interval, args.pred_len)
    pred = predictor.predict(hist[cols], hist.timestamps, y_ts, pred_len=args.pred_len,
                             T=1.0, top_p=0.9, sample_count=args.samples, verbose=False)
    last = hist.close.iloc[-1]
    print(f"\nLive forecast {args.symbol} from {hist.timestamps.iloc[-1]} UTC, last close {last:,.2f}")
    print(pred[["open", "high", "low", "close"]].iloc[[0, len(pred) // 2, -1]].round(2))
    print(f"  expected move over horizon: {pred.close.iloc[-1] / last - 1:+.2%}")


if __name__ == "__main__":
    main()
