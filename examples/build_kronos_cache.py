"""
Build a shared research cache: Bybit OHLCV + Kronos forecast paths.

For each symbol it saves
  <out>/<SYMBOL>_<interval>_ohlcv.csv    full candle history (oldest first)
  <out>/<SYMBOL>_<interval>_paths.npz    Kronos sample paths per forecast origin
      origins   (W,)          index into the ohlcv csv of the last context candle
      paths     (W, S, H, 6)  S sampled future paths, H steps, cols open..amount
Forecast origins are taken every --stride candles over the most recent history, so
everything after each origin is genuinely out-of-sample for that forecast.

Usage (from repo root):
    python examples/build_kronos_cache.py --out ~/kronos_lab/cache
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from model import Kronos, KronosTokenizer, KronosPredictor
from bybit_forecast import fetch_bybit_klines

COLS = ["open", "high", "low", "close", "volume", "amount"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT")
    p.add_argument("--interval", default="60")
    p.add_argument("--history", type=int, default=9000, help="candles to download")
    p.add_argument("--lookback", type=int, default=400)
    p.add_argument("--pred-len", type=int, default=24)
    p.add_argument("--stride", type=int, default=12)
    p.add_argument("--windows", type=int, default=600)
    p.add_argument("--paths", type=int, default=8)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--model", default="NeoQuasar/Kronos-small")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained(args.model)
    predictor = KronosPredictor(model, tokenizer, max_context=512)

    for sym in args.symbols.split(","):
        df = fetch_bybit_klines(sym, args.interval, args.history)
        df.drop(columns="ts").to_csv(f"{args.out}/{sym}_{args.interval}_ohlcv.csv", index=False)
        last_origin = len(df) - 1 - args.pred_len  # keep a realised future for evaluation
        origins = [last_origin - i * args.stride for i in range(args.windows)]
        origins = sorted(o for o in origins if o >= args.lookback - 1)
        step = df.timestamps.iloc[1] - df.timestamps.iloc[0]

        jobs = [(o, s) for o in origins for _ in range(args.paths) for s in [0]]
        out = np.zeros((len(origins), args.paths, args.pred_len, 6), dtype=np.float32)
        t0 = time.time()
        for b in range(0, len(jobs), args.batch):
            chunk = jobs[b:b + args.batch]
            dfs, xts, yts = [], [], []
            for o, _ in chunk:
                hist = df.iloc[o - args.lookback + 1:o + 1]
                dfs.append(hist[COLS].reset_index(drop=True))
                xts.append(hist.timestamps.reset_index(drop=True))
                yts.append(pd.Series([hist.timestamps.iloc[-1] + step * (k + 1)
                                      for k in range(args.pred_len)]))
            preds = predictor.predict_batch(dfs, xts, yts, pred_len=args.pred_len, T=1.0,
                                            top_p=0.9, sample_count=1, verbose=False)
            for j, pr in enumerate(preds):
                g = b + j
                out[g // args.paths, g % args.paths] = pr[COLS].values
            done = min(b + args.batch, len(jobs))
            print(f"{sym} {done}/{len(jobs)} paths, {time.time() - t0:.0f}s", flush=True)
        np.savez_compressed(f"{args.out}/{sym}_{args.interval}_paths.npz",
                            origins=np.array(origins), paths=out,
                            lookback=args.lookback, pred_len=args.pred_len)
        print(f"{sym} done: {len(origins)} origins", flush=True)
    open(f"{args.out}/READY", "w").write("ok\n")


if __name__ == "__main__":
    main()
