"""
Kronos research cache v2  (READY TO RUN - NOT RUN BY THE AUDIT AGENT; GPU is shared).

Differences vs v1 (examples/build_kronos_cache.py, run with --lookback 256 --paths 6 --windows 300):
  * untruncated sampling: top_p=1.0, top_k=0, T=1.0  (v1 used top_p=0.9 -> under-dispersed paths)
  * 30 independent paths per origin (v1: 6). Each path is its own batch row with
    sample_count=1, so KronosPredictor's internal np.mean over sample_count
    (model/kronos.py:465-467) is a no-op and the paths are NOT averaged.
  * origins every 4h (v1: 12h)
  * OLDER, disjoint period (default 2024-01-01 .. 2025-06-30 origins) fetched from Bybit v5
    /market/kline with explicit start/end, so round-1 claims (built on 2025-10..2026-10)
    can be retested out of sample.
  * unconfirmed bars dropped: any candle whose close time (open + interval) > now is removed;
    gaps / duplicates are checked and reported.
  * resumable: per-symbol shards are written every --flush-every origins.

Outputs (same layout as v1 so existing agent code works unchanged):
  <out>/<SYM>_60_ohlcv.csv   open,high,low,close,volume,amount,timestamps (UTC, oldest first)
  <out>/<SYM>_60_paths.npz   origins (W,), paths (W,S,H,6) float32, lookback, pred_len,
                             top_p, T, sample_mode="independent"
  Realised future of origin o = csv rows o+1..o+H (always present: data is fetched to
  last_origin + H).

Usage (from the Kronos repo root so `model` imports):
  cd /Users/matveypro/Kronos && /Users/matveypro/.venvs/kronos/bin/python \
      /Users/matveypro/kronos_lab/agents/r2_cache_audit_v2_spec/build_kronos_cache_v2.py \
      --out /Users/matveypro/kronos_lab/cache_v2
  Add --dry-run to only fetch data, check it and print the job count / runtime estimate.
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import requests

COLS = ["open", "high", "low", "close", "volume", "amount"]
BYBIT_URL = "https://api.bybit.com/v5/market/kline"
# v1 timing from cache_build.log: 1800 paths in 267s / 283s / 325s (BTC/ETH/SOL),
# Kronos-small, lookback 256, pred_len 24, batch 64  -> ~0.16 s per path.
SEC_PER_PATH_V1 = (267 + 283 + 325) / (3 * 1800)


def fetch_range(symbol, interval_min, start, end, category="linear"):
    """All confirmed klines with open time in [start, end] (UTC), oldest first."""
    step_ms = interval_min * 60_000
    start_ms = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    end_ms = int(pd.Timestamp(end, tz="UTC").timestamp() * 1000)
    rows, cur_end = [], end_ms
    while cur_end >= start_ms:
        params = {"category": category, "symbol": symbol, "interval": str(interval_min),
                  "start": start_ms, "end": cur_end, "limit": 1000}
        resp = None
        for attempt in range(6):
            try:
                resp = requests.get(BYBIT_URL, params=params, timeout=15).json()
                if resp.get("retCode") == 0:
                    break
            except Exception as e:
                resp = {"exception": repr(e)}
            time.sleep(2 ** attempt)
        else:
            raise RuntimeError(f"Bybit request failed 6 times: {resp}")
        batch = resp["result"]["list"]  # newest first
        if not batch:
            break
        rows.extend(batch)
        oldest = int(batch[-1][0])
        if oldest <= start_ms:
            break
        cur_end = oldest - 1
        time.sleep(0.12)
    df = pd.DataFrame(rows, columns=["ts"] + COLS).astype(float)
    df = df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
    df = df[(df.ts >= start_ms) & (df.ts <= end_ms)]
    # drop unconfirmed bar(s): close time must be in the past
    now_ms = time.time() * 1000
    n0 = len(df)
    df = df[df.ts + step_ms <= now_ms]
    if len(df) < n0:
        print(f"  {symbol}: dropped {n0 - len(df)} unconfirmed bar(s)")
    gaps = np.diff(df.ts.values)
    bad = int((gaps != step_ms).sum())
    if bad:
        print(f"  WARNING {symbol}: {bad} gaps/irregular steps (max gap {gaps.max() / step_ms:.0f} bars)")
    df["timestamps"] = pd.to_datetime(df.ts.astype("int64"), unit="ms")
    return df.reset_index(drop=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT")
    p.add_argument("--interval", type=int, default=60)
    p.add_argument("--first-origin", default="2024-01-01 00:00")
    p.add_argument("--last-origin", default="2025-06-30 23:00")
    p.add_argument("--lookback", type=int, default=256, help="256 = same as v1 cache")
    p.add_argument("--pred-len", type=int, default=24)
    p.add_argument("--stride", type=int, default=4, help="candles between origins")
    p.add_argument("--paths", type=int, default=30)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--T", type=float, default=1.0)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--flush-every", type=int, default=200, help="origins per resumable shard")
    p.add_argument("--model", default="NeoQuasar/Kronos-small")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    step = pd.Timedelta(minutes=args.interval)

    # fetch + plan
    plans = {}
    for sym in args.symbols.split(","):
        start = pd.Timestamp(args.first_origin) - step * (args.lookback - 1)
        end = pd.Timestamp(args.last_origin) + step * args.pred_len
        csv = f"{args.out}/{sym}_{args.interval}_ohlcv.csv"
        if os.path.exists(csv):
            df = pd.read_csv(csv, parse_dates=["timestamps"])
        else:
            df = fetch_range(sym, args.interval, start, end)
            df.drop(columns="ts").to_csv(csv, index=False)
        ts = df.timestamps
        lo = max(int(np.searchsorted(ts, pd.Timestamp(args.first_origin))), args.lookback - 1)
        hi = min(int(np.searchsorted(ts, pd.Timestamp(args.last_origin), side="right")) - 1,
                 len(df) - 1 - args.pred_len)
        origins = list(range(lo, hi + 1, args.stride))
        plans[sym] = (df, origins)
        print(f"{sym}: {len(df)} candles {ts.iloc[0]} -> {ts.iloc[-1]}; "
              f"{len(origins)} origins x {args.paths} paths = {len(origins) * args.paths} paths")

    total = sum(len(o) for _, o in plans.values()) * args.paths
    scale = args.lookback / 256  # attention cost grows ~linearly-to-quadratically; linear is optimistic
    est = total * SEC_PER_PATH_V1 * scale
    print(f"TOTAL {total} paths; est. runtime ~{est / 3600:.1f} h at v1 speed "
          f"({SEC_PER_PATH_V1:.3f} s/path, batch {args.batch}); "
          f"larger --batch may be faster if memory allows")
    if args.dry_run:
        return

    sys.path.insert(0, os.getcwd())
    import torch
    from model import Kronos, KronosTokenizer, KronosPredictor
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained(args.model)
    predictor = KronosPredictor(model, tokenizer, max_context=512)
    print(f"device: {predictor.device}")

    for sym, (df, origins) in plans.items():
        final = f"{args.out}/{sym}_{args.interval}_paths.npz"
        if os.path.exists(final):
            print(f"{sym}: already done"); continue
        t0 = time.time()
        shards = []
        for s0 in range(0, len(origins), args.flush_every):
            shard_f = f"{args.out}/.{sym}_shard{s0:06d}.npz"
            sub = origins[s0:s0 + args.flush_every]
            if os.path.exists(shard_f):
                shards.append(shard_f); continue
            jobs = [(i, k) for i in range(len(sub)) for k in range(args.paths)]
            out = np.zeros((len(sub), args.paths, args.pred_len, 6), dtype=np.float32)
            for b in range(0, len(jobs), args.batch):
                chunk = jobs[b:b + args.batch]
                dfs, xts, yts = [], [], []
                for i, _ in chunk:
                    o = sub[i]
                    hist = df.iloc[o - args.lookback + 1:o + 1]
                    assert len(hist) == args.lookback and hist.index[-1] == o
                    dfs.append(hist[COLS].reset_index(drop=True))
                    xts.append(hist.timestamps.reset_index(drop=True))
                    yts.append(pd.Series([hist.timestamps.iloc[-1] + step * (h + 1)
                                          for h in range(args.pred_len)]))
                # sample_count=1 on purpose: each row is one independent path (no averaging)
                preds = predictor.predict_batch(dfs, xts, yts, pred_len=args.pred_len, T=args.T,
                                                top_k=0, top_p=args.top_p, sample_count=1,
                                                verbose=False)
                for (i, k), pr in zip(chunk, preds):
                    out[i, k] = pr[COLS].values
            np.savez_compressed(shard_f, origins=np.array(sub), paths=out)
            shards.append(shard_f)
            done = s0 + len(sub)
            el = time.time() - t0
            print(f"{sym} {done}/{len(origins)} origins, {el:.0f}s, "
                  f"ETA {el / done * (len(origins) - done) / 60:.0f} min", flush=True)
        o_all = np.concatenate([np.load(f)["origins"] for f in shards])
        p_all = np.concatenate([np.load(f)["paths"] for f in shards])
        np.savez_compressed(final, origins=o_all, paths=p_all, lookback=args.lookback,
                            pred_len=args.pred_len, top_p=args.top_p, T=args.T,
                            sample_mode="independent")
        for f in shards:
            os.remove(f)
        print(f"{sym} done: {len(o_all)} origins", flush=True)
    open(f"{args.out}/READY", "w").write("ok\n")


if __name__ == "__main__":
    main()
