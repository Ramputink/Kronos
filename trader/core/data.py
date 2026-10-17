"""Market data: Bybit v5 fetch + incremental, idempotent CSV store (Wave 1, PROTECTED).

Port of research/kronos_lab/indicators/fetch.py (same endpoints, same pagination, same retry policy,
forming candle never written), extended with incremental updates:

    <data root>/<source>/<COIN>_<tf>.csv        time,open,high,low,close,volume,turnover
    <data root>/linear/<COIN>_funding.csv       time,fundingRate

`update()` reads the existing CSV, fetches only from the last stored bar (2-bar overlap), appends the new closed
bars and rewrites atomically (tmp + os.replace) under an exclusive flock. Running it twice in a row is a no-op
(byte-identical file). The exchange is the source of truth: if a stored bar in the overlap disagrees with the
exchange (`overlap_mismatch` > 0) the WHOLE series is re-fetched and the store replaced by the exchange history,
and the key is reported in `errors` (so the CLI exits 1). `verify` does the same comparison over the full history.

`seed_from()` copies only from the protected `data.seed_dir` (never from the repo, TRADER_ROOT, the strategy or
journal dirs or the data store itself) and validates every file (OHLC sanity, sorted unique closed bars on the tf
grid, nothing dated in the future) before copying anything.

The public loader is `vault.load` (re-exported here as `load`). `load_raw` is INTERNAL: only vault.py may call
it, because it returns the full, unsealed history.

Python >= 3.9. Dependencies: numpy, pandas, requests.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

if __package__ in (None, ""):  # allow `python trader/core/data.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from trader.core import config

try:
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None


class DataError(Exception):
    pass


COLS = ["open", "high", "low", "close", "volume", "turnover"]
TF_INTERVAL = {"1d": "D", "4h": "240"}
INTERVAL_MS = {"D": 86_400_000, "240": 14_400_000}
TF_DELTA = {"1d": pd.Timedelta("1D"), "4h": pd.Timedelta("4h")}
TF_DATE_FMT = {"1d": "%Y-%m-%d", "4h": "%Y-%m-%d %H:%M:%S"}
FUND_DATE_FMT = "%Y-%m-%d %H:%M:%S"
SOURCES = ("linear", "spot")
OVERLAP_BARS = 2
KLINE_LIMIT = 1000
FUND_LIMIT = 200
RETRIES = 6


# ── HTTP (port of fetch.py `get`) ────────────────────────────────────────────
def _api():
    return os.environ.get("TRADER_BYBIT_API") or config.load_config()["data"]["api"]


def _http_get(path: str, params: dict) -> dict:
    """GET <api>/<path>; returns `result` when retCode == 0. 6 attempts, back-off 1,2,3,4,5 s (as fetch.py)."""
    import requests  # lazy: tests and sweep hosts never need it

    url = "%s/%s" % (_api().rstrip("/"), path)
    last = None
    for a in range(RETRIES):
        try:
            r = requests.get(url, params=params, timeout=15).json()
            if r.get("retCode") == 0:
                return r["result"]
            raise DataError("bybit %s: %s" % (path, r))
        except Exception as e:  # noqa: BLE001 - network, JSON or API error: retry
            last = e
            if a == RETRIES - 1:
                break
            _sleep(1 + a)
    raise DataError("bybit %s failed after %d attempts: %s" % (path, RETRIES, last))


def _sleep(s):  # patched in tests
    time.sleep(s)


def _now_ms() -> int:  # patched in tests
    return int(time.time() * 1000)


def _to_ms(ts) -> int:
    return int(pd.Timestamp(ts).value // 1_000_000)


# ── fetch ────────────────────────────────────────────────────────────────────
def fetch_klines(symbol: str, interval: str, category: str, start_ms: Optional[int] = None,
                 end_ms: Optional[int] = None) -> pd.DataFrame:
    """Bybit v5 /market/kline, paginated backwards from `end_ms` (or now) down to `start_ms` (or listing).

    interval "D" or "240" (a tf "1d"/"4h" is accepted too). Returns closed bars only (open + interval <= now),
    index `time` (naive UTC bar open), float columns open,high,low,close,volume,turnover.
    """
    interval = TF_INTERVAL.get(interval, interval)
    if interval not in INTERVAL_MS:
        raise DataError("unsupported interval %r" % interval)
    if category not in SOURCES:
        raise DataError("unsupported category %r" % category)
    rows, end = [], end_ms
    while True:
        p = dict(category=category, symbol=symbol, interval=interval, limit=KLINE_LIMIT)
        if start_ms is not None:
            p["start"] = int(start_ms)
        if end is not None:
            p["end"] = int(end)
        b = _http_get("kline", p).get("list") or []
        if not b:
            break
        rows += b
        oldest = int(b[-1][0])
        end = oldest - 1
        if len(b) < KLINE_LIMIT or (start_ms is not None and oldest <= start_ms):
            break
        _sleep(0.05)
    if not rows:
        return _empty_klines()
    df = pd.DataFrame([r[:7] for r in rows], columns=["ts"] + COLS).astype(float)
    ts = df.ts.astype("int64")
    # drop forming candle(s): fetch.py dropped the newest row; here the rule is explicit so a past `end_ms`
    # never loses a closed bar.
    df = df[(ts + INTERVAL_MS[interval]) <= _now_ms()]
    if start_ms is not None:
        df = df[df.ts.astype("int64") >= int(start_ms)]
    df["time"] = pd.to_datetime(df.ts.astype("int64"), unit="ms")
    df = df.drop_duplicates("time").set_index("time").sort_index().drop(columns="ts")
    return df[COLS]


def _empty_klines() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype=float) for c in COLS},
                        index=pd.DatetimeIndex([], name="time"))


def fetch_funding(symbol: str, start_ms: Optional[int] = None) -> pd.Series:
    """Bybit v5 /market/funding/history (linear only), paginated backwards; Series `fundingRate` by event time."""
    rows, end = [], None
    while True:
        p = dict(category="linear", symbol=symbol, limit=FUND_LIMIT)
        if start_ms is not None:
            # Bybit rejects startTime without endTime
            p["startTime"] = int(start_ms)
            p["endTime"] = int(end) if end is not None else _now_ms()
        elif end is not None:
            p["endTime"] = int(end)
        b = _http_get("funding/history", p).get("list") or []
        if not b:
            break
        rows += b
        oldest = int(b[-1]["fundingRateTimestamp"])
        end = oldest - 1
        if len(b) < FUND_LIMIT or (start_ms is not None and oldest <= start_ms):
            break
        _sleep(0.05)
    if not rows:
        return pd.Series(dtype=float, name="fundingRate", index=pd.DatetimeIndex([], name="time"))
    f = pd.DataFrame(rows)
    t = f.fundingRateTimestamp.astype("int64")
    if start_ms is not None:
        f = f[t >= int(start_ms)]
    f["time"] = pd.to_datetime(f.fundingRateTimestamp.astype("int64"), unit="ms")
    s = f.drop_duplicates("time").set_index("time").fundingRate.astype(float).sort_index()
    s.name = "fundingRate"
    return s


# ── CSV store ────────────────────────────────────────────────────────────────
def kline_path(coin: str, tf: str, source: str) -> Path:
    return config.data_dir(source) / ("%s_%s.csv" % (coin, tf))


def funding_path(coin: str) -> Path:
    return config.data_dir("linear") / ("%s_funding.csv" % coin)


def read_klines(path, exact: bool = True) -> pd.DataFrame:
    """exact=True parses floats round-trip (store maintenance: rewrites are byte-identical); exact=False uses
    pandas' default parser, which is what research/kronos_lab used — load_raw uses it so numbers match."""
    df = pd.read_csv(path, index_col=0, parse_dates=True, float_precision="round_trip" if exact else None)
    df.index.name = "time"
    return df


def read_funding(path, exact: bool = True) -> pd.Series:
    s = pd.read_csv(path, index_col=0, parse_dates=True,
                    float_precision="round_trip" if exact else None).iloc[:, 0].astype(float)
    s.index.name = "time"
    s.name = "fundingRate"
    return s


def _atomic_write(text_writer, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(".%s.tmp-%d" % (path.name, os.getpid()))
    try:
        text_writer(tmp)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def write_klines(df: pd.DataFrame, path, tf: str):
    path = Path(path)
    out = df[COLS].copy()
    out.index.name = "time"
    _atomic_write(lambda p: out.to_csv(p, date_format=TF_DATE_FMT[tf]), path)


def write_funding(s: pd.Series, path):
    path = Path(path)
    out = s.astype(float).rename("fundingRate")
    out.index.name = "time"
    _atomic_write(lambda p: out.to_frame().to_csv(p, date_format=FUND_DATE_FMT), path)


class _Lock:
    """Exclusive flock on <file>.lock so two updaters (cron + manual) never interleave."""

    def __init__(self, path: Path):
        self.path = path.with_name(".%s.lock" % path.name)
        self.fh = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = open(self.path, "a")
        if fcntl:
            fcntl.flock(self.fh, fcntl.LOCK_EX)
        return self

    def __exit__(self, *exc):
        if fcntl:
            fcntl.flock(self.fh, fcntl.LOCK_UN)
        self.fh.close()


def _merge_append(old, new):
    """Existing rows are kept verbatim; only rows strictly after the last stored index are appended."""
    if old is None or len(old) == 0:
        return new.sort_index(), len(new)
    add = new[new.index > old.index.max()]
    add = add[~add.index.duplicated(keep="last")].sort_index()
    if len(add) == 0:
        return old, 0
    return pd.concat([old, add]), len(add)


def _overlap_mismatch(old: pd.DataFrame, new: pd.DataFrame) -> int:
    common = old.index.intersection(new.index)
    if len(common) == 0:
        return 0
    a, b = old.loc[common, ["open", "high", "low", "close"]], new.loc[common, ["open", "high", "low", "close"]]
    return int(((a - b).abs() > 1e-9 * (a.abs() + 1)).any(axis=1).sum())


def _diff_klines(old: pd.DataFrame, full: pd.DataFrame) -> dict:
    """How the stored series differs from the exchange history `full`."""
    common = old.index.intersection(full.index)
    changed = _overlap_mismatch(old.loc[common], full.loc[common]) if len(common) else 0
    return {"changed": int(changed), "dropped": int(len(old.index.difference(full.index))),
            "added": int(len(full.index.difference(old.index)))}


def _refetch_replace_klines(coin: str, tf: str, source: str, path: Path, old: Optional[pd.DataFrame]) -> tuple:
    """Re-fetch the whole history from the exchange and replace the store with it (caller holds the lock).
    Rows the exchange does not have (e.g. made-up bars) are dropped. Returns (frame, diff)."""
    full = fetch_klines("%sUSDT" % coin, TF_INTERVAL[tf], source)
    if len(full) == 0:
        raise DataError("%s_%s: re-fetch returned no bars; store left untouched" % (coin, tf))
    diff = _diff_klines(old, full) if old is not None and len(old) else {"changed": 0, "dropped": 0,
                                                                          "added": int(len(full))}
    if diff["changed"] or diff["dropped"] or diff["added"] or old is None:
        write_klines(full, path, tf)
    return full, diff


def update_one(coin: str, tf: str, source: str) -> dict:
    if tf not in TF_INTERVAL:
        raise DataError("unsupported tf %r" % tf)
    interval = TF_INTERVAL[tf]
    path = kline_path(coin, tf, source)
    repaired = None
    with _Lock(path):
        old = read_klines(path) if path.exists() else None
        start_ms = None
        if old is not None and len(old):
            start_ms = _to_ms(old.index.max()) - OVERLAP_BARS * INTERVAL_MS[interval]
        new = fetch_klines("%sUSDT" % coin, interval, source, start_ms=start_ms)
        mism = _overlap_mismatch(old, new) if old is not None else 0
        if mism:
            # a stored bar disagrees with the exchange: the store is not trustworthy -> exchange history wins
            merged, repaired = _refetch_replace_klines(coin, tf, source, path, old)
            n = repaired["added"]
        else:
            merged, n = _merge_append(old, new)
            if n:
                write_klines(merged, path, tf)
        last = merged.index.max() if len(merged) else None
    out = {"rows_added": int(n), "last": config.iso(last) if last is not None else None}
    if mism:
        out["overlap_mismatch"] = mism
        out["repaired"] = repaired
    return out


def verify_one(coin: str, tf: str, source: str) -> dict:
    """Compare the whole stored series with a full exchange re-fetch; replace the store if anything differs."""
    if tf not in TF_INTERVAL:
        raise DataError("unsupported tf %r" % tf)
    path = kline_path(coin, tf, source)
    with _Lock(path):
        old = read_klines(path) if path.exists() else None
        full, diff = _refetch_replace_klines(coin, tf, source, path, old)
    return dict(diff, last=config.iso(full.index.max()), replaced=bool(old is None or any(diff.values())))


def _funding_mismatch(old: pd.Series, new: pd.Series) -> int:
    common = old.index.intersection(new.index)
    if len(common) == 0:
        return 0
    a, b = old.loc[common], new.loc[common]
    return int(((a - b).abs() > 1e-12 + 1e-9 * a.abs()).sum())


def update_funding(coin: str) -> dict:
    path = funding_path(coin)
    with _Lock(path):
        old = read_funding(path) if path.exists() else None
        start_ms = _to_ms(old.index.max()) - 1 if old is not None and len(old) else None
        new = fetch_funding("%sUSDT" % coin, start_ms=start_ms)
        mism = _funding_mismatch(old, new) if old is not None and len(old) else 0
        if mism:
            full = fetch_funding("%sUSDT" % coin)
            if len(full) == 0:
                raise DataError("%s_funding: re-fetch returned no events; store left untouched" % coin)
            merged, n = full, int(len(full.index.difference(old.index)))
            write_funding(merged, path)
        else:
            merged, n = _merge_append(old, new)
            if n:
                write_funding(merged, path)
        last = merged.index.max() if len(merged) else None
    out = {"rows_added": int(n), "last": config.iso(last) if last is not None else None}
    if mism:
        out["overlap_mismatch"] = mism
        out["repaired"] = True
    return out


def update(coins: Optional[List[str]] = None, tfs: Optional[List[str]] = None, source: str = "spot",
           funding: bool = True) -> dict:
    """Incremental, idempotent update of <data root>/<source>. Funding is fetched only for source "linear"
    (spot has no funding). Errors are collected per key, never raised (one bad coin never blocks the rest)."""
    if source not in SOURCES:
        raise DataError("unsupported source %r" % source)
    cfg = config.load_config()
    coins = list(coins) if coins else list(cfg["universe"])
    tfs = list(tfs) if tfs else list(cfg["timeframes"])
    res = {"source": source, "updated": {}, "errors": {}}
    for coin in coins:
        for tf in tfs:
            key = "%s_%s" % (coin, tf)
            try:
                res["updated"][key] = r = update_one(coin, tf, source)
                if r.get("overlap_mismatch"):
                    res["errors"][key] = ("overlap_mismatch: %d stored bar(s) disagreed with the exchange; store "
                                          "re-fetched and replaced %s" % (r["overlap_mismatch"], r.get("repaired")))
            except Exception as e:  # noqa: BLE001
                res["errors"][key] = "%s: %s" % (type(e).__name__, e)
        if funding and source == "linear":
            key = "%s_funding" % coin
            try:
                res["updated"][key] = r = update_funding(coin)
                if r.get("overlap_mismatch"):
                    res["errors"][key] = ("overlap_mismatch: %d stored funding event(s) disagreed with the "
                                          "exchange; store re-fetched and replaced" % r["overlap_mismatch"])
            except Exception as e:  # noqa: BLE001
                res["errors"][key] = "%s: %s" % (type(e).__name__, e)
    return res


def verify(coins: Optional[List[str]] = None, tfs: Optional[List[str]] = None, source: str = "spot") -> dict:
    """Full-history check of the kline store against the exchange; any difference replaces the file and is
    reported in `errors`."""
    if source not in SOURCES:
        raise DataError("unsupported source %r" % source)
    cfg = config.load_config()
    coins = list(coins) if coins else list(cfg["universe"])
    tfs = list(tfs) if tfs else list(cfg["timeframes"])
    res = {"source": source, "verified": {}, "errors": {}}
    for coin in coins:
        for tf in tfs:
            key = "%s_%s" % (coin, tf)
            try:
                res["verified"][key] = r = verify_one(coin, tf, source)
                if r["changed"] or r["dropped"]:
                    res["errors"][key] = "store disagreed with the exchange and was replaced: %s" % r
            except Exception as e:  # noqa: BLE001
                res["errors"][key] = "%s: %s" % (type(e).__name__, e)
    return res


# ── seeding ──────────────────────────────────────────────────────────────────
_PKG_ROOT = Path(__file__).resolve().parents[2]  # directory that contains the `trader` package


def _forbidden_seed_roots() -> List[Path]:
    roots = [_PKG_ROOT, config.TRADER_ROOT, config.REPO_ROOT, config.strategy_dir(), config.journal_dir(),
             config.data_dir(None)]
    out = []
    for r in roots:
        try:
            out.append(Path(r).expanduser().resolve())
        except Exception:  # noqa: BLE001
            pass
    return out


def _inside(p: Path, root: Path) -> bool:
    return p == root or root in p.parents


def _seed_source(path: Optional[str]) -> Path:
    """The protected data.seed_dir, resolved. `path`, if given, must resolve to that same directory."""
    cfg_dir = Path(config.load_config()["data"]["seed_dir"]).expanduser().resolve()
    if path is not None and Path(path).expanduser().resolve() != cfg_dir:
        raise DataError("seed source must be the protected data.seed_dir (%s), not %s" % (cfg_dir, path))
    for root in _forbidden_seed_roots():
        if _inside(cfg_dir, root) or _inside(root, cfg_dir):
            raise DataError("seed dir %s overlaps %s (repo / TRADER_ROOT / strategy / journal / data store)"
                            % (cfg_dir, root))
    if not cfg_dir.is_dir():
        raise DataError("seed dir not found: %s" % cfg_dir)
    return cfg_dir


def check_klines(df: pd.DataFrame, tf: str, now_ms: Optional[int] = None) -> List[str]:
    """Problems of a kline frame read from CSV ([] = ok): columns, sorted unique closed bars on the tf grid,
    nothing in the future, finite positive prices with low <= open,close <= high, non-negative volume."""
    p = []
    if list(df.columns) != COLS:
        return ["columns %s != %s" % (list(df.columns), COLS)]
    idx = df.index
    if not isinstance(idx, pd.DatetimeIndex):
        return ["index is not a datetime index"]
    if len(idx) == 0:
        return ["no rows"]
    if not idx.is_monotonic_increasing or idx.has_duplicates:
        p.append("index not sorted/unique")
    if (idx != idx.floor(TF_DELTA[tf])).any():
        p.append("bar times off the %s grid" % tf)
    now = pd.Timestamp(now_ms if now_ms is not None else _now_ms(), unit="ms")
    if (idx + TF_DELTA[tf] > now).any():
        p.append("bars not closed yet / dated in the future (last %s)" % idx.max())
    v = df.astype(float)
    if not pd.notna(v.values).all() or not (abs(v.values) < float("inf")).all():
        p.append("non-finite values")
    o, h, l, c = v.open, v.high, v.low, v.close
    # Bybit listing bars can carry open=low=0 (e.g. linear BCH 2020-12-14), so only close/high must be > 0
    if (v[["open", "high", "low", "close"]] < 0).any().any() or (c <= 0).any() or (h <= 0).any():
        p.append("negative or zero prices")
    if ((h < o) | (h < c) | (l > o) | (l > c) | (l > h)).any():
        p.append("OHLC inconsistent (low <= open,close <= high violated)")
    if (v[["volume", "turnover"]] < 0).any().any():
        p.append("negative volume/turnover")
    return p


def check_funding(s: pd.Series, now_ms: Optional[int] = None) -> List[str]:
    p = []
    if len(s) == 0:
        return ["no rows"]
    if not isinstance(s.index, pd.DatetimeIndex):
        return ["index is not a datetime index"]
    if not s.index.is_monotonic_increasing or s.index.has_duplicates:
        p.append("index not sorted/unique")
    now = pd.Timestamp(now_ms if now_ms is not None else _now_ms(), unit="ms")
    if (s.index > now).any():
        p.append("events dated in the future (last %s)" % s.index.max())
    if not pd.notna(s.values).all() or (s.abs() > 0.1).any():
        p.append("non-finite or implausible funding rates (|rate| > 10%)")
    return p


def seed_from(path: Optional[str] = None, source: str = "linear") -> int:
    """Copy (never symlink) lab CSVs `<COIN>_<tf>.csv` / `<COIN>_funding.csv` from the protected data.seed_dir
    into <data root>/<source>. `path`, if given, must resolve to data.seed_dir (kept for API compatibility).
    Every candidate file is validated first; if any is invalid nothing is copied (DataError). Existing
    destination files are left untouched (seeding never rewinds an updated store)."""
    if source not in SOURCES:
        raise DataError("unsupported source %r" % source)
    src = _seed_source(path)
    dst = config.data_dir(source)
    forbidden = _forbidden_seed_roots()
    plan, bad = [], []
    for f in sorted(src.glob("*.csv")):
        stem = f.stem
        if "_" not in stem:
            continue
        kind = stem.rsplit("_", 1)[1]
        if kind == "funding":
            target = config.data_dir("linear") / f.name
        elif kind in TF_INTERVAL:
            target = dst / f.name
        else:
            continue
        if target.exists():
            continue
        real = f.resolve()
        if any(_inside(real, r) for r in forbidden):
            bad.append("%s -> %s is inside the repo / data store" % (f.name, real))
            continue
        try:
            if kind == "funding":
                probs = check_funding(read_funding(real))
            else:
                probs = check_klines(read_klines(real), kind)
        except Exception as e:  # noqa: BLE001
            probs = ["unreadable: %s: %s" % (type(e).__name__, e)]
        if probs:
            bad.append("%s: %s" % (f.name, "; ".join(probs)))
        else:
            plan.append((real, target))
    if bad:
        raise DataError("seed refused, invalid files (nothing copied): " + " | ".join(bad))
    n = 0
    for real, target in plan:
        dst_dir = target.parent
        dst_dir.mkdir(parents=True, exist_ok=True)
        _atomic_write(lambda p, real=real: shutil.copyfile(str(real), str(p)), target)
        n += 1
    return n


# ── loading (INTERNAL) ───────────────────────────────────────────────────────
_RAW_CACHE: Dict[tuple, tuple] = {}


def _mtime(p: Path):
    try:
        st = p.stat()
        return (st.st_mtime_ns, st.st_size)
    except FileNotFoundError:
        return None


def per_bar_funding(f: pd.Series, index: pd.DatetimeIndex, tf: str) -> pd.Series:
    """worker.py rule: an event at time t belongs to bar (t - 1ms).floor(tf); summed; 0.0 where none."""
    if f is None or len(f) == 0:
        return pd.Series(0.0, index=index, name="funding")
    key = (f.index - pd.Timedelta(milliseconds=1)).floor(TF_DELTA[tf])
    out = f.groupby(key).sum().reindex(index).fillna(0.0)
    out.name = "funding"
    return out


def load_raw(coin: str, tf: str, source: str) -> Tuple[pd.DataFrame, pd.Series]:
    """INTERNAL — only vault.py may call this. Full stored history (no cutoff!) + per-bar funding
    (0.0 for spot or when no funding file exists). Returns copies; cached by file mtime."""
    if tf not in TF_INTERVAL:
        raise DataError("unsupported tf %r" % tf)
    if source not in SOURCES:
        raise DataError("unsupported source %r" % source)
    kp = kline_path(coin, tf, source)
    if not kp.exists():
        raise DataError("no data file %s (run data.update or data.seed_from)" % kp)
    fp = funding_path(coin) if source == "linear" else None
    key = (str(kp), tf, source)
    stamp = (_mtime(kp), _mtime(fp) if fp is not None else None)
    hit = _RAW_CACHE.get(key)
    if hit is None or hit[0] != stamp:
        df = read_klines(kp, exact=False)
        df = df[~df.index.duplicated(keep="first")].sort_index()
        missing = [c for c in COLS if c not in df.columns]
        if missing:
            raise DataError("%s lacks columns %s" % (kp, missing))
        df = df[COLS].astype(float)
        f = read_funding(fp, exact=False) if fp is not None and fp.exists() else None
        fund = per_bar_funding(f, df.index, tf)
        _RAW_CACHE[key] = (stamp, df, fund)
        hit = _RAW_CACHE[key]
    return hit[1].copy(), hit[2].copy()


def load(*args, **kwargs):
    """Re-export of the public loader `vault.load` (lazy to avoid a circular import)."""
    from trader.core import vault
    return vault.load(*args, **kwargs)


# ── CLI ──────────────────────────────────────────────────────────────────────
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m trader.core.data")
    sub = ap.add_subparsers(dest="cmd")
    u = sub.add_parser("update")
    u.add_argument("--source", default="spot", choices=SOURCES)
    u.add_argument("--coins", default=None)
    u.add_argument("--tfs", default=None)
    u.add_argument("--no-funding", action="store_true")
    v = sub.add_parser("verify")
    v.add_argument("--source", default="spot", choices=SOURCES)
    v.add_argument("--coins", default=None)
    v.add_argument("--tfs", default=None)
    s = sub.add_parser("seed")  # source is always the protected data.seed_dir (no --path)
    s.add_argument("--source", default="linear", choices=SOURCES)
    a = ap.parse_args(argv)
    try:
        if a.cmd == "update":
            res = update(a.coins.split(",") if a.coins else None, a.tfs.split(",") if a.tfs else None,
                         source=a.source, funding=not a.no_funding)
            print(json.dumps(res, sort_keys=True))
            return 1 if res["errors"] else 0
        if a.cmd == "verify":
            res = verify(a.coins.split(",") if a.coins else None, a.tfs.split(",") if a.tfs else None,
                         source=a.source)
            print(json.dumps(res, sort_keys=True))
            return 1 if res["errors"] else 0
        if a.cmd == "seed":
            n = seed_from(None, a.source)
            print(json.dumps({"source": a.source, "copied": n, "dir": str(config.data_dir(a.source))}))
            return 0
    except DataError as e:
        print(json.dumps({"error": str(e)}))
        return 1
    ap.print_help(sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
