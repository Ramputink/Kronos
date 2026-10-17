"""Market watchers (CONTRACTS.md §9 "Watchers"). Runs on berry every 15 min. Python >= 3.9, stdlib + requests.

Reads ``strategy/watchlist.json``, fetches the latest Bybit SPOT candles (public v5 API, no key) for the coins its
rules name, computes each rule's metric, and when a rule fires:

* appends a row ``{"ts","rule_id","coin","metric","value","threshold","action","host"}`` to
  ``$PHIL_OUTBOX/watch-triggers-<host>.jsonl`` (``fleet/inbox.py merge`` on air copies new rows into
  ``journal/watch-triggers.jsonl``, which this module never writes);
* for ``action == "trigger_forward"`` also writes ``$PHIL_OUTBOX/trigger-<YYYYmmddTHHMMSSZ>-<rule_id>.json``
  (the same object). ``fleet/health.py push`` (ExecStartPost of phil-watch.service, or ``--push``) moves it into
  air's inbox, and loop.sh then runs a ``triggered`` cycle (settle / retro / pause-resume only).

Suppression / dedupe, all kept in ``~/.phil-crypto/watch-state.json`` (``$PHIL_WATCH_STATE``):

* per-rule ``cooldown_min``: a rule that fired is silent until the cooldown has elapsed;
* ``cross_above`` / ``cross_below`` ops only fire on the run where the metric crosses the threshold relative to the
  value seen on the previous run (an RSI that sits above 55 for days fires once, not every 15 min);
* a global ``max_trigger_files_per_day`` (watchlist, default 4): past it a ``trigger_forward`` is logged as a
  ``note`` with a ``detail`` so a volatile day cannot launch an unbounded number of Claude cycles.

Metrics (contract closed set): ``price``, ``ret_1h``, ``ret_4h``, ``ret_24h`` (15m bars; last = forming bar's close
vs the close k bars earlier), ``vol_ratio_24h_30d`` (std of the last 24 closed 1h log returns / std of the last 720),
``dist_from_ema200_1d`` (price / EMA200 of closed daily closes − 1). Extensions (pending contract amendment):
``rsi14_1d`` (Wilder RSI(14) of the closed daily closes plus the forming day's price, i.e. the same RSI the rsi_reg
strategy will see at 00:10 if the price stays here; formula identical to research engine.rsi) and
``dist_from_high50_1d`` (price / max high of the previous 50 closed days − 1; > 0 ⇒ new 50-day high).
A metric whose bars are stale (newest bar older than 2 intervals), gapped or too short is skipped and reported,
never guessed.

CLI (prints one JSON object): ``python -m trader.fleet.watch run [--dry-run] [--push] [--watchlist P]``
| ``check [--watchlist P]`` (validate only) | ``show [--coins BTC,ETH]`` (metrics only, writes nothing).
Exit 0 = ok / partial, 1 = no metric could be computed or error, 2 = invalid watchlist.
Environment: ``PHIL_OUTBOX``, ``PHIL_WATCH_STATE``, ``TRADER_BYBIT_API`` (else protected ``data.api``),
``TRADER_STRATEGY_DIR``.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from trader.core import config  # noqa: E402

try:
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None


class WatchError(Exception):
    """Invalid watchlist or unusable market data."""


CORE_METRICS = ("price", "ret_1h", "ret_4h", "ret_24h", "vol_ratio_24h_30d", "dist_from_ema200_1d")
EXT_METRICS = ("rsi14_1d", "dist_from_high50_1d")
METRICS = CORE_METRICS + EXT_METRICS
OPS = ("<", "<=", ">", ">=", "cross_above", "cross_below")
ACTIONS = ("note", "trigger_forward")
DEFAULT_MAX_TRIGGERS_DAY = 4

INTERVAL_MS = {"15": 15 * 60_000, "60": 3_600_000, "D": 86_400_000}
# metric -> (interval, bars requested)
METRIC_SOURCE = {
    "price": ("15", 200), "ret_1h": ("15", 200), "ret_4h": ("15", 200), "ret_24h": ("15", 200),
    "vol_ratio_24h_30d": ("60", 1000),
    "dist_from_ema200_1d": ("D", 1000), "rsi14_1d": ("D", 1000), "dist_from_high50_1d": ("D", 1000),
}
RET_HOURS = {"ret_1h": 1, "ret_4h": 4, "ret_24h": 24}
ID_RE = re.compile(r"^[a-z0-9_]{1,64}$")
COIN_RE = re.compile(r"^[A-Z0-9]{2,12}$")

# Bar = (open_ms, open, high, low, close), ascending by open time.
Bar = Tuple[int, float, float, float, float]


# ── paths ────────────────────────────────────────────────────────────────────
def outbox_dir() -> Path:
    return Path(os.environ.get("PHIL_OUTBOX", "~/.phil-crypto/outbox")).expanduser()


def state_path() -> Path:
    return Path(os.environ.get("PHIL_WATCH_STATE", "~/.phil-crypto/watch-state.json")).expanduser()


def watchlist_path() -> Path:
    return config.strategy_dir() / "watchlist.json"


# ── watchlist ────────────────────────────────────────────────────────────────
def validate_watchlist(wl: dict) -> List[dict]:
    """Return the enabled rules; raise WatchError on any schema problem (fail closed: no partial watchlists)."""
    if not isinstance(wl, dict) or wl.get("version") != 1 or not isinstance(wl.get("rules"), list):
        raise WatchError("watchlist must be {\"version\":1,\"rules\":[...]}")
    mt = wl.get("max_trigger_files_per_day", DEFAULT_MAX_TRIGGERS_DAY)
    if not isinstance(mt, int) or isinstance(mt, bool) or not 0 <= mt <= 24:
        raise WatchError("max_trigger_files_per_day must be an int in [0, 24]")
    seen, out = set(), []
    for i, r in enumerate(wl["rules"]):
        where = "rule %d (%s)" % (i, r.get("id") if isinstance(r, dict) else "?")
        if not isinstance(r, dict):
            raise WatchError("%s: not an object" % where)
        missing = [k for k in ("id", "coin", "metric", "op", "value", "cooldown_min", "action") if k not in r]
        if missing:
            raise WatchError("%s: missing %s" % (where, ",".join(missing)))
        if not isinstance(r["id"], str) or not ID_RE.match(r["id"]):
            raise WatchError("%s: id must match %s" % (where, ID_RE.pattern))
        if r["id"] in seen:
            raise WatchError("%s: duplicate id" % where)
        seen.add(r["id"])
        if not isinstance(r["coin"], str) or not COIN_RE.match(r["coin"]):
            raise WatchError("%s: bad coin %r" % (where, r["coin"]))
        if r["metric"] not in METRICS:
            raise WatchError("%s: metric %r not in %s" % (where, r["metric"], METRICS))
        if r["op"] not in OPS:
            raise WatchError("%s: op %r not in %s" % (where, r["op"], OPS))
        if r["action"] not in ACTIONS:
            raise WatchError("%s: action %r not in %s" % (where, r["action"], ACTIONS))
        v = r["value"]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            raise WatchError("%s: value must be a finite number" % where)
        cd = r["cooldown_min"]
        if isinstance(cd, bool) or not isinstance(cd, (int, float)) or not 15 <= cd <= 7 * 1440:
            raise WatchError("%s: cooldown_min must be in [15, 10080]" % where)
        if r.get("enabled", True) is not False:
            out.append(r)
    return out


def load_watchlist(path=None) -> dict:
    p = Path(path) if path else watchlist_path()
    try:
        wl = json.loads(p.read_text())
    except (OSError, ValueError) as e:
        raise WatchError("cannot read watchlist %s: %s" % (p, e))
    validate_watchlist(wl)
    return wl


# ── market data ──────────────────────────────────────────────────────────────
def _api() -> str:
    return os.environ.get("TRADER_BYBIT_API") or config.load_config()["data"]["api"]


def bybit_fetch(coin: str, interval: str, limit: int, *, category: str = "spot", retries: int = 3) -> List[Bar]:
    """Latest `limit` klines (forming bar included), ascending. Raises WatchError after `retries` attempts."""
    import requests  # lazy: tests inject a fake fetch

    url = "%s/kline" % _api().rstrip("/")
    params = {"category": category, "symbol": "%sUSDT" % coin, "interval": interval, "limit": int(limit)}
    last = None
    for a in range(retries):
        try:
            r = requests.get(url, params=params, timeout=10).json()
            if r.get("retCode") != 0:
                raise WatchError("bybit kline %s %s: %s" % (coin, interval, r.get("retMsg")))
            rows = (r.get("result") or {}).get("list") or []
            return parse_klines(rows)
        except Exception as e:  # noqa: BLE001 - network/JSON/API: retry
            last = e
            if a < retries - 1:
                time.sleep(1 + 2 * a)
    raise WatchError("bybit kline %s %s failed after %d attempts: %s" % (coin, interval, retries, last))


def parse_klines(rows) -> List[Bar]:
    """Bybit list rows ([startMs, o, h, l, c, vol, turnover] strings, newest first) → ascending unique Bars."""
    out = {}
    for x in rows:
        t = int(x[0])
        out[t] = (t, float(x[1]), float(x[2]), float(x[3]), float(x[4]))
    return [out[t] for t in sorted(out)]


class Market:
    """Lazy per-(coin, interval) cache so each run fetches each series at most once."""

    def __init__(self, fetch: Callable[[str, str, int], List[Bar]], now_ms: int):
        self.fetch, self.now_ms = fetch, now_ms
        self._c: Dict[Tuple[str, str], object] = {}

    def bars(self, coin: str, interval: str, limit: int) -> List[Bar]:
        k = (coin, interval)
        if k not in self._c:
            try:
                self._c[k] = self.fetch(coin, interval, limit)
            except Exception as e:  # noqa: BLE001 - stored so the coin's other metrics fail fast, once
                self._c[k] = e
        v = self._c[k]
        if isinstance(v, Exception):
            raise WatchError(str(v))
        return v


def _split(bars: List[Bar], interval: str, now_ms: int) -> Tuple[List[Bar], Optional[Bar]]:
    """(closed bars, forming bar or None) after freshness/gap checks."""
    step = INTERVAL_MS[interval]
    if not bars:
        raise WatchError("no bars")
    if bars[-1][0] < now_ms - 2 * step:
        raise WatchError("stale: newest %s bar opened %s" % (interval, config.iso(_ms_dt(bars[-1][0]))))
    closed = [b for b in bars if b[0] + step <= now_ms]
    forming = bars[-1] if bars[-1][0] + step > now_ms else None
    return closed, forming


def _contiguous(bars: List[Bar], interval: str, n: int, what: str) -> List[Bar]:
    """Last n bars, required to be gap-free."""
    step = INTERVAL_MS[interval]
    if len(bars) < n:
        raise WatchError("%s: need %d %s bars, have %d" % (what, n, interval, len(bars)))
    tail = bars[-n:]
    for a, b in zip(tail, tail[1:]):
        if b[0] - a[0] != step:
            raise WatchError("%s: gap in %s bars at %s" % (what, interval, config.iso(_ms_dt(a[0]))))
    return tail


def _ms_dt(ms: int) -> _dt.datetime:
    return _dt.datetime.fromtimestamp(ms / 1000.0, tz=_dt.timezone.utc)


def ema_last(xs: List[float], n: int) -> float:
    """pandas ewm(span=n, adjust=False).mean() last value (seeded with the first element)."""
    a = 2.0 / (n + 1)
    e = xs[0]
    for x in xs[1:]:
        e = (1 - a) * e + a * x
    return e


def rsi_last(closes: List[float], n: int = 14) -> Optional[float]:
    """Last value of research engine.rsi (Wilder: ewm(alpha=1/n, adjust=False) of gains/losses).
    None where engine.rsi is NaN (no losses in the window)."""
    if len(closes) < 2:
        return None
    a = 1.0 / n
    up = dn = None
    for p, c in zip(closes, closes[1:]):
        d = c - p
        g, l_ = max(d, 0.0), max(-d, 0.0)
        if up is None:
            up, dn = g, l_
        else:
            up = (1 - a) * up + a * g
            dn = (1 - a) * dn + a * l_
    if not dn:
        return None
    return 100.0 - 100.0 / (1.0 + up / dn)


def _std(xs: List[float]) -> float:
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def compute_metric(metric: str, coin: str, mk: Market) -> float:
    """Value of one metric for one coin, or WatchError (stale, gapped, too short, fetch failed)."""
    interval, limit = METRIC_SOURCE[metric]
    bars = mk.bars(coin, interval, limit)
    closed, forming = _split(bars, interval, mk.now_ms)
    if metric == "price" or metric in RET_HOURS:
        series = closed + ([forming] if forming else [])
        price = series[-1][4]
        if metric == "price":
            return price
        k = RET_HOURS[metric] * 3_600_000 // INTERVAL_MS[interval]
        tail = _contiguous(series, interval, k + 1, metric)
        return price / tail[0][4] - 1.0
    if metric == "vol_ratio_24h_30d":
        tail = _contiguous(closed, interval, 721, metric)
        lr = [math.log(b[4] / a[4]) for a, b in zip(tail, tail[1:])]
        s30 = _std(lr)
        if s30 <= 0:
            raise WatchError("%s: zero 30d volatility" % metric)
        return _std(lr[-24:]) / s30
    # daily metrics: price = forming day's close (latest trade), else last close
    price = forming[4] if forming else closed[-1][4]
    if metric == "dist_from_ema200_1d":
        _contiguous(closed, interval, 200, metric)
        return price / ema_last([b[4] for b in closed], 200) - 1.0
    if metric == "rsi14_1d":
        _contiguous(closed, interval, 60, metric)
        v = rsi_last([b[4] for b in closed] + [price], 14)
        if v is None:
            raise WatchError("%s: undefined (no losses)" % metric)
        return v
    if metric == "dist_from_high50_1d":
        tail = _contiguous(closed, interval, 50, metric)
        return price / max(b[2] for b in tail) - 1.0
    raise WatchError("unknown metric %s" % metric)  # unreachable: validated


# ── rule evaluation ──────────────────────────────────────────────────────────
def condition(op: str, value: float, threshold: float, prev: Optional[float]) -> bool:
    if op == "<":
        return value < threshold
    if op == "<=":
        return value <= threshold
    if op == ">":
        return value > threshold
    if op == ">=":
        return value >= threshold
    if op == "cross_above":
        return prev is not None and prev <= threshold < value
    if op == "cross_below":
        return prev is not None and prev >= threshold > value
    raise WatchError("unknown op %s" % op)


def _parse_iso(s) -> Optional[_dt.datetime]:
    try:
        return _dt.datetime.strptime(str(s), "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=_dt.timezone.utc)
    except (TypeError, ValueError):
        return None


def evaluate(rules: List[dict], values: Dict[Tuple[str, str], float], state: dict, now: _dt.datetime, *,
             max_triggers_day: int = DEFAULT_MAX_TRIGGERS_DAY, host: Optional[str] = None) -> dict:
    """Pure rule evaluation. Mutates `state` (cooldowns, last values, trigger history) and returns
    {"fired": [rows], "trigger_rows": [rows needing a trigger file], "suppressed": [ids], "skipped": [ids]}."""
    host = host or config.runner_id()
    nowz = config.iso(now)
    st_rules = state.setdefault("rules", {})
    day_ago = now - _dt.timedelta(hours=24)
    hist = [t for t in state.get("trigger_files", []) if (_parse_iso(t) or now) > day_ago]
    res = {"fired": [], "trigger_rows": [], "suppressed": [], "skipped": []}
    for r in rules:
        rs = st_rules.setdefault(r["id"], {})
        if rs.get("metric") != r["metric"] or rs.get("coin") != r["coin"]:
            # rule redefined: forget the previous value (a cross must be seen on this metric) but keep cooldown
            rs.pop("last_value", None)
            rs["metric"], rs["coin"] = r["metric"], r["coin"]
        v = values.get((r["coin"], r["metric"]))
        if v is None:
            res["skipped"].append(r["id"])
            continue
        prev = rs.get("last_value")
        hit = condition(r["op"], v, float(r["value"]), prev)
        rs["last_value"], rs["last_eval"] = v, nowz
        if not hit:
            continue
        lf = _parse_iso(rs.get("last_fired"))
        if lf is not None and now < lf + _dt.timedelta(minutes=float(r["cooldown_min"])):
            res["suppressed"].append(r["id"])
            continue
        row = {"ts": nowz, "rule_id": r["id"], "coin": r["coin"], "metric": r["metric"],
               "value": round(float(v), 6), "threshold": r["value"], "action": r["action"], "host": host}
        if r["action"] == "trigger_forward":
            if len(hist) >= max_triggers_day:
                row["action"] = "note"
                row["detail"] = "trigger_forward capped: %d trigger files in 24h" % len(hist)
            else:
                hist.append(nowz)
                res["trigger_rows"].append(row)
        rs["last_fired"] = nowz
        rs["fires"] = int(rs.get("fires", 0)) + 1
        res["fired"].append(row)
    state["trigger_files"] = hist
    return res


def collect_values(rules: List[dict], mk: Market) -> Tuple[Dict[Tuple[str, str], float], List[str]]:
    values, errors = {}, []
    for coin, metric in sorted({(r["coin"], r["metric"]) for r in rules}):
        try:
            v = compute_metric(metric, coin, mk)
            if v is None or not math.isfinite(v):
                raise WatchError("non-finite value")
            values[(coin, metric)] = v
        except WatchError as e:
            errors.append("%s %s: %s" % (coin, metric, e))
    return values, errors


# ── state / outbox IO ────────────────────────────────────────────────────────
def load_state(path=None) -> dict:
    p = Path(path) if path else state_path()
    try:
        s = json.loads(p.read_text())
        return s if isinstance(s, dict) else {}
    except (OSError, ValueError):
        return {}


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(str(tmp), str(path))


def save_state(state: dict, path=None) -> Path:
    p = Path(path) if path else state_path()
    _atomic_write(p, json.dumps(state, indent=1, sort_keys=True) + "\n")
    return p


def trigger_name(row: dict) -> str:
    stamp = row["ts"].replace("-", "").replace(":", "")  # 2026-10-20T00:10:03Z → 20261020T001003Z
    return "trigger-%s-%s.json" % (stamp, row["rule_id"])


def emit(res: dict, host: str) -> dict:
    ob = outbox_dir()
    ob.mkdir(parents=True, exist_ok=True)
    log = ob / ("watch-triggers-%s.jsonl" % host)
    for row in res["fired"]:
        config.jsonl_append(log, row)
    files = []
    for row in res["trigger_rows"]:
        p = ob / trigger_name(row)
        _atomic_write(p, json.dumps(row, sort_keys=True) + "\n")
        files.append(str(p))
    return {"log": str(log), "trigger_files": files}


# ── run ──────────────────────────────────────────────────────────────────────
def run(*, watchlist=None, fetch: Optional[Callable] = None, now: Optional[_dt.datetime] = None,
        dry_run: bool = False, push: bool = False) -> dict:
    wl = watchlist if isinstance(watchlist, dict) else load_watchlist(watchlist)
    rules = validate_watchlist(wl)
    now = now or config.utcnow()
    host = config.runner_id()
    sp = state_path()
    sp.parent.mkdir(parents=True, exist_ok=True)
    lockf = open(str(sp) + ".lock", "a")
    try:
        if fcntl:
            try:
                fcntl.flock(lockf, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                return {"ok": True, "skipped": "another watch run holds the lock"}
        mk = Market(fetch or bybit_fetch, int(now.timestamp() * 1000))
        values, errors = collect_values(rules, mk)
        state = load_state()
        res = evaluate(rules, values, state, now, host=host,
                       max_triggers_day=int(wl.get("max_trigger_files_per_day", DEFAULT_MAX_TRIGGERS_DAY)))
        out = {"ok": bool(values) or not rules, "ts": config.iso(now), "host": host, "dry_run": dry_run,
               "n_rules": len(rules), "n_values": len(values),
               "values": {"%s.%s" % k: round(v, 6) for k, v in sorted(values.items())},
               "fired": res["fired"], "suppressed": res["suppressed"], "skipped": res["skipped"],
               "errors": errors}
        if dry_run:
            out["would_write_trigger_files"] = [trigger_name(r) for r in res["trigger_rows"]]
            return out
        out.update(emit(res, host))
        state.update({"version": 1, "last_run": config.iso(now), "host": host, "errors": errors[-50:]})
        if out["ok"]:
            state["last_ok"] = config.iso(now)
        save_state(state)
        if push:
            from trader.fleet import health
            out["push"] = health.push()
        return out
    finally:
        if fcntl:
            try:
                fcntl.flock(lockf, fcntl.LOCK_UN)
            except OSError:
                pass
        lockf.close()


def show(coins: Optional[List[str]] = None, fetch: Optional[Callable] = None) -> dict:
    now = config.utcnow()
    mk = Market(fetch or bybit_fetch, int(now.timestamp() * 1000))
    rules = [{"coin": c, "metric": m} for c in (coins or ["BTC"]) for m in METRICS]
    values, errors = collect_values(rules, mk)
    return {"ok": bool(values), "ts": config.iso(now),
            "values": {"%s.%s" % k: round(v, 6) for k, v in sorted(values.items())}, "errors": errors}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m trader.fleet.watch")
    ap.add_argument("cmd", choices=["run", "check", "show"], nargs="?", default="run")
    ap.add_argument("--watchlist", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--push", action="store_true", help="also run fleet.health push after writing")
    ap.add_argument("--coins", default=None)
    a = ap.parse_args(argv)
    try:
        if a.cmd == "check":
            wl = load_watchlist(a.watchlist)
            res = {"ok": True, "n_rules": len(validate_watchlist(wl)),
                   "extension_metrics_used": sorted({r["metric"] for r in wl["rules"]} & set(EXT_METRICS))}
        elif a.cmd == "show":
            res = show(a.coins.split(",") if a.coins else None)
        else:
            res = run(watchlist=a.watchlist, dry_run=a.dry_run, push=a.push)
    except WatchError as e:
        print(json.dumps({"ok": False, "invalid_watchlist": True, "error": str(e)}))
        return 2
    except Exception as e:  # CLI boundary
        print(json.dumps({"ok": False, "error": "%s: %s" % (type(e).__name__, e)}))
        return 1
    print(json.dumps(res, default=str))
    return 0 if res.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
