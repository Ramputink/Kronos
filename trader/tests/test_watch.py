"""Watchers tests (CONTRACTS.md §9 "Watchers"): watchlist validation, metrics on recorded candles (checked against
the pandas formulas of research/kronos_lab/indicators/engine.py), rule logic, suppression (cooldown, cross,
daily trigger cap), outbox/trigger files and the round trip through fleet/inbox.py.

Offline (fake fetch, no network). Run: /Users/matveypro/.venvs/kronos/bin/python trader/tests/test_watch.py
"""
import bisect
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

TMP = Path(tempfile.mkdtemp(prefix="phil-watch-test-"))
os.environ["TRADER_JOURNAL_DIR"] = str(TMP / "journal")
os.environ["TRADER_STRATEGY_DIR"] = str(TMP / "strategy")
os.environ["PHIL_OUTBOX"] = str(TMP / "outbox")
os.environ["PHIL_INBOX"] = str(TMP / "inbox")
os.environ["PHIL_WATCH_STATE"] = str(TMP / "state" / "watch-state.json")
os.environ["PHIL_RUNNER"] = "berry"
os.environ["TRADER_BYBIT_API"] = "http://127.0.0.1:9/never"  # any accidental network call fails fast

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from trader.core import config  # noqa: E402
from trader.fleet import inbox, watch  # noqa: E402

LAB = Path("/Users/matveypro/kronos_lab/indicators/data")
REPO = Path(__file__).resolve().parents[2]
UTC = dt.timezone.utc
H = 3_600_000
D = 86_400_000
M15 = 900_000


def reset():
    for p in (TMP / "outbox", TMP / "inbox", TMP / "state", TMP / "journal"):
        shutil.rmtree(str(p), ignore_errors=True)


def load_csv(coin, tf):
    df = pd.read_csv(LAB / ("%s_%s.csv" % (coin, tf)), parse_dates=["time"])
    ms = (df.time.values.astype("datetime64[ms]").astype("int64")).tolist()
    return [(int(t), float(o), float(h), float(l_), float(c))
            for t, o, h, l_, c in zip(ms, df.open, df.high, df.low, df.close)]


def ms(ts):
    return int(pd.Timestamp(ts).tz_localize(None).value // 1_000_000)


# engine.py reference formulas (research/kronos_lab/indicators/engine.py, copied verbatim)
def ref_ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def ref_rsi(c, n):
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


BTC_D = load_csv("BTC", "1d")
BTC_4H = load_csv("BTC", "4h")


def step_15m(bars4h):
    """15m bars from recorded 4h bars: each 4h bar becomes 16 flat 15m bars at its close (step function)."""
    out = []
    for t, o, h, l_, c in bars4h:
        for i in range(16):
            out.append((t + i * M15, c, c, c, c))
    return out


_KEYS = {}  # id(bars list) -> open times (bars lists live for the whole test run)


def make_fetch(series, now_ms, *, forming_price=None, calls=None):
    """Fake bybit_fetch over recorded bars: returns the last `limit` bars with open <= now (the last one is the
    forming bar when open + interval > now). `series` = {interval: bars}; same series for every coin."""
    def fetch(coin, interval, limit):
        if calls is not None:
            calls.append((coin, interval))
        bars = series[interval]
        keys = _KEYS.get(id(bars))
        if keys is None:
            keys = _KEYS[id(bars)] = [b[0] for b in bars]
        i = bisect.bisect_right(keys, now_ms)
        out = list(bars[max(0, i - limit):i])
        if forming_price is not None and out and out[-1][0] + watch.INTERVAL_MS[interval] > now_ms:
            t, o, h, l_, _ = out[-1]
            out[-1] = (t, o, max(h, forming_price), min(l_, forming_price), forming_price)
        return out
    return fetch


# ── 1. watchlist validation ─────────────────────────────────────────────────
def test_watchlist():
    wl = json.loads((REPO / "trader" / "strategy" / "watchlist.json").read_text())
    rules = watch.validate_watchlist(wl)
    assert len(rules) == len(wl["rules"]) >= 10
    assert {r["metric"] for r in rules} <= set(watch.METRICS)
    assert any(r["action"] == "trigger_forward" for r in rules)
    assert all(r["coin"] in config.load_config()["universe"] for r in rules)
    # contract example rules validate
    ex = {"version": 1, "rules": [
        {"id": "btc_drop_6pct_24h", "coin": "BTC", "metric": "ret_24h", "op": "<", "value": -0.06,
         "cooldown_min": 360, "action": "trigger_forward"},
        {"id": "eth_vol_spike", "coin": "ETH", "metric": "vol_ratio_24h_30d", "op": ">", "value": 2.5,
         "cooldown_min": 720, "action": "note"}]}
    assert len(watch.validate_watchlist(ex)) == 2
    base = ex["rules"][0]
    bad = [dict(base, metric="rsi_99"), dict(base, op="=="), dict(base, action="buy"), dict(base, value="x"),
           dict(base, value=float("nan")), dict(base, cooldown_min=0), dict(base, coin="btc"),
           dict(base, id="Bad Id"), {k: v for k, v in base.items() if k != "cooldown_min"}]
    for b in bad:
        try:
            watch.validate_watchlist({"version": 1, "rules": [b]})
        except watch.WatchError:
            continue
        raise AssertionError("accepted bad rule %r" % b)
    try:
        watch.validate_watchlist({"version": 1, "rules": [base, base]})
        raise AssertionError("duplicate id accepted")
    except watch.WatchError:
        pass
    try:
        watch.validate_watchlist({"version": 2, "rules": []})
        raise AssertionError("bad version accepted")
    except watch.WatchError:
        pass
    assert watch.validate_watchlist({"version": 1, "rules": [dict(base, enabled=False)]}) == []
    print("ok watchlist")


# ── 2. metrics on recorded candles vs pandas reference ──────────────────────
def test_metrics_recorded():
    df = pd.DataFrame(BTC_D, columns=["t", "o", "h", "l", "c"])
    for asof in ("2021-05-19", "2022-11-09", "2024-03-05", "2025-01-15"):
        day = ms(asof)
        now = day + 13 * H  # 13:00 UTC: the asof day is the forming bar
        price = df.c[df.t == day].iloc[0] * 0.97  # forming price differs from the recorded close
        mk = watch.Market(make_fetch({"D": BTC_D}, now, forming_price=price), now)
        closed = df[df.t < day].tail(999)  # limit 1000 incl. forming
        # rsi on closed closes + forming price
        ref = ref_rsi(pd.concat([closed.c, pd.Series([price])], ignore_index=True), 14).iloc[-1]
        got = watch.compute_metric("rsi14_1d", "BTC", mk)
        assert abs(got - ref) < 1e-9, (asof, got, ref)
        ref = price / ref_ema(closed.c, 200).iloc[-1] - 1
        got = watch.compute_metric("dist_from_ema200_1d", "BTC", mk)
        assert abs(got - ref) < 1e-12, (asof, got, ref)
        ref = price / closed.h.tail(50).max() - 1
        got = watch.compute_metric("dist_from_high50_1d", "BTC", mk)
        assert abs(got - ref) < 1e-12, (asof, got, ref)
    # rsi_last == engine.rsi on the full series (no forming bar), many endpoints
    c = df.c
    r = ref_rsi(c, 14)
    for i in (20, 100, 500, len(c) - 1):
        assert abs(watch.rsi_last(c.iloc[:i + 1].tolist(), 14) - r.iloc[i]) < 1e-9
    assert watch.rsi_last([1, 2, 3, 4, 5], 14) is None  # engine: NaN when no losses

    # returns on 15m bars: built from recorded 4h closes (step), checked against the 4h series
    b15 = step_15m(BTC_4H)
    df4 = pd.DataFrame(BTC_4H, columns=["t", "o", "h", "l", "c"]).set_index("t")
    t0 = ms("2021-05-19 12:00")
    now = t0 + 4 * H - 1  # last 15m bar of the 08:00..12:00 4h block is forming... use the block of t0
    mk = watch.Market(make_fetch({"15": b15}, now), now)
    price = watch.compute_metric("price", "BTC", mk)
    assert price == df4.c[t0]
    r24 = watch.compute_metric("ret_24h", "BTC", mk)
    # 96 15m bars back from the forming bar = the 15m bar 24h earlier = inside the 4h block t0-24h
    assert abs(r24 - (df4.c[t0] / df4.c[t0 - 24 * H] - 1)) < 1e-12, r24
    r4 = watch.compute_metric("ret_4h", "BTC", mk)
    assert abs(r4 - (df4.c[t0] / df4.c[t0 - 4 * H] - 1)) < 1e-12
    assert watch.compute_metric("ret_1h", "BTC", mk) == 0.0  # flat inside the 4h block
    assert r24 < -0.06, "2021-05-19 crash should be a >6% 24h drop, got %.3f" % r24

    # vol ratio on 1h bars (4h closes relabelled hourly): reference with numpy
    b1h = [(ms("2023-01-01") + i * H,) + b[1:] for i, b in enumerate(BTC_4H[-1200:])]
    now = b1h[-1][0] + 30 * 60_000  # last bar forming
    mk = watch.Market(make_fetch({"60": b1h}, now), now)
    closes = np.array([b[4] for b in b1h[:-1]][-721:])
    lr = np.diff(np.log(closes))
    ref = lr[-24:].std(ddof=1) / lr.std(ddof=1)
    got = watch.compute_metric("vol_ratio_24h_30d", "BTC", mk)
    assert abs(got - ref) < 1e-9, (got, ref)
    print("ok metrics_recorded")


# ── 3. data guards: stale, gap, short, fetch failure ────────────────────────
def test_guards():
    day = ms("2024-03-05")
    now = day + 13 * H
    stale_now = day + 5 * D
    mk = watch.Market(make_fetch({"D": BTC_D}, day + 13 * H), stale_now)  # bars end 5 days before "now"
    try:
        watch.compute_metric("rsi14_1d", "BTC", mk)
        raise AssertionError("stale accepted")
    except watch.WatchError as e:
        assert "stale" in str(e)
    gapped = [b for b in BTC_D if b[0] != day - 10 * D]
    mk = watch.Market(make_fetch({"D": gapped}, now), now)
    try:
        watch.compute_metric("dist_from_high50_1d", "BTC", mk)
        raise AssertionError("gap accepted")
    except watch.WatchError as e:
        assert "gap" in str(e)
    short = BTC_D[:40]
    n2 = short[-1][0] + 2 * H
    mk = watch.Market(make_fetch({"D": short}, n2), n2)
    try:
        watch.compute_metric("dist_from_ema200_1d", "BTC", mk)
        raise AssertionError("short accepted")
    except watch.WatchError as e:
        assert "need 200" in str(e)

    calls = []

    def flaky(coin, interval, limit):
        calls.append((coin, interval))
        if coin == "ETH":
            raise watch.WatchError("boom")
        return make_fetch({"D": BTC_D}, now)(coin, interval, limit)
    rules = [{"coin": c, "metric": m} for c in ("BTC", "ETH") for m in ("rsi14_1d", "dist_from_high50_1d")]
    vals, errs = watch.collect_values(rules, watch.Market(flaky, now))
    assert set(vals) == {("BTC", "rsi14_1d"), ("BTC", "dist_from_high50_1d")}
    assert len(errs) == 2 and all("ETH" in e for e in errs)
    assert calls.count(("ETH", "D")) == 1 and calls.count(("BTC", "D")) == 1  # cached per (coin, interval)
    print("ok guards")


# ── 4. rule logic + suppression (pure evaluate) ─────────────────────────────
def T(minutes):
    return dt.datetime(2026, 10, 20, tzinfo=UTC) + dt.timedelta(minutes=minutes)


def test_evaluate_threshold_cooldown():
    rule = {"id": "btc_drop", "coin": "BTC", "metric": "ret_24h", "op": "<", "value": -0.06,
            "cooldown_min": 60, "action": "note"}
    st = {}
    seq = [-0.01, -0.07, -0.08, -0.09, -0.07, -0.02, -0.07, -0.10]  # every 15 min
    fired = []
    for i, v in enumerate(seq):
        res = watch.evaluate([rule], {("BTC", "ret_24h"): v}, st, T(15 * i), host="berry")
        fired += [(i, r["value"]) for r in res["fired"]]
    # fires at i=1; i=2,3,4 inside 60 min cooldown; i=5 condition false; i=6 = 75 min after → fires again
    assert [i for i, _ in fired] == [1, 6, ], fired
    row = watch.evaluate([rule], {("BTC", "ret_24h"): -0.2}, st, T(15 * 6 + 60), host="berry")["fired"][0]
    assert set(row) == {"ts", "rule_id", "coin", "metric", "value", "threshold", "action", "host"}
    assert row["host"] == "berry" and row["threshold"] == -0.06 and row["ts"] == "2026-10-20T02:30:00Z"
    # missing value → skipped, never fired
    res = watch.evaluate([rule], {}, st, T(1000))
    assert res["skipped"] == ["btc_drop"] and not res["fired"]
    print("ok evaluate_threshold_cooldown")


def test_evaluate_cross():
    on = {"id": "rsi_on", "coin": "BTC", "metric": "rsi14_1d", "op": "cross_above", "value": 55,
          "cooldown_min": 15, "action": "note"}
    off = {"id": "rsi_off", "coin": "BTC", "metric": "rsi14_1d", "op": "cross_below", "value": 50,
           "cooldown_min": 15, "action": "note"}
    st = {}
    seq = [60, 57, 54, 56, 58, 53, 49, 48, 51, 49, 56]
    got = []
    for i, v in enumerate(seq):
        res = watch.evaluate([on, off], {("BTC", "rsi14_1d"): float(v)}, st, T(30 * i))
        got += [(i, r["rule_id"]) for r in res["fired"]]
    # i=0 first observation never fires (no previous value) although 60 > 55
    assert got == [(3, "rsi_on"), (6, "rsi_off"), (9, "rsi_off"), (10, "rsi_on")], got
    # redefining the rule's metric forgets the previous value: no spurious cross
    on2 = dict(on, metric="dist_from_high50_1d", value=0.0)
    res = watch.evaluate([on2], {("BTC", "dist_from_high50_1d"): 0.01}, st, T(1000))
    assert not res["fired"] and st["rules"]["rsi_on"]["last_value"] == 0.01
    print("ok evaluate_cross")


def test_trigger_cap():
    rules = [{"id": "r%d" % i, "coin": "BTC", "metric": "ret_1h", "op": "<", "value": -0.01,
              "cooldown_min": 15, "action": "trigger_forward"} for i in range(6)]
    st = {}
    res = watch.evaluate(rules, {("BTC", "ret_1h"): -0.05}, st, T(0), max_triggers_day=4)
    assert len(res["fired"]) == 6 and len(res["trigger_rows"]) == 4
    capped = [r for r in res["fired"] if r["action"] == "note"]
    assert len(capped) == 2 and all("capped" in r["detail"] for r in capped)
    res = watch.evaluate(rules, {("BTC", "ret_1h"): -0.05}, st, T(60), max_triggers_day=4)
    assert len(res["trigger_rows"]) == 0 and len(res["fired"]) == 6
    res = watch.evaluate(rules, {("BTC", "ret_1h"): -0.05}, st, T(24 * 60 + 1), max_triggers_day=4)
    assert len(res["trigger_rows"]) == 4  # window rolled
    print("ok trigger_cap")


# ── 5. full run: outbox, trigger files, state, dry-run, inbox round trip ────
def test_run_and_inbox():
    reset()
    wl = {"version": 1, "max_trigger_files_per_day": 4, "rules": [
        {"id": "btc_drop_6pct_24h", "coin": "BTC", "metric": "ret_24h", "op": "<", "value": -0.06,
         "cooldown_min": 360, "action": "trigger_forward"},
        {"id": "btc_rsi_low", "coin": "BTC", "metric": "rsi14_1d", "op": "<", "value": 101,
         "cooldown_min": 720, "action": "note"}]}
    t0 = ms("2021-05-19 12:00")
    now_ms = t0 + 4 * H - 1
    now = dt.datetime.fromtimestamp(now_ms // 1000, tz=UTC)
    fetch = make_fetch({"15": step_15m(BTC_4H), "D": BTC_D}, now_ms)

    dry = watch.run(watchlist=wl, fetch=fetch, now=now, dry_run=True)
    assert dry["ok"] and len(dry["fired"]) == 2 and len(dry["would_write_trigger_files"]) == 1
    assert not (TMP / "outbox").exists() and not watch.state_path().exists(), "dry-run wrote files"

    res = watch.run(watchlist=wl, fetch=fetch, now=now)
    assert res["ok"] and not res["errors"], res
    assert len(res["fired"]) == 2 and len(res["trigger_files"]) == 1
    tf = Path(res["trigger_files"][0])
    assert inbox.TRIGGER_RE.match(tf.name) and ":" not in tf.name, tf.name
    trow = json.loads(tf.read_text())
    assert trow["rule_id"] == "btc_drop_6pct_24h" and trow["action"] == "trigger_forward" and trow["value"] < -0.06
    log = config.jsonl_read(TMP / "outbox" / "watch-triggers-berry.jsonl")
    assert len(log) == 2 and trow in log
    st = json.loads(watch.state_path().read_text())
    assert st["last_run"] == config.iso(now) and st["last_ok"] == st["last_run"]
    assert not list((TMP / "outbox").glob("*.tmp"))

    # 15 min later: same conditions → suppressed by cooldown, no new rows/files
    res2 = watch.run(watchlist=wl, fetch=fetch, now=now + dt.timedelta(minutes=15))
    assert not res2["fired"] and sorted(res2["suppressed"]) == ["btc_drop_6pct_24h", "btc_rsi_low"]
    assert len(config.jsonl_read(TMP / "outbox" / "watch-triggers-berry.jsonl")) == 2
    assert len(list((TMP / "outbox").glob("trigger-*.json"))) == 1

    # round trip: berry outbox → air inbox (simulate rsync) → inbox.merge → journal/watch-triggers.jsonl
    ib = TMP / "inbox"
    ib.mkdir(parents=True, exist_ok=True)
    for p in (TMP / "outbox").iterdir():
        shutil.copy(str(p), str(ib / p.name))
    m = inbox.merge()
    assert m["new_trigger_rows"] == 2 and len(m["pending"]) == 1 and not m["errors"], m
    j = config.jsonl_read(TMP / "journal" / "watch-triggers.jsonl")
    assert len(j) == 2 and trow in j
    assert inbox.merge()["new_trigger_rows"] == 0  # idempotent re-send

    # lock held by another run → skipped, nothing written
    import fcntl
    with open(str(watch.state_path()) + ".lock", "a") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        r = watch.run(watchlist=wl, fetch=fetch, now=now + dt.timedelta(days=2))
        assert r.get("skipped"), r

    # every fetch failing → ok False, state still records last_run (health sees the watcher ran) but no last_ok
    def dead(coin, interval, limit):
        raise watch.WatchError("network down")
    later = now + dt.timedelta(days=3)
    r = watch.run(watchlist=wl, fetch=dead, now=later)
    assert not r["ok"] and r["errors"] and not r["fired"]
    st = json.loads(watch.state_path().read_text())
    assert st["last_run"] == config.iso(later) and st["last_ok"] == config.iso(now + dt.timedelta(minutes=15))
    print("ok run_and_inbox")


# ── 6. replay May 2021 on recorded data: firing and suppression over 12 days ─
def test_replay_may_2021():
    reset()
    wl = {"version": 1, "max_trigger_files_per_day": 2, "rules": [
        {"id": "btc_drop_6pct_24h", "coin": "BTC", "metric": "ret_24h", "op": "<", "value": -0.06,
         "cooldown_min": 360, "action": "trigger_forward"},
        {"id": "btc_rsi_regime_off", "coin": "BTC", "metric": "rsi14_1d", "op": "cross_below", "value": 50,
         "cooldown_min": 720, "action": "note"},
        {"id": "btc_below_ema200", "coin": "BTC", "metric": "dist_from_ema200_1d", "op": "cross_below",
         "value": 0.0, "cooldown_min": 1440, "action": "note"}]}
    b15 = step_15m(BTC_4H)
    start, end = ms("2021-05-08"), ms("2021-05-24")
    t, n_cond, fired = start, 0, []
    while t < end:
        now = dt.datetime.fromtimestamp(t // 1000, tz=UTC)
        fetch = make_fetch({"15": b15, "D": BTC_D}, t)
        res = watch.run(watchlist=wl, fetch=fetch, now=now)
        assert res["ok"] and not res["errors"], res["errors"]
        if res["values"]["BTC.ret_24h"] < -0.06:
            n_cond += 1
        fired += res["fired"]
        t += M15
    drops = [r for r in fired if r["rule_id"] == "btc_drop_6pct_24h"]
    assert len(drops) >= 3, "May 2021 had several >6%% 24h drops, got %d" % len(drops)
    assert n_cond > 4 * len(drops), (n_cond, len(drops))  # suppression did its job
    ts = [watch._parse_iso(r["ts"]) for r in drops]
    assert all((b - a) >= dt.timedelta(minutes=360) for a, b in zip(ts, ts[1:])), ts
    # trigger files respect the 24h cap
    tfiles = sorted((TMP / "outbox").glob("trigger-*.json"))
    tts = [watch._parse_iso(json.loads(p.read_text())["ts"]) for p in tfiles]
    for i, a in enumerate(tts):
        assert sum(1 for b in tts if a - dt.timedelta(hours=24) < b <= a) <= 2
    assert len(tfiles) == sum(1 for r in drops if r["action"] == "trigger_forward") <= len(drops)
    assert all(r.get("detail") for r in drops if r["action"] == "note")
    # crosses fire at most once per crossing: the regime-off and ema200 rules fired, each only a few times
    offs = [r for r in fired if r["rule_id"] == "btc_rsi_regime_off"]
    emas = [r for r in fired if r["rule_id"] == "btc_below_ema200"]
    assert 1 <= len(offs) <= 3 and 1 <= len(emas) <= 2, (len(offs), len(emas))
    assert emas[0]["ts"] < "2021-05-24" and emas[0]["value"] < 0
    print("ok replay_may_2021 (cond %d steps, %d drop rows, %d trigger files, %d rsi-off, %d ema200)"
          % (n_cond, len(drops), len(tfiles), len(offs), len(emas)))


# ── 7. CLI + python 3.9 compatibility ───────────────────────────────────────
def test_cli_and_py39():
    py = sys.executable
    env = dict(os.environ)
    env["TRADER_STRATEGY_DIR"] = str(REPO / "trader" / "strategy")
    p = subprocess.run([py, "-m", "trader.fleet.watch", "check"], cwd=str(REPO), env=env,
                       capture_output=True, text=True, timeout=60)
    out = json.loads(p.stdout)
    assert p.returncode == 0 and out["ok"] and out["n_rules"] >= 10, p.stdout + p.stderr
    bad = TMP / "bad.json"
    bad.write_text(json.dumps({"version": 1, "rules": [{"id": "x"}]}))
    p = subprocess.run([py, "-m", "trader.fleet.watch", "check", "--watchlist", str(bad)], cwd=str(REPO),
                       env=env, capture_output=True, text=True, timeout=60)
    assert p.returncode == 2 and json.loads(p.stdout)["invalid_watchlist"]
    py39 = "/usr/bin/python3"  # macOS CLT python 3.9 (berry/bigsur run 3.9): stdlib-only import + check
    if Path(py39).exists():
        v = subprocess.run([py39, "-c", "import sys;print(sys.version_info[:2])"], capture_output=True, text=True)
        if v.stdout.strip() == "(3, 9)":
            p = subprocess.run([py39, "-m", "trader.fleet.watch", "check"], cwd=str(REPO), env=env,
                               capture_output=True, text=True, timeout=60)
            assert p.returncode == 0 and json.loads(p.stdout)["ok"], p.stdout + p.stderr
            print("ok py39 check")
    print("ok cli")


if __name__ == "__main__":
    try:
        test_watchlist()
        test_metrics_recorded()
        test_guards()
        test_evaluate_threshold_cooldown()
        test_evaluate_cross()
        test_trigger_cap()
        test_run_and_inbox()
        test_replay_may_2021()
        test_cli_and_py39()
    finally:
        shutil.rmtree(str(TMP), ignore_errors=True)
    print("OK test_watch")
