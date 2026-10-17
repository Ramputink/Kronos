"""Tests for core/forecast.py and strategy/screener.py (no network, no model calls). Run:
    /Users/matveypro/.venvs/kronos/bin/python trader/tests/test_screener.py
"""
import copy
import datetime as dt
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="phil-screener-test-"))
for d in ("data/spot", "strategy"):
    (TMP / d).mkdir(parents=True)
os.environ["TRADER_DATA_DIR"] = str(TMP / "data")
os.environ["TRADER_STRATEGY_DIR"] = str(TMP / "strategy")  # empty: screener falls back to the shipped questions

from trader.core import config  # noqa: E402

BASE_CFG = json.loads((config.TRADER_ROOT / "config" / "protected.json").read_text())
assert BASE_CFG["forward"]["forward_start"] is None
BASE_CFG["forward"]["forward_start"] = "2026-03-01"


def use_config(name, mutate=None):
    cfg = copy.deepcopy(BASE_CFG)
    if mutate:
        mutate(cfg)
    p = TMP / ("%s.json" % name)
    p.write_text(json.dumps(cfg))
    os.environ["TRADER_CONFIG"] = str(p)
    return cfg


def use_journal(name):
    d = TMP / ("journal-" + name)
    d.mkdir(exist_ok=True)
    os.environ["TRADER_JOURNAL_DIR"] = str(d)
    return d


# ── synthetic spot candles (2024-01-01 .. 2026-10-14), deterministic ───────────────────────────────────────
IDX = pd.date_range("2024-01-01", "2026-10-14", freq="1D")
CLOSES = {}
for i, coin in enumerate(("BTC", "ETH")):
    rng = np.random.default_rng(10 + i)
    r = rng.normal(0.0005, 0.035, len(IDX))
    c = 100.0 * (i + 1) * np.exp(np.cumsum(r))
    o = np.r_[c[0], c[:-1]]
    h = np.maximum(o, c) * (1 + rng.uniform(0, 0.02, len(IDX)))
    lo = np.minimum(o, c) * (1 - rng.uniform(0, 0.02, len(IDX)))
    v = rng.uniform(1000, 3000, len(IDX))
    df = pd.DataFrame({"open": o, "high": h, "low": lo, "close": c, "volume": v, "turnover": v * c}, index=IDX)
    df.index.name = "time"
    df.index = df.index.strftime("%Y-%m-%d")
    df.to_csv(TMP / "data" / "spot" / ("%s_1d.csv" % coin))
    # what load_raw will see (pandas default float parser)
    CLOSES[coin] = pd.read_csv(TMP / "data" / "spot" / ("%s_1d.csv" % coin), index_col=0, parse_dates=True)["close"]

# ── fake clock (vault, forecast and config.iso all go through config.utcnow) ────────────────────────────────
CLOCK = {"t": None}
_real_utcnow = config.utcnow


def set_clock(s):
    CLOCK["t"] = pd.Timestamp(s).to_pydatetime().replace(tzinfo=dt.timezone.utc)


config.utcnow = lambda: CLOCK["t"] if CLOCK["t"] is not None else _real_utcnow()

from trader.core import forecast  # noqa: E402
from trader.strategy import screener  # noqa: E402

D = pd.Timedelta("1D")
SCORED, CTX = screener.load_questions()
Q = {q["id"]: q for q in SCORED}
assert {"q_up_1d", "q_drop5_1d", "q_up_7d", "q_ret_bucket_7d", "q_score_trend_14d", "q_highvol_7d"} <= set(Q)
assert [q["id"] for q in CTX] == ["ctx_event_24h"]


def expect_raises(exc, fn, *a, **k):
    try:
        fn(*a, **k)
    except exc:
        return
    raise AssertionError("expected %s" % exc.__name__)


# ═══ 1. question validation ═════════════════════════════════════════════════════════════════════════════════
assert forecast.validate_question(Q["q_ret_bucket_7d"]) == []
bad = dict(Q["q_up_1d"], resolver={"type": "news"})
assert forecast.validate_question(bad)
bad = dict(Q["q_ret_bucket_7d"], options=["a", "b"])
assert forecast.validate_question(bad)
assert forecast.validate_question(dict(Q["q_highvol_7d"], horizon_bars=1))
assert forecast.normalise_p(Q["q_up_1d"], 0.7) == [0.3, 0.7]
expect_raises(forecast.ForecastError, forecast.normalise_p, Q["q_up_1d"], 1.3)
expect_raises(forecast.ForecastError, forecast.normalise_p, Q["q_ret_bucket_7d"], [0.5, 0.2, 0.1])
expect_raises(forecast.ForecastError, forecast.normalise_p, Q["q_ret_bucket_7d"], 0.5)
assert abs(sum(forecast.normalise_p(Q["q_ret_bucket_7d"], [0.33, 0.33, 0.33])) - 1) < 1e-5
assert abs(forecast.brier([0.42, 0.58], 1) - 0.3528) < 1e-12  # CONTRACTS example


# ═══ 2. outcomes, baseline (no lookahead), settlement and Brier on synthetic data ════════════════════════
use_config("main")
use_journal("settle")
c = CLOSES["BTC"]
A = pd.Timestamp("2026-06-09")

# manual outcomes
assert forecast.outcome_series(c, {"type": "close_gt"}, 1, "1d")[0][A] == float(c[A + D] > c[A])
ret7 = c[A + 7 * D] / c[A] - 1
bucket = 0 if ret7 < -0.05 else (1 if ret7 < 0.05 else 2)
assert forecast.outcome_series(c, Q["q_ret_bucket_7d"]["resolver"], 7, "1d")[0][A] == bucket
lr = np.log(c[A:A + 7 * D]).diff().dropna()
assert len(lr) == 7
vol7 = float(np.std(lr.values, ddof=1) * np.sqrt(365))
o, real = forecast.outcome_series(c, {"type": "vol_gt", "x": 0.6}, 7, "1d")
assert abs(real[A] - vol7) < 1e-9 and o[A] == float(vol7 > 0.6)

# baseline uses only windows fully closed by asof: manual frequency
hist = c[c.index <= A]
wins = [t for t in hist.index if A - pd.Timedelta(days=365) <= t <= A - D]
freq_up = np.mean([hist[t + D] > hist[t] for t in wins])
bp, bname, bn = forecast.baseline_probs(c, Q["q_up_1d"], A)
assert bname == "base_rate_365d" and bn == len(wins) == 365
assert abs(bp[1] - freq_up) < 1e-6 and abs(sum(bp) - 1) < 1e-6
# changing the future must not change the baseline
c_future = c.copy()
c_future[c_future.index > A] *= 3.0
assert forecast.baseline_probs(c_future, Q["q_up_1d"], A)[0] == bp
assert forecast.baseline_probs(c, dict(Q["q_up_1d"], baseline="uniform"), A)[0] == [0.5, 0.5]

# logging rules
set_clock("2026-06-09 12:00")  # asof bar still forming
expect_raises(forecast.ForecastError, forecast.log_forecast, Q["q_up_1d"], "BTC", A, 0.7, model="stub",
              prompt_hash="h")
set_clock("2026-06-10 00:30")
fid = forecast.log_forecast(Q["q_up_1d"], "BTC", A, 0.7, model="stub", prompt_hash="h")
assert fid == "F-20260610-BTC-q_up_1d"
expect_raises(forecast.DuplicateForecast, forecast.log_forecast, Q["q_up_1d"], "BTC", A, 0.4, model="stub",
              prompt_hash="h2")
expect_raises(forecast.ForecastError, forecast.log_forecast, Q["q_up_1d"], "BTC", A - D, 0.7, model="stub",
              prompt_hash="h")  # late: the asof bar closed > 1 bar ago
fid_b = forecast.log_forecast(Q["q_ret_bucket_7d"], "BTC", A, [0.2, 0.5, 0.3], model="stub", prompt_hash="h")
fid_v = forecast.log_forecast(Q["q_highvol_7d"], "BTC", A, 0.4, model="stub", prompt_hash="h")
fid_s = forecast.log_forecast(dict(Q["q_drop5_1d"], _score=10), "BTC", A, 0.10, model="stub", prompt_hash="h")
rows = {r["forecast_id"]: r for r in config.jsonl_read(forecast.forecasts_path())}
r0 = rows[fid]
assert r0["p"] == [0.3, 0.7] and r0["baseline_p"] == bp and r0["resolves_at_bar"] == "2026-06-10T00:00:00Z"
assert r0["resolver"] == {"type": "close_gt"} and r0["asof_bar"] == "2026-06-09T00:00:00Z" and r0["lag_s"] == 1800
assert rows[fid_s]["score"] == 10 and rows[fid_s]["kind"] == "score"
assert rows[fid_b]["resolves_at_bar"] == "2026-06-16T00:00:00Z"

# settlement
set_clock("2026-06-10 12:00")
assert forecast.resolve_due() == []  # resolution bar still forming
set_clock("2026-06-11 00:05")
res = {r["forecast_id"]: r for r in forecast.resolve_due()}
assert set(res) == {fid, fid_s}
y = int(c[A + D] > c[A])
assert res[fid]["outcome"] == y
assert abs(res[fid]["brier"] - 2 * (0.7 - y) ** 2) < 1e-6
assert abs(res[fid]["brier_baseline"] - 2 * (bp[1] - y) ** 2) < 1e-5
assert abs(res[fid]["brier_uniform"] - 0.5) < 1e-12
ys = int(c[A + D] / c[A] - 1 < -0.05)
assert res[fid_s]["outcome"] == ys
assert forecast.resolve_due() == []  # never resolved twice
set_clock("2026-06-17 00:05")
res = {r["forecast_id"]: r for r in forecast.resolve_due()}
assert set(res) == {fid_b, fid_v}
assert res[fid_b]["outcome"] == bucket
pb = [0.2, 0.5, 0.3]
assert abs(res[fid_b]["brier"] - sum((pb[i] - (i == bucket)) ** 2 for i in range(3))) < 1e-6
assert res[fid_v]["outcome"] == int(vol7 > 0.6) and abs(res[fid_v]["realised"] - vol7) < 1e-5

s = forecast.brier_summary()
assert s["n"] == 4 and s["days"] == 1 and set(s["by_question"]) == {"q_up_1d", "q_ret_bucket_7d", "q_highvol_7d",
                                                                   "q_drop5_1d"}
assert abs(s["brier"] - np.mean([r["brier"] for r in config.jsonl_read(forecast.forecasts_path())
                                 if r["type"] == "resolve"])) < 1e-6
assert forecast.brier_summary(question_id="q_up_1d")["n"] == 1
assert forecast.veto_allowed() is False and forecast.veto_allowed("q_up_1d") is False

# ═══ 3. veto gate: needs >= 30 forward days, >= 100 resolved, and beating BOTH baselines ═════════════════════
def fill(journal, good, days=40, start="2026-04-01"):
    use_journal(journal)
    t0 = pd.Timestamp(start)
    for k in range(days):
        a = t0 + k * D
        set_clock(a + D + pd.Timedelta("10min"))
        for coin in ("BTC", "ETH"):
            cc = CLOSES[coin]
            for qid in ("q_up_1d", "q_drop5_1d"):
                q = Q[qid]
                yy = int(forecast.outcome_series(cc[cc.index <= a + D], q["resolver"], 1, "1d")[0][a])
                p = (0.85 if yy else 0.15) if good else (0.15 if yy else 0.85)
                forecast.log_forecast(q, coin, a, p, model="stub", prompt_hash="x")
    set_clock(t0 + (days + 2) * D)
    forecast.resolve_due()


fill("veto-good", good=True)
st = forecast.veto_status()
assert st["summary"]["n"] == 160 and st["summary"]["days"] == 40, st
assert st["allowed"] and forecast.veto_allowed() is True, st
assert forecast.veto_allowed("q_up_1d") is False  # 80 resolved < 100 for a single question
sq = forecast.veto_status("q_up_1d")
assert any("resolved forecasts" in r for r in sq["reasons"])

use_config("strict", lambda cfg: cfg["screener"].update(veto_min_forward_days=41))
assert forecast.veto_allowed() is False
use_config("main")

fill("veto-bad", good=False)
st = forecast.veto_status()
assert not st["allowed"] and any("does not beat" in r for r in st["reasons"])

fill("veto-short", good=True, days=20)  # 80 forecasts over 20 days
st = forecast.veto_status()
assert not st["allowed"] and len(st["reasons"]) == 2, st

# forecasts dated before forward_start never count
use_config("late-start", lambda cfg: cfg["forward"].update(forward_start="2026-05-01"))
use_journal("veto-good")
assert forecast.veto_status()["summary"]["n"] == 160 - 2 * 2 * 30
assert forecast.veto_allowed() is False
use_config("main")


# ═══ 4. screener: features, prompt, dry run and logging with a stubbed model ═════════════════════════════
def stub_reply(coins, *, fence=False, broken=None):
    out = {"forecasts": []}
    for cn in coins:
        ans = {"q_up_1d": {"p": 0.55}, "q_drop5_1d": {"score": 8}, "q_up_7d": {"p": 0.52},
               "q_ret_bucket_7d": {"p": [0.25, 0.5, 0.25]}, "q_score_trend_14d": {"score": 40},
               "q_highvol_7d": {"p": 0.3}, "ctx_event_24h": {"p": 0.1, "note": "none known"}}
        if broken and cn in broken:
            ans["q_up_1d"] = {"p": 1.7}
            ans["q_ret_bucket_7d"] = {"p": [0.9, 0.9, 0.9]}
        out["forecasts"].append({"coin": cn, "answers": ans})
    txt = json.dumps(out)
    return "```json\n%s\n```" % txt if fence else txt


CALLS = []


def make_stub(**kw):
    def call(prompt, model):
        CALLS.append((prompt, model))
        coins = [cn for cn in ("BTC", "ETH") if '"coin":"%s"' % cn in prompt]
        return stub_reply(coins, **kw)
    return call


use_journal("screener")
set_clock("2026-07-01 00:20")
ASOF = "2026-06-30T00:00:00Z"
df = screener.load_bars("BTC")
assert df.index[-1] == pd.Timestamp("2026-06-30")  # closed bars only (the 07-01 bar is forming)
assert df.index[0] >= pd.Timestamp("2026-03-01") - pd.Timedelta(days=BASE_CFG["vault"]["warmup_days"])  # windowed
f = screener.features("BTC", df, 0.0001)
assert f["asof_bar"] == ASOF and abs(f["last_close"] - CLOSES["BTC"]["2026-06-30"]) / f["last_close"] < 1e-5
assert abs(f["ret_1d"] - (CLOSES["BTC"]["2026-06-30"] / CLOSES["BTC"]["2026-06-29"] - 1)) < 1e-4
assert 0 <= f["rsi14"] <= 100 and 0 <= f["adx14"] <= 100 and f["funding_rate_last"] == 0.0001

# dry run without a model: prompts only, nothing written, no call counted
r = screener.run(["BTC", "ETH"], dry_run=True, funding_fn=None)
assert r["calls"] == 0 and len(r["prompts"]) == 1 and r["prompts"][0]["coins"] == ["BTC", "ETH"]
pr = r["prompts"][0]["prompt"]
assert "q_ret_bucket_7d" in pr and "ctx_event_24h" in pr and ASOF in pr and "2026-07-01" not in pr
assert not forecast.forecasts_path().exists() and screener.calls_today() == 0

# dry run with the stub: parsed, still nothing written
r = screener.run(["BTC", "ETH"], dry_run=True, call_model=make_stub(fence=True), funding_fn=None)
assert r["calls"] == 1 and r["errors"] == [] and len(r["prompts"][0]["answers"]) == 2, r
assert r["context"]["BTC"]["ctx_event_24h"]["p"] == 0.1
assert not forecast.forecasts_path().exists() and screener.calls_today() == 0

# real run with the stub, one coin answering badly
CALLS.clear()
r = screener.run(["BTC", "ETH", "NOPE"], call_model=make_stub(broken={"ETH"}), funding_fn=None)
assert len(CALLS) == 1 and CALLS[0][1] == BASE_CFG["screener"]["model"]
assert r["skipped"]["NOPE"] == "not in universe"
assert len(r["logged"]) == 6 + 4, r  # ETH lost 2 invalid answers
assert any("ETH/q_up_1d" in e for e in r["errors"]) and any("ETH/q_ret_bucket_7d" in e for e in r["errors"])
assert screener.calls_today() == 1
rows = [x for x in config.jsonl_read(forecast.forecasts_path()) if x["type"] == "forecast"]
assert all(x["asof_bar"] == ASOF and x["model"] == BASE_CFG["screener"]["model"] for x in rows)
assert len({x["prompt_hash"] for x in rows}) == 1
sc = [x for x in rows if x["question_id"] == "q_score_trend_14d" and x["coin"] == "BTC"][0]
assert sc["p"] == [0.6, 0.4] and sc["score"] == 40 and sc["resolves_at_bar"] == "2026-07-14T00:00:00Z"
assert not any(x["question_id"] == "ctx_event_24h" for x in rows)  # context questions are never logged

# second run the same day: BTC complete -> not asked; ETH asked again only for the missing questions,
# whose duplicates for the answered ones are refused
CALLS.clear()
r = screener.run(["BTC", "ETH"], call_model=make_stub(), funding_fn=None)
assert r["skipped"]["BTC"].startswith("already forecast") and len(CALLS) == 1
assert '"coin":"BTC"' not in CALLS[0][0]
assert sorted(r["logged"]) == ["F-20260701-ETH-q_ret_bucket_7d", "F-20260701-ETH-q_up_1d"], r
assert sum(1 for k, v in r["skipped"].items() if v == "duplicate") == 4
CALLS.clear()
r = screener.run(["BTC", "ETH"], call_model=make_stub(), funding_fn=None)
assert CALLS == [] and r["calls"] == 0 and r["logged"] == []

# call budget: max_calls_per_day = 2 already reached today
use_config("budget", lambda cfg: cfg["screener"].update(max_calls_per_day=2))
assert screener.calls_today() == 2  # same UTC day as the two real runs above
use_journal("screener-budget")
CALLS.clear()
r = screener.run(["BTC", "ETH"], call_model=make_stub(), funding_fn=None, coins_per_call=1)
assert len(CALLS) == 0 and any("budget" in e for e in r["errors"]) and not r["ok"]
use_config("main")

# a garbage reply is an error, never a forecast
use_journal("screener-garbage")
r = screener.run(["BTC"], call_model=lambda p, m: "I think BTC goes up", funding_fn=None)
assert r["logged"] == [] and r["errors"] and not forecast.forecasts_path().exists()

# settle the screener's forecasts later
use_journal("screener")
set_clock("2026-07-15 00:30")
done = forecast.resolve_due()
assert len(done) == 12 and all(not x.get("void") for x in done)

# the screener cannot read the paper view while forward_start is null (everything >= cutoff is sealed)
use_config("sealed", lambda cfg: cfg["forward"].update(forward_start=None))
use_journal("sealed")
r = screener.run(["BTC"], call_model=make_stub(), funding_fn=None)
assert "VaultSealed" in r["skipped"]["BTC"] and r["logged"] == []
use_config("main")

# CLI-level JSON extraction
assert screener.extract_json('noise {"forecasts": []} trailing') == {"forecasts": []}
expect_raises(screener.ScreenerError, screener.extract_json, "no json here")

print("OK test_screener")
