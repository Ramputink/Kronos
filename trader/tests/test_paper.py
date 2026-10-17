"""Tests for core/paper.py and core/real.py (no network). Run:
    /Users/matveypro/.venvs/kronos/bin/python trader/tests/test_paper.py
"""
import copy
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

LAB = "/Users/matveypro/kronos_lab/indicators/data"
TMP = Path(tempfile.mkdtemp(prefix="phil-paper-test-"))
(TMP / "data").mkdir()
os.symlink(LAB, TMP / "data" / "spot")
os.symlink(LAB, TMP / "data" / "linear")
os.environ["TRADER_DATA_DIR"] = str(TMP / "data")
os.environ["TRADER_STRATEGY_DIR"] = str(TMP / "strategy")
(TMP / "strategy").mkdir()
for k in ("BYBIT_API_KEY", "BYBIT_API_SECRET"):
    os.environ.pop(k, None)
# real.py honours TRADER_CONFIG / TRADER_JOURNAL_DIR only with PHIL_CI=1 and only for dry runs
os.environ["PHIL_CI"] = "1"
# the test harness reads the paper view as caller "test" (vault requires the operator/test-harness env flag)
os.environ["TRADER_VAULT_OPERATOR"] = "1"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from trader.core import config  # noqa: E402

BASE_CFG = json.loads((config.TRADER_ROOT / "config" / "protected.json").read_text())
# the shipped config has forward_start=null (paper refuses to run, see section 0); the tests simulate an operator
# who started paper on 2026-08-01, after the vault cutoff
assert BASE_CFG["forward"]["forward_start"] is None
BASE_CFG["forward"]["forward_start"] = "2026-08-01"


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


from trader.core import backtest, paper, real, vault  # noqa: E402

# keep the module's real fixed paths for the regression checks, but never write into the repo journal from tests
REAL_FIXED_CONFIG = real._FIXED_CONFIG
assert REAL_FIXED_CONFIG == Path(__file__).resolve().parents[1] / "config" / "protected.json"
REAL_FIXED_LEDGER_DIR = TMP / "fixed-journal"
real._FIXED_LEDGER = REAL_FIXED_LEDGER_DIR / "real-ledger.jsonl"
real._FIXED_PAPER_LEDGER = REAL_FIXED_LEDGER_DIR / "paper-ledger.jsonl"

SPEC = {"tf": "1d", "mode": "SPOT", "comps": [{"t": "rsi_reg", "p": {"n": 14, "up": 55, "gap": 5}}],
        "comb": "and", "k": 1, "adx": 0, "vt": 0.5, "stop": 0}
COINS = ["BTC", "ETH", "SOL", "BNB"]
ALLOC = 800.0
START, END = pd.Timestamp("2026-08-31"), pd.Timestamp("2026-09-30")  # asof bars; first fill at close of 08-31
DAYS = pd.date_range(START, END, freq="D")


def write_active(strategies):
    p = TMP / "strategy" / "active.json"
    p.write_text(json.dumps({"version": 1, "updated": config.iso(), "strategies": strategies}))
    return str(p)


def strat(**kw):
    s = {"id": "rsi_regime_l4", "status": "paper", "spec": SPEC, "coins": COINS, "allocation_usdt": ALLOC,
         "activated_at": "2026-09-01", "hypothesis_id": "H-0001", "evidence": "test", "screener_veto": False}
    s.update(kw)
    return s


def replay(active_path):
    out = []
    for d in DAYS:
        out.append(paper.run_paper(active_path, asof=d.strftime("%Y-%m-%d")))
    return out


def reference(cost):
    """Backtest per coin over the same window: returns of bars 09-01..09-30 + the targets decided at 08-31..09-30."""
    ref = {}
    for coin in COINS:
        df = vault.load(coin, "1d", view="paper", asof=END, caller="test")
        res = backtest.evaluate_df(SPEC, df, "all", cost_side=cost)
        r = res["returns"]
        ref[coin] = {"r": r[(r.index > START) & (r.index <= END)], "target": res["target"],
                     "close": df["close"]}
    return ref


def ledger(journal):
    return config.jsonl_read(journal / "paper-ledger.jsonl")


# ── 0. honesty seal: no paper run/settle on the sealed TEST window ──────────
# Regression (review): with forward_start=null, `paper run --asof <cutoff+1d> --dry-run --active probe.json`
# printed sealed spot closes / weights / equity without a look or a trials row.
import contextlib  # noqa: E402
import io  # noqa: E402

use_config("shipped", lambda c: c["forward"].update(forward_start=None))
j_seal = use_journal("seal")
CUT = pd.Timestamp(BASE_CFG["vault"]["vault_cutoff"])
probe = TMP / "strategy" / "probe.json"
probe.write_text(json.dumps({"version": 1, "strategies": [
    {"id": "probe_x", "status": "paper", "spec": SPEC, "coins": ["BTC", "ETH"], "allocation_usdt": 800,
     "activated_at": CUT.strftime("%Y-%m-%d"), "hypothesis_id": "H-0001", "evidence": "x"}]}))
day1 = (CUT + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
for argv in (["run", "--asof", day1, "--dry-run", "--active", str(probe)],
             ["run", "--asof", day1, "--active", str(probe)],
             ["run", "--dry-run", "--active", str(probe)],  # default asof = last closed bar
             ["settle", "--asof", day1]):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = paper.main(argv)
    out = json.loads(buf.getvalue())
    assert rc == 1 and out["ok"] is False and "forward_start" in out["error"], (argv, out)
    assert "close" not in buf.getvalue() and "equity_usdt" not in buf.getvalue(), buf.getvalue()
for call in (lambda: paper.run_paper(str(probe), asof=day1, dry_run=True),
             lambda: paper.settle(day1, active_path=str(probe))):
    try:
        call()
        raise AssertionError("paper must refuse while forward_start is null")
    except paper.PaperError as e:
        assert "forward_start" in str(e), e
assert not (j_seal / "paper-ledger.jsonl").exists(), "refused runs must not write the ledger"
# validate_active: activated_at inside the sealed window is invalid even before paper starts
assert any("vault_cutoff" in e for e in paper.validate_active(json.loads(probe.read_text())))

# forward_start set: asof before it is refused (dry run included); activated_at before it is invalid
use_config("fs_set", lambda c: c["forward"].update(forward_start="2026-08-01"))
for a in ("2026-07-31", day1):
    try:
        paper.run_paper(str(probe), asof=a, dry_run=True)
        raise AssertionError("asof < forward_start must be refused")
    except paper.PaperError as e:
        assert "forward_start" in str(e), e
    try:
        paper.settle(a, active_path=str(probe))
        raise AssertionError("settle asof < forward_start must be refused")
    except paper.PaperError as e:
        assert "forward_start" in str(e), e
early = {"version": 1, "strategies": [dict(json.loads(probe.read_text())["strategies"][0], activated_at="2026-07-31")]}
assert any("forward_start" in e for e in paper.validate_active(early)), paper.validate_active(early)
probe.write_text(json.dumps(early))
try:  # valid asof, but the sleeve claims to have started inside the sealed window
    paper.run_paper(str(probe), asof="2026-08-05", dry_run=True)
    raise AssertionError("activated_at < forward_start must be refused")
except paper.PaperError as e:
    assert "forward_start" in str(e), e
try:  # _plan refuses on its own (defence in depth for callers that skip validate_active)
    paper._plan(early, pd.Timestamp("2026-08-05"), [], config.load_config(), "P-x")
    raise AssertionError("_plan must refuse activated_at < forward_start")
except paper.PaperError as e:
    assert "forward_start" in str(e), e
# market data is loaded through the vault forward window: no sealed TEST bars before forward_start - warmup
dfp = paper._load_paper_df("BTC", pd.Timestamp("2026-08-05"))
assert dfp.index.min() >= pd.Timestamp("2026-08-01") - pd.Timedelta(days=BASE_CFG["vault"]["warmup_days"]), \
    dfp.index.min()
assert dfp.index.max() == pd.Timestamp("2026-08-05")
assert not (j_seal / "paper-ledger.jsonl").exists()
probe.unlink()
print("ok honesty seal (forward_start gate on run/settle/dry-run, activated_at, vault forward window)")

# ── 1. validate_active ───────────────────────────────────────────────────────
use_config("base")
assert paper.validate_active({"version": 1, "strategies": [strat()]}) == [], \
    paper.validate_active({"version": 1, "strategies": [strat()]})
bad = [
    strat(spec=dict(SPEC, mode="PERP")),
    strat(spec=dict(SPEC, tf="4h")),
    strat(id="Bad-Id"),
    strat(allocation_usdt=1500),
    strat(coins=["BTC", "FOO"]),
    strat(status="live"),
    strat(activated_at="21/10/2026"),
]
for b in bad:
    assert paper.validate_active({"version": 1, "strategies": [b]}), b
over = [strat(id="s_%d" % i, allocation_usdt=900) for i in range(3)]  # 2700 > bankroll 2000
assert any("bankroll" in e for e in paper.validate_active({"version": 1, "strategies": over}))
dup = [strat(), strat()]
assert any("duplicate" in e for e in paper.validate_active({"version": 1, "strategies": dup}))
print("ok validate_active")

# ── 2. zero costs, no minimum: paper equity == backtest equity exactly ───────
def zero(cfg):
    cfg["costs"]["spot"]["fee_side"] = 0.0
    cfg["costs"]["spot"]["slippage_side"] = 0.0
    cfg["paper"]["min_order_usdt"] = 0.0


use_config("zero", zero)
j0 = use_journal("zero")
ap = write_active([strat()])
res0 = replay(ap)
assert all(r["status"] == "ok" for r in res0), res0[:2]
ref0 = reference(0.0)
eq0 = paper.equity("rsi_regime_l4")
assert eq0.index[0] == START and eq0.index[-1] == END, (eq0.index[0], eq0.index[-1])
sleeve0 = ALLOC / len(COINS)
exp_total = 0.0
pos = paper.positions()
for coin in COINS:
    exp = sleeve0 * float(np.prod(1 + ref0[coin]["r"].values))
    got = pos["rsi_regime_l4:%s" % coin]["equity_usdt"]
    assert abs(got / exp - 1) < 1e-9, (coin, got, exp)
    # every mark weight equals the backtest target decided at that close
    marks = [m for m in ledger(j0) if m["type"] == "mark" and m["coin"] == coin]
    for m in marks:
        t = pd.Timestamp(m["asof_bar"]).tz_localize(None)
        tgt = float(ref0[coin]["target"].loc[t])
        assert abs(m["weight"] - min(max(tgt, 0), 1)) < 1e-6, (coin, t, m["weight"], tgt)
    exp_total += exp
assert abs(eq0.iloc[-1] / exp_total - 1) < 1e-9, (eq0.iloc[-1], exp_total)
assert abs(res0[-1]["equity_usdt"] - exp_total) < 1e-4
n_fills0 = sum(1 for r in ledger(j0) if r["type"] == "fill")
assert n_fills0 > 4, n_fills0  # init fills + vol-target rebalances
print("ok zero-cost replay == backtest (total %.4f, fills %d)" % (exp_total / ALLOC - 1, n_fills0))

# ── 3. idempotency + dry run + no trading backwards ──────────────────────────
n_rows = len(ledger(j0))
again = paper.run_paper(ap, asof=END.strftime("%Y-%m-%d"))
assert again["status"] == "skipped", again
assert sum(1 for r in ledger(j0) if r["type"] in ("fill", "mark")) == \
    sum(1 for r in ledger(j0)[:n_rows] if r["type"] in ("fill", "mark"))
n_rows = len(ledger(j0))
dr = paper.run_paper(ap, asof="2026-10-01", dry_run=True)
assert dr["status"] == "dry_run" and len(ledger(j0)) == n_rows
try:
    paper.run_paper(ap, asof="2026-08-20")
    raise AssertionError("trading backwards must fail")
except paper.PaperError:
    pass
print("ok idempotent / dry-run / no backwards")

# ── 4. settle: mark only, idempotent ─────────────────────────────────────────
s1 = paper.settle("2026-10-01", active_path=ap)
assert s1["marks"] == 4, s1
s2 = paper.settle("2026-10-01", active_path=ap)
assert s2["marks"] == 0 and s2["already_marked"] == 4
assert sum(1 for r in ledger(j0) if r["type"] == "fill" and r["asof_bar"].startswith("2026-10-01")) == 0
print("ok settle")

# ── 5. realistic Bybit EU costs: paper == backtest within costs ──────────────
cfg = use_config("real_costs")
j1 = use_journal("costs")
cost = cfg["costs"]["spot"]["fee_side"] + cfg["costs"]["spot"]["slippage_side"]
min_order = cfg["paper"]["min_order_usdt"]
ap = write_active([strat()])
res1 = replay(ap)
rows = ledger(j1)
fills = [r for r in rows if r["type"] == "fill"]
skips = [r for r in rows if r["type"] == "skip"]
assert fills and all(f["notional_usdt"] >= min_order for f in fills), "min order violated"
assert all(s["reason"] == "below_min_order" and s["notional_usdt"] < min_order for s in skips), skips[:3]
for f in fills:  # fee and slippage exactly as protected config
    assert abs(f["fee_usdt"] - f["notional_usdt"] * cfg["costs"]["spot"]["fee_side"]) < 1e-6
    sgn = 1 if f["side"] == "buy" else -1
    assert abs(f["fill_price"] / f["price"] - 1 - sgn * cfg["costs"]["spot"]["slippage_side"]) < 1e-9
ref1 = reference(cost)
pos = paper.positions()
worst = 0.0
for coin in COINS:
    sid = "rsi_regime_l4:%s" % coin
    paper_tot = pos[sid]["equity_usdt"] / sleeve0 - 1
    zero_tot = float(np.prod(1 + ref0[coin]["r"].values)) - 1
    bt_tot = float(np.prod(1 + ref1[coin]["r"].values)) - 1
    traded = sum(f["notional_usdt"] for f in fills if f["sleeve_id"] == sid) / sleeve0
    skipped = sum(s["notional_usdt"] for s in skips if s["sleeve_id"] == sid) / sleeve0
    turn = ref1[coin]["target"]
    w = turn[(turn.index >= START - pd.Timedelta(days=1)) & (turn.index <= END)].clip(0, 1).values
    bt_turn = float(np.abs(np.diff(w)).sum()) + abs(w[1])  # incl. the initial entry the backtest never charges
    # (a) paper loses vs the frictionless run roughly by cost x traded notional
    assert paper_tot <= zero_tot + 1e-12, (coin, paper_tot, zero_tot)
    assert abs((zero_tot - paper_tot) - cost * traded) <= cost * traded * 0.25 + skipped * 0.2 + 1e-4, \
        (coin, zero_tot - paper_tot, cost * traded, skipped)
    # (b) paper vs backtest with the same costs: equal up to cost differences (drift rebalancing, initial entry,
    #     min-order skips) -> bounded by cost x (paper + backtest turnover) + the skipped notional's exposure
    bound = cost * (traded + bt_turn) + skipped * 0.2 + 1e-4
    diff = abs(paper_tot - bt_tot)
    worst = max(worst, diff)
    assert diff <= bound, (coin, paper_tot, bt_tot, bound)
print("ok realistic costs: paper within costs of backtest (worst diff %.5f, fills %d, skips %d)"
      % (worst, len(fills), len(skips)))

# ── 6. retire closes sleeves; paused never increases ─────────────────────────
ap = write_active([strat(status="retired")])
rr = paper.run_paper(ap, asof="2026-10-01")
sells = [r for r in ledger(j1) if r["type"] == "fill" and r["asof_bar"].startswith("2026-10-01")]
assert all(s["side"] == "sell" and s["reason"] == "retire" for s in sells), sells
assert paper.positions() == {} or all(p["qty"] == 0 for p in paper.positions().values()), paper.positions()
eq_after = rr["equity_usdt"]
rr2 = paper.run_paper(ap, asof="2026-10-02")
assert rr2["fills"] == 0 and abs(rr2["equity_usdt"] - eq_after) < 1e-6, (rr2, eq_after)
print("ok retire")

j2 = use_journal("paused")
ap = write_active([strat(activated_at="2026-09-01")])
paper.run_paper(ap, asof="2026-08-31")
ap = write_active([strat(status="paused")])
for d in pd.date_range("2026-09-01", "2026-09-20"):
    paper.run_paper(ap, asof=d.strftime("%Y-%m-%d"))
assert not [r for r in ledger(j2) if r["type"] == "fill" and r["side"] == "buy" and r["reason"] != "init"]
print("ok paused never buys")

# activation gate: nothing before activated_at
j3 = use_journal("activation")
ap = write_active([strat(activated_at="2026-09-10")])
r = paper.run_paper(ap, asof="2026-09-05")
assert r["fills"] == 0 and not [x for x in ledger(j3) if x["type"] == "mark"], r
print("ok activation gate")

# ── 7. real.py refuses while disabled; never logs secrets ────────────────────
use_config("base")
jr = use_journal("real")
FAKE_KEY, FAKE_SECRET = "FAKEKEY1234567890", "FAKESECRETabcdefghijklmnop"
for env in (False, True):
    if env:
        os.environ["BYBIT_API_KEY"], os.environ["BYBIT_API_SECRET"] = FAKE_KEY, FAKE_SECRET
    for dry in (True, False):
        try:
            real.place_order("BTC", "buy", 10, client_order_id="t-%s-%s" % (env, dry), strategy_id="rsi_regime_l4",
                             dry_run=dry)
            raise AssertionError("must refuse when disabled")
        except real.RealTradingDisabled:
            pass
rl = config.jsonl_read(jr / "real-ledger.jsonl")
assert len(rl) == 2 and all(r["status"] == "refused" and r["dry_run"] for r in rl), rl  # dry runs: test journal
rl_live = config.jsonl_read(real._FIXED_LEDGER)                                           # live: fixed ledger
assert len(rl_live) == 2 and all(r["status"] == "refused" and not r["dry_run"] for r in rl_live), rl_live
doc = real.doctor(offline=True)
assert doc["enabled"] is False and doc["ok"] is False
assert any(c["name"] == "config_flag" and not c["ok"] for c in doc["checks"])

# enabled in a temp config, but strategy not allowed / coin not allowed / no keys -> still refused
def enable(cfg):
    cfg["real"]["real_trading_enabled"] = True
    cfg["real"]["allowed_strategy_ids"] = ["rsi_regime_l4"]


use_config("enabled", enable)
for coin, sid in (("BTC", "other_strategy"), ("DOGE", "rsi_regime_l4")):
    try:
        real.place_order(coin, "buy", 10, client_order_id="x1", strategy_id=sid, dry_run=True)
        raise AssertionError("must refuse")
    except real.RealTradingDisabled:
        pass
os.environ.pop("BYBIT_API_KEY")
try:
    real.place_order("BTC", "buy", 10, client_order_id="x2", strategy_id="rsi_regime_l4", dry_run=True)
    raise AssertionError("must refuse without keys")
except real.RealTradingDisabled:
    pass
os.environ["BYBIT_API_KEY"] = FAKE_KEY
# caps (dry run, no network)
for args, exc in (((25,), real.RealCapExceeded),       # > per_order_eur 20
                  ((1,), real.RealCapExceeded)):        # 1 EUR * 1.16 < 5 USDT min
    try:
        real.place_order("BTC", "buy", *args, client_order_id="cap%s" % args[0], strategy_id="rsi_regime_l4")
        raise AssertionError("cap not enforced")
    except exc:
        pass
ok = real.place_order("BTC", "buy", 20, client_order_id="dry1", strategy_id="rsi_regime_l4", dry_run=True)
assert ok["status"] == "dry_run" and abs(ok["notional_usdt"] - 20 * 1.16) < 1e-9, ok
try:
    real.place_order("BTC", "buy", 20, client_order_id="dry1", strategy_id="rsi_regime_l4", dry_run=True)
    raise AssertionError("duplicate client id")
except real.RealCapExceeded:
    pass
# daily cap counts only submitted/filled rows: simulate 40 EUR already spent today
config.jsonl_append(jr / "real-ledger.jsonl", {"ts": config.iso(), "client_order_id": "past", "coin": "ETH",
                                               "side": "buy", "notional_eur": 40, "status": "filled"})
try:
    real.place_order("BTC", "buy", 15, client_order_id="dc", strategy_id="rsi_regime_l4", dry_run=True)
    raise AssertionError("daily cap not enforced")
except real.RealCapExceeded as e:
    assert "daily" in str(e)
raw = (jr / "real-ledger.jsonl").read_text()
doc = real.doctor(offline=True)
assert FAKE_KEY not in raw and FAKE_SECRET not in raw and FAKE_SECRET not in json.dumps(doc), "secret leaked"
# mirror_paper while disabled -> all refused, nothing raised
use_config("base")
os.environ["TRADER_JOURNAL_DIR"] = str(j0)
some_run = next(r["run_id"] for r in ledger(j0) if r["type"] == "fill")
assert real.mirror_paper(some_run) == []  # no strategy allowed -> no twins at all
use_config("allowed_but_disabled", lambda c: c["real"].update(allowed_strategy_ids=["rsi_regime_l4"]))
mir = real.mirror_paper(some_run)
assert mir and all(m["status"] == "refused" for m in mir), mir
print("ok real refuses when disabled, caps enforced, no secrets logged")

# ── 8. regression: env redirects cannot enable real trading; HARD ceilings; git provenance ─────────
# (review issue: TRADER_CONFIG + TRADER_JOURNAL_DIR pointing at an agent-written config turned real trading on)
from trader.core import validate  # noqa: E402

assert json.loads(REAL_FIXED_CONFIG.read_text())["real"]["real_trading_enabled"] is False
os.environ["BYBIT_API_KEY"], os.environ["BYBIT_API_SECRET"] = FAKE_KEY, FAKE_SECRET


def evil(cfg):
    cfg["real"].update(real_trading_enabled=True, allowed_strategy_ids=["evil", "rsi_regime_l4"],
                       allowed_coins=["BTC"], per_order_eur=1e9, daily_eur=1e9)


use_config("evil", evil)
je = use_journal("evil")
fixed_before = len(config.jsonl_read(real._FIXED_LEDGER))
for ci in (None, "1"):
    if ci:
        os.environ["PHIL_CI"] = ci
    else:
        os.environ.pop("PHIL_CI", None)
    for dry in ((True, False) if not ci else (False,)):
        try:
            real.place_order("BTC", "buy", 5000, client_order_id="evil-%s-%s" % (ci, dry), strategy_id="evil",
                             dry_run=dry)
            raise AssertionError("env-redirected config enabled real trading (ci=%s dry=%s)" % (ci, dry))
        except real.RealTradingDisabled as e:
            assert "real_trading_enabled is false" in str(e), e
    # live orders always read the fixed config and log to the fixed ledger, never TRADER_JOURNAL_DIR
    assert real.ledger_path(dry_run=False) == real._FIXED_LEDGER
    assert real.config_file(dry_run=False) == REAL_FIXED_CONFIG
os.environ.pop("PHIL_CI", None)
assert real.ledger_path(dry_run=True) == real._FIXED_LEDGER  # without PHIL_CI even dry runs ignore env
assert real.doctor(offline=True)["enabled"] is False
assert not [r for r in config.jsonl_read(je / "real-ledger.jsonl") if not r["dry_run"]], "live row in env journal"
assert len(config.jsonl_read(real._FIXED_LEDGER)) == fixed_before + 3  # 2 (no CI) + 1 (CI, live)
os.environ["PHIL_CI"] = "1"
# with the PHIL_CI dry-run override the evil config still hits the HARD ceilings re-checked inside real.py
try:
    real.place_order("BTC", "buy", 5000, client_order_id="evil-hard", strategy_id="evil", dry_run=True)
    raise AssertionError("HARD ceilings not re-checked")
except real.RealCapExceeded as e:
    assert "hard ceiling" in str(e), e
# live mirror ignores TRADER_JOURNAL_DIR for the paper ledger too (fixed paper ledger is empty here)
os.environ["TRADER_JOURNAL_DIR"] = str(j0)
assert real.mirror_paper(some_run, dry_run=False) == []
assert real.mirror_paper(some_run, dry_run=True)  # dry-run override still sees the test paper ledger

# HARD ceilings one by one (each just above the ceiling is refused; at the ceiling is fine)
assert real.hard_limits() == {"per_order_eur": 100.0, "daily_eur": 250.0, "max_open_positions": 10}
for key, bad in (("per_order_eur", 100.01), ("daily_eur", 250.5), ("max_open_positions", 11),
                 ("per_order_eur", float("nan")), ("daily_eur", "1e9"), ("max_open_positions", True)):
    use_config("hard_" + key, lambda c, k=key, b=bad: (enable(c), c["real"].update({k: b})))
    try:
        real.place_order("BTC", "buy", 10, client_order_id="h-%s" % key, strategy_id="rsi_regime_l4")
        raise AssertionError("hard ceiling %s=%r not enforced" % (key, bad))
    except real.RealCapExceeded as e:
        assert key in str(e), e
use_journal("hard_ok")
use_config("hard_ok", lambda c: (enable(c), c["real"].update(per_order_eur=100, daily_eur=250,
                                                              max_open_positions=10)))
assert real.place_order("BTC", "buy", 100, client_order_id="h-ok", strategy_id="rsi_regime_l4")["status"] == "dry_run"
try:
    real.place_order("BTC", "buy", 100.5, client_order_id="h-over", strategy_id="rsi_regime_l4")
    raise AssertionError("per-order cap")
except real.RealCapExceeded:
    pass

# provenance: a live order with the (fixed) flag true still needs validate.check_real_flag to pass
os.environ.pop("PHIL_CI", None)
good = copy.deepcopy(BASE_CFG)
enable(good)
fixed_cfg = TMP / "fixed-config" / "protected.json"
fixed_cfg.parent.mkdir()
fixed_cfg.write_text(json.dumps(good))
orig = (real._FIXED_CONFIG, validate.CONFIG_FILE, validate.check_real_flag, real._http, real._eur_usdt)
calls = []
ORIG_REPO_ROOT = config.REPO_ROOT


class FakeSession:
    def get_instruments_info(self, **kw):
        return {"result": {"list": [{"lotSizeFilter": {"basePrecision": "0.000001"},
                                     "priceFilter": {"tickSize": "0.01"}}]}}

    def get_tickers(self, **kw):
        return {"result": {"list": [{"lastPrice": "60000"}]}}

    def place_order(self, **kw):
        calls.append(kw)
        return {"result": {"orderId": "OID1"}}

    def get_order_history(self, **kw):
        return {"result": {"list": [{"cumExecQty": "0.0003", "avgPrice": "60100", "cumExecFee": "0.01",
                                     "cumExecValue": "18.03", "orderStatus": "Filled"}]}}


def flag_check(result):
    def f(cfg):
        assert cfg["real"]["real_trading_enabled"] is True
        assert "GIT_DIR" not in os.environ, "git env redirect not neutralised"
        assert config.REPO_ROOT == REAL_FIXED_CONFIG.parents[2]
        f.called = True
        return list(result)
    f.called = False
    return f


try:
    real._FIXED_CONFIG = fixed_cfg
    real._http = lambda rc: FakeSession()
    real._eur_usdt = lambda rc, live: 1.16
    os.environ["GIT_DIR"] = str(TMP / "evil.git")
    os.environ["TRADER_CONFIG"] = str(TMP / "evil.json")  # ignored for live orders
    # a mismatch between the gate file and validate's provenance target fails closed
    validate.check_real_flag = flag_check([])
    try:
        real.place_order("BTC", "buy", 20, client_order_id="p0", strategy_id="rsi_regime_l4", dry_run=False)
        raise AssertionError("provenance target mismatch must refuse")
    except real.RealTradingDisabled as e:
        assert "provenance" in str(e) and not validate.check_real_flag.called, e
    validate.CONFIG_FILE = str(fixed_cfg)  # absolute: repo_root()/CONFIG_FILE == fixed config
    validate.check_real_flag = flag_check(["last commit touching protected.json is not 'operator:'"])
    try:
        real.place_order("BTC", "buy", 20, client_order_id="p1", strategy_id="rsi_regime_l4", dry_run=False)
        raise AssertionError("live order without operator provenance")
    except real.RealTradingDisabled as e:
        assert "provenance" in str(e) and validate.check_real_flag.called, e
    # nothing reached the exchange; env and config.REPO_ROOT restored after the check
    assert not calls and os.environ["GIT_DIR"] == str(TMP / "evil.git") and config.REPO_ROOT == ORIG_REPO_ROOT
    validate.check_real_flag = flag_check([])
    res = real.place_order("BTC", "buy", 20, client_order_id="p2", strategy_id="rsi_regime_l4", dry_run=False)
    assert res["status"] == "filled" and len(calls) == 1 and calls[0]["category"] == "spot", res
    assert calls[0]["timeInForce"] == "IOC" and calls[0]["orderType"] == "Limit"
    assert config.jsonl_read(real._FIXED_LEDGER)[-1]["client_order_id"] == "p2"
    try:  # live per-order cap from the fixed config (20 EUR)
        real.place_order("BTC", "buy", 25, client_order_id="p3", strategy_id="rsi_regime_l4", dry_run=False)
        raise AssertionError("live per-order cap")
    except real.RealCapExceeded:
        pass
    assert len(calls) == 1
finally:
    real._FIXED_CONFIG, validate.CONFIG_FILE, validate.check_real_flag, real._http, real._eur_usdt = orig
    os.environ.pop("GIT_DIR", None)
    os.environ["PHIL_CI"] = "1"
assert FAKE_SECRET not in real._FIXED_LEDGER.read_text()
print("ok real gate reads fixed protected.json, HARD ceilings re-checked, provenance required for live")

# ── 9. regression: mirror_paper sizes twins by WEIGHT; buy, rebalance, exit leaves real flat ──────────
# (review issue: twins were min(paper notional, per_order_eur): 166/15/181 USDT paper fills became 20/12.93/20 EUR
#  twins and left 12.93 EUR of coin held in real while paper was flat)
SID = "rsi_regime_l4"


def mirror_scenario(name, fills, mutate=None):
    use_config("mirror_" + name, lambda c: (enable(c), mutate and mutate(c)))
    jm = use_journal("mirror_" + name)
    for i, (side, notional, pw, tw) in enumerate(fills):
        config.jsonl_append(jm / "paper-ledger.jsonl", {
            "type": "fill", "run_id": "P-M%s%d" % (name[:6], i), "strategy_id": SID, "sleeve_id": SID + ":BTC",
            "coin": "BTC", "side": side, "notional_usdt": notional, "prev_weight": pw, "target_weight": tw})
    twins = []
    for i in range(len(fills)):
        twins += real.mirror_paper("P-M%s%d" % (name[:6], i), dry_run=True)
    return twins, real.holdings(dry_run=True)


# the reviewer's case: buy 166 (0 -> 0.83), rebalance 15 (0.83 -> 0.90), full exit 181 (0.90 -> 0)
tw_, book = mirror_scenario("review", [("buy", 166.0, 0.0, 0.83), ("buy", 15.0, 0.83, 0.90),
                                       ("sell", 181.0, 0.90, 0.0)])
assert [t["status"] for t in tw_] == ["dry_run", "refused", "dry_run"], tw_
assert abs(tw_[0]["notional_eur"] - 0.83 * 20) < 1e-6, tw_[0]          # weight x real sleeve (20 EUR)
assert "min" in tw_[1]["reason"], tw_[1]                                # 0.07 x 20 EUR < 5 USDT: refused, logged
assert tw_[2]["closed"] and abs(tw_[2]["notional_eur"] - 0.83 * 20) < 1e-6, tw_[2]  # sells the whole holding
assert book == {}, book                                                 # net real position 0 once paper is flat
# buy, rebalance, partial sell, full exit: every twin scaled the same way, net 0 at the end
tw_, book = mirror_scenario("steps", [("buy", 100.0, 0.0, 0.5), ("buy", 60.0, 0.5, 0.8),
                                      ("sell", 80.0, 0.8, 0.4), ("sell", 80.0, 0.4, 0.0)])
assert [t["status"] for t in tw_] == ["dry_run"] * 4, tw_
got = [round(t["notional_eur"], 6) for t in tw_]
assert got == [10.0, 6.0, 8.0, 8.0], got
assert not tw_[2].get("closed") and tw_[3]["closed"] and book == {}, (tw_, book)
# real sleeve larger than the per-order cap: the scaled order is capped, never replaced by the cap
tw_, book = mirror_scenario("cap", [("buy", 100.0, 0.0, 0.5), ("buy", 40.0, 0.5, 0.7), ("sell", 140.0, 0.7, 0.0)],
                            lambda c: c["real"].update(sleeve_eur=60))
assert [t["status"] for t in tw_] == ["dry_run"] * 3, tw_
assert [round(t["notional_eur"], 6) for t in tw_] == [20.0, 12.0, 32.0], tw_  # min(30, cap 20); 0.2 x 60; close
assert book == {}, book
# a fill without weights is refused (never sized from its notional)
tw_, _ = mirror_scenario("noweights", [("buy", 166.0, None, None)])
assert tw_[0]["status"] == "refused" and "weight" in tw_[0]["reason"], tw_
# sells never exceed the simulated holding; nothing held -> refused
try:
    real.place_order("BTC", "sell", None, client_order_id="s-none", strategy_id=SID, sell_fraction=1.0)
    raise AssertionError("sell with nothing held")
except real.RealCapExceeded as e:
    assert "nothing held" in str(e), e
print("ok mirror_paper scales by weight; buy + rebalance + exit leaves the real position at 0")

# ── 10. regression: live sells are sized from the free base-coin balance, not from EUR notional ──────
# (review issue: sell qty = notional / px; Bybit spot takes the buy fee in the base coin, so the account holds
#  less coin than 20 EUR buys -> insufficient balance, position never closed; or leftover coin when price rose)
os.environ.pop("PHIL_CI", None)
orig = (real._FIXED_CONFIG, real._FIXED_LEDGER, validate.CONFIG_FILE, validate.check_real_flag, real._http,
        real._eur_usdt)


class LiveSession:
    def __init__(self, free, last="60000", min_amt="5", fill_all=True):
        self.free, self.last, self.min_amt, self.fill_all, self.orders = free, last, min_amt, fill_all, []

    def get_instruments_info(self, **kw):
        return {"result": {"list": [{"lotSizeFilter": {"basePrecision": "0.000001", "minOrderAmt": self.min_amt,
                                                       "minOrderQty": "0.000001"},
                                     "priceFilter": {"tickSize": "0.01"}}]}}

    def get_tickers(self, **kw):
        return {"result": {"list": [{"lastPrice": self.last}]}}

    def get_wallet_balance(self, **kw):
        assert kw.get("coin") == "BTC", kw
        return {"result": {"list": [{"coin": [{"coin": "BTC", "walletBalance": str(self.free), "locked": "0"}]}]}}

    def place_order(self, **kw):
        q = float(kw["qty"])
        assert kw["side"] != "Sell" or q <= self.free + 1e-12, "sell above free balance: %s > %s" % (q, self.free)
        self.orders.append(kw)
        return {"result": {"orderId": "O%d" % len(self.orders)}}

    def get_order_history(self, **kw):
        kw_ = self.orders[-1]
        q, px = float(kw_["qty"]), float(kw_["price"])
        fee = q * 0.001 if kw_["side"] == "Buy" else q * px * 0.001
        return {"result": {"list": [{"cumExecQty": str(q), "avgPrice": str(px), "cumExecFee": str(fee),
                                     "cumExecValue": str(q * px), "orderStatus": "Filled"}]}}


def live_ledger(name, rows=()):
    real._FIXED_LEDGER = TMP / ("live-" + name) / "real-ledger.jsonl"
    real._FIXED_LEDGER.parent.mkdir()
    for r in rows:
        config.jsonl_append(real._FIXED_LEDGER, dict({"ts": "2026-10-01T00:00:00Z", "dry_run": False,
                                                      "status": "filled", "coin": "BTC"}, **r))


try:
    real._FIXED_CONFIG = fixed_cfg
    validate.CONFIG_FILE = str(fixed_cfg)
    validate.check_real_flag = flag_check([])
    real._eur_usdt = lambda rc, live: 1.16

    # (a) buy 20 EUR: the base-coin fee is subtracted from the recorded qty
    live_ledger("buy")
    ses = LiveSession(free=0.0)
    real._http = lambda rc: ses
    b = real.place_order("BTC", "buy", 20, client_order_id="L-buy", strategy_id=SID, dry_run=False)
    bq = float(ses.orders[-1]["qty"])
    assert b["status"] == "filled" and abs(b["qty"] - bq * 0.999) < 1e-12, b
    # (b) close at an unchanged price: sells the free balance (bq - fee), not 20 EUR worth of coin
    ses.free = b["qty"]
    s1 = real.place_order("BTC", "sell", None, client_order_id="L-close", strategy_id=SID, dry_run=False,
                          sell_fraction=1.0)
    sq = float(ses.orders[-1]["qty"])
    assert s1["status"] == "filled" and s1["closed"] and sq <= ses.free + 1e-12 and sq < bq, (s1, sq, bq)
    assert s1["residual_qty"] < 1e-6 and real.holdings(dry_run=False) == {}, (s1, real.holdings(dry_run=False))
    # (c) a EUR-sized sell after the price fell is capped by the free balance (never insufficient balance)
    live_ledger("eur", [{"client_order_id": "x", "strategy_id": SID, "side": "buy", "notional_eur": 20, "qty": 0.000386}])
    ses = LiveSession(free=0.000386, last="55000")
    real._http = lambda rc: ses
    s2 = real.place_order("BTC", "sell", 20, client_order_id="L-eur", strategy_id=SID, dry_run=False)
    assert s2["status"] == "filled" and float(ses.orders[-1]["qty"]) <= 0.000386, (s2, ses.orders)
    # (d) close after the price rose: sells all the coin (no leftover), whatever 20 EUR would buy now
    live_ledger("up", [{"client_order_id": "x", "strategy_id": SID, "side": "buy", "notional_eur": 20, "qty": 0.000386}])
    ses = LiveSession(free=0.000386, last="80000")
    real._http = lambda rc: ses
    s3 = real.place_order("BTC", "sell", None, client_order_id="L-up", strategy_id=SID, dry_run=False,
                          sell_fraction=1.0)
    assert float(ses.orders[-1]["qty"]) == 0.000386 and s3["closed"] and s3["residual_qty"] < 1e-9, s3
    # (e) a partial sell that would strand < minOrderAmt is enlarged to the whole balance (sole holder) ...
    live_ledger("dust", [{"client_order_id": "x", "strategy_id": SID, "side": "buy", "notional_eur": 20, "qty": 0.000386}])
    ses = LiveSession(free=0.000386, last="60000")
    real._http = lambda rc: ses
    s4 = real.place_order("BTC", "sell", None, client_order_id="L-dust", strategy_id=SID, dry_run=False,
                          sell_fraction=0.9)
    assert float(ses.orders[-1]["qty"]) == 0.000386 and s4["closed"], s4
    # ... but refused, with the residual recorded, when another strategy shares the coin
    live_ledger("shared", [{"client_order_id": "x", "strategy_id": SID, "side": "buy", "notional_eur": 20,
                            "qty": 0.000386},
                           {"client_order_id": "y", "strategy_id": "other_one", "side": "buy", "notional_eur": 20,
                            "qty": 0.000386}])
    ses = LiveSession(free=0.000772, last="60000")
    real._http = lambda rc: ses
    try:
        real.place_order("BTC", "sell", None, client_order_id="L-shared", strategy_id=SID, dry_run=False,
                         sell_fraction=0.9)
        raise AssertionError("dust-stranding sell must be refused")
    except real.RealCapExceeded as e:
        assert "minOrderAmt" in str(e), e
    last_row = config.jsonl_read(real._FIXED_LEDGER)[-1]
    assert last_row["status"] == "refused" and last_row["residual_qty"] > 0 and not ses.orders, last_row
    # ... and with another strategy holding the coin, closing sells only this strategy's qty
    s5 = real.place_order("BTC", "sell", None, client_order_id="L-shared2", strategy_id=SID, dry_run=False,
                          sell_fraction=1.0)
    assert float(ses.orders[-1]["qty"]) == 0.000386 and s5["closed"], s5
    # (f) whole balance below minOrderAmt: refused with a clear reason, residual recorded, nothing sent
    live_ledger("tiny", [{"client_order_id": "x", "strategy_id": SID, "side": "buy", "notional_eur": 3, "qty": 0.00005}])
    ses = LiveSession(free=0.00005, last="60000")
    real._http = lambda rc: ses
    try:
        real.place_order("BTC", "sell", None, client_order_id="L-tiny", strategy_id=SID, dry_run=False,
                         sell_fraction=1.0)
        raise AssertionError("sub-minimum sell must be refused")
    except real.RealCapExceeded as e:
        assert "below minOrderAmt" in str(e) and "dust" in str(e), e
    last_row = config.jsonl_read(real._FIXED_LEDGER)[-1]
    assert abs(last_row["residual_qty"] - 0.00005) < 1e-12 and not ses.orders, last_row
finally:
    (real._FIXED_CONFIG, real._FIXED_LEDGER, validate.CONFIG_FILE, validate.check_real_flag, real._http,
     real._eur_usdt) = orig
    os.environ["PHIL_CI"] = "1"
print("ok live sells sized from the free base-coin balance; dust refused or swept, residual recorded")

# ── 10. regression: allocation / coin edits move capital through recorded flows ─────────────────
# (review issue: a sleeve's cash was set once at init, so an allocation cut 800 -> 600 left the sleeves at 800 and
#  report / promotion_check (equity / current allocation) showed +44.9 % for a true +8.65 %; an added coin got
#  fresh cash with no record)
def run_days(ap_fn, days):
    for d in days:
        paper.run_paper(ap_fn(d), asof=d.strftime("%Y-%m-%d"))


CUT = pd.Timestamp("2026-09-15")
SID10 = "rsi_regime_l4"


def cut_active(new_alloc=600.0, coins=None):
    def f(d):
        if d >= CUT:
            return write_active([strat(allocation_usdt=new_alloc, coins=coins or COINS)])
        return write_active([strat()])
    return f


def check_ledger_ok(rows, min_order):
    for r in rows:
        assert r["type"] in ("run", "fill", "skip", "mark"), r  # validate.check_journal row types only
        if r["type"] == "fill":
            assert r["notional_usdt"] >= min_order - 1e-9, r


# (a) zero costs: an allocation cut is a pure withdrawal -> flow-adjusted daily returns identical to the uncut run
use_config("zero10", zero)
use_journal("alloc-base")
run_days(cut_active(800.0), DAYS)
r_base = paper.returns(SID10)
eq_base = paper.equity(SID10)
assert abs(paper.contributed(SID10) - 800) < 1e-6
assert abs(paper.net_return(SID10) - (eq_base.iloc[-1] / 800 - 1)) < 1e-9  # no flows: TWR == equity / capital
jc = use_journal("alloc-cut")
run_days(cut_active(600.0), DAYS)
rows = ledger(jc)
fl = paper.flows(SID10)
assert abs(fl.loc[START] - 800) < 1e-6 and abs(fl.loc[CUT] + 200) < 1e-6, fl[fl.abs() > 1e-9]
assert abs(fl.drop([START, CUT]).abs().sum()) < 1e-9
assert abs(paper.contributed(SID10) - 600) < 1e-6, paper.contributed(SID10)
assert abs(paper.contributed(SID10, asof=CUT - pd.Timedelta(days=1)) - 800) < 1e-6
for coin in COINS:  # every sleeve now carries 150 of capital, recorded on its marks
    m = [r for r in rows if r["type"] == "mark" and r["coin"] == coin]
    assert abs(m[-1]["contributed_usdt"] - 150) < 1e-6 and abs(m[0]["contributed_usdt"] - 200) < 1e-6
    assert abs(sum(r.get("flow_usdt", 0) for r in m) - 150) < 1e-6
eq_cut = paper.equity(SID10)
assert eq_cut.loc[CUT] < eq_base.loc[CUT] - 199.99, (eq_cut.loc[CUT], eq_base.loc[CUT])  # the money really left
r_cut = paper.returns(SID10)
# every sleeve gives back the same 50 USDT (capital follows allocation / n), so the sleeve mix of the strategy
# changes slightly after the cut; per sleeve, though, the flow-adjusted return must be identical to the uncut run
assert np.allclose(r_cut[r_cut.index <= CUT].values, r_base[r_base.index <= CUT].values, atol=1e-12)
base_rows = ledger(TMP / "journal-alloc-base")


def sleeve_twr(rs, coin):
    m = [r for r in rs if r["type"] == "mark" and r["coin"] == coin]
    out, prev = [], None
    for r in m:
        e, f = float(r["equity_usdt"]), float(r.get("flow_usdt") or 0.0)
        out.append(0.0 if prev is None else (e - f) / prev - 1.0)
        prev = e
    return np.array(out)


for coin in COINS:
    rb, rc = sleeve_twr(base_rows, coin), sleeve_twr(rows, coin)
    assert len(rb) == len(rc) and np.allclose(rb, rc, atol=1e-9), (coin, np.abs(rb - rc).max())
assert (r_cut - r_base).abs().max() < 1e-3, (r_cut - r_base).abs().max()  # only the sleeve mix differs
assert abs(paper.net_return(SID10) - (eq_base.iloc[-1] / 800 - 1)) < 5e-3
# the phantom P&L is gone: equity / current allocation no longer reports the never-withdrawn 200 USDT as profit
assert abs((eq_cut.iloc[-1] / 600 - 1) - (eq_base.iloc[-1] / 800 - 1)) < 0.03, (eq_cut.iloc[-1], eq_base.iloc[-1])
check_ledger_ok(rows, 0.0)
print("ok allocation cut withdraws capital (flows recorded, TWR unchanged %.4f)" % paper.net_return(SID10))

# (b) realistic costs: cut and raise; withdraw sells respect the min order and pay costs; capital follows allocation
cfg10 = use_config("costs10")
mo = cfg10["paper"]["min_order_usdt"]
jr = use_journal("alloc-cut-costs")
run_days(cut_active(600.0), DAYS)
rows = ledger(jr)
wd = [r for r in rows if r["type"] == "fill" and r["reason"] == "withdraw"]
assert wd and all(r["side"] == "sell" and r["asof_bar"].startswith("2026-09-15") and r["fee_usdt"] > 0 for r in wd)
assert abs(paper.contributed(SID10) - 600) < 1e-6
assert all(m["cash_usdt"] >= -1e-9 for m in rows if m["type"] == "mark"), "negative cash after a withdrawal"
check_ledger_ok(rows, mo)
twr = paper.net_return(SID10)
eq = paper.equity(SID10)
assert abs(twr - (eq.iloc[-1] / 600 - 1)) < 0.03, (twr, eq.iloc[-1])
assert abs(twr - (eq.iloc[-1] / 800 - 1)) > 0.15  # what the old accounting effectively reported is far off
ju = use_journal("alloc-raise-costs")
run_days(cut_active(1000.0), DAYS)
rows = ledger(ju)
assert abs(paper.contributed(SID10) - 1000) < 1e-6 and abs(paper.flows(SID10).loc[CUT] - 200) < 1e-6
assert not [r for r in rows if r["type"] == "fill" and r["reason"] == "withdraw"]
check_ledger_ok(rows, mo)
print("ok allocation cut/raise with costs: withdraw sells >= min order, contributed == allocation")

# (c) coin added to a live strategy: new sleeve funded by a deposit, old sleeves drained to alloc / n, no free money
ja = use_journal("coin-add")
COINS5 = COINS + ["XRP"]
run_days(cut_active(800.0, COINS5), DAYS)
rows = ledger(ja)
pos = paper.positions()
assert set(pos) == {"%s:%s" % (SID10, c) for c in COINS5}
for sid, p in pos.items():
    assert abs(p["contributed_usdt"] - 160) < 1e-6, (sid, p)
assert abs(paper.contributed(SID10) - 800) < 1e-6 and abs(paper.flows(SID10).loc[CUT]) < 1e-6
xrp0 = [r for r in rows if r["type"] == "mark" and r["coin"] == "XRP"][0]
assert xrp0["asof_bar"].startswith("2026-09-15") and abs(xrp0["flow_usdt"] - 160) < 1e-6
check_ledger_ok(rows, mo)
eq = paper.equity(SID10)
assert abs(paper.net_return(SID10) - (eq.iloc[-1] / 800 - 1)) < 0.03

# (d) coin removed from a live strategy: sold, its cash withdrawn (sleeve equity 0), the rest re-funded to alloc / n
jd = use_journal("coin-remove")
run_days(cut_active(800.0, COINS[:3]), DAYS)
rows = ledger(jd)
bnb = [r for r in rows if r["type"] == "mark" and r["coin"] == "BNB"]
closed = [r for r in bnb if r.get("closed")]
assert len(closed) == 1 and closed[0]["equity_usdt"] == 0 and closed[0]["flow_usdt"] < -100, closed
pos = paper.positions()
assert set(pos) == {"%s:%s" % (SID10, c) for c in COINS[:3]}
assert all(abs(p["contributed_usdt"] - 800 / 3) < 1e-6 for p in pos.values())
# init 800, BNB's whole cash withdrawn (its flow), the 3 kept sleeves re-funded 200 -> 800/3 each (+200 in all)
assert abs(paper.contributed(SID10) - (800 + closed[0]["flow_usdt"] + 200)) < 1e-6, paper.contributed(SID10)
assert abs(paper.contributed(SID10) - paper.flows(SID10).sum()) < 1e-6
check_ledger_ok(rows, mo)
print("ok coin add/remove: deposits and withdrawals recorded, contributed capital tracks the allocation")

# (e) a ledger written before flows were recorded: contributed rebuilt from init, no second deposit
jl = use_journal("legacy")
ap = write_active([strat()])
run_days(lambda d: ap, DAYS[:5])
lp = jl / "paper-ledger.jsonl"
legacy_rows = [{k: v for k, v in r.items() if k not in ("contributed_usdt", "flow_usdt")} for r in ledger(jl)]
lp.write_text("".join(json.dumps(r) + "\n" for r in legacy_rows))
assert abs(paper.contributed(SID10) - 800) < 1e-6, paper.contributed(SID10)
paper.run_paper(ap, asof=DAYS[5].strftime("%Y-%m-%d"))
new_marks = [r for r in ledger(jl) if r["type"] == "mark" and r["asof_bar"].startswith(DAYS[5].strftime("%Y-%m-%d"))]
assert new_marks and all("flow_usdt" not in r and abs(r["contributed_usdt"] - 200) < 1e-6 for r in new_marks)
print("ok legacy ledger: contributed capital rebuilt, nothing re-deposited")

# ── 11. regression: a partial sell never strands a residual below the min order (mirror of real.py) ──────
# (review: 38.67 USDT sleeve at price 100, weight 0.6 then 0.1 left 3.86 USDT that the exit to 0 could never sell)
st = {"qty": 0.0, "cash": 348.0 / 9}
kw = dict(fee=0.001, slip=0.0005, min_order=5.0, allow_up=True, allow_down=True)
f1, s1, _ = paper._trade(st, 100.0, 0.6, **kw)
assert f1 and f1["side"] == "buy" and not s1
f2, s2, _ = paper._trade(st, 100.0, 0.1, **kw)
assert f2 and f2["side"] == "sell" and st["qty"] == 0.0, (f2, s2, st)  # whole position sold, no dust
f3, s3, _ = paper._trade(st, 100.0, 0.0, **kw)
assert f3 is None and s3 is None and st["qty"] == 0.0, (f3, s3, st)
st = {"qty": 0.0, "cash": 1000.0}  # a partial sell whose residual stays above the min order is untouched
paper._trade(st, 100.0, 0.6, **kw)
f4, _, _ = paper._trade(st, 100.0, 0.3, **kw)
assert f4 and st["qty"] > 0 and st["qty"] * 100.0 >= 5.0, st
st = {"qty": 0.5, "cash": 1.0}  # withdraw: no residual below the min order either
wf, ws, wfl = paper._withdraw(st, 100.0, 48.0, fee=0.001, slip=0.0005, min_order=5.0, can_sell=True)
assert wf and st["qty"] == 0.0 and abs(wfl + 48.0) < 1e-9 and st["cash"] > 0, (wf, ws, wfl, st)
print("ok partial sells and withdrawals never leave unsellable dust")

print("OK test_paper")
