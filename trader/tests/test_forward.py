"""Forward-loop tests: a dry-run harness that executes the deterministic (non-LLM) parts of FORWARD / DEEP_RETRO
cycles end to end on local data (no network, no claude, no git writes to the repo). Run:
    /Users/matveypro/.venvs/kronos/bin/python trader/tests/test_forward.py
"""
import contextlib
import copy
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

LAB = "/Users/matveypro/kronos_lab/indicators/data"
TMP = Path(tempfile.mkdtemp(prefix="phil-forward-test-"))
(TMP / "data").mkdir()
os.symlink(LAB, TMP / "data" / "spot")
os.symlink(LAB, TMP / "data" / "linear")
os.environ["TRADER_DATA_DIR"] = str(TMP / "data")
os.environ["TRADER_STRATEGY_DIR"] = str(TMP / "strategy")
(TMP / "strategy").mkdir()
os.environ.pop("PHIL_LEASE", None)
os.environ["PHIL_RUNNER"] = "testrunner"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from trader.core import config  # noqa: E402

T0 = time.time()
TRADER = Path(__file__).resolve().parents[1]
BASE_CFG = json.loads((TRADER / "config" / "protected.json").read_text())
assert BASE_CFG["forward"]["forward_start"] is None, "shipped config must keep forward_start null"
assert BASE_CFG["real"]["real_trading_enabled"] is False


def use_config(name, forward_start="2026-08-01"):
    cfg = copy.deepcopy(BASE_CFG)
    cfg["forward"]["forward_start"] = forward_start
    p = TMP / ("%s.json" % name)
    p.write_text(json.dumps(cfg))
    os.environ["TRADER_CONFIG"] = str(p)
    return cfg


def use_journal(name):
    d = TMP / ("journal-" + name)
    d.mkdir(exist_ok=True)
    os.environ["TRADER_JOURNAL_DIR"] = str(d)
    return d


SPEC = {"tf": "1d", "mode": "SPOT", "comps": [{"t": "rsi_reg", "p": {"n": 14, "up": 55, "gap": 5}}],
        "comb": "and", "k": 1, "adx": 0, "vt": 0.5, "stop": 0}
STRATS = [
    {"id": "rsi_regime_l4", "status": "paper", "spec": SPEC, "coins": ["BTC", "ETH", "SOL", "BNB"],
     "allocation_usdt": 580, "activated_at": "2026-08-15", "hypothesis_id": "H-0001", "evidence": "test",
     "screener_veto": False},
    {"id": "rsi_regime_l2_late", "status": "paper", "spec": SPEC, "coins": ["XRP", "ADA"],
     "allocation_usdt": 200, "activated_at": "2026-09-10", "hypothesis_id": "H-0002", "evidence": "test",
     "screener_veto": False},
]


def write_active(strategies):
    p = TMP / "strategy" / "active.json"
    p.write_text(json.dumps({"version": 1, "updated": config.iso(), "strategies": strategies}))
    return p


cfg = use_config("cfg")
J = use_journal("main")
write_active(STRATS)

from trader.core import paper  # noqa: E402
from trader.strategy.tools import report  # noqa: E402

assert paper.validate_active(json.loads((TMP / "strategy" / "active.json").read_text())) == []

# ── 1. simulate 2 months of daily FORWARD settlements (what loop.sh runs before the agent) ──────────────
LAST = pd.Timestamp("2026-10-12")
days = pd.date_range("2026-08-14", LAST, freq="D")
for d in days:
    r = paper.run_paper(asof=str(d.date()))
    assert r["status"] == "ok", r
again = paper.run_paper(asof=str(LAST.date()))
assert again["status"] == "skipped", again  # idempotent per bar
ledger_before = (J / "paper-ledger.jsonl").read_bytes()
trials_exists_before = (J / "trials.jsonl").exists()

# ── 2. daily report: forward score vs buy&hold and vs random baseline ──────────────────────────────────
rep = report.daily_report(str(LAST.date()), n_random=25)
assert rep["status"] == "ok", rep["warnings"]
assert rep["asof_bar"] == "2026-10-12T00:00:00Z"
assert rep["paper"]["ran_for_asof"] is True
assert not [w for w in rep["warnings"] if "active.json" in w], rep["warnings"]
by = {s["id"]: s for s in rep["strategies"]}
assert set(by) == {"rsi_regime_l4", "rsi_regime_l2_late"}
side = BASE_CFG["costs"]["spot"]["fee_side"] + BASE_CFG["costs"]["spot"]["slippage_side"]
for sid, s in by.items():
    strat = next(x for x in STRATS if x["id"] == sid)
    eq = paper.equity(sid)
    eq = eq[eq.index <= LAST]
    # first mark is the close of the bar before activated_at (the close where the first fill happens)
    first_bar = pd.Timestamp(strat["activated_at"]) - pd.Timedelta(days=1)
    assert eq.index.min() == first_bar, (sid, eq.index.min())
    assert s["forward_days"] == len(eq), (sid, s["forward_days"], len(eq))
    assert abs(s["net_return"] - (eq.iloc[-1] / strat["allocation_usdt"] - 1)) < 1e-6
    assert abs(s["equity_usdt"] - eq.iloc[-1]) < 1e-3
    # buy & hold recomputed independently from the raw lab closes
    bhs = []
    for c in strat["coins"]:
        cl = pd.read_csv(Path(LAB) / ("%s_1d.csv" % c), index_col=0, parse_dates=True)["close"]
        bhs.append(cl.loc[LAST] / cl.loc[first_bar] * (1 - side) - 1)
    assert abs(s["bh_return"] - float(np.mean(bhs))) < 1e-6, (sid, s["bh_return"], np.mean(bhs))
    assert s["beats_bh"] == (s["net_return"] > s["bh_return"])
    assert 0.0 <= s["random_percentile"] <= 1.0 and s["n_random"] == 25
    assert s["maxdd"] <= 0.0
    pr = s["promotion"]
    assert pr["eligible_preview"] is False
    assert any("forward_days" in x for x in pr["reasons"]), pr
    assert s["days_since_activation"] == (LAST - pd.Timestamp(strat["activated_at"])).days + 1
pf = rep["portfolio"]
assert abs(pf["allocation_usdt"] - 780) < 1e-9 and pf["equity_usdt"] > 0
# writes nothing: paper ledger byte-identical, no trials rows (random baseline is computed with log=False)
assert (J / "paper-ledger.jsonl").read_bytes() == ledger_before
assert (J / "trials.jsonl").exists() == trials_exists_before
# deterministic: same seed -> same percentile
rep2 = report.daily_report(str(LAST.date()), n_random=25)
assert [s["random_percentile"] for s in rep2["strategies"]] == [s["random_percentile"] for s in rep["strategies"]]
# the binding protected check (deep retro) agrees with the report and leaves a journal validate accepts
from trader.core import validate  # noqa: E402
pb = report.promotion("rsi_regime_l4", str(LAST.date()))
assert pb["eligible"] is False and pb["real_trading_enabled"] is False and pb["action"] == "none"
assert abs(pb["net_return"] - by["rsi_regime_l4"]["net_return"]) < 1e-6
assert abs(pb["bh_return"] - by["rsi_regime_l4"]["bh_return"]) < 1e-6
assert validate.check_journal() == [], validate.check_journal()
# promotion-rule mirror flips to eligible only when every protected condition holds
fake = dict(by["rsi_regime_l4"], forward_days=120, net_return=0.2, bh_return=0.05, beats_bh=True,
            random_percentile=0.9, n_random=500)
assert report.promotion_mirror(fake, cfg)["eligible_preview"] is True
assert report.promotion_mirror(dict(fake, n_random=25), cfg)["eligible_preview"] is False
assert report.promotion_mirror(dict(fake, beats_bh=False), cfg)["eligible_preview"] is False
assert report.promotion_mirror(dict(fake, status="paused"), cfg)["eligible_preview"] is False

# ── 3. screener forecasts: log one, settle it through core/forecast (sole writer) ──────────────────────
fc_mod = report._forecast_module()
if fc_mod is not None:
    qs = json.loads((TRADER / "strategy" / "screener_questions.json").read_text())["questions"]
    q = next(x for x in qs if x.get("kind") == "binary" and x.get("scored", True) and x.get("tf", "1d") == "1d")
    real_now = fc_mod._now
    fc_mod._now = lambda: pd.Timestamp("2026-09-21 00:20:00")  # the screener runs at 00:20 UTC after the close
    try:
        fid = fc_mod.log_forecast(q, "ETH", "2026-09-20T00:00:00Z", [0.4, 0.6], model="test", prompt_hash="0" * 64)
    finally:
        fc_mod._now = real_now
    assert report.screener_summary(LAST)["pending_due"] == 1
    st = report.settle_forecasts(str(LAST.date()))
    assert st["available"] and st["resolved"] == 1, st
    assert st["rows"][0]["forecast_id"] == fid
    assert report.settle_forecasts(str(LAST.date()))["resolved"] == 0  # idempotent
    sc = report.screener_summary(LAST)
    assert sc["n_resolved"] == 1 and sc["pending_due"] == 0 and sc["veto_allowed"] is False, sc
    assert report.activity(LAST)["resolved_forecasts_24h"] == 1
    errs = report.largest_errors(24, top=3, asof=str(LAST.date()))
    assert errs["forecasts"] and errs["forecasts"][0]["forecast_id"] == fid
else:
    st = report.settle_forecasts()
    assert st["available"] is False and st["resolved"] == 0
    errs = report.largest_errors(24 * 90, top=3, asof=str(LAST.date()))
assert isinstance(errs["strategy_days"], list)

# ── 4. retro skeleton + cycles.log line ────────────────────────────────────────────────────────────────
md = report.render_markdown(rep, "forward")
for h in ("# Forward retro — bar 2026-10-12", "## Scoreboard", "rsi_regime_l4", "## Diagnosis (agent)",
          "## Edits (agent", "## Proposals"):
    assert h in md, h
line = report.format_cycles_line("forward", "ok", {"runner": "air", "fills": 3}, "settled 4 sleeves")
assert report.CYCLES_LOG_RE.match(line)
try:
    report.format_cycles_line("bogus", "ok", {})
    raise AssertionError("bad kind accepted")
except ValueError:
    pass

# ── 5. the end-to-end dry run of a forward cycle ──────────────────────────────────────────────────────
NEXT = LAST + pd.Timedelta(days=1)
ledger_before = (J / "paper-ledger.jsonl").read_bytes()
dr = report.cycle_dry_run(str(NEXT.date()), write=False, n_random=5)
assert dr["status"] == "ok", dr["paper"]
assert dr["paper"]["status"] == "dry_run" and dr["paper"]["asof_bar"] == "2026-10-13T00:00:00Z"
assert (J / "paper-ledger.jsonl").read_bytes() == ledger_before  # dry run wrote nothing to the ledger
assert report.CYCLES_LOG_RE.match(dr["cycles_line"]), dr["cycles_line"]
assert "## Scoreboard" in dr["retro_markdown"]
wr = report.cycle_dry_run(str(NEXT.date()), write=True, n_random=5)
assert wr["paper"]["status"] == "ok", wr["paper"]
assert len((J / "paper-ledger.jsonl").read_bytes()) > len(ledger_before)
assert report.cycle_dry_run(str(NEXT.date()), write=True, n_random=0)["paper"]["status"] == "skipped"

# ── 6. TRIGGERED-style pause: paused strategy only de-risks, report still scores it ────────────────────
paused = copy.deepcopy(STRATS)
paused[1]["status"] = "paused"
write_active(paused)
rp = report.daily_report(str(NEXT.date()), n_random=0)
assert {s["id"]: s["status"] for s in rp["strategies"]}["rsi_regime_l2_late"] == "paused"
assert next(s for s in rp["strategies"] if s["id"] == "rsi_regime_l2_late")["promotion"]["eligible_preview"] is False
write_active(STRATS)

# ── 7. forward not started: everything degrades to a clear status, nothing is read ────────────────────
use_config("cfg-null", forward_start=None)
J2 = use_journal("null")
rn = report.daily_report(str(LAST.date()), n_random=5)
assert rn["status"] == "not_started" and all(s["forward_days"] == 0 for s in rn["strategies"])
dn = report.cycle_dry_run(str(LAST.date()), n_random=5)
assert dn["status"] == "error" and "forward_start is null" in dn["paper"]["error"], dn["paper"]
assert not (J2 / "paper-ledger.jsonl").exists()
use_config("cfg")
use_journal("main")

# ── 8. deep retro helpers: active.json semantic diff over git history, proposals parsing ───────────────
old = {"strategies": copy.deepcopy(STRATS)}
new = copy.deepcopy(old)
new["strategies"][0]["allocation_usdt"] = 500
new["strategies"][1]["status"] = "retired"
new["strategies"].append(dict(STRATS[0], id="rsi_regime_new", allocation_usdt=100, evidence="retro 2026-10-12"))
ch = report.active_diff(old, new)
kinds = {(c["id"], c["change"]) for c in ch}
assert ("rsi_regime_l4", "allocation_usdt") in kinds and ("rsi_regime_l2_late", "status") in kinds
assert ("rsi_regime_new", "added") in kinds and len(ch) == 3
assert report.active_diff(new, old)[-1]["change"].startswith("deleted")

repo = TMP / "repo"
(repo / "trader" / "strategy").mkdir(parents=True)
genv = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t", GIT_COMMITTER_NAME="t",
            GIT_COMMITTER_EMAIL="t@t")


def git(*a):
    subprocess.run(["git", "-c", "core.hooksPath=/dev/null", "-c", "commit.gpgsign=false", "-C", str(repo)] + list(a),
                   check=True, capture_output=True, env=genv)


git("init", "-q")
(repo / "trader" / "strategy" / "active.json").write_text(json.dumps(old))
git("add", "-A")
git("commit", "-q", "-m", "operator: seed")
(repo / "trader" / "strategy" / "active.json").write_text(json.dumps(new))
(repo / "trader" / "strategy" / "playbook.md").write_text("x\n")
git("add", "-A")
git("commit", "-q", "-m", "cycle(forward): 2026-10-13 trim l4, retire l2")
ed = report.recent_edits(24, repo=repo)
assert ed["ok"] and len(ed["commits"]) == 2 and ed["n_agent_commits"] == 1, ed
top = ed["commits"][0]
assert top["subject"].startswith("cycle(forward)") and top["agent"]
assert {(c["id"], c["change"]) for c in top["active_changes"]} == kinds
assert ed["commits"][1]["active_changes"][0]["change"] == "added"  # seed commit: everything added
assert report.recent_edits(24, repo=TMP / "not-a-repo")["ok"] is False

(J / "proposals.md").write_text(
    "# Proposals\n\n## P-20261013-01 Promote rsi_regime_l4 to real\n- Status: open\n- Evidence: ...\n\n"
    "## P-20261001-01 Lower spot slippage\nStatus: rejected\n\n## P-20261002-01 broken\nno status here\n")
pp = report.proposals()
assert pp["exists"] and pp["open"] == 1 and [p["id"] for p in pp["proposals"]][0] == "P-20261013-01"
assert pp["malformed"] == ["P-20261002-01"]

# ── 9. CLI prints one JSON object and exits 0 ─────────────────────────────────────────────────────────
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    rc = report.main(["daily", "--asof", str(LAST.date()), "--n-random", "3"])
assert rc == 0 and json.loads(buf.getvalue())["asof_bar"] == "2026-10-12T00:00:00Z"
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    rc = report.main(["daily", "--asof", str(LAST.date()), "--n-random", "0", "--md"])
assert rc == 0 and "## Scoreboard" in buf.getvalue()
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    rc = report.main(["proposals"])
assert rc == 0 and json.loads(buf.getvalue())["open"] == 1
os.environ["PHIL_LEASE"] = "acquired"
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    rc = report.main(["cycle-dry-run", "--write"])
assert rc == 1 and "refused inside a cycle" in buf.getvalue()  # the agent can never write via the harness
os.environ.pop("PHIL_LEASE")

# ── 10. the cycle procedures are consistent with the code they drive ───────────────────────────────────
SUBCMDS = {"daily", "settle-forecasts", "edits", "errors", "proposals", "promotion", "cycle-dry-run"}
for name, must in (("CYCLE_FORWARD.md", ("real_trading_enabled", "proposals.md", "retros/", "active.json",
                                          "TRIGGERED", "evidence", "report daily", "settle-forecasts")),
                   ("CYCLE_DEEP_RETRO.md", ("keep", "sharpen", "revert", "promotion", "proposals.md",
                                            "real_trading_enabled", "report edits", "report errors"))):
    text = (TRADER / name).read_text()
    for m in must:
        assert m in text, (name, m)
    for sub in re.findall(r"trader\.strategy\.tools\.report ([a-z-]+)", text):
        assert sub in SUBCMDS, (name, sub)
    for mod in set(re.findall(r"-m (trader(?:\.[a-z_]+)+)", text)):
        assert (TRADER.parent / (mod.replace(".", "/") + ".py")).exists(), (name, mod)
    assert "Never run git" in text and "Do not commit" in text  # loop.sh owns git, lease and commits
    assert "Never flip real trading" in text or "Real money is out of scope" in text

assert time.time() - T0 < 180, "test too slow"
print("OK test_forward (%.1fs)" % (time.time() - T0))
