"""Research loop tests (research area, Wave 2): strategy/tools/sweep.py + CYCLE_RESEARCH.md.

Run: /Users/matveypro/.venvs/kronos/bin/python trader/tests/test_research.py   (< 3 min, no network)

Everything is redirected to temp dirs (journal, strategy, data symlink to the lab CSVs, dispatch registry, remote
root), so the real journal is never written.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

TMP = Path(tempfile.mkdtemp(prefix="phil-research-test-"))
os.environ["TRADER_JOURNAL_DIR"] = str(TMP / "journal")
os.environ["TRADER_STRATEGY_DIR"] = str(TMP / "strategy")
os.environ["PHIL_STATE_DIR"] = str(TMP / "jobs-registry")
os.environ["PHIL_REMOTE_ROOT"] = str(TMP / "remote" / "runner")
os.environ["PHIL_RUNNER"] = "testrunner"
LAB = Path("/Users/matveypro/kronos_lab/indicators/data")
(TMP / "data").mkdir()
os.symlink(str(LAB), str(TMP / "data" / "linear"))
os.environ["TRADER_DATA_DIR"] = str(TMP / "data")
(TMP / "strategy" / "hypotheses").mkdir(parents=True)
(TMP / "journal").mkdir()

from trader.core import catalog, config, trials, validate  # noqa: E402
from trader.strategy.tools import sweep  # noqa: E402

PY = sys.executable
REPO = config.REPO_ROOT
RSI = {"tf": "1d", "mode": "SPOT", "comps": [{"t": "rsi_reg", "p": {"n": 14, "up": 55, "gap": 5}}],
       "comb": "and", "k": 1, "adx": 0, "vt": 0.5, "stop": 0}
LENIENT = {"min_coins_passing": 1, "cagr_tr": -0.99, "cagr_va": -0.99, "sh_tr": -9.0, "sh_va": -9.0,
           "mdd": -1.01, "entries": 1, "beat_random": False}


def draft(hid, **kw):
    h = {"id": hid, "title": "RSI regime grid test " + hid,
         "rationale": "Test hypothesis: research/README.md finds RSI(14) regime 55/50 + vol-target the best simple "
                      "spot rule; check neighbouring thresholds.",
         "family": "rsi_reg", "base_spec": RSI, "grid": {"comps.0.p.up": [55, 60], "adx": [0, 20]},
         "coins": ["BTC", "ETH"], "trainval_criterion": LENIENT,
         "test_criterion": {"portfolio_cagr_min": -0.99, "beat_bh": False, "beat_random_rate": False,
                            "dsr_min": 0.0},
         "max_trials": 100}
    h.update(kw)
    p = TMP / "strategy" / "hypotheses" / ("%s.json" % hid)
    p.write_text(json.dumps(h, indent=1))
    return p


def cli(*args):
    p = subprocess.run([PY, "-m", "trader.strategy.tools.sweep"] + list(args), cwd=str(REPO), capture_output=True,
                       text=True, env=dict(os.environ), timeout=170)
    try:
        out = json.loads(p.stdout.strip().splitlines()[-1])
    except Exception:
        raise AssertionError("CLI did not print JSON: rc=%s out=%r err=%r" % (p.returncode, p.stdout, p.stderr[-800:]))
    return p.returncode, out


# ----------------------------------------------------------------------------------------------- expand
def test_expand():
    h = {"base_spec": RSI, "grid": {"comps.0.p.up": [50, 55, 60], "adx": [0, 15, 20, 25]}, "coins": ["BTC", "ETH"],
         "max_trials": 100}
    specs = sweep.expand(h)
    assert len(specs) == 12, len(specs)
    assert {s["comps"][0]["p"]["up"] for s in specs} == {50, 55, 60}
    assert {s["adx"] for s in specs} == {0, 15, 20, 25}
    assert all(not catalog.validate_spec(s) for s in specs)
    assert len({catalog.spec_hash(s) for s in specs}) == 12
    # base_spec untouched
    assert RSI["comps"][0]["p"]["up"] == 55 and RSI["adx"] == 0
    # duplicates dropped (same value twice)
    assert len(sweep.expand(dict(h, grid={"adx": [0, 0, 20]}))) == 2
    # comb "and": k follows the number of comps when comps is gridded
    two = [{"t": "rsi_reg", "p": {"n": 14, "up": 55, "gap": 5}}, {"t": "ma", "p": {"kind": "sma", "n": 50}}]
    sp = sweep.expand(dict(h, grid={"comps": [RSI["comps"], two]}))
    assert sorted(s["k"] for s in sp) == [1, 2], sp
    # trial cap: 12 specs x 2 coins = 24 > 20
    try:
        sweep.expand(dict(h, max_trials=20))
        raise AssertionError("cap not enforced")
    except sweep.SweepError as e:
        assert "cap" in str(e)
    # invalid grid point / bad path / coin outside universe
    for bad in ({"grid": {"vt": [0.5, 5.0]}}, {"grid": {"comps.3.p.up": [50]}}, {"coins": ["BTC", "NOPE"]},
                {"grid": {"comps.0.t": ["no_such_component"]}}):
        try:
            sweep.expand(dict(h, **bad))
            raise AssertionError("accepted %s" % bad)
        except sweep.SweepError:
            pass
    print("ok expand")


# ----------------------------------------------------------------------------------------------- prereg
def test_prereg_and_immutability():
    assert sweep.next_id() == "H-0001"
    p = draft("H-0001")
    res = sweep.prereg("H-0001")
    assert res["n_specs"] == 4 and res["n_trials_planned"] == 8, res
    h = json.loads(p.read_text())
    assert h["hash"] == trials.hypothesis_hash(h) and h["created"] and h["author"].startswith("research-cycle@")
    assert trials.load_hypothesis("H-0001")["hash"] == h["hash"]
    assert (trials.hypothesis_status("H-0001") or {}).get("status") == "registered"
    # idempotent
    assert sweep.prereg("H-0001")["hash"] == h["hash"]
    assert len(trials.hypothesis_rows("H-0001")) == 1
    # editing a registered hypothesis is refused (immutable); the file is left as the agent wrote it
    saved = p.read_text()
    h2 = dict(h, coins=["BTC", "ETH", "SOL"])
    p.write_text(json.dumps(h2))
    try:
        sweep.prereg("H-0001")
        raise AssertionError("edited registered hypothesis accepted")
    except sweep.SweepRefused:
        pass
    p.write_text(saved)
    # the strategy check of the guard accepts the file
    errs, _warns = validate.check_strategy()
    assert not [e for e in errs if "H-0001" in e], errs
    # bad drafts
    draft("H-0002", rationale="too short")
    try:
        sweep.prereg("H-0002")
        raise AssertionError("short rationale accepted")
    except sweep.SweepError:
        pass
    (TMP / "strategy" / "hypotheses" / "H-0002.json").unlink()
    assert sweep.next_id() == "H-0002"
    print("ok prereg")


# ----------------------------------------------------------------------------------------------- run + open
def test_run_inproc_and_open_vault():
    t0 = time.time()
    n0 = trials.dsr_context()["n_trials"]
    res = sweep.run("H-0001", host="inproc", procs=2, n_random=12)
    assert res["passed"] is True, res["reasons"]
    assert res["n_trials"] == 8, res["n_trials"]
    assert res["batch_trial_id"] and res["batch_trial_id"].startswith("B-")
    assert res["random"]["n_random"] > 0
    assert res["selected"] and res["selected"][0]["coins"], res["selected"]
    assert res["summary_file"] and Path(config.TRADER_ROOT / res["summary_file"]).exists() or \
        Path(res["summary_file"]).exists()
    # ledger: batch row (research view, hypothesis hash), baseline rows, re-evaluation singles
    b = trials.read_trials(kind="batch", hypothesis_id="H-0001")
    assert b and b[-1]["view"] == "research" and b[-1]["n_trials"] == 8
    assert b[-1]["hypothesis_hash"] == trials.load_hypothesis("H-0001")["hash"]
    assert len(trials.read_trials(kind="baseline")) == res["random"]["n_random"]
    singles = trials.read_trials(kind="single", hypothesis_id="H-0001")
    assert singles and all(r["view"] == "research" and r["period"] == "trainval" for r in singles)
    assert trials.dsr_context()["n_trials"] >= n0 + 8
    st = trials.hypothesis_status("H-0001")
    assert st["status"] == "passed_trainval" and st["selected"] == res["selected"]
    # the sweep result never contains sealed bars: the stored result is research view
    sw = json.loads((Path(os.environ["TRADER_JOURNAL_DIR"]) / "sweeps" / ("%s.json" % res["job_id"])).read_text())
    assert sw["view"] == "research"
    # a hypothesis is swept once
    try:
        sweep.run("H-0001", host="inproc")
        raise AssertionError("second sweep accepted")
    except sweep.SweepRefused:
        pass
    # ONE look at the sealed test
    rep = sweep.open_vault("H-0001", "passed lenient TRAIN/VAL criterion (test)", n_random=15)
    assert rep["look_id"].startswith("L-"), rep
    assert rep["pairs"] and all(p["trial_id"] for p in rep["pairs"])
    assert rep["portfolio"]["n_bars"] > 300, rep["portfolio"]       # 2025-07-01 .. 2026-10 on 1d
    assert rep["buy_and_hold"]["total"] is not None
    assert rep["random_baseline"]["n_random"] > 0
    assert rep["portfolio"]["dsr"] is not None and rep["dsr_context"]["n_trials"] >= 8
    assert rep["passed"] is True, rep["reasons"]                     # criterion is trivially lenient
    looks = config.jsonl_read(trials.looks_path())
    assert len(looks) == 1 and looks[0]["hypothesis_id"] == "H-0001"
    assert set(looks[0]["coins"]) == set(rep["selected"][0]["coins"])
    full = trials.read_trials(kind="single", hypothesis_id="H-0001", view="full")
    assert full and all(r["look_id"] == rep["look_id"] and r["period"] == "test" for r in full)
    assert trials.hypothesis_status("H-0001")["status"] == "test_pass"
    # never a second look for the same hypothesis
    try:
        sweep.open_vault("H-0001", "again")
        raise AssertionError("second look accepted")
    except (sweep.SweepRefused, trials.LookBudgetExceeded, trials.HypothesisError):
        pass
    assert len(config.jsonl_read(trials.looks_path())) == 1
    print("ok run+open (%.0fs)" % (time.time() - t0))


def test_failing_hypothesis_cannot_open_vault():
    draft("H-0002", trainval_criterion=dict(LENIENT, cagr_tr=5.0), grid={"adx": [0, 20]})
    sweep.prereg("H-0002")
    res = sweep.run("H-0002", host="inproc", procs=1, n_random=0)
    assert res["passed"] is False and res["selected"] == [] and res["reasons"]
    assert trials.hypothesis_status("H-0002")["status"] == "failed_trainval"
    n_looks = len(config.jsonl_read(trials.looks_path()))
    try:
        sweep.open_vault("H-0002", "should be refused")
        raise AssertionError("vault opened for a failed hypothesis")
    except sweep.SweepRefused:
        pass
    # trials.request_look refuses on its own as well (not only the tool)
    try:
        trials.request_look("H-0002", [catalog.spec_hash(RSI)], ["BTC"], "bypass attempt")
        raise AssertionError("request_look accepted a failed hypothesis")
    except (trials.HypothesisError, trials.LookBudgetExceeded):
        pass
    assert len(config.jsonl_read(trials.looks_path())) == n_looks
    print("ok failed hypothesis sealed")


def test_beat_random_gate():
    # strict gate on a grid that passes nothing: grid pass rate 0 can never beat the random rate
    draft("H-0003", trainval_criterion=dict(LENIENT, cagr_tr=5.0, beat_random=True), grid={"adx": [0]})
    sweep.prereg("H-0003")
    res = sweep.run("H-0003", host="inproc", procs=1, n_random=6)
    assert res["passed"] is False
    assert any("random" in r for r in res["reasons"]), res["reasons"]
    print("ok beat-random gate")


def test_dispatch_local_host_path():
    """The fleet path (dispatch.submit -> runner via /bin/sh nohup -> status poll -> collect)."""
    draft("H-0004", grid={"adx": [0, 20]}, coins=["BTC"])
    sweep.prereg("H-0004")
    res = sweep.run("H-0004", host="local", n_random=0, poll_s=1.0, timeout_min=2.5)
    assert res.get("passed") is True, res
    assert res["n_trials"] == 2 and res["batch_trial_id"]
    rows = [r for r in trials.read_trials(kind="batch", hypothesis_id="H-0004")]
    assert len({r["job_id"] for r in rows}) == 1        # provisional + final rows of ONE job
    print("ok dispatch local host")


def test_daily_sweep_budget_and_cli():
    (TMP / "strategy" / "schedule.json").write_text(json.dumps(
        {"version": 1, "research": {"enabled": True, "every_hours": 6, "max_minutes": 60, "max_sweeps_per_day": 1}}))
    draft("H-0005", grid={"adx": [0]}, coins=["BTC"])
    rc, out = cli("prereg", "H-0005")
    assert rc == 0 and out["n_trials_planned"] == 1, out
    rc, out = cli("expand", "H-0005", "--specs")
    assert rc == 0 and out["n_specs"] == 1 and out["specs"][0]["spec_hash"], out
    rc, out = cli("run", "H-0005", "--host", "inproc", "--n-random", "0")
    assert rc == 2 and out["refused"] and "max_sweeps_per_day" in out["error"], out
    assert trials.hypothesis_status("H-0005")["status"] == "registered"   # nothing started
    rc, out = cli("open", "H-0005", "--reason", "not passed")
    assert rc == 2 and out["refused"], out
    rc, out = cli("status", "H-0001")
    assert rc == 0 and out["latest"]["status"] == "test_pass" and out["looks"]["used"] == 1, out
    rc, out = cli("next-id")
    assert rc == 0 and out["next_id"] == "H-0006", out
    (TMP / "strategy" / "schedule.json").unlink()
    print("ok budget + cli")


def test_journal_valid_and_cycle_doc():
    errs = [e for e in validate.check_journal() if "vault_access" not in e]
    assert not errs, errs
    doc = (config.TRADER_ROOT / "CYCLE_RESEARCH.md").read_text()
    for must in ("Hard rules", "lease", "prereg", "sweep run", "sweep open", "playbook", "cycles.log",
                 "request_look", "random", "Deflated Sharpe", "proposals.md", "core/", "evidence"):
        assert must.lower() in doc.lower(), must
    # every command the procedure tells the agent to run must be a trader module (loop.sh allowlist)
    for line in re.findall(r"`(python3? [^`]*)`", doc):
        assert re.match(r"python3? -m trader\.", line), line
    print("ok journal + cycle doc")


if __name__ == "__main__":
    t = time.time()
    test_expand()
    test_prereg_and_immutability()
    test_run_inproc_and_open_vault()
    test_failing_hypothesis_cannot_open_vault()
    test_beat_random_gate()
    test_dispatch_local_host_path()
    test_daily_sweep_budget_and_cli()
    test_journal_valid_and_cycle_doc()
    print("OK test_research (%.0fs)" % (time.time() - t))
