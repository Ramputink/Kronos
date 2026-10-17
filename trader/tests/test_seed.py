"""Seed area tests: strategy/active.json, schedule.json, hypotheses H-0001..H-0004, playbook.md, backlog.md.

Run: /Users/matveypro/.venvs/kronos/bin/python trader/tests/test_seed.py   (no network, < 1 min)
Reads the real trader/strategy files; writes nothing outside a temp dir.
"""
import copy
import itertools
import json
import os
import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

_TMP = tempfile.mkdtemp(prefix="seed_test_")
os.environ["TRADER_JOURNAL_DIR"] = os.path.join(_TMP, "journal")  # never touch the real journal
os.environ.pop("TRADER_STRATEGY_DIR", None)                           # read the REAL strategy dir

import pandas as pd  # noqa: E402

from trader.core import config, catalog, paper, trials, validate, backtest  # noqa: E402

SD = config.TRADER_ROOT / "strategy"
assert config.strategy_dir().resolve() == SD.resolve(), config.strategy_dir()
CFG = config.load_config()
LAB = "/Users/matveypro/kronos_lab/indicators/data"
RSI_SPEC = {"tf": "1d", "mode": "SPOT", "comps": [{"t": "rsi_reg", "p": {"n": 14, "up": 55, "gap": 5}}],
            "comb": "and", "k": 1, "adx": 0, "vt": 0.5, "stop": 0}
L9 = ["BTC", "ETH", "SOL", "XRP", "BNB", "DOGE", "ADA", "LINK", "AVAX"]  # research engine.UNIV
SEED_DATE = pd.Timestamp("2026-10-16")


def load(name):
    return json.loads((SD / name).read_text())


def set_path(spec, dotted, value):
    cur = spec
    parts = dotted.split(".")
    for p in parts[:-1]:
        cur = cur[int(p)] if isinstance(cur, list) else cur[p]
    last = parts[-1]
    if isinstance(cur, list):
        cur[int(last)] = value
    else:
        cur[last] = value


def expand(h):
    keys = sorted(h["grid"])
    out = []
    for combo in itertools.product(*[h["grid"][k] for k in keys]):
        s = copy.deepcopy(h["base_spec"])
        for k, v in zip(keys, combo):
            set_path(s, k, v)
        if s["comb"] == "and":
            s["k"] = len(s["comps"])
        out.append(s)
    return out


def test_active():
    a = load("active.json")
    errs = paper.validate_active(a)
    assert errs == [], errs
    strat = {s["id"]: s for s in a["strategies"]}
    assert set(strat) == {"rsi_regime_l4", "rsi_regime_l9"}, list(strat)
    for s in strat.values():
        assert s["status"] == "paper"
        assert s["spec"] == RSI_SPEC, s["spec"]
        assert catalog.spec_hash(s["spec"]) == "cd9ff45458116162a26b8d3c8fb059ec"
        assert pd.Timestamp(s["activated_at"]) >= SEED_DATE
        assert pd.Timestamp(s["activated_at"]) > pd.Timestamp(CFG["vault"]["vault_cutoff"])
        assert (SD / "hypotheses" / (s["hypothesis_id"] + ".json")).exists(), s["hypothesis_id"]
        assert s["evidence"] and "research/" in s["evidence"]
        assert s["allocation_usdt"] / len(s["coins"]) >= CFG["paper"]["min_order_usdt"] * 5  # room for vt trims
    assert strat["rsi_regime_l4"]["coins"] == ["BTC", "ETH", "SOL", "BNB"]
    assert strat["rsi_regime_l9"]["coins"] == L9
    # capital split 1000 EUR, weights 0.5 / 0.3 / 0.2 (C4 reserved in cash)
    cap = a["capital"]
    assert cap["capital_eur"] == 1000
    assert abs(cap["capital_usdt"] - cap["capital_eur"] * cap["eur_usdt"]) < 1e-6
    assert abs(cap["eur_usdt"] - CFG["real"]["eur_usdt_fallback"]) < 1e-9
    w = cap["weights"]
    assert abs(sum(w.values()) - 1.0) < 1e-9
    for sid, s in strat.items():
        assert abs(s["allocation_usdt"] - w[sid] * cap["capital_usdt"]) < 0.5, sid
    alloc = sum(s["allocation_usdt"] for s in strat.values())
    assert abs(alloc + cap["unallocated_usdt"] - cap["capital_usdt"]) < 0.5
    assert abs(cap["unallocated_usdt"] - w["breakout_c4_basket"] * cap["capital_usdt"]) < 0.5
    assert alloc <= CFG["paper"]["bankroll_usdt"]
    # negative control: validator catches a PERP spec and a TEST-window activation
    bad = copy.deepcopy(a)
    bad["strategies"][0]["spec"]["mode"] = "PERP"
    bad["strategies"][1]["activated_at"] = "2025-06-01"  # (2025-08-01 passes while forward_start is null: open issue)
    assert len(paper.validate_active(bad)) >= 2


def test_schedule():
    s = load("schedule.json")
    assert s["version"] == 1
    r = s["research"]
    assert r["enabled"] is True and r["every_hours"] >= CFG["schedule"]["research_every_hours"]
    assert 0 < r["max_minutes"] <= 120 and 1 <= r["max_sweeps_per_day"] <= 24 // r["every_hours"]
    assert re.match(r"^\d\d:\d\d$", s["screener"]["at_utc"])
    assert s["screener"]["at_utc"] > CFG["schedule"]["forward_cycle"]  # after the paper run
    assert set(s["screener"]["coins"]) <= set(CFG["universe"])
    assert s["data_update"]["spot_every_minutes"] >= 15


def test_hypotheses():
    files = sorted((SD / "hypotheses").glob("H-*.json"))
    assert [f.stem for f in files] == ["H-0001", "H-0002", "H-0003", "H-0004"], files
    cap = CFG["research"]["max_trials_per_hypothesis"]
    lab_cache = {}
    for f in files:
        h = trials.load_hypothesis(f)  # raises on hash mismatch
        assert h["id"] == f.stem
        assert trials.load_hypothesis(h["id"])["hash"] == h["hash"]
        for k in ("created", "author", "title", "rationale", "family", "base_spec", "grid", "coins",
                  "trainval_criterion", "test_criterion", "max_trials", "hash"):
            assert k in h, (f.name, k)
        assert catalog.validate_spec(h["base_spec"]) == [], (f.name, catalog.validate_spec(h["base_spec"]))
        assert h["base_spec"]["mode"] == "SPOT" and h["base_spec"]["tf"] == "1d"
        assert set(h["coins"]) <= set(CFG["universe"])
        specs = expand(h)
        for s in specs:
            assert catalog.validate_spec(s) == [], (f.name, s, catalog.validate_spec(s))
        assert len({catalog.spec_hash(s) for s in specs}) == len(specs), f.name
        n = len(specs) * len(h["coins"])
        assert n <= min(h["max_trials"], cap), (f.name, n, h["max_trials"])
        tv = h["trainval_criterion"]
        assert 1 <= tv["min_coins_passing"] <= len(h["coins"]) and tv["mdd"] < 0
        assert h["test_criterion"]["dsr_min"] >= CFG["research"]["dsr_min"]
        # smoke: every expanded spec runs on the research view (ETH, trainval), no lookahead past the cutoff
        if "ETH" not in lab_cache:
            df = pd.read_csv(f"{LAB}/ETH_1d.csv", index_col=0, parse_dates=True)
            df = df[df.index < pd.Timestamp(CFG["vault"]["vault_cutoff"])].copy()
            df["funding"] = 0.0
            df.attrs = {"coin": "ETH", "tf": "1d", "view": "research", "source": "linear",
                        "cutoff": CFG["vault"]["vault_cutoff"], "data_end": str(df.index[-1])}
            lab_cache["ETH"] = (df, catalog.make_d(df))
        df, d = lab_cache["ETH"]
        for s in specs:
            res = backtest.evaluate_df(s, df, "trainval", cost_side=backtest.cost_side("SPOT"), d=d)
            assert res["metrics"] is not None and res["returns"].index.max() < pd.Timestamp("2025-07-01")
    # provenance hypotheses carry exactly the active specs/coins
    a = {s["id"]: s for s in load("active.json")["strategies"]}
    h1, h2 = trials.load_hypothesis("H-0001"), trials.load_hypothesis("H-0002")
    assert h1["base_spec"] == a["rsi_regime_l4"]["spec"] and h1["coins"] == a["rsi_regime_l4"]["coins"]
    assert h2["base_spec"] == a["rsi_regime_l9"]["spec"] and h2["coins"] == a["rsi_regime_l9"]["coins"]
    # tamper check: editing a field breaks the hash
    t = json.loads(files[2].read_text())
    t["grid"]["adx"] = [0, 30]
    assert t["hash"] != trials.hypothesis_hash(t)


def test_validate_strategy_dir():
    errs, warns = validate.check_strategy(CFG)
    assert errs == [], errs


def test_playbook():
    p = (SD / "playbook.md").read_text()
    heads = re.findall(r"^## (.+)$", p, flags=re.M)
    for sec in ("Findings", "Rules of honesty", "Active strategies", "Graveyard", "Open questions"):
        assert any(h.startswith(sec) for h in heads), (sec, heads)
    for needle in ("rsi_regime_l4", "rsi_regime_l9", "breakout_c4_basket", "cd9ff45458116162a26b8d3c8fb059ec",
                   "6.3 %", "9.4 %", "Kronos", "vault_cutoff", "Deflated Sharpe", "random", "buy & hold",
                   "LLM calibration"):
        assert needle in p, needle
    # every research path the playbook cites exists
    root = config.REPO_ROOT
    for rel in set(re.findall(r"`(research/[^`\s]+)`", p)):
        assert (root / rel).exists(), rel
    for rel in set(re.findall(r"`((?:indicators|overnight|top10|goal15)/[^`\s]+)`", p)):
        assert (root / "research" / "kronos_lab" / rel).exists(), rel


def test_backlog():
    b = (SD / "hypotheses" / "backlog.md").read_text()
    rows = [ln for ln in b.splitlines() if re.match(r"^\| B-\d\d \|", ln)]
    assert 10 <= len(rows) <= 15, len(rows)
    ids = [re.match(r"^\| (B-\d\d) \|", r).group(1) for r in rows]
    assert len(set(ids)) == len(ids)
    for r in rows:
        cells = [c.strip() for c in r.strip("|").split("|")]
        assert len(cells) == 8, (len(cells), r[:60])
        assert re.search(r"\*\*[^*]*[+−≈≤≥<>][^*]*\*\*", cells[6]), ("expected sign missing", cells[0])
        assert len(cells[6]) > 40, ("why missing", cells[0])
    for hid in ("H-0001", "H-0002", "H-0003", "H-0004"):
        assert hid in b, hid


def test_seed_evidence_reproduces():
    """The active rule on ETH (lab CSV, sealed TEST window already opened in research/) at protected spot costs
    keeps the sign and rough size the playbook claims (+23 %/yr at 0.25 % fee → ~+22 % at 0.30 %)."""
    df = pd.read_csv(f"{LAB}/ETH_1d.csv", index_col=0, parse_dates=True)
    df["funding"] = 0.0
    df.attrs = {"coin": "ETH", "tf": "1d", "view": "full", "source": "linear", "cutoff": "2025-07-01",
                "data_end": str(df.index[-1])}
    res = backtest.evaluate_df(RSI_SPEC, df, "test", cost_side=backtest.cost_side("SPOT"))
    cagr = res["metrics"]["cagr"]
    assert 0.15 < cagr < 0.268, cagr
    assert res["bh_metrics"]["cagr"] < cagr


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print("ok", t.__name__)
    print("OK test_seed")
