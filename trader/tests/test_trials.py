"""Tests for core/trials.py and core/score.py (trials area). Plain script: python trader/tests/test_trials.py"""
import json
import math
import os
import random
import sys
import tempfile
import time
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

TMP = Path(tempfile.mkdtemp(prefix="phil_trials_"))
os.environ["TRADER_JOURNAL_DIR"] = str(TMP / "journal")
os.environ["TRADER_STRATEGY_DIR"] = str(TMP / "strategy")
(TMP / "data").mkdir()
LAB = Path("/Users/matveypro/kronos_lab/indicators/data")
if LAB.exists():
    os.symlink(LAB, TMP / "data" / "linear")
    os.symlink(LAB, TMP / "data" / "spot")
os.environ["TRADER_DATA_DIR"] = str(TMP / "data")
os.environ["PHIL_RUNNER"] = "test"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from trader.core import config, score, trials  # noqa: E402

T0 = time.time()


def approx(a, b, tol):
    return abs(a - b) <= tol


# ------------------------------------------------------------------------------------ score: stats + metrics
def test_norm():
    assert approx(score.norm_cdf(0), 0.5, 1e-12)
    assert approx(score.norm_ppf(0.975), 1.959964, 1e-5)
    assert approx(score.norm_cdf(score.norm_ppf(0.123)), 0.123, 1e-9)


def test_metrics():
    rng = np.random.default_rng(0)
    r = rng.normal(0.001, 0.02, 1000)
    pos = (rng.random(1000) > 0.5).astype(float)
    m = score.metrics(pd.Series(r), 365, pos)
    g = np.prod(1 + r)
    assert approx(m["cagr"], g ** (365 / 1000) - 1, 1e-12)
    assert approx(m["total"], g - 1, 1e-12)
    assert approx(m["sharpe"], r.mean() / r.std() * math.sqrt(365), 1e-12)  # numpy ddof=0 like worker.py
    eq = np.cumprod(1 + r)
    assert approx(m["maxdd"], (eq / np.maximum.accumulate(eq) - 1).min(), 1e-12)
    assert approx(m["kurt"], pd.Series(r).kurt() + 3, 1e-12)  # raw kurtosis
    assert approx(m["skew"], pd.Series(r).skew(), 1e-12)
    w = pos
    assert m["entries"] == int(((w[1:] != 0) & (np.sign(w[1:]) != np.sign(w[:-1]))).sum())
    assert approx(m["exposure"], pos.mean(), 1e-12)
    assert m["n_obs"] == 1000 and m["bpy"] == 365
    assert score.metrics(r[:29], 365) is None  # < 30 bars
    # equity wiped out -> cagr -1
    bad = np.r_[np.full(40, 0.0), [-1.0], np.zeros(10)]
    assert score.metrics(bad, 365)["cagr"] == -1.0
    # flat returns: sharpe 0, no crash
    z = score.metrics(np.zeros(50), 365, np.zeros(50))
    assert z["sharpe"] == 0.0 and z["entries"] == 0
    # metrics dict is JSON-serialisable
    json.dumps(m)


def test_buy_and_hold():
    close = pd.Series(np.linspace(100, 200, 101), index=pd.date_range("2024-01-01", periods=101))
    bh = score.buy_and_hold(close, 365, entry_cost=0.003)
    r = close.pct_change().iloc[1:].values.copy()
    r[0] -= 0.003
    assert bh["n_obs"] == 100
    assert approx(bh["total"], float(np.prod(1 + r)) - 1, 1e-12), bh
    assert approx(bh["total"], 2.0 - 1 - 0.003 * 2.0 / 1.01, 1e-9)
    bh0 = score.buy_and_hold(close, 365)  # default = protected spot side 0.0030
    assert approx(bh0["total"], bh["total"], 1e-12)
    assert bh["exposure"] == 1.0


def test_brier_portfolio():
    assert approx(score.brier([0.7, 0.2, 0.1], 0), 0.09 + 0.04 + 0.01, 1e-12)
    assert approx(score.brier([1.0, 0.0], 0), 0.0, 1e-12)
    idx = pd.date_range("2025-01-01", periods=3)
    a = {"returns": pd.Series([0.1, 0.1, 0.1], idx)}
    b = {"returns": pd.Series([-0.1, -0.1, -0.1], idx)}
    eq = score.portfolio([a, b])
    # no rebalancing: mean of 1.1^3 and 0.9^3
    assert approx(eq.iloc[-1], (1.1 ** 3 + 0.9 ** 3) / 2, 1e-12)
    pr = score.portfolio_returns([a, b])
    assert approx(float(np.prod(1 + pr.values)), eq.iloc[-1], 1e-12)


# ------------------------------------------------------------------------------------ DSR
def test_dsr():
    # final_test_output.txt: N=960, daily SR std 0.0226 -> expected max SR0 = 0.0732 (annualised 1.40)
    sr0 = score.expected_max_sr(960, 0.0226 ** 2)
    assert approx(sr0, 0.0732, 0.0005), sr0
    assert approx(sr0 * math.sqrt(365), 1.40, 0.01)
    # sr == sr0 -> DSR 0.5
    assert approx(score.deflated_sharpe(sr0, 1000, 0.0, 3.0, n_trials=960, sr_var_daily=0.0226 ** 2), 0.5, 1e-9)
    # rsi55_50 finalist: annual SR train+val 1.45 -> DSR 0.55 in final_test (n ~ 2000+ days, skew/kurt near 0/≈normal+)
    sr = 1.45 / math.sqrt(365)
    d = score.deflated_sharpe(sr, 2100, 0.0, 3.0, n_trials=960, sr_var_daily=0.0226 ** 2)
    assert 0.5 < d < 0.75, d
    # monotone: more trials -> lower DSR; more obs -> more extreme
    d_more = score.deflated_sharpe(sr, 2100, 0.0, 3.0, n_trials=96000, sr_var_daily=0.0226 ** 2)
    assert d_more < d
    # goal15 formula identity (independent reimplementation)
    N, V, em = 50, 0.0004, 0.5772156649
    s0 = math.sqrt(V) * ((1 - em) * score.norm_ppf(1 - 1 / N) + em * score.norm_ppf(1 - 1 / (N * math.e)))
    srd, n, sk, ku = 0.08, 700, -0.4, 7.0
    z = (srd - s0) * math.sqrt(n - 1) / math.sqrt(1 - sk * srd + (ku - 1) / 4 * srd ** 2)
    assert approx(score.deflated_sharpe(srd, n, sk, ku, n_trials=N, sr_var_daily=V), score.norm_cdf(z), 1e-12)
    # dsr_of_metrics converts annual -> daily and bars -> days
    m = {"sharpe": 1.45, "n_obs": 2100, "skew": 0.0, "kurt": 3.0, "bpy": 365}
    assert approx(score.dsr_of_metrics(m, n_trials=960, sr_var_daily=0.0226 ** 2), d, 1e-12)
    # DSR sanity on noise: best of many pure-noise strategies should NOT look significant
    rng = np.random.default_rng(7)
    R = rng.normal(0, 0.02, (300, 730))
    srs = R.mean(1) / R.std(1)
    i = int(np.argmax(srs))
    raw_p = score.norm_cdf(srs[i] * math.sqrt(729))  # naive (undeflated) PSR is "significant"
    best = score.deflated_sharpe(srs[i], 730, float(pd.Series(R[i]).skew()), float(pd.Series(R[i]).kurt() + 3),
                                 n_trials=300, sr_var_daily=float(srs.var(ddof=1)))
    assert raw_p > 0.95 and best < 0.9, (raw_p, best)


# ------------------------------------------------------------------------------------ hypotheses + looks
def write_hyp(hid, **over):
    h = {"id": hid, "created": "2026-10-21T02:00:00Z", "author": "test", "title": f"test {hid}",
         "rationale": "synthetic", "family": "rsi_reg",
         "base_spec": {"tf": "1d", "mode": "SPOT", "comps": [{"t": "rsi_reg", "p": {"n": 14, "up": 55, "gap": 5}}],
                       "comb": "and", "k": 1, "adx": 0, "vt": 0.5, "stop": 0},
         "grid": {"adx": [0, 20]}, "coins": ["BTC", "ETH", "SOL", "BNB"],
         "trainval_criterion": {}, "test_criterion": {}, "max_trials": 8}
    h.update(over)
    h["hash"] = trials.hypothesis_hash(h)
    d = config.strategy_dir() / "hypotheses"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{hid}.json"
    p.write_text(json.dumps(h, indent=1))
    return p


def expect(exc, fn, *a, **k):
    try:
        fn(*a, **k)
    except exc:
        return
    except Exception as e:  # noqa
        if exc.__name__ == "VaultSealed" and e.__class__.__name__ == "VaultSealed":
            return
        raise AssertionError(f"expected {exc.__name__}, got {e.__class__.__name__}: {e}")
    raise AssertionError(f"expected {exc.__name__}, nothing raised")


def test_hypotheses_and_looks():
    p = write_hyp("H-0001")
    h = trials.load_hypothesis("H-0001")
    assert h["hash"] == trials.hypothesis_hash(h)
    r1 = trials.register_hypothesis(p)
    r2 = trials.register_hypothesis(p)  # idempotent
    assert r1["hypothesis_hash"] == r2["hypothesis_hash"]
    assert len([r for r in config.jsonl_read(trials.hypotheses_path()) if r["status"] == "registered"]) == 1
    # tampered file -> mismatch
    bad = json.loads(p.read_text())
    bad["coins"].append("DOGE")
    (p.parent / "H-0099.json").write_text(json.dumps(dict(bad, id="H-0099")))
    expect(trials.HypothesisError, trials.load_hypothesis, "H-0099")
    expect(trials.HypothesisError, trials.set_status, "H-0099", "sweeping")  # unregistered
    expect(trials.HypothesisError, trials.set_status, "H-0001", "bogus")

    # a look before passed_trainval is refused
    expect(trials.HypothesisError, trials.request_look, "H-0001", ["abc"], ["ETH"], "too early")
    trials.set_status("H-0001", "sweeping")
    trials.set_status("H-0001", "passed_trainval", trials=8, summary="ok")
    # coins outside the hypothesis are refused
    expect(trials.HypothesisError, trials.request_look, "H-0001", ["abc"], ["DOGE"], "coin not registered")
    lid = trials.request_look("H-0001", ["abc"], ["ETH", "BNB"], "passed TRAIN/VAL")
    month = config.utcnow().strftime("%Y-%m")
    assert lid == f"L-{month}-0001", lid
    assert trials.looks_used() == 1
    row = trials.check_look(lid, "abc", "ETH")
    assert row["budget_used"] == 1 and row["budget_total"] == 8
    trials.check_look(lid, "abc", "BNB")
    VS = trials._vault_sealed()
    expect(VS, trials.check_look, lid, "other_spec", "ETH")
    expect(VS, trials.check_look, "L-1999-01-0001", "abc", "ETH")
    expect(VS, trials.check_look, None, "abc", "ETH")
    # regression (review: look did not bind coin): a look for ETH/BNB never opens SOL, and no coin = refused
    expect(VS, trials.check_look, lid, "abc", "SOL")
    expect(VS, trials.check_look, lid, "abc")
    # regression (review: spec_hash=None opened everything): only the baseline purpose, only covered coins
    expect(VS, trials.check_look, lid, None, "ETH")
    expect(VS, trials.check_look, lid, None)
    trials.check_look(lid, None, "ETH", purpose="baseline")
    expect(VS, trials.check_look, lid, None, "SOL", purpose="baseline")
    expect(VS, trials.check_look, lid, None, "ETH", purpose="anything")
    with trials._baseline_scope(lid, caller="test"):
        trials.check_look(lid, None, "BNB")
        expect(VS, trials.check_look, lid, None, "BTC")
        expect(VS, trials.check_look, "L-1999-01-0001", None, "ETH")
    expect(VS, trials.check_look, lid, None, "BNB")  # scope closed
    assert trials.look_coins(lid) == ["ETH", "BNB"] and trials.look_coins("nope") == []
    # second look for the same hypothesis refused
    expect(trials.LookBudgetExceeded, trials.request_look, "H-0001", ["abc"], ["ETH"], "again")

    # budget: 8 per month -> 9th refused
    for i in range(2, 9):
        hid = f"H-00{i:02d}"
        trials.register_hypothesis(write_hyp(hid))
        trials.set_status(hid, "passed_trainval")
        trials.request_look(hid, ["s"], ["BTC"], "budget fill")
    assert trials.looks_used() == 8
    trials.register_hypothesis(write_hyp("H-0009"))
    trials.set_status("H-0009", "passed_trainval")
    expect(trials.LookBudgetExceeded, trials.request_look, "H-0009", ["s"], ["BTC"], "ninth")
    assert trials.looks_used() == 8
    assert trials.looks_used("1999-01") == 0

    # expiry: rewrite a look row with a past expiry (test-only manipulation of a temp journal)
    rows = config.jsonl_read(trials.looks_path())
    rows[0]["expires"] = "2000-01-01T00:00:00Z"
    trials.looks_path().write_text("".join(json.dumps(r) + "\n" for r in rows))
    expect(VS, trials.check_look, lid, "abc")

    # hypothesis edited after registration -> refused at look time and at re-registration
    trials.register_hypothesis(write_hyp("H-0010"))
    trials.set_status("H-0010", "passed_trainval")
    write_hyp("H-0010", title="edited after the fact")
    expect(trials.HypothesisError, trials.register_hypothesis, config.strategy_dir() / "hypotheses" / "H-0010.json")
    # (budget already spent, but the hash check fires first)
    expect(trials.HypothesisError, trials.request_look, "H-0010", ["s"], ["BTC"], "edited")

    # CLI looks
    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = trials._main(["looks"])
    out = json.loads(buf.getvalue())
    assert rc == 0 and out["used"] == 8 and out["remaining"] == 0, out


# ------------------------------------------------------------------------------------ trials ledger + DSR context
def fake_result(sharpe_seed, coin="ETH", view="research", period="trainval", tf="1d", n=400, drift=0.0):
    rng = np.random.default_rng(sharpe_seed)
    idx = pd.date_range("2022-01-01", periods=n, freq="D")
    r = pd.Series(rng.normal(drift, 0.02, n), idx)
    pos = pd.Series(np.ones(n), idx)
    m = score.metrics(r, 365, pos)
    bh = score.metrics(pd.Series(rng.normal(0, 0.02, n), idx), 365)
    return {"spec_hash": f"h{sharpe_seed}", "spec": {"tf": tf}, "coin": coin, "tf": tf, "mode": "SPOT", "view": view,
            "period": period, "cost_side": 0.003, "returns": r, "positions": pos, "target": pos, "metrics": m,
            "bh_metrics": bh, "trial_id": None}


def test_ledger():
    tids = [trials.log_trial(fake_result(i, drift=0.002 if i == 3 else 0.0), caller="research",
                             hypothesis_id="H-0001") for i in range(10)]
    assert len(set(tids)) == 10 and all(t.startswith("T-") for t in tids)
    rows = trials.read_trials(kind="single")
    assert len(rows) == 10
    r0 = rows[0]
    for k in ("trial_id", "ts", "kind", "caller", "runner", "hypothesis_id", "hypothesis_hash", "spec_hash", "spec",
              "coin", "tf", "mode", "view", "period", "look_id", "cost_side", "data_end", "metrics"):
        assert k in r0, k
    assert r0["hypothesis_hash"] == trials._registered_hash("H-0001")
    assert r0["data_end"].endswith("Z") and r0["runner"] == "test"
    # baseline rows
    for i in range(20):
        trials.log_trial(fake_result(100 + i), caller="research", kind="baseline")
    assert len(trials.read_trials(kind="baseline")) == 20
    assert len(trials.read_trials(caller=["research"])) == 30
    assert trials.read_trials(since="2999-01-01") == []
    # invalid usage
    expect(ValueError, trials.log_trial, fake_result(1), caller="hacker")
    expect(trials._vault_sealed(), trials.log_trial, fake_result(1, view="full", period="test"), caller="research")
    # batch row
    job = {"job_id": "J-1", "host": "bigsur", "n_configs": 100, "n_trials": 1000, "n_pass": 3,
           "sr_moments": {"1d": [600, 120.0, 600 * (0.4 ** 2 + 0.2 ** 2)], "4h": [400, 40.0, 400 * (0.5 ** 2 + 0.1 ** 2)]},
           "periods": ["train", "val"], "cost_side": 0.003}
    bid = trials.log_batch(job, hypothesis_id="H-0001")
    assert bid.startswith("B-")
    b = trials.read_trials(kind="batch")[0]
    assert b["n_trials"] == 1000 and b["result_file"] == "journal/sweeps/J-1.json" and b["runner"] == "bigsur"
    expect(ValueError, trials.log_batch, dict(job, view="full"), hypothesis_id="H-0001")

    # DSR context: N = 10 singles + 1000 batch trials, baseline excluded
    ctx = trials.dsr_context()
    assert ctx["n_trials"] == 1010, ctx
    sh = [r["metrics"]["sharpe"] for r in rows]
    n = 10 + 600 + 400
    s = sum(sh) + 120.0 + 40.0
    ss = sum(x * x for x in sh) + 600 * (0.4 ** 2 + 0.2 ** 2) + 400 * (0.5 ** 2 + 0.1 ** 2)
    var = (ss - s * s / n) / (n - 1)
    assert approx(ctx["sr_var_daily"], var / 365, 1e-12), (ctx, var / 365)
    ctx1d = trials.dsr_context(tf="1d")
    assert ctx1d["n_trials"] == 610
    ctx4h = trials.dsr_context(tf="4h")
    assert ctx4h["n_trials"] == 400

    # report
    rep = trials.report()
    assert rep["counts"]["single"] == 10 and rep["counts"]["baseline"] == 20 and rep["counts"]["batch"] == 1
    assert rep["dsr_context"]["n_trials"] == 1010
    assert rep["best"]["spec_hash"] == "h3", rep["best"]
    assert 0.0 <= rep["best"]["dsr"] <= 1.0
    br = rep["beats_random"]
    assert br["n_selected"] == 10 and br["n_random"] == 20
    assert rep["looks"]["used"] == 8
    json.dumps(trials._jsonable(rep))


# ------------------------------------------------------------------------------------ random baseline (stubbed)
def _install_stubs():
    """Synthetic catalog/backtest/vault so random_baseline_rate mechanics are tested independently of the other
    Wave-1 areas. Each spec has an 'edge' parameter: daily drift of the strategy returns."""
    saved = {k: sys.modules.get(k) for k in ("trader.core.catalog", "trader.core.backtest", "trader.core.vault")}

    class VaultSealed(Exception):
        pass

    vault = types.ModuleType("trader.core.vault")
    vault.VaultSealed = VaultSealed
    loads = []

    def load(coin, tf, view="research", *, source=None, look_id=None, spec_hash=None, asof=None, caller=None):
        if view == "full":
            trials.check_look(look_id, spec_hash, coin)
        loads.append((coin, tf, view))
        n = 600
        idx = pd.date_range("2023-01-01", periods=n, freq="D")
        rng = np.random.default_rng(abs(hash(coin)) % 1000)
        c = 100 * np.cumprod(1 + rng.normal(0, 0.02, n))
        df = pd.DataFrame({"open": c, "high": c, "low": c, "close": c, "volume": 1.0, "turnover": 1.0, "funding": 0.0},
                          index=idx)
        df.attrs = {"coin": coin, "tf": tf, "view": view}
        return df

    vault.load = load

    catalog = types.ModuleType("trader.core.catalog")

    def rand_cfg(rng, modes=("SPOT", "PERP"), tfs=("1d", "1d", "4h")):
        return {"tf": rng.choice(tfs), "mode": rng.choice(modes), "comps": [{"t": "x", "p": {"edge": rng.uniform(-0.002, 0.002)}}],
                "comb": "and", "k": 1, "adx": 0, "vt": 0, "stop": 0}

    catalog.rand_cfg = rand_cfg

    backtest = types.ModuleType("trader.core.backtest")

    def evaluate_df(spec, df, period="all", *, cost_side, activated_at=None):
        edge = spec["comps"][0]["p"]["edge"]
        rng = np.random.default_rng(int(abs(edge) * 1e9) % 2 ** 32)
        r = pd.Series(rng.normal(edge, 0.02, len(df)), df.index)
        pos = pd.Series(np.ones(len(df)), df.index)
        return {"spec_hash": f"e{edge:.6f}", "coin": df.attrs.get("coin"), "tf": spec["tf"], "mode": spec["mode"],
                "view": df.attrs.get("view"), "period": period, "cost_side": cost_side, "returns": r, "positions": pos,
                "target": pos, "metrics": score.metrics(r, 365, pos),
                "bh_metrics": score.buy_and_hold(df["close"], 365), "trial_id": None}

    def evaluate(spec, coin, view="research", period="trainval", *, costs=None, look_id=None, caller,
                 hypothesis_id=None, activated_at=None, log=True):
        res = evaluate_df(spec, load(coin, spec["tf"], view, look_id=look_id, spec_hash="sel"), period,
                          cost_side=0.003)
        res["coin"] = coin
        if log:
            res["trial_id"] = trials.log_trial(res, caller=caller, hypothesis_id=hypothesis_id, look_id=look_id)
        return res

    backtest.evaluate_df = evaluate_df
    backtest.evaluate = evaluate
    backtest.cost_side = lambda mode, costs=None: 0.003
    sys.modules["trader.core.vault"] = vault
    sys.modules["trader.core.catalog"] = catalog
    sys.modules["trader.core.backtest"] = backtest
    import trader.core as tc
    for k in ("vault", "catalog", "backtest"):
        saved["attr_" + k] = getattr(tc, k, None)
        setattr(tc, k, sys.modules["trader.core." + k])
    return saved, loads


def _remove_stubs(saved):
    import trader.core as tc
    for k in ("vault", "catalog", "backtest"):
        mod = saved.get("trader.core." + k)
        if mod is None:
            sys.modules.pop("trader.core." + k, None)
        else:
            sys.modules["trader.core." + k] = mod
        a = saved.get("attr_" + k)
        if a is None:
            if hasattr(tc, k):
                delattr(tc, k)
        else:
            setattr(tc, k, a)


def test_random_baseline_stub():
    saved, loads = _install_stubs()
    try:
        before = len(trials.read_trials(kind="baseline"))
        sel = [({"tf": "1d", "mode": "SPOT", "comps": [{"t": "x", "p": {"edge": 0.004}}], "comb": "and", "k": 1,
                 "adx": 0, "vt": 0, "stop": 0}, c) for c in ("BTC", "ETH", "SOL")]
        out = score.random_baseline_rate(sel, period="trainval", view="research", n_random=150, seed=1,
                                         coins=["BTC", "ETH", "SOL", "BNB"], tfs=("1d",), caller="test")
        assert out["n_selected"] == 3 and out["n_random"] == 150, out
        assert out["rate_selected"] == 1.0 and out["rate_random"] < 0.6 and out["beats_random"], out
        assert len(trials.read_trials(kind="baseline")) - before == 150
        # same seed -> same random draws -> same rate (reproducible baseline)
        out2 = score.random_baseline_rate([], period="trainval", view="research", n_random=150, seed=1,
                                          coins=["BTC", "ETH", "SOL", "BNB"], tfs=("1d",), caller="test")
        assert out2["rate_random"] == out["rate_random"] and not out2["beats_random"]
        # random draws follow morning_test order: rand_cfg(rng) then rng.choice(coins)
        rng = random.Random(1)
        from trader.core import catalog
        exp = []
        for _ in range(3):
            s = catalog.rand_cfg(rng, modes=("SPOT",), tfs=("1d",))
            exp.append(rng.choice(["BTC", "ETH", "SOL", "BNB"]))
        b = trials.read_trials(kind="baseline")[before:before + 3]
        assert [x["coin"] for x in b] == exp, ([x["coin"] for x in b], exp)
        # full view without a look is sealed
        expect(sys.modules["trader.core.vault"].VaultSealed, score.random_baseline_rate, [], period="test", view="full",
               n_random=2, coins=["BTC"], tfs=("1d",), caller="test")
        # regression: full-view random baseline only draws/loads coins the look covers (logged as baseline)
        trials.register_hypothesis(write_hyp("H-0300", coins=["ETH", "BTC"]))
        trials.set_status("H-0300", "passed_trainval")
        cfg_saved = config.load_config
        try:
            def _cfg():
                c = json.loads(json.dumps(cfg_saved()))
                c["vault"]["looks_per_month"] = 99
                return c
            config.load_config = _cfg
            lb = trials.request_look("H-0300", ["sel"], ["ETH"], "baseline regression")
        finally:
            config.load_config = cfg_saved
        nb = len(trials.read_trials(kind="baseline"))
        n_loads = len(loads)
        fo = score.random_baseline_rate([], period="test", view="full", look_id=lb, n_random=20, seed=3,
                                        coins=["BTC", "ETH", "SOL"], tfs=("1d",), caller="test")
        assert fo["n_random"] == 20, fo
        assert {c for c, _, v in loads[n_loads:] if v == "full"} == {"ETH"}, loads[n_loads:]
        newb = trials.read_trials(kind="baseline")[nb:]
        assert len(newb) == 20 and all(r["coin"] == "ETH" and r["look_id"] == lb for r in newb)
        # outside the score path, spec_hash=None on the full view is refused (the old "any data" hole)
        expect(sys.modules["trader.core.vault"].VaultSealed, sys.modules["trader.core.vault"].load, "ETH", "1d",
               "full", look_id=lb, spec_hash=None)
        expect(sys.modules["trader.core.vault"].VaultSealed, score.random_baseline_rate, [], period="test",
               view="full", look_id=lb, n_random=2, coins=["SOL"], tfs=("1d",), caller="test")
        score.random_percentile(pd.Series(0.001, pd.date_range("2024-06-01", periods=50, freq="D")), ["ETH"],
                                period="test", view="full", look_id=lb, n_random=5, caller="test")
        expect(sys.modules["trader.core.vault"].VaultSealed, score.random_percentile,
               pd.Series(0.001, pd.date_range("2024-06-01", periods=50, freq="D")), ["SOL"], period="test",
               view="full", look_id=lb, n_random=5, caller="test")
        # random_percentile: a strong strategy beats most random spot configs
        idx = pd.date_range("2024-06-01", periods=200, freq="D")
        strong = pd.Series(0.004, idx)
        p = score.random_percentile(strong, ["BTC", "ETH"], period="all", view="paper", n_random=60, caller="forward")
        weak = pd.Series(-0.004, idx)
        q = score.random_percentile(weak, ["BTC", "ETH"], period="all", view="paper", n_random=60, caller="forward")
        assert p > 0.9 and q < 0.1, (p, q)
        assert trials.read_trials(kind="baseline")[-1]["summary"]["what"] == "random_percentile"
        # baselines never count in DSR N
        n_before = trials.dsr_context()["n_trials"]
        trials.log_trial(fake_result(999), caller="test", kind="baseline")
        assert trials.dsr_context()["n_trials"] == n_before
    finally:
        _remove_stubs(saved)


# ------------------------------------------------------------------------------------ promotion (stubbed paper)
def test_promotion_stub():
    saved, _ = _install_stubs()
    paper_saved = sys.modules.get("trader.core.paper")
    try:
        sd = config.strategy_dir()
        sd.mkdir(parents=True, exist_ok=True)
        (sd / "active.json").write_text(json.dumps({"version": 1, "strategies": [
            {"id": "s_good", "status": "paper", "spec": {}, "coins": ["BTC", "ETH"], "allocation_usdt": 1000,
             "activated_at": "2023-03-01"},
            {"id": "s_young", "status": "paper", "spec": {}, "coins": ["BTC"], "allocation_usdt": 500,
             "activated_at": "2023-03-01"}]}))
        paper = types.ModuleType("trader.core.paper")

        def equity(strategy_id=None, *, since=None):
            n = 120 if strategy_id == "s_good" else 20
            idx = pd.date_range("2023-03-01", periods=n, freq="D")
            alloc = 1000 if strategy_id == "s_good" else 500
            return pd.Series(alloc * 1.005 ** np.arange(1, n + 1), idx)

        paper.equity = equity
        sys.modules["trader.core.paper"] = paper
        # capital records (paper ledger): each strategy's sleeves were funded with its allocation on day 1
        _write_ledger([_mark("s_good", "BTC", "2023-03-01", 500.0), _mark("s_good", "ETH", "2023-03-01", 500.0),
                       _mark("s_young", "BTC", "2023-03-01", 500.0)])
        g = score.promotion_check("s_good")
        assert g["forward_days"] == 120 and g["net_return"] > 0.5, g
        assert g["random_percentile"] is not None and g["random_percentile"] >= 0.8, g
        assert g["eligible"], g
        # regression (review, off-by-one): B&H buys at the close of the FIRST MARK bar (not one bar earlier) and the
        # random window starts the bar after it (same anchoring as strategy/tools/report.score_strategy)
        from trader.core import vault as _v
        bh = []
        for c in ("BTC", "ETH"):
            cl = _v.load(c, "1d", "paper")["close"]
            bh.append(float(cl.loc["2023-06-28"] / cl.loc["2023-03-01"]) * (1 - score._spot_side()) - 1)
        assert approx(g["bh_return"], float(np.mean(bh)), 1e-12), (g["bh_return"], bh)
        base = trials.read_trials(kind="baseline")[-1]["summary"]
        assert base["window"][0].startswith("2023-03-02") and base["window"][1].startswith("2023-06-28"), base
        y = score.promotion_check("s_young")
        assert not y["eligible"] and any("forward_days" in r for r in y["reasons"]), y
        m = score.promotion_check("missing")
        assert not m["eligible"] and m["reasons"]
    finally:
        if paper_saved is None:
            sys.modules.pop("trader.core.paper", None)
        else:
            sys.modules["trader.core.paper"] = paper_saved
        _remove_stubs(saved)


def _ledger_file():
    return config.journal_dir() / score.PAPER_LEDGER


def _write_ledger(rows):
    config.journal_dir().mkdir(parents=True, exist_ok=True)
    with open(_ledger_file(), "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def _mark(sid, coin, day, cash, qty=0.0, price=100.0, run=None):
    run = run or f"P-{day}"
    return {"type": "mark", "run_id": run, "ts": f"{day}T00:10:00Z", "asof_bar": f"{day}T00:00:00Z",
            "strategy_id": sid, "sleeve_id": f"{sid}:{coin}", "coin": coin, "price": price, "qty": qty,
            "cash_usdt": cash, "equity_usdt": cash + qty * price, "weight": 0.0}


def test_promotion_allocation_edit():
    """Regression (review): promotion_check must score from capital actually put into the sleeves (paper ledger),
    not active.json allocation_usdt. Reproduction: 800 USDT funded, allocation later cut to 600 -> old code
    reported net = eq[-1]/600 - 1 (0.449 instead of 0.087) and could manufacture eligibility."""
    saved, _ = _install_stubs()
    paper_saved = sys.modules.get("trader.core.paper")
    sys.modules.pop("trader.core.paper", None)
    try:
        from trader.core import paper  # real equity() over the ledger we write
        fee, slip = 0.0025, 0.0005
        days = pd.date_range("2023-03-01", periods=120, freq="D")
        rows = []
        p_btc, p_eth = 100 * 1.0007 ** np.arange(120), 50 * 1.0007 ** np.arange(120)
        sleeves = {}
        for coin, px in (("BTC", p_btc), ("ETH", p_eth)):
            notional = 400.0 / (1 + fee)
            q = notional / (px[0] * (1 + slip))
            cash = 400.0 - notional - notional * fee
            sleeves[coin] = (q, cash, px)
            rows.append({"type": "fill", "run_id": "P-2023-03-01", "ts": "2023-03-01T00:10:00Z",
                         "asof_bar": "2023-03-01T00:00:00Z", "strategy_id": "s_cut", "sleeve_id": f"s_cut:{coin}",
                         "coin": coin, "side": "buy", "qty": q, "price": px[0], "fill_price": px[0] * (1 + slip),
                         "notional_usdt": notional, "fee_usdt": notional * fee, "slippage_bps": 5,
                         "prev_weight": 0.0, "target_weight": 1.0, "reason": "init"})
        for i, d in enumerate(days):
            day = str(d.date())
            for coin, (q, cash, px) in sleeves.items():
                rows.append(_mark("s_cut", coin, day, cash, q, px[i]))
        _write_ledger(rows)
        flows = score.sleeve_contributions("s_cut")
        assert list(flows) == [pd.Timestamp("2023-03-01")] and approx(flows[pd.Timestamp("2023-03-01")], 800.0, 1e-9)
        eq = paper.equity("s_cut")
        true_net = float(eq.iloc[-1] / 800.0 - 1)
        sd = config.strategy_dir()
        sd.mkdir(parents=True, exist_ok=True)
        res = {}
        for alloc in (800, 600, 2000):
            (sd / "active.json").write_text(json.dumps({"version": 1, "strategies": [
                {"id": "s_cut", "status": "paper", "spec": {}, "coins": ["BTC", "ETH"], "allocation_usdt": alloc,
                 "activated_at": "2023-03-01"}]}))
            res[alloc] = score.promotion_check("s_cut")
        for alloc, r in res.items():
            assert approx(r["net_return"], true_net, 1e-9), (alloc, r["net_return"], true_net)
            assert approx(r["contributed_usdt"], 800.0, 1e-9), r
        # the allocation cut no longer changes the random-baseline input or the buy&hold verdict
        assert res[600]["random_percentile"] == res[800]["random_percentile"]
        assert [x for x in res[600]["reasons"] if "buy&hold" in x] == [x for x in res[800]["reasons"] if "buy&hold" in x]
        assert true_net < 0.10 and (eq.iloc[-1] / 600 - 1) > 0.40  # the reviewer's inflation is gone

        # a capital flow mid-window (coin added on day 60 with 300 USDT cash) is removed, not counted as return
        add = [_mark("s_cut", "SOL", str(d.date()), 300.0) for d in days[60:]]
        _write_ledger(rows + add)
        (sd / "active.json").write_text(json.dumps({"version": 1, "strategies": [
            {"id": "s_cut", "status": "paper", "spec": {}, "coins": ["BTC", "ETH", "SOL"], "allocation_usdt": 600,
             "activated_at": "2023-03-01"}]}))
        f2 = score.sleeve_contributions("s_cut")
        assert approx(f2[pd.Timestamp(days[60])], 300.0, 1e-9)
        r2 = score.promotion_check("s_cut")
        assert approx(r2["contributed_usdt"], 1100.0, 1e-9)
        # SOL earns 0, so the time-weighted return is the 2-sleeve return, diluted only after day 60
        assert 0 < r2["net_return"] <= true_net + 1e-12, (r2["net_return"], true_net)
        assert r2["net_return"] > true_net * 0.6, (r2["net_return"], true_net)
        eq2 = paper.equity("s_cut")
        assert eq2.iloc[-1] / 600 - 1 > 0.8  # what the allocation-based formula would have reported
        # window cut (asof) is measured against the previous mark, never against allocation
        r3 = score.promotion_check("s_cut", asof="2023-04-30")
        assert r3["forward_days"] == 61, r3
        exp = float(np.prod(score.flow_adjusted_returns(eq2, f2).iloc[:61] + 1) - 1)
        assert approx(r3["net_return"], exp, 1e-12)
        # flow-recording ledger (contributed_usdt/flow_usdt on marks): an allocation raise (deposit, day 50) and a
        # cut (withdrawal, day 80) are capital flows, not P&L -> net = g^(n-1) - 1 exactly, whatever the allocation
        g, eqv, contrib, new_rows = 1.002, 0.0, 0.0, []
        flows_by_day = {0: 400.0, 50: 200.0, 80: -150.0}
        for i, d in enumerate(days):
            f = flows_by_day.get(i, 0.0)
            eqv = eqv * g + f
            contrib += f
            m = _mark("s_flow", "BTC", str(d.date()), eqv)
            m["contributed_usdt"] = contrib
            if f:
                m["flow_usdt"] = f
            new_rows.append(m)
        _write_ledger(new_rows)
        assert score.sleeve_contributions("s_flow") == {pd.Timestamp(days[i]): v for i, v in flows_by_day.items()}
        for alloc in (400, 600, 450):
            (sd / "active.json").write_text(json.dumps({"version": 1, "strategies": [
                {"id": "s_flow", "status": "paper", "spec": {}, "coins": ["BTC"], "allocation_usdt": alloc,
                 "activated_at": "2023-03-02"}]}))
            r5 = score.promotion_check("s_flow")
            # activated_at 2023-03-02 -> first mark is the 2023-03-01 bar (its close is the first trade)
            assert r5["forward_days"] == 120, r5
            assert approx(r5["net_return"], g ** 119 - 1, 1e-9), (alloc, r5["net_return"], g ** 119 - 1)
            assert approx(r5["contributed_usdt"], 450.0, 1e-9), r5
        if hasattr(paper, "net_return"):  # same number as paper's own flow-adjusted return
            assert approx(paper.net_return("s_flow"), r5["net_return"], 1e-9), (paper.net_return("s_flow"), r5)
        # no capital records -> refuse to score (never fall back to active.json allocation)
        _write_ledger([])
        r4 = score.promotion_check("s_flow")  # still in active.json, but the ledger has no capital records
        assert not r4["eligible"] and r4["net_return"] is None and any("capital" in x for x in r4["reasons"]), r4
    finally:
        _write_ledger([])
        if paper_saved is None:
            sys.modules.pop("trader.core.paper", None)
        else:
            sys.modules["trader.core.paper"] = paper_saved
        _remove_stubs(saved)


# ------------------------------------------------------------------------------------ random baseline (real modules)
def test_random_baseline_real():
    """With the real Wave-1 catalog/backtest/vault (if already present) on the research view: small sample."""
    try:
        from trader.core import backtest, catalog, vault  # noqa: F401
        needed = [hasattr(catalog, "rand_cfg"), hasattr(backtest, "evaluate_df"), hasattr(vault, "load")]
        if not all(needed) or not LAB.exists():
            raise ImportError("incomplete")
    except Exception as e:
        print(f"SKIP test_random_baseline_real (other Wave-1 areas not available yet: {e.__class__.__name__})")
        return
    t = time.time()
    spec = {"tf": "1d", "mode": "SPOT", "comps": [{"t": "rsi_reg", "p": {"n": 14, "up": 55, "gap": 5}}],
            "comb": "and", "k": 1, "adx": 0, "vt": 0.5, "stop": 0}
    out = score.random_baseline_rate([(spec, "ETH"), (spec, "BNB")], period="trainval", view="research",
                                     n_random=60, seed=1, coins=["BTC", "ETH", "SOL", "BNB"], caller="test")
    assert out["n_selected"] == 2 and out["n_random"] + out["n_skipped"] == 60, out
    assert 0.0 <= out["rate_random"] <= 1.0
    # end-to-end seal: a granted look opens the TEST period for the covered spec only (fresh journal: budget)
    old_j = os.environ["TRADER_JOURNAL_DIR"]
    os.environ["TRADER_JOURNAL_DIR"] = str(TMP / "journal_e2e")
    try:
        sh = catalog.spec_hash(spec)
        expect(vault.VaultSealed, backtest.evaluate, spec, "ETH", view="full", period="test", caller="test")
        trials.register_hypothesis(write_hyp("H-0200", coins=["ETH"]))
        trials.set_status("H-0200", "passed_trainval")
        lid = trials.request_look("H-0200", [sh], ["ETH"], "e2e test")
        res = backtest.evaluate(spec, "ETH", view="full", period="test", look_id=lid, caller="test",
                                hypothesis_id="H-0200")
        assert res["metrics"] is not None and res["trial_id"]
        row = trials.read_trials(kind="single")[-1]
        assert row["look_id"] == lid and row["view"] == "full" and row["period"] == "test"
        other = dict(spec, adx=20)
        expect(vault.VaultSealed, backtest.evaluate, other, "ETH", view="full", period="test", look_id=lid,
               caller="test")
        # regression (review): the ETH look must not open SOL TEST, nor any coin via spec_hash=None
        expect(vault.VaultSealed, backtest.evaluate, spec, "SOL", view="full", period="test", look_id=lid,
               caller="test")
        expect(vault.VaultSealed, vault.load, "BTC", "1d", "full", look_id=lid, spec_hash=None)
        expect(vault.VaultSealed, vault.load, "ETH", "1d", "full", look_id=lid, spec_hash=None)
        assert len(vault.load("ETH", "1d", "full", look_id=lid, spec_hash=sh))  # covered coin + spec still opens
    finally:
        os.environ["TRADER_JOURNAL_DIR"] = old_j
    print(f"  real baseline: {json.dumps({k: out[k] for k in ('rate_selected', 'rate_random', 'n_random')})} "
          f"in {time.time() - t:.1f}s")


if __name__ == "__main__":
    tests = [test_norm, test_metrics, test_buy_and_hold, test_brier_portfolio, test_dsr, test_hypotheses_and_looks,
             test_ledger, test_random_baseline_stub, test_promotion_stub, test_promotion_allocation_edit,
             test_random_baseline_real]
    for fn in tests:
        fn()
        print(f"  ok {fn.__name__}")
    # the real journal must never be touched by tests
    assert config.journal_dir() == TMP / "journal"
    print(f"OK test_trials ({time.time() - T0:.1f}s)")
