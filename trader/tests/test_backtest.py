"""Backtest area regression tests (CONTRACTS.md section 4). Plain script: asserts, prints OK, < 3 min.

run: /Users/matveypro/.venvs/kronos/bin/python trader/tests/test_backtest.py
"""
import importlib.util
import math
import os
import random
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
_TMP = tempfile.mkdtemp(prefix="phil-bt-")
os.environ["TRADER_JOURNAL_DIR"] = os.path.join(_TMP, "journal")  # never touch the real journal
os.environ["TRADER_STRATEGY_DIR"] = os.path.join(_TMP, "strategy")
# keep numba's on-disk cache for the imported research worker out of the repo tree
os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(tempfile.gettempdir(), "phil-numba-cache"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from trader.core import backtest as B  # noqa: E402
from trader.core import catalog as C  # noqa: E402

LAB = "/Users/matveypro/kronos_lab/indicators/data"
WORKER = ROOT / "research" / "kronos_lab" / "overnight" / "worker.py"
CUTOFF = pd.Timestamp("2025-07-01")
TFD = {"1d": pd.Timedelta("1D"), "4h": pd.Timedelta("4h")}
RSI_SPEC = {"tf": "1d", "mode": "SPOT", "comps": [{"t": "rsi_reg", "p": {"n": 14, "up": 55, "gap": 5}}],
            "comb": "and", "k": 1, "adx": 0, "vt": 0.5, "stop": 0}


def lab_frame(coin, tf, research=False):
    """vault.load-shaped frame straight from the lab CSVs (worker.py funding rule)."""
    df = pd.read_csv(f"{LAB}/{coin}_{tf}.csv", index_col=0, parse_dates=True)
    if research:
        df = df[df.index < CUTOFF]
    f = pd.read_csv(f"{LAB}/{coin}_funding.csv", index_col=0, parse_dates=True).iloc[:, 0]
    fund = f.groupby((f.index - pd.Timedelta(milliseconds=1)).floor(TFD[tf])).sum().reindex(df.index).fillna(0.0)
    df = df.copy()
    df["funding"] = fund.values
    df.attrs = {"coin": coin, "tf": tf, "view": "research" if research else "full", "source": "linear",
                "cutoff": "2025-07-01", "data_end": str(df.index[-1])}
    return df


def cagr(r, bpy=365):
    r = np.asarray(r)
    return float(np.prod(1 + r) ** (bpy / len(r)) - 1)


def test_hash_and_validation():
    assert C.spec_hash(RSI_SPEC) == "cd9ff45458116162a26b8d3c8fb059ec", C.spec_hash(RSI_SPEC)
    assert C.spec_hash(dict(RSI_SPEC, name="x", notes="y")) == C.spec_hash(RSI_SPEC)
    assert C.validate_spec(RSI_SPEC) == []
    bad = [dict(RSI_SPEC, tf="1h"), dict(RSI_SPEC, mode="MARGIN"), dict(RSI_SPEC, comps=[]),
           dict(RSI_SPEC, comps=[{"t": "nope", "p": {}}]), dict(RSI_SPEC, comps=[{"t": "rsi_reg", "p": {"n": 14}}]),
           dict(RSI_SPEC, comb="or"), dict(RSI_SPEC, k=2), dict(RSI_SPEC, vt=1.5), dict(RSI_SPEC, stop=-1),
           dict(RSI_SPEC, comps=[{"t": "rsi_reg", "p": {"n": 14.5, "up": 55, "gap": 5}}]),
           {k: v for k, v in RSI_SPEC.items() if k != "adx"}]
    for s in bad:
        assert C.validate_spec(s), s
    assert C.validate_spec(dict(RSI_SPEC, mode="PERP", vt=1.5)) == []
    assert C.validate_spec(dict(RSI_SPEC, comps=[{"t": "st", "p": {"n": 14, "m": 2.5}}])) == []  # off-grid ok
    rng = random.Random(7)
    for _ in range(300):
        cfg = C.rand_cfg(rng)
        assert C.validate_spec(cfg) == [], (cfg, C.validate_spec(cfg))
        m = C.mutate(cfg, rng)
        assert C.validate_spec(m) == [], (m, C.validate_spec(m))
    assert len(C.CATALOG) == 15


def test_costs():
    assert abs(B.cost_side("SPOT") - 0.0030) < 1e-12
    assert abs(B.cost_side("PERP") - 0.00105) < 1e-12
    assert B.cost_side("SPOT", "legacy") == 0.0015 and B.cost_side("PERP", "legacy") == 0.00105
    assert B.cost_side("SPOT", 0.002) == 0.002
    assert B.cost_side("PERP", {"SPOT": 0.1, "PERP": 0.2}) == 0.2


def test_rsi_regression():
    """Fixture 1: rsi55_50 + vt on ETH 1d SPOT, legacy cost 0.0015, TEST 2025-07-01..2026-10-13 -> CAGR 0.268."""
    df = lab_frame("ETH", "1d")
    res = B.evaluate_df(RSI_SPEC, df, "test", cost_side=B.cost_side("SPOT", "legacy"))
    r = res["returns"]
    assert str(r.index[0].date()) == "2025-07-01" and str(r.index[-1].date()) == "2026-10-13", (r.index[0], r.index[-1])
    assert len(r) == 470, len(r)
    c = cagr(r.values)
    assert abs(c - 0.268) <= 0.001, c
    assert res["spec_hash"] == "cd9ff45458116162a26b8d3c8fb059ec"
    assert res["metrics"] is not None and abs(res["metrics"]["cagr"] - c) < 1e-9, res["metrics"]
    assert res["trial_id"] is None and res["bh_metrics"] is not None
    assert len(res["positions"]) == len(res["target"]) == 470
    assert (res["target"] >= 0).all() and (res["target"] <= 1.0).all()
    # legacy engine.py path gives the same number
    sig = C.base_signals(df)["rsi55_50"]
    lr, lwx = B.legacy_bar_returns(df, df["funding"], sig, B.LEGACY_COSTS["SPOT"], "1d", 0, True, None)
    lt = lr[lr.index >= CUTOFF]
    lc = cagr(lt.values)
    assert abs(lc - 0.268) <= 0.001, lc
    assert abs(lc - c) < 2e-3, (lc, c)
    # legacy adx-gated path runs (donch100_50 + adx 25 + vt was the other finalist)
    adx_s = C.legacy_adx(df)
    lr2, _ = B.legacy_bar_returns(df, df["funding"], C.base_signals(df)["donch100_50"], "SPOT", "1d", 25, True, adx_s)
    assert abs(cagr(lr2[lr2.index >= CUTOFF].values) - 0.230) <= 0.001, cagr(lr2[lr2.index >= CUTOFF].values)
    # default (protected) spot cost is higher -> strictly lower CAGR
    res2 = B.evaluate_df(RSI_SPEC, df, "test", cost_side=B.cost_side("SPOT"))
    assert res2["metrics"]["cagr"] < c
    print("  fixture1: CAGR %.5f (legacy engine path %.5f); at protected spot cost %.5f"
          % (c, lc, res2["metrics"]["cagr"]))


def test_no_lookahead():
    """Target at bar t is unchanged when every row after t is deleted."""
    df = lab_frame("SOL", "1d", research=True)
    rng = random.Random(3)
    specs = [RSI_SPEC] + [C.rand_cfg(rng, tfs=("1d",)) for _ in range(6)]
    for spec in specs:
        full = B.target_positions(spec, C.make_d(df))
        for cut in (400, 900, len(df) - 5):
            part = B.target_positions(spec, C.make_d(df.iloc[:cut]))
            assert np.allclose(full[:cut], part, equal_nan=True, atol=1e-12), (spec, cut)
    # returns: position held in bar t is the target of t-1, costs charged one bar later
    d = C.make_d(df)
    pos = B.target_positions(RSI_SPEC, d)
    r, wx = B.bar_returns(pos, d, "SPOT", 0.01)
    assert wx[0] == 0 and np.array_equal(wx[1:], pos[:-1])


def _load_worker():
    spec = importlib.util.spec_from_file_location("worker", str(WORKER))
    w = importlib.util.module_from_spec(spec)
    sys.modules["worker"] = w  # numba cached kernels pickle a reference to the module by name
    spec.loader.exec_module(w)
    w.COST = {"SPOT": 0.0015, "PERP": 0.00105}  # = protected costs.legacy_research
    return w


def test_worker_regression():
    """Fixture 2: 20 random worker configs (random.Random(1)) match worker.evaluate exactly on research view."""
    w = _load_worker()
    # identical RNG sequences
    r1, r2 = random.Random(1), random.Random(1)
    cfgs = [w.rand_cfg(r1) for _ in range(20)]
    mine = [C.rand_cfg(r2) for _ in range(20)]
    assert cfgs == mine
    r1, r2 = random.Random(5), random.Random(5)
    assert [w.mutate(c, r1) for c in cfgs] == [C.mutate(c, r2) for c in cfgs]
    for c in cfgs:
        h = w.hashlib.md5(w.json.dumps(c, sort_keys=True).encode()).hexdigest()
        assert C.spec_hash(c) == h
    data = w.load_all()
    frames, ds = {}, {}
    for coin in w.COINS:
        for tf in ("1d", "4h"):
            frames[(coin, tf)] = lab_frame(coin, tf, research=True)
            ds[(coin, tf)] = C.make_d(frames[(coin, tf)])
    n_rows = 0
    for cfg in cfgs:
        ref = {m["coin"]: m for m in w.evaluate(cfg, data)}
        cs = B.cost_side(cfg["mode"], "legacy")
        got = {}
        for coin in w.COINS:
            m = B.worker_metrics(cfg, frames[(coin, cfg["tf"])], cost_side=cs, d=ds[(coin, cfg["tf"])])
            if m is not None:
                got[coin] = m
        assert set(ref) == set(got), (cfg, set(ref) ^ set(got))
        for coin, m in ref.items():
            for k in ("cagr_tr", "sh_tr", "cagr_va", "sh_va", "sh_tv", "mdd", "entries", "expo"):
                assert abs(m[k] - got[coin][k]) <= 1e-9, (cfg, coin, k, m[k], got[coin][k])
            n_rows += 1
        # evaluate_df trainval agrees with worker on the same segment
        for coin in list(ref)[:5]:
            fr = frames[(coin, cfg["tf"])]
            e = B.evaluate_df(cfg, fr, "trainval", cost_side=cs, d=ds[(coin, cfg["tf"])])
            mt, m = e["metrics"], ref[coin]
            assert abs(mt["maxdd"] - m["mdd"]) <= 1e-9 and mt["entries"] == m["entries"]
            assert abs(mt["exposure"] - m["expo"]) <= 1e-9
            # worker uses the population std; allow score.metrics a ddof=1 convention (~1/(2n) relative)
            assert abs(mt["sharpe"] - m["sh_tv"]) <= 1e-9 + 2e-3 * abs(m["sh_tv"]), (mt["sharpe"], m["sh_tv"])
            assert e["returns"].index.max() < CUTOFF
    assert n_rows > 300, n_rows
    print("  fixture2: %d coin-rows of 20 worker configs identical" % n_rows)


def test_periods_and_perp():
    df = lab_frame("BTC", "1d")
    tr = B.evaluate_df(RSI_SPEC, df, "train", cost_side=0.0015)["returns"]
    va = B.evaluate_df(RSI_SPEC, df, "val", cost_side=0.0015)["returns"]
    tv = B.evaluate_df(RSI_SPEC, df, "trainval", cost_side=0.0015)["returns"]
    assert tr.index.min() == df.index[0] + pd.Timedelta(days=210)
    assert tr.index.max() < pd.Timestamp("2024-01-01") <= va.index.min() and va.index.max() < CUTOFF
    assert len(tr) + len(va) == len(tv)
    al = B.evaluate_df(RSI_SPEC, df, "all", cost_side=0.0015)["returns"]
    assert al.index.max() == df.index[-1]
    # PERP pays funding: with zero funding the result differs
    perp = dict(RSI_SPEC, mode="PERP", vt=0)
    a = B.evaluate_df(perp, df, "trainval", cost_side=0.00105)["returns"]
    df0 = df.copy(); df0["funding"] = 0.0
    b = B.evaluate_df(perp, df0, "trainval", cost_side=0.00105)["returns"]
    assert not np.allclose(a.values, b.values)
    pos = B.evaluate_df(perp, df, "trainval", cost_side=0.00105)["target"]
    assert (pos < 0).any()  # PERP may short
    # too-short period -> metrics None; empty frame -> InsufficientData
    short = df.iloc[:230]
    assert B.evaluate_df(RSI_SPEC, short, "all", cost_side=0.0015)["metrics"] is None
    try:
        B.evaluate_df(RSI_SPEC, df.iloc[:1], "all", cost_side=0.0015)
        raise AssertionError("expected InsufficientData")
    except B.InsufficientData:
        pass
    try:
        B.evaluate(RSI_SPEC, "ETH", "research", "test", caller="test", log=False)
        raise AssertionError("expected refusal of test period on research view")
    except ValueError:
        pass


def test_unlogged_evaluate_restricted():
    """Regression: evaluate(log=False) must only be allowed for caller test/sweep and never on view='full'
    (an unlogged sealed TEST evaluation would be missing from DSR N with no audit trail)."""
    from trader.core import vault
    calls = []
    orig = vault.load
    vault.load = lambda *a, **k: calls.append((a, k)) or (_ for _ in ()).throw(AssertionError("vault touched"))
    try:
        bad = [dict(view="full", period="test", look_id="L-2026-10-0001", caller="research"),
               dict(view="full", period="test", look_id="L-2026-10-0001", caller="test"),
               dict(view="full", period="trainval", look_id="L-2026-10-0001", caller="sweep"),
               dict(view="research", period="trainval", caller="research"),
               dict(view="research", period="trainval", caller="operator"),
               dict(view="paper", period="forward", caller="forward"),
               dict(view="research", period="trainval", caller="screener")]
        for kw in bad:
            try:
                B.evaluate(RSI_SPEC, "ETH", log=False, **kw)
                raise AssertionError("expected refusal of log=False for %r" % (kw,))
            except ValueError as e:
                assert "log=False" in str(e), (kw, e)
        assert not calls, "refusal must happen before any data (or sealed look) is loaded"
        # allowed combination passes the guard (reaches vault.load, which we stubbed to fail)
        for caller in ("test", "sweep"):
            try:
                B.evaluate(RSI_SPEC, "ETH", "research", "trainval", caller=caller, log=False)
            except AssertionError as e:
                assert "vault touched" in str(e)
        assert len(calls) == 2
    finally:
        vault.load = orig
    assert not os.path.exists(os.path.join(_TMP, "journal", "trials.jsonl"))


def main():
    t0 = time.time()
    for fn in (test_hash_and_validation, test_costs, test_rsi_regression, test_no_lookahead,
               test_periods_and_perp, test_worker_regression, test_unlogged_evaluate_restricted):
        t = time.time()
        fn()
        print("  %-28s %.1fs" % (fn.__name__, time.time() - t))
    assert not os.path.exists(os.path.join(_TMP, "journal", "trials.jsonl"))
    print("OK test_backtest (%.0fs)" % (time.time() - t0))


if __name__ == "__main__":
    main()
