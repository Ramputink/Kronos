"""Tests for core/data.py and core/vault.py (offline: a fake Bybit API serves the lab CSVs).

Run: /Users/matveypro/.venvs/kronos/bin/python trader/tests/test_data.py
"""
import os
import shutil
import sys
import tempfile
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

TMP = Path(tempfile.mkdtemp(prefix="phil-data-test-"))
os.environ["TRADER_JOURNAL_DIR"] = str(TMP / "journal")
os.environ["TRADER_STRATEGY_DIR"] = str(TMP / "strategy")
os.environ["TRADER_DATA_DIR"] = str(TMP / "store")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from trader.core import config, data, vault  # noqa: E402

SEED = Path("/Users/matveypro/kronos_lab/indicators/data")
CFG = config.load_config()
MS = {"D": 86_400_000, "240": 14_400_000}
REAL_NOW_MS = data._now_ms


# ── fake Bybit v5 ────────────────────────────────────────────────────────────
class FakeBybit:
    """Serves /kline and /funding/history like Bybit: newest first, start/end filters, limit, plus one
    forming candle after the last closed bar so the 'never write the forming bar' rule is exercised."""

    def __init__(self, coin, fail=False):
        self.coin, self.fail, self.calls = coin, fail, 0
        self.k = {iv: data.read_klines(SEED / ("%s_%s.csv" % (coin, tf))) for tf, iv in (("1d", "D"), ("4h", "240"))}
        self.f = data.read_funding(SEED / ("%s_funding.csv" % coin))
        ends = [data._to_ms(self.k["D"].index.max()) + MS["D"], data._to_ms(self.k["240"].index.max()) + MS["240"],
                data._to_ms(self.f.index.max())]
        self.now_ms = max(ends) + 3_600_000  # every stored bar closed; the next bars are forming
        assert self.now_ms < data._to_ms(self.k["D"].index.max()) + 2 * MS["D"]

    def rows(self, iv):
        df = self.k[iv]
        ts = (df.index.values.astype("datetime64[ms]").astype("int64"))
        out = [[str(t)] + ["%r" % float(x) for x in r] for t, r in zip(ts, df[data.COLS].values)]
        # forming bar(s) — any bar whose open + interval > now
        t = int(ts[-1]) + MS[iv]
        while t <= self.now_ms:
            out.append([str(t), "1", "1", "1", "1", "1", "1"])
            t += MS[iv]
        return out

    def __call__(self, path, params):
        self.calls += 1
        if self.fail:
            raise data.DataError("simulated outage")
        assert params["symbol"] == self.coin + "USDT"
        if path == "kline":
            rows = self.rows(params["interval"])
            lo, hi = params.get("start", -1), params.get("end", 1 << 62)
            sel = [r for r in rows if lo <= int(r[0]) <= hi]
            return {"list": sel[::-1][: params["limit"]]}
        if path == "funding/history":
            assert ("startTime" not in params) or ("endTime" in params), "Bybit needs endTime with startTime"
            ts = self.f.index.values.astype("datetime64[ms]").astype("int64")
            rows = [{"symbol": self.coin + "USDT", "fundingRate": repr(float(v)), "fundingRateTimestamp": str(t)}
                    for t, v in zip(ts, self.f.values)]
            lo, hi = params.get("startTime", -1), params.get("endTime", 1 << 62)
            sel = [r for r in rows if lo <= int(r["fundingRateTimestamp"]) <= hi]
            return {"list": sel[::-1][: params["limit"]]}
        raise AssertionError(path)


def install(fake):
    data._http_get = fake
    data._now_ms = lambda: fake.now_ms
    data._sleep = lambda s: None


def head_copy(src: Path, dst: Path, drop: int):
    lines = src.read_bytes().splitlines(keepends=True)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(b"".join(lines[: len(lines) - drop]))


# ── tests ────────────────────────────────────────────────────────────────────
def test_roundtrip_bytes():
    """read + write reproduces the lab files byte for byte (so appends never reformat history)."""
    out = TMP / "rt"
    n = 0
    for f in sorted(SEED.glob("*.csv")):
        kind = f.stem.rsplit("_", 1)[1]
        dst = out / f.name
        if kind == "funding":
            data.write_funding(data.read_funding(f), dst)
        else:
            data.write_klines(data.read_klines(f), dst, kind)
        assert dst.read_bytes() == f.read_bytes(), "roundtrip changed %s" % f.name
        n += 1
    assert n >= 29 * 2
    shutil.rmtree(out)


def test_incremental_idempotent():
    store = config.data_dir("linear")
    fake = FakeBybit("BTC")
    install(fake)
    head_copy(SEED / "BTC_1d.csv", store / "BTC_1d.csv", 40)
    head_copy(SEED / "BTC_4h.csv", store / "BTC_4h.csv", 2500)  # > 1000 → pagination with `start`
    head_copy(SEED / "BTC_funding.csv", store / "BTC_funding.csv", 450)  # > 200 → funding pagination
    r = data.update(["BTC"], ["1d", "4h"], source="linear")
    assert not r["errors"], r
    assert r["updated"]["BTC_1d"]["rows_added"] == 40, r
    assert r["updated"]["BTC_4h"]["rows_added"] == 2500, r
    assert r["updated"]["BTC_funding"]["rows_added"] == 450, r
    assert "overlap_mismatch" not in r["updated"]["BTC_1d"]
    for name in ("BTC_1d.csv", "BTC_4h.csv", "BTC_funding.csv"):
        assert (store / name).read_bytes() == (SEED / name).read_bytes(), "update result differs: " + name
    assert r["updated"]["BTC_1d"]["last"] == config.iso(fake.k["D"].index.max())
    # second run: nothing new, file untouched (bytes and mtime)
    st = {n: (store / n).stat().st_mtime_ns for n in ("BTC_1d.csv", "BTC_4h.csv", "BTC_funding.csv")}
    r2 = data.update(["BTC"], ["1d", "4h"], source="linear")
    assert not r2["errors"], r2
    assert all(v["rows_added"] == 0 for v in r2["updated"].values()), r2
    for n, m in st.items():
        assert (store / n).stat().st_mtime_ns == m, "idempotent update rewrote " + n
        assert (store / n).read_bytes() == (SEED / n).read_bytes()
    # forming bar never written
    df = data.read_klines(store / "BTC_1d.csv")
    assert data._to_ms(df.index.max()) + MS["D"] <= fake.now_ms
    assert not any(t.name.startswith(".") and "tmp" in t.name for t in store.iterdir())


def test_from_scratch_and_spot_no_funding():
    fake = FakeBybit("ETH")
    install(fake)
    k = data.fetch_klines("ETHUSDT", "D", "linear")
    exp = fake.k["D"]
    assert len(k) == len(exp) and (k.index == exp.index).all()
    assert np.allclose(k.values, exp[data.COLS].values)
    # past end_ms keeps the closed bar that fetch.py's iloc[:-1] would have dropped
    end = data._to_ms(exp.index[100])
    k2 = data.fetch_klines("ETHUSDT", "D", "linear", end_ms=end)
    assert k2.index.max() == exp.index[100]
    # spot source: separate dir, no funding fetched
    r = data.update(["ETH"], ["1d"], source="spot")
    assert not r["errors"] and "ETH_funding" not in r["updated"], r
    assert (config.data_dir("spot") / "ETH_1d.csv").read_bytes() == (SEED / "ETH_1d.csv").read_bytes()
    assert not (config.data_dir("linear") / "ETH_funding.csv").exists()


def test_overlap_mismatch_refetches_and_replaces():
    """Review issue (major, data.py): a stored bar that disagrees with the exchange used to be kept verbatim and
    only reported. Now it is an error that re-fetches the whole series and replaces the store (poison removed,
    including made-up bars deeper in history and bars the exchange never had)."""
    store = config.data_dir("linear")
    fake = FakeBybit("SOL")
    install(fake)
    head_copy(SEED / "SOL_1d.csv", store / "SOL_1d.csv", 5)
    df = data.read_klines(store / "SOL_1d.csv")
    df.iloc[-1, df.columns.get_loc("close")] += 1.0  # stored row in the overlap disagrees with the exchange
    df.iloc[100, df.columns.get_loc("high")] *= 3.0  # poison deeper in history (outside the overlap)
    fake_bar = df.iloc[[0]].copy()
    fake_bar.index = fake_bar.index - pd.Timedelta(days=30)  # a made-up bar before the listing
    df = pd.concat([fake_bar, df])
    data.write_klines(df, store / "SOL_1d.csv", "1d")
    r = data.update(["SOL"], ["1d"], source="linear", funding=False)
    u = r["updated"]["SOL_1d"]
    assert u.get("overlap_mismatch") == 1 and "SOL_1d" in r["errors"], r
    assert u["repaired"] == {"changed": 2, "dropped": 1, "added": 5}, u
    assert u["rows_added"] == 5, u
    assert (store / "SOL_1d.csv").read_bytes() == (SEED / "SOL_1d.csv").read_bytes(), "store must equal exchange"
    assert data.main(["update", "--source", "linear", "--coins", "SOL", "--tfs", "1d", "--no-funding"]) == 0
    # funding: a stored event that disagrees with the exchange is repaired the same way
    head_copy(SEED / "SOL_funding.csv", store / "SOL_funding.csv", 3)
    f = data.read_funding(store / "SOL_funding.csv")
    f.iloc[-1] = 0.0123
    f.iloc[10] = -0.05
    data.write_funding(f, store / "SOL_funding.csv")
    r = data.update(["SOL"], ["1d"], source="linear")
    assert r["updated"]["SOL_funding"].get("overlap_mismatch") == 1 and "SOL_funding" in r["errors"], r
    assert (store / "SOL_funding.csv").read_bytes() == (SEED / "SOL_funding.csv").read_bytes()
    # errors are collected per key, never raised
    install(FakeBybit("SOL", fail=True))
    r = data.update(["SOL", "XRP"], ["1d"], source="linear", funding=False)
    assert set(r["errors"]) == {"SOL_1d", "XRP_1d"}, r


def test_verify_full_history():
    """`verify` compares the whole stored series (not only the overlap) with the exchange and replaces it."""
    store = config.data_dir("linear")
    fake = FakeBybit("XRP")
    install(fake)
    data.write_klines(fake.k["D"], store / "XRP_1d.csv", "1d")
    r = data.verify(["XRP"], ["1d"], source="linear")
    assert not r["errors"] and r["verified"]["XRP_1d"]["replaced"] is False, r
    df = data.read_klines(store / "XRP_1d.csv")
    df.iloc[50, df.columns.get_loc("close")] *= 1.5  # poison far from the overlap: update() would not see it
    data.write_klines(df, store / "XRP_1d.csv", "1d")
    assert not data.update(["XRP"], ["1d"], source="linear", funding=False)["errors"]
    r = data.verify(["XRP"], ["1d"], source="linear")
    assert "XRP_1d" in r["errors"] and r["verified"]["XRP_1d"]["changed"] == 1, r
    assert (store / "XRP_1d.csv").read_bytes() == (SEED / "XRP_1d.csv").read_bytes()


def test_seed_from():
    data._now_ms = REAL_NOW_MS  # seeding validates "no bar in the future" against the real clock
    root = TMP / "seedroot"
    os.environ["TRADER_DATA_DIR"] = str(root)
    try:
        n = data.seed_from(str(SEED), "linear")
        assert n == len(list(SEED.glob("*.csv"))), n
        p = config.data_dir("linear") / "BTC_1d.csv"
        assert not p.is_symlink() and p.read_bytes() == (SEED / "BTC_1d.csv").read_bytes()
        assert data.seed_from(str(SEED), "linear") == 0  # never overwrites
        assert data.seed_from(None, "spot") == len(list(SEED.glob("*_1d.csv"))) + len(list(SEED.glob("*_4h.csv")))
    finally:
        os.environ["TRADER_DATA_DIR"] = str(TMP / "store")
        shutil.rmtree(root)


def _expect(exc, fn, msg):
    try:
        fn()
    except exc:
        return
    raise AssertionError(msg)


def test_seed_only_from_protected_seed_dir():
    """Review issue (major, data.py): `seed --path` copied any <COIN>_<tf>.csv from any directory (including the
    agent-writable trader/strategy/**) into the live store, unchecked. Now: no --path, the function only accepts
    the protected data.seed_dir, and every file is validated before anything is copied."""
    root = TMP / "seedroot2"
    os.environ["TRADER_DATA_DIR"] = str(root)
    cfg_saved = os.environ.get("TRADER_CONFIG")
    try:
        # (1) a poisoned dir inside the repo (agent-writable strategy dir) or anywhere else is refused
        evil = config.strategy_dir() / "evil_seed"  # TRADER_STRATEGY_DIR (temp) stands in for trader/strategy
        other = TMP / "evil"
        for d in (evil, other):
            d.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(str(SEED / "ETH_1d.csv"), str(d / "ETH_1d.csv"))
        for d in (evil, other, config.TRADER_ROOT / "strategy", config.TRADER_ROOT / "tests"):
            _expect(data.DataError, lambda d=d: data.seed_from(str(d), "spot"), "seed from %s must fail" % d)
        assert not (root / "spot" / "ETH_1d.csv").exists()
        # (2) the CLI has no --path any more
        _expect(SystemExit, lambda: data.main(["seed", "--path", str(other), "--source", "spot"]),
                "seed --path must be rejected by the CLI")
        # (3) a protected seed_dir inside the repo / data store is refused too
        cfg = config.load_config()
        for bad_dir in (str(config.TRADER_ROOT / "strategy"), str(root)):
            cfg["data"]["seed_dir"] = bad_dir
            p = TMP / "cfg_seed.json"
            p.write_text(__import__("json").dumps(cfg))
            os.environ["TRADER_CONFIG"] = str(p)
            _expect(data.DataError, lambda: data.seed_from(None, "spot"), "seed_dir %s must be refused" % bad_dir)
        # (4) invalid bars in the seed dir: nothing is copied
        bad = TMP / "badseed"
        bad.mkdir(exist_ok=True)
        shutil.copyfile(str(SEED / "BTC_1d.csv"), str(bad / "BTC_1d.csv"))
        cases = {}
        e = data.read_klines(SEED / "ETH_1d.csv")
        e1 = e.copy()
        e1.iloc[-1, e1.columns.get_loc("high")] = e1.iloc[-1]["low"] * 0.5  # high < low
        cases["ohlc"] = e1
        e2 = e.copy()
        fut = e2.iloc[[-1]].copy()
        fut.index = pd.DatetimeIndex([pd.Timestamp.utcnow().tz_localize(None).normalize() + pd.Timedelta(days=3)])
        cases["future"] = pd.concat([e2, fut])
        e3 = data.read_klines(SEED / "ETH_4h.csv").iloc[:500]
        e3.index = e3.index + pd.Timedelta(hours=1)  # off the 4h grid
        cases["grid"] = e3
        cfg["data"]["seed_dir"] = str(bad)
        p.write_text(__import__("json").dumps(cfg))
        data._now_ms = REAL_NOW_MS
        for name, frame in cases.items():
            if name == "grid":
                shutil.copyfile(str(SEED / "ETH_1d.csv"), str(bad / "ETH_1d.csv"))
                data.write_klines(frame, bad / "ETH_4h.csv", "4h")
            else:
                data.write_klines(frame, bad / "ETH_1d.csv", "1d")
            _expect(data.DataError, lambda: data.seed_from(None, "spot"), "invalid seed (%s) must be refused" % name)
            assert not (root / "spot" / "BTC_1d.csv").exists(), "nothing may be copied when a file is invalid"
        (bad / "ETH_4h.csv").unlink()
        shutil.copyfile(str(SEED / "ETH_1d.csv"), str(bad / "ETH_1d.csv"))
        assert data.seed_from(None, "spot") == 2
        assert (root / "spot" / "ETH_1d.csv").read_bytes() == (SEED / "ETH_1d.csv").read_bytes()
    finally:
        if cfg_saved is None:
            os.environ.pop("TRADER_CONFIG", None)
        else:
            os.environ["TRADER_CONFIG"] = cfg_saved
        os.environ["TRADER_DATA_DIR"] = str(TMP / "store")
        shutil.rmtree(root, ignore_errors=True)


def _use_symlinked_seed():
    root = TMP / "linked"
    root.mkdir(exist_ok=True)
    for s in ("linear", "spot"):
        if not (root / s).exists():
            os.symlink(str(SEED), str(root / s))
    os.environ["TRADER_DATA_DIR"] = str(root)


def test_research_view_never_crosses_cutoff():
    _use_symlinked_seed()
    co = vault.cutoff()
    assert co == pd.Timestamp(CFG["vault"]["vault_cutoff"])
    checked = 0
    for coin in CFG["universe"] + ["POL"]:
        for tf in CFG["timeframes"]:
            if not (SEED / ("%s_%s.csv" % (coin, tf))).exists():
                continue
            df = vault.load(coin, tf)
            assert len(df) > 0, (coin, tf)
            assert df.index.max() < co, (coin, tf, df.index.max())
            assert list(df.columns) == vault.COLUMNS
            assert df.attrs["view"] == "research" and df.attrs["source"] == "linear"
            assert df.attrs["coin"] == coin and df.attrs["tf"] == tf
            assert df.attrs["data_end"] == config.iso(df.index.max()) and df.attrs["cutoff"] == config.iso(co)
            assert not vault.period_mask(df.index, "test").any()
            checked += 1
    assert checked >= 29 + 28, checked
    # public re-export behaves the same
    assert data.load("BTC", "1d").index.max() < co


def test_funding_matches_worker_rule():
    _use_symlinked_seed()
    for coin, tf, delta in (("BTC", "1d", "1D"), ("ETH", "4h", "4h")):
        df = vault.load(coin, tf)
        raw = pd.read_csv(SEED / ("%s_%s.csv" % (coin, tf)), index_col=0, parse_dates=True)
        raw = raw[raw.index < pd.Timestamp("2025-07-01")]
        f = pd.read_csv(SEED / ("%s_funding.csv" % coin), index_col=0, parse_dates=True).iloc[:, 0]
        fund = f.groupby((f.index - pd.Timedelta(milliseconds=1)).floor(pd.Timedelta(delta))).sum() \
            .reindex(raw.index).fillna(0.0)
        assert (df.index == raw.index).all()
        assert np.array_equal(df.funding.values, fund.values), (coin, tf)
        assert np.array_equal(df.close.values, raw.close.values)
    sp = vault.load("BTC", "1d", source="spot")
    assert (sp.funding == 0.0).all()


def test_full_view_requires_look():
    _use_symlinked_seed()
    co = vault.cutoff()
    for kw in ({}, {"look_id": None}, {"look_id": ""}):
        try:
            vault.load("BTC", "1d", "full", **kw)
            raise AssertionError("full view without look must raise")
        except vault.VaultSealed:
            pass
    fake = types.ModuleType("trader.core.trials")
    granted = {"L-ok": {"spec_hashes": ["abc"]}}

    def check_look(look_id, spec_hash):
        g = granted.get(look_id)
        if g is None or (spec_hash is not None and spec_hash not in g["spec_hashes"]):
            raise vault.VaultSealed("refused")
        return {"look_id": look_id}

    fake.check_look = check_look
    saved = sys.modules.get("trader.core.trials")
    sys.modules["trader.core.trials"] = fake
    try:
        for lid, sh in (("L-bad", "abc"), ("L-ok", "zzz")):
            try:
                vault.load_full("BTC", "1d", look_id=lid, spec_hash=sh)
                raise AssertionError("refused look must raise")
            except vault.VaultSealed:
                pass
        fake.check_look = lambda l, s: (_ for _ in ()).throw(KeyError("boom"))  # any error → sealed
        try:
            vault.load_full("BTC", "1d", look_id="L-ok", spec_hash="abc")
            raise AssertionError("verification error must fail closed")
        except vault.VaultSealed:
            pass
        fake.check_look = check_look
        full = vault.load_full("BTC", "1d", look_id="L-ok", spec_hash="abc")
        assert full.index.max() >= co and full.attrs["view"] == "full"
        m = vault.period_mask(full.index, "test")
        assert m.any() and (full.index[m] >= co).all()
        res = vault.load("BTC", "1d")
        assert full.loc[res.index].equals(res), "research view must be a prefix of the full view"
    finally:
        if saved is None:
            sys.modules.pop("trader.core.trials", None)
        else:
            sys.modules["trader.core.trials"] = saved


def test_paper_view_and_periods():
    _use_symlinked_seed()
    with _forward_start("2025-08-01"), _operator_env():
        p = _paper_load("BTC", None)  # the real paper module claims caller='paper'
        assert p.attrs["source"] == "spot" and p.index.max() >= vault.cutoff()
        p2 = _paper_load("BTC", pd.Timestamp("2025-09-01"))
        assert p2.index.max() == pd.Timestamp("2025-09-01")
        p3 = vault.load("BTC", "4h", "paper", asof=pd.Timestamp("2025-09-01 10:00:00"), caller="operator")
        assert p3.index.max() == pd.Timestamp("2025-09-01 08:00:00")
    try:
        vault.load("BTC", "1d", "sealed")
        raise AssertionError("unknown view must raise")
    except ValueError:
        pass
    idx = vault.load("BTC", "1d").index
    tr, va, tv, al = (vault.period_mask(idx, k) for k in ("train", "val", "trainval", "all"))
    start = idx[0] + pd.Timedelta(days=CFG["vault"]["warmup_days"])
    assert tr.any() and va.any() and not (tr & va).any()
    assert np.array_equal(tr | va, tv) and np.array_equal(tv, al)  # research view: all == trainval
    assert idx[tr].min() >= start and idx[tr].max() < pd.Timestamp("2024-01-01")
    assert idx[va].min() == pd.Timestamp("2024-01-01")
    # worker.py masks reproduced
    assert np.array_equal(tr, (idx >= start) & (idx < pd.Timestamp("2024-01-01")))
    assert not vault.period_mask(idx, "forward").any()  # forward_start is null
    assert vault.period_mask(pd.DatetimeIndex([]), "train").shape == (0,)


def _sealed(fn, msg):
    try:
        fn()
    except vault.VaultSealed:
        return
    raise AssertionError(msg)


class _forward_start:
    """Temporarily patch vault.forward_start (None or a date string)."""

    def __init__(self, fs):
        self.fs = pd.Timestamp(fs) if fs else None

    def __enter__(self):
        self.orig = vault.forward_start
        vault.forward_start = lambda: self.fs
        return self.fs

    def __exit__(self, *exc):
        vault.forward_start = self.orig


class _operator_env:
    """Temporarily set (or clear) the operator/test env flag."""

    def __init__(self, on=True):
        self.on = on

    def __enter__(self):
        self.saved = os.environ.get(vault.OPERATOR_ENV)
        if self.on:
            os.environ[vault.OPERATOR_ENV] = "1"
        else:
            os.environ.pop(vault.OPERATOR_ENV, None)

    def __exit__(self, *exc):
        if self.saved is None:
            os.environ.pop(vault.OPERATOR_ENV, None)
        else:
            os.environ[vault.OPERATOR_ENV] = self.saved


def _paper_load(coin, asof, mod="trader.core.paper", caller="paper"):
    """vault.load(view='paper') from a frame that carries `mod`'s name AND real source file (what the module
    itself would produce), so the tests do not depend on which caller paper.py currently passes."""
    fn = str(vault._module_file(mod))
    g = {"__name__": mod, "vault": vault}
    exec(compile("def f(coin, asof, caller):\n    return vault.load(coin, '1d', view='paper', asof=asof, "
                 "caller=caller)", fn, "exec"), g)
    return g["f"](coin, asof, caller)


def _access_rows():
    p = vault._access_path()
    return config.jsonl_read(p) if p.exists() else []


def test_paper_callers_are_bound_and_windowed():
    """Review issue (major, vault.py): view='paper' trusted any caller= string; screener/watch/operator/test/paper
    got every bar after the cutoff (the whole sealed TEST window) with no look and no log. Reproduction of the
    attack: vault.load('ETH','1d','paper',caller='screener') + backtest.evaluate_df(..., 'test') gave sealed TEST
    metrics with an empty ledger."""
    _use_symlinked_seed()
    co, wd = vault.cutoff(), CFG["vault"]["warmup_days"]
    # (a) the reproduced attack, every caller string, forward_start null, flag set or not: all sealed
    for env in (False, True):
        with _forward_start(None), _operator_env(env):
            for c in ("screener", "watch", "operator", "test", "paper", "forward", "deep_retro"):
                _sealed(lambda c=c: vault.load("ETH", "1d", "paper", caller=c),
                        "caller %r (env=%s) must be sealed while forward_start is null" % (c, env))
            # even the real paper module: asof >= cutoff while forward_start is null is sealed TEST data
            _sealed(lambda: _paper_load("ETH", pd.Timestamp("2025-09-01")), "paper before forward_start")
            # entirely before the cutoff there is nothing sealed: the real paper module may replay it
            assert _paper_load("ETH", co - pd.Timedelta(days=1)).index.max() < co
    fs = pd.Timestamp("2025-10-01")
    with _forward_start(fs):
        # (b) an explicit caller is bound to its module: screener/watch/paper from elsewhere are refused
        with _operator_env(True):
            for c in ("screener", "watch", "paper"):
                _sealed(lambda c=c: vault.load("ETH", "1d", "paper", caller=c), "%r from a test module" % c)
            # spoofing __name__ (exec under the module's name) is not enough: the source file is checked too
            for mod, c in (("trader.strategy.screener", "screener"), ("trader.fleet.watch", "watch"),
                           ("trader.core.paper", "paper"), ("trader.core.paper", None)):
                g = {"__name__": mod, "vault": vault, "c": c}
                exec("def f():\n    return vault.load('ETH', '1d', view='paper', caller=c)", g)
                _sealed(g["f"], "spoofed %s/%r must be refused" % (mod, c))
            # a real module that is not bound to the caller is refused; the bound modules are accepted (windowed)
            _sealed(lambda: _paper_load("ETH", None, mod="trader.core.backtest"), "'paper' from backtest")
            _sealed(lambda: _paper_load("ETH", None, mod="trader.core.paper", caller="screener"), "screener@paper")
            assert _paper_load("ETH", None, mod="trader.core.score").index.min() < co  # 'paper': full history
            for mod, c in (("trader.strategy.screener", "screener"), ("trader.fleet.watch", "watch")):
                df = _paper_load("ETH", None, mod=mod, caller=c)
                assert df.index.min() >= fs - pd.Timedelta(days=wd), (c, df.index.min())
        # (c) operator/test only with the env flag
        with _operator_env(False):
            for c in ("operator", "test"):
                _sealed(lambda c=c: vault.load("ETH", "1d", "paper", caller=c), "%r without the env flag" % c)
        # (d) with the flag, every non-'paper' caller is windowed like forward callers
        lo = fs - pd.Timedelta(days=wd)
        with _operator_env(True):
            for c in ("operator", "test", "forward", "deep_retro"):
                df = vault.load("ETH", "1d", "paper", caller=c)
                assert df.index.min() >= lo and df.index.min() < fs, (c, df.index.min())
                assert not ((df.index >= co) & (df.index < lo)).any(), "TEST bars before the warmup leaked (%s)" % c
                assert df.index[vault.period_mask(df.index, "all")].min() >= fs
        # (e) the real paper module: refused for asof before forward_start (one-bar tolerance: the bar whose
        #     close starts the forward period), full history otherwise
        _sealed(lambda: _paper_load("ETH", fs - pd.Timedelta(days=2)), "paper asof < forward_start")
        assert _paper_load("ETH", fs - pd.Timedelta(days=1)).index.max() == fs - pd.Timedelta(days=1)
        full = _paper_load("ETH", fs + pd.Timedelta(days=10))
        assert full.index.min() < co and full.index.max() == fs + pd.Timedelta(days=10)


def test_paper_view_sealed_access_is_logged():
    """Every paper-view load that returns bars in [cutoff, forward_start) leaves an "access" row."""
    _use_symlinked_seed()
    co, wd = vault.cutoff(), CFG["vault"]["warmup_days"]
    vault._ACCESS_SEEN.clear()
    n0 = len(_access_rows())
    fs = pd.Timestamp("2025-10-01")
    with _forward_start(fs), _operator_env(True):
        _paper_load("ETH", fs + pd.Timedelta(days=5))
        rows = _access_rows()[n0:]
        assert len(rows) == 1, rows
        r = rows[0]
        assert r["kind"] == "access" and r["caller"] == "paper" and r["coin"] == "ETH" and r["view"] == "paper"
        assert r["module"] == "trader.core.paper"
        assert r["sealed_first"] == config.iso(co) and r["sealed_last"] == config.iso(fs - pd.Timedelta(days=1))
        assert r["sealed_rows"] == (fs - co).days, r
        _paper_load("ETH", fs + pd.Timedelta(days=5))  # same process, same range: deduplicated
        assert len(_access_rows()) == n0 + 1
        # windowed caller: the warmup bars inside [cutoff, forward_start) are logged too
        vault.load("ETH", "1d", "paper", caller="test")
        r = _access_rows()[-1]
        assert r["caller"] == "test" and r["sealed_first"] == config.iso(max(co, fs - pd.Timedelta(days=wd))), r
        assert r["sealed_rows"] == (fs - max(co, fs - pd.Timedelta(days=wd))).days, r
    # an asof before forward_start (warmup-only frame of a windowed caller) is logged too
    with _forward_start("2026-01-01"), _operator_env(True):
        n1 = len(_access_rows())
        vault.load("ETH", "1d", "paper", asof="2025-12-01", caller="operator")
        assert len(_access_rows()) == n1 + 1
    # the access ledger never breaks the trials ledger schema: unknown kinds stay out of trials.jsonl
    from trader.core import trials
    if not vault._access_path().name == "trials.jsonl":
        assert all(r.get("kind") != "access" for r in config.jsonl_read(trials.trials_path())) \
            if trials.trials_path().exists() else True


def test_paper_view_gate_regression():
    """Review issue (critical): view='paper' handed research callers every post-cutoff bar with no look, and
    source='linear' turned it into the full sealed research series. Each hole has its own assertion."""
    _use_symlinked_seed()
    co = vault.cutoff()
    # (2) caller is mandatory for the paper view; research/sweep/unknown callers are refused
    _sealed(lambda: vault.load("ETH", "1d", "paper"), "paper view without caller must be sealed")
    _sealed(lambda: data.load("ETH", "1d", "paper"), "data.load re-export without caller must be sealed")
    for c in ("research", "sweep", "bogus", ""):
        _sealed(lambda c=c: vault.load("ETH", "1d", "paper", caller=c), "caller %r must be refused" % c)
    # (1) the reproduced leak: source='linear' through the paper view, without a look
    for c in ("research", "paper", "operator", "test"):
        _sealed(lambda c=c: vault.load("ETH", "1d", "paper", source="linear", caller=c),
                "paper view with source='linear' and no look must be sealed (caller %s)" % c)
    # allowed operational callers keep working on the paper source (once forward_start is set; windowed)
    with _forward_start("2025-10-01"), _operator_env(True):
        for c in ("operator", "test"):
            df = vault.load("ETH", "1d", "paper", caller=c)
            assert df.attrs["source"] == CFG["data"]["paper_source"] and df.index.max() >= co, c
        # linear through the paper view only with an accepted look
        fake = types.ModuleType("trader.core.trials")
        fake.check_look = lambda l, s: ({"look_id": l} if l == "L-ok"
                                        else (_ for _ in ()).throw(vault.VaultSealed("no")))
        saved = sys.modules.get("trader.core.trials")
        sys.modules["trader.core.trials"] = fake
        try:
            _sealed(lambda: vault.load("ETH", "1d", "paper", source="linear", caller="test", look_id="L-bad"),
                    "refused look must not open the linear paper view")
            lin = vault.load("ETH", "1d", "paper", source="linear", caller="test", look_id="L-ok")
            assert lin.attrs["source"] == "linear" and lin.index.max() >= co
        finally:
            if saved is None:
                sys.modules.pop("trader.core.trials", None)
            else:
                sys.modules["trader.core.trials"] = saved
    # implicit caller only for the real CONTRACTS §3 paper modules (a spoofed __name__ is not enough)
    g = {"__name__": "trader.core.paper", "vault": vault}
    exec("def f():\n    return vault.load('ETH', '1d', view='paper', asof='2025-06-01')", g)
    _sealed(g["f"], "exec under trader.core.paper's name must not claim the implicit paper caller")
    g = {"__name__": "trader.strategy.research_agent", "vault": vault}
    exec("def f():\n    return vault.load('ETH', '1d', view='paper')", g)
    _sealed(g["f"], "an unknown module without caller= must be refused")
    # (3) forward/deep_retro callers: sealed while forward_start is null; windowed once it is set
    orig_fs = vault.forward_start
    try:
        vault.forward_start = lambda: None
        for c in ("forward", "deep_retro"):
            _sealed(lambda c=c: vault.load("ETH", "1d", "paper", caller=c),
                    "forward caller with forward_start=null must be sealed")
        fs = pd.Timestamp("2025-10-01")
        vault.forward_start = lambda: fs
        df = vault.load("ETH", "1d", "paper", caller="forward")
        lo = fs - pd.Timedelta(days=CFG["vault"]["warmup_days"])
        assert df.index.min() >= lo and df.index.min() < fs, df.index.min()
        assert not ((df.index >= co) & (df.index < lo)).any(), "TEST bars before the warmup must stay sealed"
        al = vault.period_mask(df.index, "all")
        assert df.index[al].min() >= fs, "period 'all' of a forward frame must be the forward period"
        # forward_start before the cutoff: the window starts at cutoff - warmup (no sealed bar is exposed early)
        vault.forward_start = lambda: pd.Timestamp("2020-01-01")
        df2 = vault.load("ETH", "1d", "paper", caller="deep_retro")
        assert df2.index.min() >= co - pd.Timedelta(days=CFG["vault"]["warmup_days"])
    finally:
        vault.forward_start = orig_fs
    # research view is unchanged and ignores caller
    assert vault.load("ETH", "1d", caller="research").index.max() < co


def test_export_research():
    _use_symlinked_seed()
    out = TMP / "export"
    n = vault.export_research(out, coins=["BTC", "POL"], tfs=["1d", "4h"])
    assert n == 2 + 1 + 1 + 1, n  # BTC 1d,4h,funding; POL 1d,funding
    for f in out.rglob("*.csv"):
        t = pd.read_csv(f, index_col=0, parse_dates=True).index
        assert t.max() <= vault.cutoff(), f
        if not f.stem.endswith("funding"):
            assert t.max() < vault.cutoff(), f
    local = vault.load("BTC", "4h")
    os.environ["TRADER_DATA_DIR"] = str(out)
    try:
        remote = vault.load("BTC", "4h")
    finally:
        _use_symlinked_seed()
    assert remote.equals(local), "exported research view must reproduce the local one"


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    order = ["test_roundtrip_bytes", "test_incremental_idempotent", "test_from_scratch_and_spot_no_funding",
             "test_overlap_mismatch_refetches_and_replaces", "test_verify_full_history", "test_seed_from",
             "test_seed_only_from_protected_seed_dir"]
    tests.sort(key=lambda f: (order.index(f.__name__) if f.__name__ in order else len(order), f.__name__))
    try:
        for t in tests:
            os.environ["TRADER_DATA_DIR"] = str(TMP / "store")
            t()
            print("  ok", t.__name__, flush=True)
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
    print("OK test_data")


if __name__ == "__main__":
    main()
