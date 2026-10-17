"""Fleet area tests (CONTRACTS.md §8): lease, sweep runner, dispatch (dry-run + localhost), health, inbox, setup.

Offline (no network, no remote hosts). Run: /Users/matveypro/.venvs/kronos/bin/python trader/tests/test_fleet.py
Everything is redirected to temp dirs: TRADER_JOURNAL_DIR, TRADER_STRATEGY_DIR, TRADER_DATA_DIR, PHIL_STATE_DIR,
PHIL_REMOTE_ROOT, PHIL_INBOX, PHIL_OUTBOX. Lease tests use a local bare git "origin" with two clones.
"""
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

TMP = Path(tempfile.mkdtemp(prefix="phil-fleet-test-"))
os.environ["TRADER_JOURNAL_DIR"] = str(TMP / "journal")
os.environ["TRADER_STRATEGY_DIR"] = str(TMP / "strategy")
os.environ["PHIL_STATE_DIR"] = str(TMP / "jobs-registry")
os.environ["PHIL_REMOTE_ROOT"] = str(TMP / "remote" / ".phil-crypto" / "runner")
os.environ["PHIL_INBOX"] = str(TMP / "inbox")
os.environ["PHIL_OUTBOX"] = str(TMP / "outbox")
os.environ["PHIL_RUNNER"] = "testrunner"
LAB = Path("/Users/matveypro/kronos_lab/indicators/data")
(TMP / "data").mkdir()
os.symlink(str(LAB), str(TMP / "data" / "linear"))
os.environ["TRADER_DATA_DIR"] = str(TMP / "data")
for k, v in {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
             "GIT_COMMITTER_EMAIL": "t@t"}.items():
    os.environ.setdefault(k, v)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from trader.core import config, lease  # noqa: E402
from trader.fleet import dispatch, health, inbox, sweep_runner  # noqa: E402

PY = sys.executable
REPO = config.REPO_ROOT
FLEET = config.TRADER_ROOT / "fleet"
CFG = config.load_config()
CUTOFF = pd.Timestamp(CFG["vault"]["vault_cutoff"])
PASSED = []
CASES = []


def case(fn):
    """Register a test; tests run only under __main__ (spawned sweep workers re-import this module)."""
    CASES.append(fn)
    return fn


def run_all():
    register_hyp("H-9001")      # dispatch refuses unregistered hypotheses; small_job() uses H-9001
    for fn in CASES:
        t0 = time.time()
        fn()
        PASSED.append(fn.__name__)
        print("  ok %-45s %.1fs" % (fn.__name__, time.time() - t0), flush=True)


def git(cwd, *args, check=True):
    p = subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, text=True)
    if check and p.returncode != 0:
        raise RuntimeError("git %s: %s" % (args, p.stderr))
    return p.stdout.strip()


# ── lease: pure logic ─────────────────────────────────────────────────────────
@case
def test_lease_describe_offline():
    now = dt.datetime(2026, 10, 21, 0, 30, tzinfo=dt.timezone.utc)
    held = {"runner": "air", "cycle": "forward", "started": "2026-10-21T00:10:00Z", "ttl_s": 3000, "pid": 1}
    d = lease.describe(held, "bigsur", now)
    assert d["held"] and d["fresh"] and not d["mine"] and d["age_s"] == 1200 and d["cycle"] == "forward"
    assert lease.held_by_other(d)
    assert d["expires"] == "2026-10-21T01:00:00Z"
    assert not lease.held_by_other(lease.describe(held, "air", now))           # mine
    later = now + dt.timedelta(seconds=1801)                                      # age 3001 > ttl 3000
    assert not lease.describe(held, "bigsur", later)["fresh"]
    assert not lease.held_by_other(lease.describe(held, "bigsur", later))        # expired ⇒ takeable
    skew = now - dt.timedelta(seconds=1205)                                       # clock skew: started in future
    assert lease.describe(held, "bigsur", skew)["fresh"]
    assert not lease.describe({}, "x", now)["held"]
    assert not lease.describe({"runner": None, "released_by": "air"}, "x", now)["held"]
    assert not lease.describe({"runner": "air", "started": "garbage"}, "x", now)["fresh"]   # corrupt ⇒ expired
    assert not lease.describe("not a dict", "x", now)["held"]
    p = lease.held_payload("air", "research", 5400, now, pid=42)
    assert p == {"runner": "air", "cycle": "research", "started": "2026-10-21T00:30:00Z", "ttl_s": 5400, "pid": 42}
    f = lease.free_payload("air", now)
    assert f == {"runner": None, "released_by": "air", "released": "2026-10-21T00:30:00Z"}
    assert lease.default_ttl() == int(CFG["fleet"]["lease_ttl_s"])


# ── lease: two runners against a local bare origin ────────────────────────────
def _use_clone(clone):
    os.environ["TRADER_LEASE_REPO"] = str(clone)
    os.environ["TRADER_JOURNAL_DIR"] = str(clone / "trader" / "journal")


@case
def test_lease_git_two_runners():
    saved = os.environ["TRADER_JOURNAL_DIR"]
    root = TMP / "lease"
    origin, a, b = root / "origin.git", root / "a", root / "b"
    root.mkdir()
    git(root, "init", "--bare", "-q", "-b", "phil-crypto", str(origin))
    git(root, "clone", "-q", str(origin), str(a))
    git(a, "checkout", "-q", "-b", "phil-crypto")
    (a / "trader" / "journal").mkdir(parents=True)
    (a / "trader" / "journal" / ".gitkeep").write_text("")
    (a / "README").write_text("x\n")
    git(a, "add", "-A")
    git(a, "commit", "-q", "-m", "operator: init")
    git(a, "push", "-q", "origin", "phil-crypto")
    git(root, "clone", "-q", "-b", "phil-crypto", str(origin), str(b))
    rel = "trader/journal/lease.json"
    try:
        _use_clone(a)
        assert lease.lease_relpath() == rel
        c = lease.check(runner="air")
        assert c["exit"] == 0 and not c["held"], c
        r = lease.acquire("forward", runner="air", ttl_s=3000)
        assert r["acquired"] and r["written"] and r["exit"] == 0, r
        # the lease commit touches only lease.json, has the agent prefix, and HEAD was fast-forwarded
        assert git(a, "log", "-1", "--format=%s") == "cycle(lease): air acquire"
        assert git(a, "show", "--name-only", "--format=", "HEAD").split() == [rel]
        assert git(a, "rev-parse", "HEAD") == git(a, "rev-parse", "origin/phil-crypto") == r["sha"]
        on_disk = json.loads((a / rel).read_text())
        assert on_disk["runner"] == "air" and on_disk["cycle"] == "forward" and on_disk["ttl_s"] == 3000
        assert set(on_disk) == {"runner", "cycle", "started", "ttl_s", "pid"}

        _use_clone(b)                      # runner bigsur-like second brain, still on the old tip
        r2 = lease.acquire("research", runner="m2")
        assert r2["exit"] == 3 and not r2["acquired"] and r2["runner"] == "air", r2
        c2 = lease.check(runner="m2")
        assert c2["held"] and c2["fresh"] and c2["runner"] == "air"
        rel_b = lease.release(runner="m2")
        assert rel_b["exit"] == 3 and not rel_b["released"], rel_b      # not ours

        # expiry: 3001 s later the lease is takeable, but b's HEAD is behind origin ⇒ not written (4)
        real_now = lease._now
        lease._now = lambda: real_now() + dt.timedelta(seconds=3001)
        try:
            r3 = lease.acquire("research", runner="m2")
            assert r3["exit"] == 4 and "behind" in r3["reason"], r3
            git(b, "pull", "-q", "--ff-only")
            r4 = lease.acquire("research", runner="m2")
            assert r4["acquired"] and r4["written"] and r4["replaced"] == "air", r4
        finally:
            lease._now = real_now

        _use_clone(a)                       # air comes back: the lease is m2's now
        git(a, "pull", "-q", "--ff-only")
        r5 = lease.acquire("forward", runner="air")
        assert r5["exit"] == 3, r5
        _use_clone(b)
        r6 = lease.release(runner="m2")
        assert r6["released"] and r6["exit"] == 0, r6
        assert git(b, "log", "-1", "--format=%s", "origin/phil-crypto") == "cycle(lease): m2 release"
        r7 = lease.release(runner="m2")
        assert r7["exit"] == 0 and r7["status"] == "free"
        _use_clone(a)
        git(a, "pull", "-q", "--ff-only")
        r8 = lease.acquire("forward", runner="air")
        assert r8["acquired"] and r8["exit"] == 0

        # local uncommitted edit to lease.json ⇒ refuse (4)
        _use_clone(b)
        git(b, "pull", "-q", "--ff-only")
        (b / rel).write_text("{}\n")
        r9 = lease.acquire("forward", runner="b2", ttl_s=1)
        assert r9["exit"] in (3, 4), r9
        git(b, "checkout", "--", rel)

        # CLI exit codes
        env = dict(os.environ, TRADER_LEASE_REPO=str(b), TRADER_JOURNAL_DIR=str(b / "trader" / "journal"))
        p = subprocess.run([PY, "-m", "trader.core.lease", "acquire", "--cycle", "research", "--runner", "m2"],
                           cwd=str(REPO), capture_output=True, text=True, env=env)
        assert p.returncode == 3 and json.loads(p.stdout)["runner"] == "air", (p.returncode, p.stdout, p.stderr)
        p = subprocess.run([PY, "-m", "trader.core.lease", "check"], cwd=str(REPO), capture_output=True,
                           text=True, env=env)
        assert p.returncode == 0 and json.loads(p.stdout)["held"]

        # no remote ⇒ acquired, not written, nothing committed
        solo = root / "solo"
        solo.mkdir()
        git(solo, "init", "-q")
        _use_clone(solo)
        r10 = lease.acquire("research", runner="air")
        assert r10["acquired"] and not r10["written"] and r10["exit"] == 0
        assert lease.release(runner="air")["exit"] == 0
    finally:
        os.environ.pop("TRADER_LEASE_REPO", None)
        os.environ["TRADER_JOURNAL_DIR"] = saved


@case
def test_lease_never_publishes_local_commits_and_refuses_agent():
    """Regressions: acquire built on local HEAD (pushing unpushed agent commits); the agent could drive the lease."""
    saved = os.environ["TRADER_JOURNAL_DIR"]
    root = TMP / "lease2"
    origin, a = root / "origin.git", root / "a"
    root.mkdir()
    git(root, "init", "--bare", "-q", "-b", "phil-crypto", str(origin))
    git(root, "clone", "-q", str(origin), str(a))
    git(a, "checkout", "-q", "-b", "phil-crypto")
    (a / "trader" / "journal").mkdir(parents=True)
    (a / "trader" / "journal" / ".gitkeep").write_text("")
    git(a, "add", "-A")
    git(a, "commit", "-q", "-m", "operator: init")
    git(a, "push", "-q", "origin", "phil-crypto")
    try:
        _use_clone(a)
        base = git(a, "rev-parse", "HEAD")
        # an unpushed local commit (e.g. made by agent tools mid-cycle) on top of the remote tip
        (a / "trader" / "journal" / "sneaky.txt").write_text("unvalidated\n")
        git(a, "add", "-A")
        git(a, "commit", "-q", "-m", "cycle(research): sneaky")
        sneaky = git(a, "rev-parse", "HEAD")
        r = lease.acquire("research", runner="air")
        assert r["exit"] == 4 and not r["acquired"] and "ahead" in r["reason"], r
        assert git(origin, "rev-parse", "phil-crypto") == base                    # nothing published
        assert git(a, "rev-parse", "HEAD") == sneaky
        # once HEAD == remote tip, the lease commit's parent is the fetched remote base
        git(a, "reset", "-q", "--hard", base)
        r = lease.acquire("research", runner="air")
        assert r["acquired"] and r["written"], r
        assert git(origin, "rev-parse", "%s^" % r["sha"]) == base
        assert git(origin, "show", "--name-only", "--format=", r["sha"]).split() == ["trader/journal/lease.json"]
        tip = git(origin, "rev-parse", "phil-crypto")
        # agent context: PHIL_LEASE=acquired => acquire/release refused (4), nothing fetched or pushed
        os.environ["PHIL_LEASE"] = "acquired"
        try:
            ra = lease.acquire("research", runner="air")
            assert ra["exit"] == 4 and not ra["acquired"] and "PHIL_LEASE" in ra["reason"], ra
            rr = lease.release(runner="air")
            assert rr["exit"] == 4 and not rr["released"] and "PHIL_LEASE" in rr["reason"], rr
            assert lease.check(runner="air")["exit"] == 0                          # read-only stays allowed
            env = dict(os.environ, TRADER_LEASE_REPO=str(a), TRADER_JOURNAL_DIR=str(a / "trader" / "journal"))
            for cmd in ("acquire", "release"):
                p = subprocess.run([PY, "-m", "trader.core.lease", cmd, "--runner", "air"], cwd=str(REPO),
                                   capture_output=True, text=True, env=env)
                assert p.returncode == 4 and "PHIL_LEASE" in json.loads(p.stdout)["reason"], (cmd, p.stdout)
        finally:
            os.environ.pop("PHIL_LEASE", None)
        assert git(origin, "rev-parse", "phil-crypto") == tip
        assert lease.release(runner="air")["released"]
    finally:
        os.environ.pop("TRADER_LEASE_REPO", None)
        os.environ.pop("PHIL_LEASE", None)
        os.environ["TRADER_JOURNAL_DIR"] = saved


# ── data helpers for sweep tests ──────────────────────────────────────────────
def lab_loader(coin, tf):
    """Research-view frame straight from the lab CSVs (worker.py load_all rule), < cutoff."""
    df = pd.read_csv(LAB / ("%s_%s.csv" % (coin, tf)), index_col=0, parse_dates=True)
    df = df[df.index < CUTOFF]
    f = pd.read_csv(LAB / ("%s_funding.csv" % coin), index_col=0, parse_dates=True).iloc[:, 0]
    delta = pd.Timedelta("1D") if tf == "1d" else pd.Timedelta("4h")
    df["funding"] = f.groupby((f.index - pd.Timedelta(milliseconds=1)).floor(delta)).sum() \
        .reindex(df.index).fillna(0.0).values
    return df


STUB_DIR = TMP / "stub"
STUB_DIR.mkdir()
(STUB_DIR / "phil_stub_eval.py").write_text('''
import numpy as np, pandas as pd
def evaluate(spec, df, cost_side):
    n = int(spec["comps"][0]["p"].get("n", 20)) * 3
    c = df["close"]
    pos = (c > c.rolling(n).mean()).astype(float)
    wx = pos.shift(1).fillna(0.0)
    turn = pos.diff().abs().fillna(pos.abs()).shift(1).fillna(0.0)
    r = wx * c.pct_change().fillna(0.0) - turn * cost_side
    return r, wx
''')
STUB = "phil_stub_eval:evaluate"
sys.path.insert(0, str(STUB_DIR))


def rsi_spec(n=14, up=55, tf="1d", adx=0):
    return {"tf": tf, "mode": "SPOT", "comps": [{"t": "rsi_reg", "p": {"n": n, "up": up, "gap": 5}}],
            "comb": "and", "k": 1, "adx": adx, "vt": 0.5, "stop": 0}


def small_job(**kw):
    job = {"hypothesis_id": "H-9001", "specs": [rsi_spec(7), rsi_spec(14), rsi_spec(21, 60)],
           "coins": ["BTC", "ETH"], "host": "local", "top_k": 5,
           "quota": {"max_disk_mb": 256, "max_ram_mb": 4096, "max_minutes": 5, "max_procs": 2}}
    job.update(kw)
    return job


def register_hyp(hid, title="fleet test hypothesis"):
    """Write strategy/hypotheses/<hid>.json (hash-verified) and pre-register it in the trials registry."""
    from trader.core import trials
    h = {"id": hid, "title": title, "family": "rsi_reg", "max_trials": 5000}
    h["hash"] = trials.hypothesis_hash(h)
    d = config.strategy_dir() / "hypotheses"
    d.mkdir(parents=True, exist_ok=True)
    f = d / ("%s.json" % hid)
    f.write_text(json.dumps(h, sort_keys=True))
    return trials.register_hypothesis(f)


def trials_rows(**flt):
    p = Path(os.environ["TRADER_JOURNAL_DIR"]) / "trials.jsonl"
    rows = config.jsonl_read(p) if p.exists() else []
    return [r for r in rows if all(r.get(k) == v for k, v in flt.items())]


# ── sweep runner ──────────────────────────────────────────────────────────────
@case
def test_sweep_runner_rows_match_worker_rules():
    jd = TMP / "sr1"
    job = dispatch._prepare(small_job(), CFG)
    st = dispatch.stage(job, jd, loader=lab_loader, cfg=CFG)
    assert st["files"] == 2 and not st["coins_missing"]
    for p in (jd / "data").glob("*.csv"):
        assert pd.read_csv(p, index_col=0, parse_dates=True).index.max() < CUTOFF
    res = sweep_runner.run(jd / "job.json", procs=1, evaluator_env=STUB)
    assert res["n_configs"] == 3 and res["n_trials"] == 6 and res["errors"] == 0, res
    mom = res["sr_moments"]["1d"]
    assert mom[0] == 6
    rows = [json.loads(x) for x in (jd / "rows.jsonl").read_text().splitlines()]
    assert len(rows) == 6 and abs(sum(r["sh_tv"] for r in rows) - mom[1]) < 1e-9
    assert all(r["pass"] == sweep_runner.passes(r, sweep_runner.DEFAULT_CRITERION) for r in rows)
    # recompute one row by hand with worker.py formulas
    df = pd.read_csv(jd / "data" / "ETH_1d.csv", index_col=0, parse_dates=True)
    sys.modules.pop("phil_stub_eval", None)
    import phil_stub_eval
    r, wx = phil_stub_eval.evaluate(rsi_spec(14), df, 0.003)
    start = df.index[0] + pd.Timedelta(days=210)
    m = r.index >= start
    re_, w = r.values[m], wx.values[m]
    tr = r.index[m] < pd.Timestamp("2024-01-01")
    cagr_tr, sh_tr = sweep_runner.seg_stats(re_[tr], 365)
    _, sh_tv = sweep_runner.seg_stats(re_, 365)
    row = [x for x in rows if x["coin"] == "ETH" and x["i"] == 1][0]
    assert abs(row["cagr_tr"] - cagr_tr) < 1e-12 and abs(row["sh_tr"] - sh_tr) < 1e-12
    assert abs(row["sh_tv"] - sh_tv) < 1e-12
    assert row["entries"] == int(((w[1:] != 0) & (np.sign(w[1:]) != np.sign(w[:-1]))).sum())
    assert row["spec_hash"] == sweep_runner.spec_hash(rsi_spec(14))
    # result rows carry the spec, are ranked, and the file is within max_result_kb
    assert all("spec" in x and "i" not in x for x in res["rows"])
    assert (jd / "result.json").stat().st_size <= CFG["fleet"]["max_result_kb"] * 1024
    assert json.loads((jd / "state.json").read_text())["state"] == "done"
    # spec_hash is the worker.py cfg_hash of the 8 core keys (extra keys ignored)
    assert sweep_runner.spec_hash(dict(rsi_spec(14), adx=0, name="x")) == "cd9ff45458116162a26b8d3c8fb059ec"


@case
def test_sweep_runner_truncates_and_refuses_sealed():
    res = {"job_id": "J-x", "rows": [{"spec": {"pad": "x" * 500}, "sh_tv": i} for i in range(100)]}
    out = sweep_runner._truncate_rows(res, 20000)
    assert len(json.dumps(out, sort_keys=True).encode()) <= 20000 and out["rows_truncated"] > 0
    assert out["rows"][0]["sh_tv"] == 0                                         # keeps the head (best first)
    jd = TMP / "sr-sealed"
    job = dispatch._prepare(small_job(coins=["BTC"]), CFG)
    dispatch.stage(job, jd, loader=lab_loader, cfg=CFG)
    full = pd.read_csv(LAB / "BTC_1d.csv", index_col=0, parse_dates=True)
    full.assign(funding=0.0).to_csv(jd / "data" / "BTC_1d.csv", date_format="%Y-%m-%d")   # sealed bars leak in
    p = subprocess.run([PY, "-m", "trader.fleet.sweep_runner", str(jd / "job.json")], cwd=str(REPO),
                       capture_output=True, text=True, env=dict(os.environ, PHIL_SWEEP_EVALUATOR=STUB,
                                                                PYTHONPATH="%s:%s" % (REPO, STUB_DIR)))
    assert p.returncode == 2 and "vault_cutoff" in p.stdout, (p.returncode, p.stdout, p.stderr[-500:])
    assert json.loads((jd / "state.json").read_text())["state"] == "failed"


@case
def test_sweep_runner_quota_kill():
    jd = TMP / "sr-quota"
    job = dispatch._prepare(small_job(quota={"max_disk_mb": 0.001, "max_ram_mb": 4096, "max_minutes": 5,
                                             "max_procs": 1}), CFG)
    dispatch.stage(job, jd, loader=lab_loader, cfg=CFG)
    p = subprocess.run([PY, "-m", "trader.fleet.sweep_runner", str(jd / "job.json")], cwd=str(REPO),
                       capture_output=True, text=True, env=dict(os.environ, PHIL_SWEEP_EVALUATOR=STUB,
                                                                PYTHONPATH="%s:%s" % (REPO, STUB_DIR)))
    assert p.returncode == 3, (p.returncode, p.stdout, p.stderr[-800:])
    st = json.loads((jd / "state.json").read_text())
    assert st["state"] == "killed" and "MB" in st["reason"]
    assert not (jd / "result.json").exists()


@case
def test_sweep_runner_multiprocess_real_backtest():
    """2 procs (spawn) with the real backtest.evaluate_df when the backtest area is importable."""
    try:
        from trader.core import backtest  # noqa: F401
        env = None
    except Exception as e:  # concurrent wave: backtest not there yet
        print("    (backtest unavailable: %s; using stub)" % e)
        env = STUB
    jd = TMP / "sr-mp"
    job = dispatch._prepare(small_job(top_k=10), CFG)
    dispatch.stage(job, jd, loader=lab_loader, cfg=CFG)
    env_run = dict(os.environ, PYTHONPATH="%s:%s" % (REPO, STUB_DIR))
    env_run.pop("PHIL_SWEEP_EVALUATOR", None)
    if env:
        env_run["PHIL_SWEEP_EVALUATOR"] = env
    # through the CLI (production path): spawned workers re-import sweep_runner, not this test module
    p = subprocess.run([PY, "-m", "trader.fleet.sweep_runner", str(jd / "job.json"), "--procs", "2"],
                       cwd=str(REPO), capture_output=True, text=True, env=env_run, timeout=170)
    assert p.returncode == 0, (p.stdout, p.stderr[-1500:])
    res = json.loads((jd / "result.json").read_text())
    assert json.loads((jd / "state.json").read_text())["procs"] == 2
    assert res["n_configs"] == 3 and res["errors"] == 0 and res["n_trials"] == 6, res
    if env is None:
        # cross-check one row against a direct evaluate_df call on the same frame
        from trader.core import backtest
        df = pd.read_csv(jd / "data" / "BTC_1d.csv", index_col=0, parse_dates=True)
        out = backtest.evaluate_df(rsi_spec(14), df, period="all", cost_side=0.003)
        r = out["returns"]
        r = r[r.index >= df.index[0] + pd.Timedelta(days=210)].values
        _, sh_tv = sweep_runner.seg_stats(r, 365)
        row = [x for x in res["rows"] if x["coin"] == "BTC" and x["spec_hash"] == sweep_runner.spec_hash(rsi_spec(14))]
        assert row and abs(row[0]["sh_tv"] - sh_tv) < 1e-9, (row, sh_tv)


# ── dispatch ──────────────────────────────────────────────────────────────────
@case
def test_dispatch_validation_and_quota():
    register_hyp("H-9001")
    bad = [
        (small_job(view="full"), "view must be 'research'"),
        (small_job(coins=["BTC", "NOPE"]), "not in universe"),
        (small_job(specs=[rsi_spec()] * 2600, coins=["BTC", "ETH"]), "max_trials_per_hypothesis"),
        (small_job(specs=[{"tf": "1w", "mode": "SPOT", "comps": []}]), "spec 0"),
        (small_job(periods=["test"]), "research periods"),
        (dict(small_job(), hypothesis_id=None), "missing hypothesis_id"),
    ]
    for job, needle in bad:
        try:
            dispatch._prepare(job, CFG)
        except dispatch.DispatchError as e:
            assert needle in str(e), (needle, str(e))
        else:
            raise AssertionError("not refused: %s" % needle)
    q = dispatch.effective_quota(small_job(quota={"max_disk_mb": 1024}), "bigsur", CFG)
    assert q["max_disk_mb"] == 1024 and q["max_procs"] == 2
    for over in ({"max_disk_mb": 6000}, {"max_ram_mb": 9000}, {"max_procs": 8}):
        try:
            dispatch.effective_quota(small_job(quota=over), "bigsur", CFG)
        except dispatch.QuotaError:
            pass
        else:
            raise AssertionError("quota not enforced: %s" % over)
    try:
        dispatch.choose_host(small_job(host="berry"), CFG)
    except dispatch.DispatchError:
        pass
    else:
        raise AssertionError("berry is not a sweep host")


@case
def test_dispatch_quota_rejects_nan_inf_negative_and_unbounded_minutes():
    """Regression: NaN/inf/negative/str quotas made every `>` check False (unlimited disk/RAM on bigsur)."""
    bad = [{"max_disk_mb": "nan", "max_ram_mb": "nan"}, {"max_disk_mb": float("nan")},
           json.loads('{"max_ram_mb": NaN}'), {"max_disk_mb": float("inf")}, {"max_disk_mb": -1},
           {"max_disk_mb": 0}, {"max_ram_mb": "100"}, {"max_procs": True}, {"max_procs": 1.5},
           {"max_minutes": 1e9}, {"max_minutes": float("nan")}, {"max_minutes": -5}, "nan"]
    for quota in bad:
        for host in ("bigsur", "local"):
            try:
                dispatch.effective_quota(small_job(quota=quota), host, CFG)
            except dispatch.QuotaError:
                pass
            else:
                raise AssertionError("quota not refused on %s: %r" % (host, quota))
    # unknown keys are ignored, max_procs comes back as int, max_minutes at the default host ceiling is fine
    q = dispatch.effective_quota(small_job(quota={"max_minutes": dispatch.HOST_MAX_MINUTES_DEFAULT,
                                                  "max_procs": 3.0, "evil": float("nan")}), "bigsur", CFG)
    assert "evil" not in q and q["max_procs"] == 3 and isinstance(q["max_procs"], int)
    assert set(q) == set(sweep_runner.DEFAULT_QUOTA)
    # a host-level max_minutes overrides the default ceiling
    cfg2 = json.loads(json.dumps(CFG))
    cfg2["fleet"]["hosts"]["bigsur"]["max_minutes"] = 30
    try:
        dispatch.effective_quota(small_job(quota={"max_minutes": 31}), "bigsur", cfg2)
    except dispatch.QuotaError as e:
        assert "max_minutes" in str(e)
    else:
        raise AssertionError("host max_minutes not enforced")
    # submit (dry run) refuses before staging or copying anything
    register_hyp("H-9050")
    try:
        dispatch.submit(small_job(hypothesis_id="H-9050", host="local",
                                  quota={"max_disk_mb": "nan", "max_ram_mb": "nan", "max_minutes": 1e9}),
                        loader=lab_loader, dry_run=True)
    except dispatch.QuotaError:
        pass
    else:
        raise AssertionError("submit accepted a NaN quota")
    # sweep_runner on the host refuses the same job.json with EXIT_BAD_INPUT (defence in depth)
    for i, quota in enumerate(({"max_disk_mb": "nan", "max_ram_mb": "nan"}, {"max_minutes": float("inf")},
                               {"max_ram_mb": -1})):
        jd = TMP / ("sr-badquota-%d" % i)
        job = dispatch._prepare(small_job(), CFG)
        dispatch.stage(job, jd, loader=lab_loader, cfg=CFG)
        raw = json.loads((jd / "job.json").read_text())
        raw["quota"] = quota
        (jd / "job.json").write_text(json.dumps(raw))
        p = subprocess.run([PY, "-m", "trader.fleet.sweep_runner", str(jd / "job.json")], cwd=str(REPO),
                           capture_output=True, text=True, env=dict(os.environ, PHIL_SWEEP_EVALUATOR=STUB,
                                                                    PYTHONPATH="%s:%s" % (REPO, STUB_DIR)))
        assert p.returncode == 2 and "quota" in p.stdout, (quota, p.returncode, p.stdout, p.stderr[-500:])
        assert not (jd / "result.json").exists()
    for quota in ({"max_disk_mb": "nan"}, {"max_ram_mb": float("nan")}, {"max_procs": 0}, [1]):
        try:
            sweep_runner.sanitize_quota(quota)
        except ValueError:
            pass
        else:
            raise AssertionError("sanitize_quota accepted %r" % (quota,))


@case
def test_dispatch_dry_run_plan_for_bigsur():
    rec = dispatch.submit_plan(small_job(host="bigsur", quota={"max_procs": 6}), dry_run=True, loader=lab_loader)
    cmds = rec["commands"]
    flat = [" ".join(c) for c in cmds]
    assert rec["host"] == "bigsur" and len(cmds) == 4, flat
    assert cmds[0][0] == "ssh" and "BatchMode=yes" in cmds[0] and cmds[0][-2] == "bigsur"
    root = os.environ["PHIL_REMOTE_ROOT"]
    assert cmds[1][0] == "rsync" and "--delete" in cmds[1] and cmds[1][-1] == "bigsur:%s/code/trader/" % root
    assert "journal/" in cmds[1] and "tests/" in cmds[1]                        # never ship the journal
    assert cmds[2][-1] == "bigsur:%s/jobs/%s/" % (root, rec["job_id"])
    start = cmds[3][-1]
    assert "nohup" in start and "trader.fleet.sweep_runner" in start and ".phil-crypto/venv/bin/python" in start
    assert "PHIL_RUNNER=bigsur" in start and "TRADER_JOURNAL_DIR=" in start
    assert not Path(root).exists()                                              # nothing executed
    assert dispatch.status(rec["job_id"])["state"] == "queued"
    # with the configured (non-test) remote root the paths are home-relative for ssh/rsync
    saved = os.environ.pop("PHIL_REMOTE_ROOT")
    try:
        rec2 = dispatch.submit_plan(small_job(host="bigsur"), dry_run=True, loader=lab_loader)
        assert rec2["commands"][1][-1] == "bigsur:.phil-crypto/runner/code/trader/"
        assert '"$HOME"/' in rec2["commands"][0][-1]
    finally:
        os.environ["PHIL_REMOTE_ROOT"] = saved
    plan = dispatch.sync_code("berry", dry_run=True)
    assert any("berry:" in c for c in plan)
    # CLI dry run
    jf = TMP / "job-cli.json"
    jf.write_text(json.dumps(small_job(host="bigsur")))
    p = subprocess.run([PY, "-m", "trader.fleet.dispatch", "submit", str(jf), "--dry-run"], cwd=str(REPO),
                       capture_output=True, text=True, env=os.environ.copy())
    # the CLI uses vault.load for staging: OK when the data area is importable, else a clean JSON error
    out = json.loads(p.stdout)
    assert (p.returncode == 0 and out["ok"] and out["dry_run"]) or (p.returncode == 1 and not out["ok"]), p.stdout


@case
def test_dispatch_localhost_end_to_end():
    env = {"PHIL_SWEEP_EVALUATOR": STUB,
           "PYTHONPATH": "%s/code:%s" % (os.environ["PHIL_REMOTE_ROOT"], STUB_DIR)}
    jid = dispatch.submit(small_job(host="local"), loader=lab_loader, env=env)
    jd = Path(os.environ["PHIL_REMOTE_ROOT"]) / "jobs" / jid
    code = Path(os.environ["PHIL_REMOTE_ROOT"]) / "code" / "trader"
    assert (code / "fleet" / "sweep_runner.py").exists() and (code / "config" / "protected.json").exists()
    assert not (code / "journal").exists() and not (code / "tests").exists()
    for p in (jd / "data").glob("*.csv"):                                       # research view only on the host
        assert pd.read_csv(p, index_col=0, parse_dates=True).index.max() < CUTOFF
    deadline = time.time() + 120
    st = dispatch.status(jid)
    while st["state"] in ("queued", "running") and time.time() < deadline:
        time.sleep(0.5)
        st = dispatch.status(jid)
    assert st["state"] == "done", (st, (jd / "runner.log").read_text()[-1500:] if (jd / "runner.log").exists() else "")
    assert st["progress"] == 1.0
    assert "n_pass" not in st                                                    # no results before logging
    prov = trials_rows(job_id=jid)                                               # counted since submit
    assert len(prov) == 1 and prov[0]["result_file"].startswith("pending:") and prov[0]["n_trials"] == 6, prov
    res = dispatch.collect(jid)
    # regression: collect logs the batch BEFORE returning results (no separate ingest needed)
    rows = [r for r in trials_rows(job_id=jid) if not r["result_file"].startswith("pending:")]
    assert len(rows) == 1 and rows[0]["trial_id"] == res["batch_trial_id"], rows
    assert trials_rows(job_id=jid)[-1]["trial_id"] == res["batch_trial_id"]     # final row supersedes (last wins)
    assert rows[0]["hypothesis_hash"] and rows[0]["n_trials"] == 6
    sweep_file = Path(os.environ["TRADER_JOURNAL_DIR"]) / "sweeps" / ("%s.json" % jid)
    assert sweep_file.exists() and sweep_file.stat().st_size <= CFG["fleet"]["max_result_kb"] * 1024
    assert res["n_trials"] == 6 and res["job_id"] == jid and res["view"] == "research"
    assert res["host"] == "local" and res["cost_side"] == 0.003
    assert dispatch.collect(jid)["n_trials"] == 6                               # idempotent (add-only file)
    try:
        from trader.core import trials  # noqa: F401
        has_trials = True
    except Exception:
        has_trials = False
    if has_trials:
        tid = dispatch.ingest(jid)
        assert tid == dispatch.ingest(jid)                                       # logged once
        rows = [r for r in config.jsonl_read(Path(os.environ["TRADER_JOURNAL_DIR"]) / "trials.jsonl")
                if r.get("job_id") == jid and not r["result_file"].startswith("pending:")]
        assert len(rows) == 1 and rows[0]["kind"] == "batch" and rows[0]["n_trials"] == 6, rows
        assert rows[0]["sr_moments"]["1d"][0] == 6
    dispatch.cleanup(jid)
    assert not jd.exists()
    # CLI status of a cleaned job: the registry knows it
    p = subprocess.run([PY, "-m", "trader.fleet.dispatch", "status", jid], cwd=str(REPO), capture_output=True,
                       text=True, env=os.environ.copy())
    assert p.returncode == 0 and json.loads(p.stdout)["job_id"] == jid


@case
def test_dispatch_submit_counts_trials_before_runner():
    """Regression: a local-host sweep's rows.jsonl/result.json are readable without collect, so the trials must be
    in the ledger from submit on (provisional batch row), even if collect never runs."""
    from trader.core import trials
    register_hyp("H-9010", "submit-time counting regression")
    n0 = trials.dsr_context()["n_trials"]
    # the runner must never start uncounted: log first, then start (spy on the order)
    order = []
    real_log, real_exec = dispatch._log_provisional, dispatch._exec
    dispatch._log_provisional = lambda job, host: (order.append("log"), real_log(job, host))[1]

    def spy_exec(host, cmd, **kw):
        if "sweep_runner" in cmd:
            order.append("start")
        return real_exec(host, cmd, **kw)
    dispatch._exec = spy_exec
    env = {"PHIL_SWEEP_EVALUATOR": STUB,
           "PYTHONPATH": "%s/code:%s" % (os.environ["PHIL_REMOTE_ROOT"], STUB_DIR)}
    try:
        jid = dispatch.submit(small_job(hypothesis_id="H-9010", host="local"), loader=lab_loader, env=env)
    finally:
        dispatch._log_provisional, dispatch._exec = real_log, real_exec
    assert order == ["log", "start"], order
    rows = trials_rows(job_id=jid)
    assert len(rows) == 1, rows
    r = rows[0]
    assert r["kind"] == "batch" and r["caller"] == "sweep" and r["hypothesis_id"] == "H-9010"
    assert r["n_trials"] == 6 and r["n_configs"] == 3 and r["result_file"].startswith("pending:")
    assert dispatch._reg_read(jid)["provisional_trial_id"] == r["trial_id"]
    assert trials.dsr_context()["n_trials"] >= n0 + 6                           # in the DSR N with no collect
    # wait for the runner: results are now readable on disk while collect has not run -> still counted
    jd = Path(os.environ["PHIL_REMOTE_ROOT"]) / "jobs" / jid
    deadline = time.time() + 120
    while dispatch.status(jid)["state"] in ("queued", "running") and time.time() < deadline:
        time.sleep(0.5)
    assert (jd / "result.json").exists()
    assert trials.dsr_context()["n_trials"] >= n0 + 6
    # collect: the provisional row is NOT mistaken for the final one (crash-recovery path included)
    res = dispatch.collect(jid)
    final = [x for x in trials_rows(job_id=jid) if not x["result_file"].startswith("pending:")]
    assert len(final) == 1 and final[0]["trial_id"] == res["batch_trial_id"] != r["trial_id"]
    assert final[0]["sr_moments"]["1d"][0] == 6
    e = dispatch._reg_read(jid)
    e.pop("batch_trial_id")
    dispatch._reg_write(jid, e)
    assert dispatch.collect(jid)["batch_trial_id"] == res["batch_trial_id"]
    assert len(trials_rows(job_id=jid)) == 2
    # dry runs never log
    n_rows = len(trials_rows())
    dispatch.submit(small_job(hypothesis_id="H-9010", host="local"), loader=lab_loader, dry_run=True)
    assert len(trials_rows()) == n_rows
    dispatch.cleanup(jid)


@case
def test_dispatch_run_local_and_disk_quota():
    res = dispatch.run_local(small_job(), loader=lab_loader, evaluator_env=STUB)
    os.environ.pop("PHIL_SWEEP_EVALUATOR", None)
    assert res["n_trials"] == 6 and dispatch.status(res["job_id"])["state"] == "done"
    assert len(trials_rows(job_id=res["job_id"])) == 1
    assert dispatch.collect(res["job_id"])["result_file"].endswith("%s.json" % res["job_id"])
    # host disk quota: pretend the local host only has 1 MB of runner space
    saved = dispatch.LOCAL_HOST["max_disk_gb"]
    dispatch.LOCAL_HOST["max_disk_gb"] = 0.25
    try:
        dispatch.submit(small_job(host="local", quota={"max_disk_mb": 256}), loader=lab_loader)
    except dispatch.QuotaError as e:
        assert "max_disk_gb" in str(e)
    else:
        raise AssertionError("host disk quota not enforced")
    finally:
        dispatch.LOCAL_HOST["max_disk_gb"] = saved


@case
def test_dispatch_run_local_logs_before_return():
    """Regression: run_local used to return n_trials/n_pass/rows with 0 rows in trials.jsonl."""
    from trader.core import trials
    reg = register_hyp("H-9002", "run_local logging regression")
    before = len(trials_rows())
    res = dispatch.run_local(small_job(hypothesis_id="H-9002"), loader=lab_loader, evaluator_env=STUB)
    os.environ.pop("PHIL_SWEEP_EVALUATOR", None)
    jid = res["job_id"]
    rows = trials_rows(job_id=jid)
    assert len(trials_rows()) == before + 1 and len(rows) == 1, rows
    r = rows[0]
    assert r["kind"] == "batch" and r["caller"] == "sweep" and r["trial_id"] == res["batch_trial_id"]
    assert r["hypothesis_id"] == "H-9002" and r["hypothesis_hash"] == reg["hypothesis_hash"]
    assert r["n_trials"] == 6 == res["n_trials"] and r["n_pass"] == res["n_pass"]
    assert r["result_file"] and r["result_file"].endswith("%s.json" % jid)
    assert trials.dsr_context()["n_trials"] >= 6
    # idempotent: collect / ingest / CLI never log twice
    assert dispatch.collect(jid)["batch_trial_id"] == res["batch_trial_id"]
    assert dispatch.ingest(jid) == res["batch_trial_id"]
    p = subprocess.run([PY, "-m", "trader.fleet.dispatch", "collect", jid], cwd=str(REPO), capture_output=True,
                       text=True, env=os.environ.copy())
    assert p.returncode == 0 and json.loads(p.stdout)["batch_trial_id"] == res["batch_trial_id"], p.stdout
    assert len(trials_rows(job_id=jid)) == 1
    # crash between log_batch and the registry write: the existing ledger row is reused, not duplicated
    e = dispatch._reg_read(jid)
    e.pop("batch_trial_id")
    dispatch._reg_write(jid, e)
    assert dispatch.collect(jid)["batch_trial_id"] == res["batch_trial_id"]
    assert len(trials_rows(job_id=jid)) == 1
    assert dispatch._reg_read(jid)["batch_trial_id"] == res["batch_trial_id"]


@case
def test_dispatch_refuses_unregistered_or_closed_hypothesis():
    """Regression: dispatch ran sweeps for a never-registered hypothesis (batch row hypothesis_hash null)."""
    from trader.core import trials
    sweeps = Path(os.environ["TRADER_JOURNAL_DIR"]) / "sweeps"
    n_sweeps = len(list(sweeps.glob("*.json"))) if sweeps.exists() else 0
    n_rows = len(trials_rows())

    def refused(job, needle, fn):
        try:
            fn(job)
        except dispatch.DispatchError as e:
            assert needle in str(e), (needle, str(e))
        else:
            raise AssertionError("not refused: %s" % needle)

    run = lambda j: dispatch.run_local(j, loader=lab_loader, evaluator_env=STUB)  # noqa: E731
    dry = lambda j: dispatch.submit_plan(j, dry_run=True, loader=lab_loader)     # noqa: E731
    refused(small_job(hypothesis_id="H-9999"), "not registered", run)
    refused(small_job(hypothesis_id="H-9999", host="bigsur"), "not registered", dry)
    reg = register_hyp("H-9003", "closed hypothesis")
    refused(small_job(hypothesis_id="H-9003", hypothesis_hash="0" * 64), "registered hash", run)
    ok = dispatch._prepare(small_job(hypothesis_id="H-9003"), CFG)
    assert ok["hypothesis_hash"] == reg["hypothesis_hash"]                       # filled from the registry
    trials.set_status("H-9003", "sweeping")
    dispatch._prepare(small_job(hypothesis_id="H-9003"), CFG)                     # still sweepable
    trials.set_status("H-9003", "failed_trainval")
    refused(small_job(hypothesis_id="H-9003"), "failed_trainval", run)
    # CLI: refused job => exit 2, nothing evaluated
    jf = TMP / "job-unreg.json"
    jf.write_text(json.dumps(small_job(hypothesis_id="H-9998")))
    p = subprocess.run([PY, "-m", "trader.fleet.dispatch", "run-local", str(jf)], cwd=str(REPO),
                       capture_output=True, text=True, env=os.environ.copy())
    assert p.returncode == 2 and "not registered" in json.loads(p.stdout)["error"], (p.returncode, p.stdout)
    os.environ.pop("PHIL_SWEEP_EVALUATOR", None)
    assert len(trials_rows()) == n_rows
    assert (len(list(sweeps.glob("*.json"))) if sweeps.exists() else 0) == n_sweeps


@case
def test_dispatch_stage_with_vault_if_available():
    try:
        from trader.core import vault
        df = vault.load("BTC", "1d", "research")
    except Exception as e:
        print("    (vault unavailable: %s; skipped)" % type(e).__name__)
        return
    assert df.index.max() < CUTOFF
    jd = TMP / "stage-vault"
    st = dispatch.stage(dispatch._prepare(small_job(coins=["BTC"]), CFG), jd, cfg=CFG)
    got = pd.read_csv(jd / "data" / "BTC_1d.csv", index_col=0, parse_dates=True)
    assert st["files"] == 1 and len(got) == len(df) and got.index.max() < CUTOFF
    assert np.allclose(got["close"].values, df["close"].values)


# ── health ────────────────────────────────────────────────────────────────────
@case
def test_health_evaluate_and_collect():
    now = dt.datetime(2026, 10, 21, 1, 0, tzinfo=dt.timezone.utc)
    hosts = CFG["fleet"]["hosts"]
    e, al = health.evaluate_host("bigsur", hosts["bigsur"],
                                 {"now": "1", "load": "1.5", "disk_free_kb": str(16 * 1024 * 1024),
                                  "python": "1", "runner_kb": str(int(4.6 * 1024 * 1024)), "jobs_running": "1"},
                                 CFG, {}, now)
    assert e["ok"] and e["runner_disk_gb"] == 4.6 and e["jobs_running"] == 1 and e["checks"]["python"]
    assert [a["level"] for a in al] == ["warn"] and "runner disk" in al[0]["msg"]
    assert health._f("2,10") == 2.1 and health._f(None, 0.0) == 0.0 and health._f("x") is None
    e, al = health.evaluate_host("bigsur", hosts["bigsur"], {"now": "1", "runner_kb": str(6 * 1024 * 1024),
                                                             "disk_free_kb": str(1024 * 1024)}, CFG, {}, now)
    assert not e["ok"] and sorted(a["level"] for a in al) == ["crit", "crit"]
    e, al = health.evaluate_host("bigsur", hosts["bigsur"], None, CFG, {"last_seen": "2026-10-20T00:00:00Z"}, now)
    assert not e["ok"] and e["last_seen"] == "2026-10-20T00:00:00Z" and al[0]["level"] == "crit"
    e, al = health.evaluate_host("m2", hosts["m2"], None, CFG, {}, now)
    assert e == {"last_seen": None, "ok": False, "checks": {"ssh": False}, "optional": True} and not al
    lease_json = json.dumps({"runner": "air", "cycle": "forward", "started": "2026-10-20T00:10:00Z", "ttl_s": 5400})
    e, al = health.evaluate_host("air", hosts["air"], {"now": "1", "claude": "1", "disk_free_kb": str(60 << 20),
                                                       "lease_json": lease_json,
                                                       "last_cycle": "2026-10-19T00:10:03Z forward error runner=air"},
                                 CFG, {}, now)
    assert e["checks"]["claude"] and e["checks"]["lease"] == "expired:air"
    msgs = " | ".join(a["msg"] for a in al)
    assert "expired" in msgs and "h ago" in msgs and "forward error" in msgs and e["ok"]
    fresh = str(int(now.timestamp()) - 3600)
    e, al = health.evaluate_host("berry", hosts["berry"], {"now": "1", "data_newest": fresh,
                                                           "watch_mtime": fresh, "python": "1"}, CFG, {}, now)
    assert e["checks"]["data_fresh"] and e["checks"]["watch_last_run"] == "2026-10-21T00:00:00Z" and not al

    def fake_probe(name, h, cfg):
        return None if name in ("m2",) else {"now": "1", "disk_free_kb": str(40 << 20), "python": "1",
                                              "claude": "1", "runner_kb": "1000",
                                              "data_newest": str(int(time.time()))}
    doc = health.collect(probe=fake_probe)
    assert set(doc["hosts"]) == set(hosts) and doc["by"] == "testrunner"
    assert doc["hosts"]["m2"]["ok"] is False and doc["hosts"]["bigsur"]["ok"]
    out = health.write(doc, TMP / "outbox" / "fleet-health.json")
    assert json.loads(out.read_text())["updated"] == doc["updated"]


@case
def test_health_probe_local_shell():
    """The probe script runs on this Mac (m2's ssh alias is 'local') and parses."""
    h = dict(CFG["fleet"]["hosts"]["m2"])
    h["roles"] = ["sweep", "brain", "data"]
    kv = health.run_probe("m2", h, CFG)
    assert kv is not None and "disk_free_kb" in kv and int(kv["disk_free_kb"]) > 0, kv
    assert "runner_kb" in kv and "jobs_running" in kv and "claude" in kv and "data_newest" in kv
    entry, _ = health.evaluate_host("m2", h, kv, CFG, {}, config.utcnow())
    assert entry["checks"]["ssh"] and entry["disk_free_gb"] > 0
    p = subprocess.run([PY, "-m", "trader.fleet.health", "run", "--hosts", "m2", "--dry-run"], cwd=str(REPO),
                       capture_output=True, text=True, env=os.environ.copy(), timeout=120)
    assert p.returncode == 0 and "m2" in json.loads(p.stdout)["hosts"], p.stderr[-500:]


@case
def test_health_push_dry_run():
    ob = Path(os.environ["PHIL_OUTBOX"])
    ob.mkdir(parents=True, exist_ok=True)
    (ob / "trigger-20261021T0100.json").write_text("{}")
    res = health.push(dry_run=True)
    cmds = [" ".join(c) for c in res["commands"]]
    assert res["ok"] and len(cmds) == 3
    assert cmds[1].endswith("air:.phil-crypto/inbox/") and "--exclude trigger-*.json" in cmds[1]
    assert "--remove-source-files" in cmds[2] and "trigger-20261021T0100.json" in cmds[2]
    assert (ob / "trigger-20261021T0100.json").exists()


# ── inbox ─────────────────────────────────────────────────────────────────────
@case
def test_inbox_merge_dedup_and_consume():
    ib = Path(os.environ["PHIL_INBOX"])
    ib.mkdir(parents=True, exist_ok=True)
    jd = Path(os.environ["TRADER_JOURNAL_DIR"])
    (ib / "fleet-health.json").write_text(json.dumps({"updated": "2026-10-21T01:00:00Z", "by": "berry",
                                                      "hosts": {}, "alerts": []}))
    rows = [{"ts": "2026-10-21T01:00:00Z", "rule_id": "btc_drop", "coin": "BTC", "metric": "ret_24h",
             "value": -0.07, "threshold": -0.06, "action": "trigger_forward", "host": "berry"},
            {"ts": "2026-10-21T01:15:00Z", "rule_id": "eth_vol", "coin": "ETH", "metric": "vol_ratio_24h_30d",
             "value": 2.7, "threshold": 2.5, "action": "note", "host": "berry"}]
    (ib / "watch-triggers-berry.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    (ib / "trigger-20261021T010000Z.json").write_text(json.dumps(rows[0]))
    (ib / "trigger-bad.json").write_text("{torn")
    r1 = inbox.merge()
    assert r1["health_copied"] and r1["new_trigger_rows"] == 2 and len(r1["pending"]) == 1 and r1["errors"]
    assert json.loads((jd / "fleet-health.json").read_text())["by"] == "berry"
    r2 = inbox.merge()                                                          # re-sent log ⇒ no duplicates
    assert not r2["health_copied"] and r2["new_trigger_rows"] == 0
    assert len(config.jsonl_read(jd / "watch-triggers.jsonl")) == 2
    moved = inbox.consume(r2["pending"][0])
    assert Path(moved).parent.name == "done" and not inbox.merge()["pending"]
    p = subprocess.run([PY, "-m", "trader.fleet.inbox", "merge"], cwd=str(REPO), capture_output=True, text=True,
                       env=os.environ.copy())
    assert json.loads(p.stdout)["new_trigger_rows"] == 0


@case
def test_inbox_consume_refuses_non_trigger_paths():
    """Regression: `inbox consume FILE` moved any reachable path (e.g. the spot data store's BTC_1d.csv) into
    <parent>/done/, which let a poisoned CSV be seeded in its place."""
    ib = Path(os.environ["PHIL_INBOX"])
    ib.mkdir(parents=True, exist_ok=True)
    victim_dir = TMP / "victim" / "spot"
    victim_dir.mkdir(parents=True)
    victim = victim_dir / "BTC_1d.csv"
    victim.write_text("time,close\n2026-10-01,83137.0\n")
    outside_trigger = victim_dir / "trigger-20261021T000000Z.json"              # right name, wrong dir
    outside_trigger.write_text("{}")
    (ib / "fleet-health.json").write_text("{}")                                  # right dir, wrong name
    (ib / "sub").mkdir(exist_ok=True)
    (ib / "sub" / "trigger-x.json").write_text("{}")                             # nested
    link = ib / "trigger-link.json"
    if link.exists() or link.is_symlink():
        link.unlink()
    os.symlink(str(victim), str(link))                                           # symlink out of the inbox
    dotdot = str(ib / ".." / "victim" / "spot" / "trigger-20261021T000000Z.json")
    for bad in (victim, outside_trigger, ib / "fleet-health.json", ib / "sub" / "trigger-x.json", link, dotdot,
                ib / "trigger-missing.json"):
        try:
            inbox.consume(bad)
        except inbox.InboxError:
            pass
        else:
            raise AssertionError("consume accepted %s" % bad)
    assert victim.exists() and outside_trigger.exists() and (ib / "fleet-health.json").exists()
    assert not (victim_dir / "done").exists()
    # CLI: refused, exit 1, nothing moved (even when mixed with a valid trigger)
    good = ib / "trigger-20261022T000000Z.json"
    good.write_text("{}")
    p = subprocess.run([PY, "-m", "trader.fleet.inbox", "consume", str(good), str(victim)], cwd=str(REPO),
                       capture_output=True, text=True, env=os.environ.copy())
    assert p.returncode == 1 and "refusing" in json.loads(p.stdout)["error"], p.stdout
    assert victim.exists() and good.exists()
    # a genuine trigger is still consumed
    assert Path(inbox.consume(good)).parent == (ib / "done").resolve()
    link.unlink()
    shutil.rmtree(str(ib / "sub"))


@case
def test_inbox_merge_refuses_symlinks_and_foreign_json():
    """Regression: merge() followed symlinks and accepted any JSON, so a symlink fleet-health.json -> ~/.claude.json
    or watch-triggers-x.jsonl -> a transcript got copied into the journal (then committed and pushed)."""
    ib = Path(os.environ["PHIL_INBOX"])
    ib.mkdir(parents=True, exist_ok=True)
    jd = Path(os.environ["TRADER_JOURNAL_DIR"])
    for f in list(ib.iterdir()):
        if f.is_file() or f.is_symlink():
            f.unlink()
    secrets = TMP / "secrets"
    secrets.mkdir(exist_ok=True)
    (secrets / "claude.json").write_text(json.dumps({"secret": "TOPSECRET-json"}))
    (secrets / "transcript.jsonl").write_text(json.dumps({"token": "TOPSECRET-jsonl"}) + "\n")
    good_health = {"updated": "2026-10-21T02:00:00Z", "by": "berry", "hosts": {"berry": {"ok": True}}, "alerts": []}
    (secrets / "health.json").write_text(json.dumps(good_health))       # valid schema, but reached via symlink
    good_row = {"ts": "2026-10-21T02:00:00Z", "rule_id": "btc_drop", "coin": "BTC", "metric": "ret_24h",
                "value": -0.08, "threshold": -0.06, "action": "trigger_forward", "host": "berry"}
    (secrets / "rows.jsonl").write_text(json.dumps(good_row) + "\n")
    (secrets / "trig.json").write_text(json.dumps(good_row))

    def journal_text():
        return "".join(f.read_text() for f in (jd / "fleet-health.json", jd / "watch-triggers.jsonl") if f.exists())

    before_health = (jd / "fleet-health.json").read_bytes() if (jd / "fleet-health.json").exists() else None
    n_rows = len(config.jsonl_read(jd / "watch-triggers.jsonl"))
    # 1) symlinks out of the inbox (secret and schema-valid targets alike)
    os.symlink(str(secrets / "claude.json"), str(ib / "fleet-health.json"))
    os.symlink(str(secrets / "transcript.jsonl"), str(ib / "watch-triggers-x.jsonl"))
    os.symlink(str(secrets / "rows.jsonl"), str(ib / "watch-triggers-y.jsonl"))
    os.symlink(str(secrets / "trig.json"), str(ib / "trigger-20261021T020000Z.json"))
    r = inbox.merge()
    assert not r["health_copied"] and r["new_trigger_rows"] == 0 and not r["pending"], r
    assert any("symlink" in e for e in r["errors"]), r["errors"]
    assert "TOPSECRET" not in journal_text()
    assert ((jd / "fleet-health.json").read_bytes() if (jd / "fleet-health.json").exists() else None) == before_health
    for f in list(ib.iterdir()):
        if f.is_symlink():
            f.unlink()
    # 2) regular files with foreign JSON are refused by the schema checks
    (ib / "fleet-health.json").write_text(json.dumps({"secret": "TOPSECRET-json"}))
    (ib / "watch-triggers-x.jsonl").write_text(json.dumps({"token": "TOPSECRET-jsonl"}) + "\n"
                                               + json.dumps(dict(good_row, token="TOPSECRET-extra")) + "\n")
    (ib / "trigger-20261021T020000Z.json").write_text(json.dumps({"token": "TOPSECRET"}))
    r = inbox.merge()
    assert not r["health_copied"] and r["new_trigger_rows"] == 0 and not r["pending"], r
    assert any("schema" in e for e in r["errors"]) and "TOPSECRET" not in journal_text()
    bad_health = [dict(good_health, hosts={"evil": {}}), dict(good_health, hosts=[]), dict(good_health, by=5),
                  dict(good_health, extra="x"), {k: v for k, v in good_health.items() if k != "updated"}]
    for d in bad_health:
        assert inbox.valid_health(d, ["air", "bigsur", "berry", "m2"]), d
    assert inbox.valid_health(good_health, list(CFG["fleet"]["hosts"])) is None
    for row in (dict(good_row, value="x"), dict(good_row, action="rm -rf"), {k: v for k, v in good_row.items()
                if k != "coin"}, dict(good_row, value=True), dict(good_row, coin="x" * 600)):
        assert inbox.valid_trigger_row(row), row
    assert inbox.valid_trigger_row(dict(good_row, detail="capped")) is None
    # 3) size caps
    (ib / "fleet-health.json").write_text(json.dumps(dict(good_health, by="b" * 100,
                                                           alerts=[{"ts": "t", "host": "berry", "level": "warn",
                                                                    "msg": "m" * 500}] * 1000)))
    r = inbox.merge()
    assert not r["health_copied"] and any("cap" in e for e in r["errors"]), r["errors"]
    # 4) a well-formed regular inbox still merges
    (ib / "fleet-health.json").write_text(json.dumps(good_health))
    (ib / "watch-triggers-x.jsonl").write_text(json.dumps(good_row) + "\n")
    (ib / "trigger-20261021T020000Z.json").write_text(json.dumps(good_row))
    r = inbox.merge()
    assert r["health_copied"] and r["new_trigger_rows"] == 1 and len(r["pending"]) == 1 and not r["errors"], r
    assert len(config.jsonl_read(jd / "watch-triggers.jsonl")) == n_rows + 1
    inbox.consume(r["pending"][0])
    for f in list(ib.iterdir()):
        if f.is_file() and not f.is_symlink():
            f.unlink()


@case
def test_health_push_skips_symlinks():
    """Regression: health.push used `rsync -a`, which recreates outbox symlinks inside air's inbox."""
    res = health.push(dry_run=True)
    for c in res["commands"][1:]:
        assert c[0] == "rsync" and "-a" not in c and "-l" not in c and "--links" not in c, c
    # real local rsync: a symlink in the outbox must not arrive in the destination
    if shutil.which("rsync"):
        src, dst = TMP / "rs-src", TMP / "rs-dst"
        src.mkdir(exist_ok=True)
        dst.mkdir(exist_ok=True)
        (TMP / "rs-secret.json").write_text("{}")
        (src / "fleet-health.json").symlink_to(TMP / "rs-secret.json")
        (src / "watch-triggers-berry.jsonl").write_text("{}\n")
        argv = [a for a in res["commands"][1] if a not in ("-e", " ".join(["ssh"] + health.SSH_OPTS))]
        argv = argv[:-2] + [str(src) + "/", str(dst) + "/"]
        subprocess.run(argv, capture_output=True, text=True, timeout=60)
        assert (dst / "watch-triggers-berry.jsonl").exists() and not (dst / "fleet-health.json").exists()
        assert not (dst / "fleet-health.json").is_symlink()


@case
def test_health_out_restricted():
    """Regression: `health --out PATH` overwrote any writable file (secrets.env, data CSVs) with JSON."""
    victim = TMP / "victim-secrets.env"
    victim.write_text("BYBIT_KEY=x\n")
    for bad in (str(victim), str(TMP / "fleet-health.json"),
                str(Path(os.environ["PHIL_OUTBOX"]) / "other.json"),
                str(Path(os.environ["PHIL_OUTBOX"]) / ".." / "victim-secrets.env")):
        p = subprocess.run([PY, "-m", "trader.fleet.health", "run", "--hosts", "nonexistent", "--out", bad],
                           cwd=str(REPO), capture_output=True, text=True, env=os.environ.copy(), timeout=120)
        assert p.returncode == 2 and json.loads(p.stdout)["refused"], (bad, p.stdout, p.stderr)
    assert victim.read_text() == "BYBIT_KEY=x\n"
    ok = Path(os.environ["PHIL_OUTBOX"]) / "fleet-health.json"
    assert health.allowed_out(str(ok)) == ok.resolve()
    assert health.allowed_out(str(Path(os.environ["TRADER_JOURNAL_DIR"]) / "fleet-health.json"))


# ── setup scripts and units ───────────────────────────────────────────────────
@case
def test_setup_scripts_and_units():
    for s in ("setup_air.sh", "setup_bigsur.sh", "setup_berry.sh"):
        p = subprocess.run(["bash", "-n", str(FLEET / s)], capture_output=True, text=True)
        assert p.returncode == 0, (s, p.stderr)
    # dry runs with a throw-away HOME: print commands, create nothing
    home = TMP / "home"
    home.mkdir()
    env = dict(os.environ, HOME=str(home), PHIL_REPO=str(home / "phil-crypto"))
    for s, extra in (("setup_air.sh", []), ("setup_bigsur.sh", []), ("setup_berry.sh", ["--no-systemd"])):
        p = subprocess.run(["bash", str(FLEET / s), "--dry-run"] + extra, capture_output=True, text=True, env=env,
                           timeout=60)
        assert p.returncode == 0, (s, p.stdout[-800:], p.stderr[-800:])
        assert "+ " in p.stdout and "pip install" in p.stdout, (s, p.stdout)
    assert not any(home.iterdir()), list(home.iterdir())
    pins = (FLEET / "setup_bigsur.sh").read_text()
    assert "numba==0.60.0" in pins and "--only-binary=:all:" in pins
    assert "numba" not in (FLEET / "setup_berry.sh").read_text().split("PINS=")[1].split("\n")[0]
    # launchd templates render to valid plists
    plists = sorted((FLEET / "launchd").glob("*.plist"))
    assert {p.name for p in plists} >= {"com.phil-crypto.tick.plist", "com.phil-crypto.forward.plist",
                                        "com.phil-crypto.deep_retro.plist"}
    for pl in plists:
        txt = pl.read_text()
        for k, v in {"@HOME@": "/Users/x", "@REPO@": "/Users/x/phil-crypto", "@VENV@": "/Users/x/.phil-crypto/venv",
                     "@RUNNER@": "air", "@HOUR@": "2", "@MINUTE@": "10"}.items():
            txt = txt.replace(k, v)
        assert "@" not in txt.replace("Apple//DTD", ""), pl.name
        out = TMP / pl.name
        out.write_text(txt)
        if shutil.which("plutil"):
            p = subprocess.run(["plutil", "-lint", str(out)], capture_output=True, text=True)
            assert p.returncode == 0, (pl.name, p.stdout)
    assert "<integer>900</integer>" in (FLEET / "launchd" / "com.phil-crypto.tick.plist").read_text()
    # systemd units: every service has a timer, oneshot, the venv python and a 15-min/hourly schedule
    sysd = FLEET / "systemd"
    for job in ("watch", "health", "data-update"):
        svc = (sysd / ("phil-%s.service" % job)).read_text()
        tmr = (sysd / ("phil-%s.timer" % job)).read_text()
        assert "[Service]" in svc and "Type=oneshot" in svc and "%h/.phil-crypto/venv/bin/python" in svc
        assert "[Timer]" in tmr and "OnCalendar=" in tmr and "WantedBy=timers.target" in tmr
    assert "*:0/15" in (sysd / "phil-watch.timer").read_text()


def test_py39_syntax():
    """health/sweep_runner/inbox/lease must parse on py3.9 (no match, no runtime X|Y)."""
    import ast
    for f in ("core/lease.py", "fleet/sweep_runner.py", "fleet/health.py", "fleet/inbox.py", "fleet/dispatch.py"):
        src = (config.TRADER_ROOT / f).read_text()
        tree = ast.parse(src, feature_version=(3, 9))
        assert not any(type(n).__name__ == "Match" for n in ast.walk(tree)), f
        assert "from __future__ import annotations" in src, f


case(test_py39_syntax)


if __name__ == "__main__":
    try:
        run_all()
        print("OK test_fleet (%d cases)" % len(PASSED))
    finally:
        shutil.rmtree(str(TMP), ignore_errors=True)
