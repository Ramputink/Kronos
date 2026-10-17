"""Remote sweep runner (CONTRACTS.md section 8). PROTECTED. Runs on bigsur / m2 / locally, Python >= 3.9.

Usage: ``python -m trader.fleet.sweep_runner <job_dir>/job.json [--procs N]``

The job directory (written by ``fleet/dispatch.py``) holds ``job.json`` and ``data/<COIN>_<tf>.csv``: research-view
frames only (index < vault_cutoff, columns open,high,low,close,volume,turnover,funding). The runner refuses to
start if any CSV reaches the cutoff, so a sweep host can never evaluate sealed data even by mistake.

For every (spec, coin) it computes the worker.py row on the research view: the strategy is evaluated once over
``[start + warmup, cutoff)`` and split at ``train_end`` into TRAIN / VAL exactly like
``research/kronos_lab/overnight/worker.evaluate`` (``seg_stats``, ``mdd``, ``entries``, ``expo``, and a coin is skipped
when TRAIN is shorter than 180 days). Rows stream to ``rows.jsonl`` (compact, one line per trial); progress goes to
``state.json``; at the end ``result.json`` (<= fleet.max_result_kb) holds the counts, the Sharpe moments for the
Deflated Sharpe N, and the passing rows plus the best ``top_k`` by ``sh_tv``.

Quotas (job ``quota``): wall clock ``max_minutes``, job-directory size ``max_disk_mb``, summed RSS of the runner
and its workers ``max_ram_mb``, worker count ``max_procs``. A breach terminates the workers, writes
``state.json`` with ``state="killed"`` and exits 3. Other failures exit 1 (``state="failed"``), bad input exits 2.

The evaluation itself is ``trader.core.backtest.evaluate_df`` (same code as research and paper). Tests may inject
another evaluator with ``PHIL_SWEEP_EVALUATOR=module:function`` (signature ``f(spec, df, cost_side) ->
(returns, positions)``, both pd.Series over the warm-up-trimmed view).
"""
from __future__ import annotations

import argparse
import hashlib
import heapq
import importlib
import json
import math
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Optional

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from trader.core import config  # noqa: E402

EXIT_OK, EXIT_FAILED, EXIT_BAD_INPUT, EXIT_QUOTA = 0, 1, 2, 3
CORE_KEYS = ("tf", "mode", "comps", "comb", "k", "adx", "vt", "stop")
DEFAULT_CRITERION = {"cagr_tr": 0.20, "cagr_va": 0.20, "sh_tr": 0.8, "sh_va": 0.8, "entries": 8, "mdd": -0.45}
DEFAULT_QUOTA = {"max_disk_mb": 512, "max_ram_mb": 4096, "max_minutes": 120, "max_procs": 2}


def sanitize_quota(raw) -> dict:
    """DEFAULT_QUOTA overlaid with the job's quota: unknown keys dropped, every value a finite number > 0.

    Raises ValueError otherwise (NaN/inf/negative/zero/bool/str would silently disable the checks below, since
    every comparison with NaN is False). ``max_procs`` is returned as an int >= 1.
    """
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("quota must be an object, got %s" % type(raw).__name__)
    q = dict(DEFAULT_QUOTA)
    for k in DEFAULT_QUOTA:
        if k in raw:
            q[k] = raw[k]
    for k, v in q.items():
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError("quota %s=%r must be a number" % (k, v))
        f = float(v)
        if not math.isfinite(f) or f <= 0:
            raise ValueError("quota %s=%r must be finite and > 0" % (k, v))
        q[k] = f
    if q["max_procs"] != int(q["max_procs"]) or q["max_procs"] < 1:
        raise ValueError("quota max_procs=%r must be an integer >= 1" % q["max_procs"])
    q["max_procs"] = int(q["max_procs"])
    return q
STATE_EVERY_S = 5.0


class QuotaExceeded(Exception):
    pass


# ── helpers shared with dispatch ──────────────────────────────────────────────
def spec_hash(spec: dict) -> str:
    """md5 of the core keys, byte-identical to worker.py cfg_hash (catalog.spec_hash when available)."""
    try:
        from trader.core import catalog  # noqa: WPS433
        return catalog.spec_hash(spec)
    except Exception:
        core = {k: spec[k] for k in CORE_KEYS if k in spec}
        return hashlib.md5(json.dumps(core, sort_keys=True).encode()).hexdigest()


def cost_for(mode: str, costs, cfg: dict) -> float:
    """Per-side cost: None → protected costs[mode] fee+slippage; "legacy" → legacy_research; number → as given."""
    if isinstance(costs, (int, float)) and not isinstance(costs, bool):
        return float(costs)
    if isinstance(costs, dict) and mode in costs:
        return float(costs[mode])
    c = cfg["costs"]
    if costs == "legacy":
        return float(c["legacy_research"]["spot_side" if mode == "SPOT" else "perp_side"])
    k = "spot" if mode == "SPOT" else "perp"
    return float(c[k]["fee_side"]) + float(c[k]["slippage_side"])


def passes(row: dict, crit: dict) -> bool:
    """worker.py is_candidate generalised to a criterion dict."""
    return (row["cagr_tr"] > crit["cagr_tr"] and row["cagr_va"] > crit["cagr_va"]
            and row["sh_tr"] > crit["sh_tr"] and row["sh_va"] > crit["sh_va"]
            and row["entries"] >= crit["entries"] and row["mdd"] > crit["mdd"])


def seg_stats(r: np.ndarray, bpy: int):
    """Verbatim worker.py seg_stats: (cagr, annualised sharpe) or None if < 30 bars."""
    n = len(r)
    if n < 30:
        return None
    g = np.prod(1 + r)
    cagr = g ** (bpy / n) - 1 if g > 0 else -1.0
    sd = r.std()
    return cagr, (r.mean() / sd * math.sqrt(bpy)) if sd > 0 else 0.0


def read_csv_frame(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index.name = "time"
    if "funding" not in df.columns:
        df["funding"] = 0.0
    return df


def dir_size_bytes(path: Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(str(path)):
        for f in files:
            try:
                total += os.lstat(os.path.join(root, f)).st_size
            except OSError:
                pass
    return total


def rss_mb(pids) -> float:
    """Summed resident set size (MB) of `pids` via ps (macOS + Linux)."""
    pids = [str(p) for p in pids if p]
    if not pids:
        return 0.0
    try:
        out = subprocess.run(["ps", "-o", "rss=", "-p", ",".join(pids)], capture_output=True, text=True,
                             timeout=10).stdout
    except Exception:
        return 0.0
    return sum(int(x) for x in out.split() if x.isdigit()) / 1024.0


# ── evaluation (runs inside each worker process) ──────────────────────────────
_W = {}  # per-process state: data frames, evaluator, cfg


def _resolve_evaluator():
    spec = os.environ.get("PHIL_SWEEP_EVALUATOR")
    if spec:
        mod, fn = spec.split(":")
        return getattr(importlib.import_module(mod), fn)

    from trader.core import backtest  # lazy: numba compile only in workers

    def _default(spec_, df, cost_side):
        res = backtest.evaluate_df(spec_, df, period="all", cost_side=cost_side)
        return res["returns"], res["positions"]
    return _default


def _init_worker(job_dir: str, coins, tfs):
    cfg = config.load_config()
    data = {}
    for tf in tfs:
        for coin in coins:
            p = Path(job_dir) / "data" / ("%s_%s.csv" % (coin, tf))
            if p.exists():
                df = read_csv_frame(p)
                df.attrs = {"coin": coin, "tf": tf, "view": "research", "source": "job",
                            "cutoff": cfg["vault"]["vault_cutoff"], "data_end": str(df.index.max())}
                data[(coin, tf)] = df
    _W.clear()
    _W.update(cfg=cfg, data=data, evaluator=_resolve_evaluator(),
              train_end=pd.Timestamp(cfg["vault"]["train_end"]),
              warm=pd.Timedelta(days=int(cfg["vault"]["warmup_days"])))


def _eval_one(spec: dict, coin: str, costs) -> Optional[dict]:
    cfg = _W["cfg"]
    df = _W["data"].get((coin, spec["tf"]))
    if df is None or len(df) == 0:
        return None
    bpy = int(cfg["bars_per_year"][spec["tf"]])
    returns, positions = _W["evaluator"](spec, df, cost_for(spec["mode"], costs, cfg))
    start = df.index[0] + _W["warm"]
    returns = returns[returns.index >= start]
    positions = positions.reindex(returns.index).fillna(0.0)
    idx = returns.index
    r = returns.values.astype(float)
    m_train = idx < _W["train_end"]
    rt, rv = r[m_train], r[~m_train]
    st, sv, se = seg_stats(rt, bpy), seg_stats(rv, bpy), seg_stats(r, bpy)
    if st is None or sv is None or se is None or len(rt) < 180 * bpy / 365:
        return None
    eq = np.cumprod(1 + r)
    mdd = float((eq / np.maximum.accumulate(eq) - 1).min())
    w = positions.values.astype(float)
    entries = int(((w[1:] != 0) & (np.sign(w[1:]) != np.sign(w[:-1]))).sum())
    return {"coin": coin, "cagr_tr": float(st[0]), "sh_tr": float(st[1]), "cagr_va": float(sv[0]),
            "sh_va": float(sv[1]), "sh_tv": float(se[1]), "mdd": mdd, "entries": entries,
            "expo": float(np.abs(w).mean())}


def _eval_spec(task):
    """One spec on every coin → (spec index, rows, n_errors, first error text)."""
    i, spec, coins, costs = task
    rows, errors, first = [], 0, None
    for coin in coins:
        try:
            row = _eval_one(spec, coin, costs)
        except Exception as e:  # one broken (spec, coin) must not kill the sweep
            errors += 1
            first = first or "%s %s: %s" % (spec.get("tf"), coin, e)
            continue
        if row is not None:
            rows.append(row)
    return i, rows, errors, first


# ── driver ────────────────────────────────────────────────────────────────────
def _write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, sort_keys=True, default=str))
    os.replace(str(tmp), str(path))


def _check_research_only(job_dir: Path, cutoff: pd.Timestamp) -> None:
    for p in sorted((job_dir / "data").glob("*.csv")):
        last = None
        with open(p, "rb") as fh:  # cheap tail read: last non-empty line
            fh.seek(0, 2)
            size = fh.tell()
            fh.seek(max(0, size - 4096))
            lines = [ln for ln in fh.read().decode().splitlines() if ln.strip()]
            if lines:
                last = lines[-1].split(",")[0]
        if last and last != "time" and pd.Timestamp(last) >= cutoff:
            raise ValueError("%s reaches %s >= vault_cutoff: sweep hosts may only hold research-view data"
                             % (p.name, last))


def _truncate_rows(result: dict, max_bytes: int) -> dict:
    """Drop the lowest-ranked rows until the serialised result fits max_bytes."""
    rows = result["rows"]
    def size(n):
        return len(json.dumps(dict(result, rows=rows[:n]), sort_keys=True, default=str).encode())
    if size(len(rows)) <= max_bytes:
        return result
    lo, hi = 0, len(rows)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if size(mid) <= max_bytes:
            lo = mid
        else:
            hi = mid - 1
    out = dict(result, rows=rows[:lo])
    out["rows_truncated"] = len(rows) - lo
    return out


def run(job_path, *, procs: Optional[int] = None, evaluator_env: Optional[str] = None) -> dict:
    """Run the job; returns result dict (also written to result.json). Raises QuotaExceeded / ValueError."""
    job_path = Path(job_path)
    job_dir = job_path.parent
    job = json.loads(job_path.read_text())
    cfg = config.load_config()
    if evaluator_env:
        os.environ["PHIL_SWEEP_EVALUATOR"] = evaluator_env
    if job.get("view", "research") != "research":
        raise ValueError("view must be 'research'")
    cutoff = pd.Timestamp(cfg["vault"]["vault_cutoff"])
    _check_research_only(job_dir, cutoff)

    specs, coins = list(job["specs"]), list(job["coins"])
    quota = sanitize_quota(job.get("quota"))  # defence in depth: ValueError -> EXIT_BAD_INPUT
    crit = dict(DEFAULT_CRITERION, **(job.get("criterion") or {}))
    top_k = int(job.get("top_k", 200))
    n_procs = max(1, min(int(procs or quota["max_procs"]), int(quota["max_procs"]), len(specs) or 1))
    tfs = sorted({s["tf"] for s in specs})
    max_bytes = int(cfg["fleet"]["max_result_kb"]) * 1024
    t0 = time.time()
    deadline = t0 + float(quota["max_minutes"]) * 60

    state_path = job_dir / "state.json"
    rows_path = job_dir / "rows.jsonl"
    state = {"job_id": job["job_id"], "state": "running", "pid": os.getpid(), "started": config.iso(),
             "progress": 0.0, "n_specs": len(specs), "done_specs": 0, "procs": n_procs}
    _write_json(state_path, state)

    hashes = [spec_hash(s) for s in specs]
    tasks = [(i, s, coins, job.get("costs")) for i, s in enumerate(specs)]
    moments = {tf: [0, 0.0, 0.0] for tf in cfg["bars_per_year"]}
    n_trials = n_configs = n_pass = errors = 0
    first_error = None
    passing, best = [], []  # best = min-heap of (sh_tv, seq, row) of size top_k
    seq = 0
    last_state = 0.0

    def consume(i, rows, errs, first):
        nonlocal n_trials, n_configs, n_pass, errors, first_error, seq
        n_configs += 1
        errors += errs
        first_error = first_error or first
        spec = specs[i]
        mom = moments.setdefault(spec["tf"], [0, 0.0, 0.0])
        with open(rows_path, "a") as fh:
            for m in rows:
                n_trials += 1
                mom[0] += 1
                mom[1] += m["sh_tv"]
                mom[2] += m["sh_tv"] ** 2
                ok = passes(m, crit)
                n_pass += int(ok)
                row = dict({"i": i, "spec_hash": hashes[i]}, **m)
                row["pass"] = ok
                fh.write(json.dumps(row, separators=(",", ":")) + "\n")
                seq += 1
                if ok:
                    passing.append(row)
                item = (m["sh_tv"], seq, row)
                if len(best) < top_k:
                    heapq.heappush(best, item)
                elif top_k > 0 and item[0] > best[0][0]:
                    heapq.heapreplace(best, item)

    def check_quota(pids):
        if time.time() > deadline:
            raise QuotaExceeded("wall clock > %s min" % quota["max_minutes"])
        disk_mb = dir_size_bytes(job_dir) / 1e6
        if disk_mb > float(quota["max_disk_mb"]):
            raise QuotaExceeded("job dir %.1f MB > %s MB" % (disk_mb, quota["max_disk_mb"]))
        ram = rss_mb([os.getpid()] + list(pids))
        if ram > float(quota["max_ram_mb"]):
            raise QuotaExceeded("RSS %.0f MB > %s MB" % (ram, quota["max_ram_mb"]))

    def tick(pids, force=False):
        nonlocal last_state
        if force or time.time() - last_state > STATE_EVERY_S:
            check_quota(pids)
            state.update(progress=round(n_configs / max(1, len(specs)), 4), done_specs=n_configs,
                         elapsed_s=int(time.time() - t0), n_trials=n_trials, n_pass=n_pass, errors=errors)
            _write_json(state_path, state)
            last_state = time.time()

    try:
        if n_procs == 1:
            _init_worker(str(job_dir), coins, tfs)
            for t in tasks:
                consume(*_eval_spec(t))
                tick([])
        else:
            import multiprocessing as mp
            ctx = mp.get_context("spawn")  # same behaviour on macOS and Linux; numba-safe
            pool = ctx.Pool(n_procs, initializer=_init_worker, initargs=(str(job_dir), coins, tfs))
            try:
                it = pool.imap_unordered(_eval_spec, tasks, chunksize=1)
                seen_pids = set()
                remaining = len(tasks)

                def live_pids():
                    # the pool silently replaces dead workers (e.g. an initializer crash): count them, and
                    # fail instead of respawning until the wall-clock quota
                    ps = [w.pid for w in getattr(pool, "_pool", []) if w.pid]
                    seen_pids.update(ps)
                    if len(seen_pids) > 2 * n_procs + 2:
                        raise RuntimeError("sweep workers keep dying (%d started for %d slots); see runner.log"
                                           % (len(seen_pids), n_procs))
                    return ps

                while remaining:
                    try:
                        out = it.next(timeout=STATE_EVERY_S)
                    except mp.TimeoutError:
                        tick(live_pids(), force=True)
                        continue
                    consume(*out)
                    remaining -= 1
                    tick(live_pids())
                pool.close()
            except BaseException:
                pool.terminate()
                raise
            finally:
                pool.join()
        tick([], force=True)
    except QuotaExceeded as e:
        state.update(state="killed", reason=str(e), elapsed_s=int(time.time() - t0), finished=config.iso())
        _write_json(state_path, state)
        raise

    ranked = sorted(best, key=lambda x: (-x[0], x[1]))
    chosen, seen = [], set()
    for row in sorted(passing, key=lambda r: -r["sh_tv"]) + [x[2] for x in ranked]:
        key = (row["i"], row["coin"])
        if key in seen:
            continue
        seen.add(key)
        chosen.append(row)
    out_rows = []
    for row in chosen:
        r = dict(row)
        r["spec"] = specs[r.pop("i")]
        out_rows.append(r)
    modes = sorted({s["mode"] for s in specs})
    cost_sides = {m: cost_for(m, job.get("costs"), cfg) for m in modes}
    result = {"job_id": job["job_id"], "host": job.get("host_resolved") or config.runner_id(),
              "hypothesis_id": job.get("hypothesis_id"), "hypothesis_hash": job.get("hypothesis_hash"),
              "view": "research", "periods": job.get("periods", ["train", "val"]),
              "cost_side": cost_sides[modes[0]] if len(modes) == 1 else cost_sides,
              "cost_sides": cost_sides, "criterion": crit,
              "n_configs": n_configs, "n_trials": n_trials, "elapsed_s": int(time.time() - t0),
              "sr_moments": {tf: m for tf, m in moments.items() if m[0]}, "n_pass": n_pass,
              "rows": out_rows, "errors": errors, "first_error": first_error, "finished": config.iso()}
    result = _truncate_rows(result, max_bytes)
    _write_json(job_dir / "result.json", result)
    state.update(state="done", progress=1.0, done_specs=n_configs, elapsed_s=result["elapsed_s"],
                 n_trials=n_trials, n_pass=n_pass, errors=errors, finished=result["finished"])
    _write_json(state_path, state)
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m trader.fleet.sweep_runner")
    ap.add_argument("job")
    ap.add_argument("--procs", type=int, default=None)
    a = ap.parse_args(argv)
    job_path = Path(a.job).expanduser()
    try:
        res = run(job_path, procs=a.procs)
    except QuotaExceeded as e:
        print(json.dumps({"ok": False, "state": "killed", "reason": str(e)}))
        return EXIT_QUOTA
    except (ValueError, KeyError, FileNotFoundError) as e:
        _mark_failed(job_path, "bad input: %s" % e)
        print(json.dumps({"ok": False, "state": "failed", "reason": str(e)}))
        return EXIT_BAD_INPUT
    except Exception as e:
        _mark_failed(job_path, "%s\n%s" % (e, traceback.format_exc()[-2000:]))
        print(json.dumps({"ok": False, "state": "failed", "reason": str(e)}))
        return EXIT_FAILED
    print(json.dumps({"ok": True, "state": "done", "n_trials": res["n_trials"], "n_pass": res["n_pass"],
                      "errors": res["errors"]}))
    return EXIT_OK


def _mark_failed(job_path: Path, reason: str) -> None:
    try:
        sp = job_path.parent / "state.json"
        st = json.loads(sp.read_text()) if sp.exists() else {}
        st.update(state="failed", reason=reason[-2000:], finished=config.iso())
        _write_json(sp, st)
    except Exception:
        pass


if __name__ == "__main__":
    sys.exit(main())
