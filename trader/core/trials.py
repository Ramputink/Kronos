"""Trials ledger, sealed-data look budget and hypothesis registry (PROTECTED CORE, trials area, Wave 1).

Interface: trader/CONTRACTS.md section 5. Sole run-time writer of journal/trials.jsonl, journal/looks.jsonl and
journal/hypotheses.jsonl. Python 3.9, numpy + pandas + stdlib.

Honesty rules enforced here:
- every evaluation is a row in trials.jsonl (single / batch / baseline); single+batch rows form the DSR N;
- a look at sealed data (view "full") needs a look_id from request_look(), which refuses when the hypothesis is
  not registered with a verified hash, has not passed TRAIN/VAL, already had a look, or the month's budget
  (vault.looks_per_month) is spent. check_look() is what core/vault.py calls before serving sealed rows.

CLI: python -m trader.core.trials looks | stats | report [--hypothesis H-0007] [--tf 1d]
"""
from __future__ import annotations

import contextlib
import contextvars
import datetime as _dt
import json
import math
import os
import sys
import uuid
from pathlib import Path
from typing import Dict, List, Optional

from trader.core import config

try:
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None

STATUSES = ("registered", "sweeping", "failed_trainval", "passed_trainval", "test_pass", "test_fail", "abandoned")
CALLERS = ("research", "forward", "deep_retro", "sweep", "screener", "test", "operator")
KINDS = ("single", "batch", "baseline")


class LookBudgetExceeded(Exception):
    pass


class HypothesisError(Exception):
    pass


class _FallbackVaultSealed(Exception):
    pass


def _vault_sealed():
    """VaultSealed from core/vault.py (lazy: vault imports trials). Fallback only while vault.py is absent."""
    try:
        from trader.core.vault import VaultSealed
        return VaultSealed
    except Exception:
        return _FallbackVaultSealed


# ----------------------------------------------------------------------------------------------- paths / io
def trials_path() -> Path:
    return config.journal_dir() / "trials.jsonl"


def looks_path() -> Path:
    return config.journal_dir() / "looks.jsonl"


def hypotheses_path() -> Path:
    return config.journal_dir() / "hypotheses.jsonl"


@contextlib.contextmanager
def _lock(name: str):
    """Exclusive lock around read-check-append sequences (look budget), across processes."""
    p = config.journal_dir() / f".{name}.lock"
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as fh:
        if fcntl:
            fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            if fcntl:
                fcntl.flock(fh, fcntl.LOCK_UN)


def _compact_ts(now: _dt.datetime) -> str:
    return now.strftime("%Y%m%dT%H%M%S")


def _parse_iso(s) -> _dt.datetime:
    if isinstance(s, _dt.datetime):
        return s if s.tzinfo else s.replace(tzinfo=_dt.timezone.utc)
    s = str(s).strip()
    if len(s) == 10:
        s += "T00:00:00Z"
    return _dt.datetime.strptime(s.replace("Z", ""), "%Y-%m-%dT%H:%M:%S").replace(tzinfo=_dt.timezone.utc)


def _jsonable(x):
    """Floats -> python float (nan/inf -> None), numpy scalars -> python, dicts/lists recursively."""
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if hasattr(x, "item") and not hasattr(x, "__len__"):
        x = x.item()
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if hasattr(x, "isoformat") and hasattr(x, "year"):
        return config.iso(x)
    return x


# ----------------------------------------------------------------------------------------------- hypotheses
def hypothesis_hash(h: dict) -> str:
    """sha256(canonical_json(h minus {"hash","status"}))."""
    import hashlib
    body = {k: v for k, v in h.items() if k not in ("hash", "status")}
    return hashlib.sha256(config.canonical_json(body).encode()).hexdigest()


def _hyp_file(path_or_id) -> Path:
    p = Path(str(path_or_id)).expanduser()
    if p.suffix == ".json" or p.exists():
        return p
    return config.strategy_dir() / "hypotheses" / f"{path_or_id}.json"


def load_hypothesis(path_or_id) -> dict:
    p = _hyp_file(path_or_id)
    if not p.exists():
        raise HypothesisError(f"hypothesis file not found: {p}")
    try:
        h = json.loads(p.read_text())
    except ValueError as e:
        raise HypothesisError(f"hypothesis {p} is not valid JSON: {e}")
    if not h.get("id"):
        raise HypothesisError(f"hypothesis {p} has no id")
    if h.get("hash") != hypothesis_hash(h):
        raise HypothesisError(f"hypothesis {h.get('id')} hash mismatch (file edited after pre-registration?)")
    return h


def hypothesis_rows(hypothesis_id: str) -> List[dict]:
    return [r for r in config.jsonl_read(hypotheses_path()) if r.get("hypothesis_id") == hypothesis_id]


def hypothesis_status(hypothesis_id: str) -> Optional[dict]:
    """Latest registry row for the hypothesis (None if never registered)."""
    rows = hypothesis_rows(hypothesis_id)
    return rows[-1] if rows else None


def _registered_hash(hypothesis_id: str) -> Optional[str]:
    for r in hypothesis_rows(hypothesis_id):
        if r.get("status") == "registered":
            return r.get("hypothesis_hash")
    return None


def register_hypothesis(path) -> dict:
    """Verify the file hash and append a "registered" row. Idempotent: re-registering the same hash returns the
    existing registered row; a different hash for an already registered id raises HypothesisError."""
    h = load_hypothesis(path)
    hid, hh = h["id"], h["hash"]
    with _lock("hypotheses"):
        prev = _registered_hash(hid)
        if prev is not None:
            if prev != hh:
                raise HypothesisError(f"{hid} already registered with a different hash (hypotheses are immutable)")
            return next(r for r in hypothesis_rows(hid) if r.get("status") == "registered")
        row = {"ts": config.iso(), "hypothesis_id": hid, "hypothesis_hash": hh, "status": "registered",
               "trials": 0, "look_id": None, "summary": str(h.get("title", ""))[:200], "evidence": str(_hyp_file(path))}
        config.jsonl_append(hypotheses_path(), row)
    return row


def set_status(hypothesis_id: str, status: str, **fields) -> dict:
    if status not in STATUSES:
        raise HypothesisError(f"unknown status {status!r}; allowed {STATUSES}")
    hh = _registered_hash(hypothesis_id)
    if hh is None:
        raise HypothesisError(f"{hypothesis_id} is not registered")
    if status == "registered":
        raise HypothesisError("use register_hypothesis() to register")
    row = {"ts": config.iso(), "hypothesis_id": hypothesis_id, "hypothesis_hash": hh, "status": status,
           "trials": int(fields.pop("trials", 0) or 0), "look_id": fields.pop("look_id", None),
           "summary": str(fields.pop("summary", ""))[:500], "evidence": fields.pop("evidence", None)}
    row.update(_jsonable(fields))
    config.jsonl_append(hypotheses_path(), row)
    return row


# ----------------------------------------------------------------------------------------------- trials
def _new_id(prefix: str) -> str:
    return f"{prefix}-{_compact_ts(config.utcnow())}-{uuid.uuid4().hex[:6]}"


def log_trial(result: dict, *, caller, hypothesis_id=None, kind="single", look_id=None) -> str:
    """Append one evaluation row (shape: CONTRACTS.md section 5) and return its trial_id.

    `result` is a backtest.evaluate()/evaluate_df() dict; optional extra keys used if present: "spec",
    "data_end", "bh_metrics" (stored compactly as {"cagr","total","sharpe","maxdd"}).
    """
    if caller not in CALLERS:
        raise ValueError(f"caller {caller!r} not in {CALLERS}")
    if kind not in ("single", "baseline"):
        raise ValueError("log_trial kind must be 'single' or 'baseline' (batches use log_batch)")
    view = result.get("view")
    if view == "full" and not look_id:
        raise _vault_sealed()("a full-view evaluation can only be logged with the look_id that unlocked it")
    hh = _registered_hash(hypothesis_id) if hypothesis_id else None
    data_end = result.get("data_end")
    rets = result.get("returns")
    if data_end is None and rets is not None and len(rets):
        data_end = rets.index.max()
    bh = result.get("bh_metrics")
    trial_id = _new_id("T")
    row = {"trial_id": trial_id, "ts": config.iso(), "kind": kind, "caller": caller, "runner": config.runner_id(),
           "hypothesis_id": hypothesis_id, "hypothesis_hash": hh, "spec_hash": result.get("spec_hash"),
           "spec": result.get("spec"), "coin": result.get("coin"), "tf": result.get("tf"), "mode": result.get("mode"),
           "view": view, "period": result.get("period"), "look_id": look_id, "cost_side": result.get("cost_side"),
           "data_end": config.iso(data_end) if data_end is not None and not isinstance(data_end, str) else data_end,
           "metrics": result.get("metrics"),
           "bh_metrics": ({k: bh.get(k) for k in ("cagr", "total", "sharpe", "maxdd")} if bh else None)}
    config.jsonl_append(trials_path(), _jsonable(row))
    return trial_id


def log_batch(job_result: dict, *, hypothesis_id, caller="sweep") -> str:
    """One row per sweep job (sweep_runner result.json, optionally merged with job.json fields)."""
    if caller not in CALLERS:
        raise ValueError(f"caller {caller!r} not in {CALLERS}")
    view = job_result.get("view", "research")
    if view != "research":
        raise ValueError("sweep batches must be research view")
    job_id = job_result.get("job_id")
    trial_id = _new_id("B")
    costs = job_result.get("cost_side", job_result.get("costs"))
    row = {"trial_id": trial_id, "ts": config.iso(), "kind": "batch", "caller": caller,
           "runner": job_result.get("host") or config.runner_id(), "job_id": job_id,
           "hypothesis_id": hypothesis_id,
           "hypothesis_hash": _registered_hash(hypothesis_id) if hypothesis_id else None,
           "view": view, "periods": job_result.get("periods", ["train", "val"]),
           "cost_side": costs, "n_configs": int(job_result.get("n_configs", 0)),
           "n_trials": int(job_result.get("n_trials", 0)),
           "sr_moments": job_result.get("sr_moments", {}), "n_pass": int(job_result.get("n_pass", 0)),
           "result_file": job_result.get("result_file") or (f"journal/sweeps/{job_id}.json" if job_id else None)}
    config.jsonl_append(trials_path(), _jsonable(row))
    return trial_id


def log_baseline_summary(summary: dict, *, caller, look_id=None) -> str:
    """One compact kind="baseline" row for an aggregate baseline computation (e.g. score.random_percentile).
    Excluded from DSR N like every baseline row."""
    if caller not in CALLERS:
        raise ValueError(f"caller {caller!r} not in {CALLERS}")
    trial_id = _new_id("T")
    row = {"trial_id": trial_id, "ts": config.iso(), "kind": "baseline", "caller": caller,
           "runner": config.runner_id(), "look_id": look_id, "view": summary.get("view"),
           "period": summary.get("period"), "summary": summary, "metrics": None}
    config.jsonl_append(trials_path(), _jsonable(row))
    return trial_id


def read_trials(**filters) -> List[dict]:
    """Filters: kind, caller, hypothesis_id, since (ISO; rows with ts >= since) and any other top-level key
    (equality). A filter value that is a list/tuple/set matches any of its members."""
    since = filters.pop("since", None)
    since_dt = _parse_iso(since) if since else None
    out = []
    for r in config.jsonl_read(trials_path()):
        if since_dt is not None:
            try:
                if _parse_iso(r.get("ts")) < since_dt:
                    continue
            except Exception:
                continue
        ok = True
        for k, v in filters.items():
            if v is None:
                continue
            if isinstance(v, (list, tuple, set)):
                if r.get(k) not in v:
                    ok = False
                    break
            elif r.get(k) != v:
                ok = False
                break
        if ok:
            out.append(r)
    return out


def dsr_context(*, tf=None, since=None) -> dict:
    """{"n_trials": N, "sr_var_daily": v} pooled over single + batch rows (baseline excluded).

    single rows contribute their annualised sharpe; batch rows contribute sr_moments [n, sum, sumsq] of annualised
    Sharpes. Variance (ddof=1) of the pooled annualised Sharpes / 365 = variance of daily-equivalent SR.
    N = number of single rows + batch n_trials (with tf given: that tf's single rows + batch sr_moments[tf][0]).
    """
    n_tr = 0
    m_n, m_s, m_ss = 0.0, 0.0, 0.0
    for r in read_trials(since=since):
        k = r.get("kind")
        if k == "single":
            if tf is not None and r.get("tf") != tf:
                continue
            n_tr += 1
            sh = (r.get("metrics") or {}).get("sharpe")
            if sh is not None:
                m_n += 1
                m_s += sh
                m_ss += sh * sh
        elif k == "batch":
            mom = r.get("sr_moments") or {}
            if tf is not None:
                x = mom.get(tf)
                if not x:
                    continue
                n_tr += int(x[0])
                m_n, m_s, m_ss = m_n + x[0], m_s + x[1], m_ss + x[2]
            else:
                n_tr += int(r.get("n_trials") or sum(int(x[0]) for x in mom.values()))
                for x in mom.values():
                    m_n, m_s, m_ss = m_n + x[0], m_s + x[1], m_ss + x[2]
    var_ann = (m_ss - m_s * m_s / m_n) / (m_n - 1) if m_n > 1 else 0.0
    var_ann = max(var_ann, 0.0)
    return {"n_trials": int(n_tr), "sr_var_daily": var_ann / 365.0, "n_sr": int(m_n), "tf": tf, "since": since}


# ----------------------------------------------------------------------------------------------- looks
def _month(month=None) -> str:
    return month or config.utcnow().strftime("%Y-%m")


def looks_used(month: Optional[str] = None) -> int:
    m = _month(month)
    return sum(1 for r in config.jsonl_read(looks_path()) if r.get("month") == m)


def looks_budget() -> int:
    return int(config.load_config()["vault"]["looks_per_month"])


def request_look(hypothesis_id: str, spec_hashes: List[str], coins: List[str], reason: str, *,
                 caller="research") -> str:
    """Grant one look at sealed data (returns the look_id that vault.load(view="full") requires).

    Requires: hypothesis registered, file hash == registered hash, latest status passed_trainval, no previous look
    for this hypothesis, looks_used(this month) < vault.looks_per_month; otherwise raises (HypothesisError or
    LookBudgetExceeded). Appends a looks.jsonl row.
    """
    if caller not in CALLERS:
        raise ValueError(f"caller {caller!r} not in {CALLERS}")
    if not spec_hashes or not coins:
        raise HypothesisError("a look must name the spec_hashes and coins it will evaluate")
    if not str(reason).strip():
        raise HypothesisError("a look needs a reason")
    reg = _registered_hash(hypothesis_id)
    if reg is None:
        raise HypothesisError(f"{hypothesis_id} is not registered")
    h = load_hypothesis(hypothesis_id)  # verifies the file hash
    if h["hash"] != reg:
        raise HypothesisError(f"{hypothesis_id} file hash differs from the registered hash")
    hc = h.get("coins")
    if hc and not set(coins) <= set(hc):
        raise HypothesisError(f"coins {sorted(set(coins) - set(hc))} are not in the pre-registered hypothesis")
    cfg = config.load_config()["vault"]
    total = int(cfg["looks_per_month"])
    ttl = float(cfg["look_ttl_hours"])
    with _lock("looks"):
        st = hypothesis_status(hypothesis_id)
        if not st or st.get("status") != "passed_trainval":
            raise HypothesisError(f"{hypothesis_id} latest status is {st and st.get('status')!r}, "
                                  "a look needs 'passed_trainval'")
        rows = config.jsonl_read(looks_path())
        if any(r.get("hypothesis_id") == hypothesis_id for r in rows):
            raise LookBudgetExceeded(f"{hypothesis_id} already used its one look at the sealed data")
        now = config.utcnow()
        month = now.strftime("%Y-%m")
        used = sum(1 for r in rows if r.get("month") == month)
        if used >= total:
            raise LookBudgetExceeded(f"look budget for {month} exhausted ({used}/{total})")
        seq = sum(1 for r in rows if str(r.get("look_id", "")).startswith(f"L-{month}-")) + 1
        look_id = f"L-{month}-{seq:04d}"
        row = {"look_id": look_id, "ts": config.iso(now), "month": month, "runner": config.runner_id(),
               "caller": caller, "hypothesis_id": hypothesis_id, "hypothesis_hash": reg,
               "spec_hashes": list(spec_hashes), "coins": list(coins), "reason": str(reason)[:1000],
               "expires": config.iso(now + _dt.timedelta(hours=ttl)), "budget_used": used + 1,
               "budget_total": total}
        config.jsonl_append(looks_path(), row)
    return look_id


_BASELINE_SCOPE: "contextvars.ContextVar" = contextvars.ContextVar("trials_baseline_scope", default=None)


@contextlib.contextmanager
def _baseline_scope(look_id: str, *, caller: str):
    """Private path for score.random_baseline_rate / random_percentile: inside this block (same thread/context),
    check_look(look_id, None) is accepted for coins the look covers, as purpose="baseline". Every baseline row
    the caller computes is logged by score (kind "baseline"); entering the scope is itself not a look."""
    if caller not in CALLERS:
        raise ValueError(f"caller {caller!r} not in {CALLERS}")
    tok = _BASELINE_SCOPE.set(look_id)
    try:
        yield
    finally:
        _BASELINE_SCOPE.reset(tok)


def _coin_from_vault_frame() -> Optional[str]:
    """Interim: recover the coin being loaded from the calling core/vault.py load() frame, for vault versions
    whose _check_look does not forward `coin` yet. Returns None when not called from vault.load."""
    suffix = os.path.join("core", "vault.py")
    f = sys._getframe(1)
    depth = 0
    while f is not None and depth < 12:
        if f.f_code.co_name == "load" and f.f_code.co_filename.endswith(suffix):
            c = f.f_locals.get("coin")
            return str(c) if c is not None else None
        f = f.f_back
        depth += 1
    return None


def check_look(look_id: str, spec_hash: Optional[str], coin: Optional[str] = None, *,
               purpose: Optional[str] = None) -> dict:
    """Return the look row if `look_id` is known, unexpired, covers `coin` (row["coins"]) and covers `spec_hash`
    (row["spec_hashes"]); else raise VaultSealed.

    - `coin` is mandatory in effect: when None it is recovered from the calling vault.load frame; if it still
      cannot be determined the look is refused (a look never unlocks "any coin").
    - `spec_hash=None` is accepted only for purpose="baseline" (or inside trials._baseline_scope(look_id), the
      private path score uses for the random baseline), and still only for coins the look covers.
    """
    VS = _vault_sealed()
    if not look_id:
        raise VS("no look_id: sealed data needs a granted look")
    row = next((r for r in config.jsonl_read(looks_path()) if r.get("look_id") == look_id), None)
    if row is None:
        raise VS(f"unknown look {look_id}")
    if config.utcnow() > _parse_iso(row["expires"]):
        raise VS(f"look {look_id} expired at {row['expires']}")
    if coin is None:
        coin = _coin_from_vault_frame()
    if coin is None:
        raise VS(f"look {look_id}: the coin being loaded must be given (check_look(look_id, spec_hash, coin))")
    if str(coin) not in [str(c) for c in (row.get("coins") or [])]:
        raise VS(f"look {look_id} does not cover coin {coin} (covers {row.get('coins')})")
    if spec_hash is None:
        in_scope = _BASELINE_SCOPE.get() == look_id
        if purpose != "baseline" and not in_scope:
            raise VS(f"look {look_id}: spec_hash=None is only allowed for the logged random baseline")
    elif spec_hash not in (row.get("spec_hashes") or []):
        raise VS(f"look {look_id} does not cover spec {spec_hash}")
    return row


def look_coins(look_id: str) -> List[str]:
    """Coins a granted look covers ([] if unknown)."""
    row = next((r for r in config.jsonl_read(looks_path()) if r.get("look_id") == look_id), None)
    return list(row.get("coins") or []) if row else []


# ----------------------------------------------------------------------------------------------- report
def report(*, hypothesis_id=None, tf=None, since=None, criterion: Optional[dict] = None) -> dict:
    """Honesty summary of the ledger: counts per kind/caller, DSR context (global N), the best single trial
    (by annualised Sharpe, optionally within one hypothesis) with its DSR, and the beats-random rate:
    share of single trials meeting `criterion` vs share of baseline trials meeting it (same view/period
    as the best trial when baseline rows for it exist)."""
    from trader.core import score

    crit = dict(score.DEFAULT_CRITERION if criterion is None else criterion)
    rows = read_trials(since=since)
    counts: Dict[str, int] = {}
    callers: Dict[str, int] = {}
    for r in rows:
        counts[r.get("kind")] = counts.get(r.get("kind"), 0) + 1
        callers[r.get("caller")] = callers.get(r.get("caller"), 0) + 1
    ctx = dsr_context(tf=tf, since=since)
    singles = [r for r in rows if r.get("kind") == "single" and (r.get("metrics") or {}).get("sharpe") is not None
               and (tf is None or r.get("tf") == tf) and (hypothesis_id is None or r.get("hypothesis_id") == hypothesis_id)]
    best = max(singles, key=lambda r: r["metrics"]["sharpe"]) if singles else None
    best_out = None
    if best:
        best_out = {k: best.get(k) for k in ("trial_id", "spec_hash", "coin", "tf", "mode", "view", "period",
                                              "hypothesis_id")}
        best_out["metrics"] = best["metrics"]
        best_out["dsr"] = score.dsr_of_metrics(best["metrics"], n_trials=ctx["n_trials"],
                                               sr_var_daily=ctx["sr_var_daily"])
    base = [r for r in rows if r.get("kind") == "baseline" and r.get("metrics")]
    if best is not None:
        same = [r for r in base if r.get("view") == best.get("view") and r.get("period") == best.get("period")]
        base = same or base
        singles_cmp = [r for r in singles if r.get("view") == best.get("view") and r.get("period") == best.get("period")]
    else:
        singles_cmp = singles
    rs = (sum(score.passes(r["metrics"], r.get("bh_metrics"), crit) for r in singles_cmp) / len(singles_cmp)
          if singles_cmp else None)
    rr = sum(score.passes(r["metrics"], r.get("bh_metrics"), crit) for r in base) / len(base) if base else None
    look_m = _month()
    return {"n_rows": len(rows), "counts": counts, "callers": callers, "dsr_context": ctx, "best": best_out,
            "beats_random": {"criterion": crit, "rate_selected": rs, "n_selected": len(singles_cmp),
                             "rate_random": rr, "n_random": len(base),
                             "beats": (rs is not None and rr is not None and rs > rr)},
            "looks": {"month": look_m, "used": looks_used(look_m), "total": looks_budget()}}


# ----------------------------------------------------------------------------------------------- CLI
def _main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    cmd = argv[0] if argv else "stats"

    def opt(name):
        return argv[argv.index(name) + 1] if name in argv and argv.index(name) + 1 < len(argv) else None

    try:
        if cmd == "looks":
            m = _month(opt("--month"))
            used, total = looks_used(m), looks_budget()
            out = {"month": m, "used": used, "total": total, "remaining": max(total - used, 0)}
        elif cmd == "stats":
            rows = read_trials()
            counts, callers = {}, {}
            for r in rows:
                counts[r.get("kind")] = counts.get(r.get("kind"), 0) + 1
                callers[r.get("caller")] = callers.get(r.get("caller"), 0) + 1
            out = {"counts": counts, "callers": callers, "dsr_context": dsr_context(tf=opt("--tf"))}
        elif cmd == "report":
            out = report(hypothesis_id=opt("--hypothesis"), tf=opt("--tf"), since=opt("--since"))
        else:
            print(json.dumps({"error": f"unknown command {cmd!r}", "usage": "looks | stats | report"}))
            return 2
    except Exception as e:
        print(json.dumps({"error": f"{e.__class__.__name__}: {e}"}))
        return 1
    print(json.dumps(_jsonable(out), default=str))
    return 0


if __name__ == "__main__":
    sys.exit(_main())
