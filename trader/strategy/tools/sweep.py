"""Research sweep tool (research-loop area, Wave 2). Interface: trader/CONTRACTS.md section 9.

The headless research cycle (trader/CYCLE_RESEARCH.md) drives a pre-registered hypothesis through this tool:

    prereg  H-0007            fill the hash of strategy/hypotheses/H-0007.json and register it (trials registry)
    expand  H-0007            grid -> specs, validated, counted (read-only, no ledger rows)
    run     H-0007            sweep TRAIN/VAL on the research view (in-process pool or fleet/dispatch to bigsur/m2),
                              batch row in trials.jsonl, random-config baseline, pre-registered gate,
                              status passed_trainval | failed_trainval (+ the specs/coins fixed for the test)
    open    H-0007 --reason   ONE look at the sealed TEST period for the specs fixed by `run`: buy&hold,
                              random-config baseline, Deflated Sharpe with the global N -> test_pass | test_fail
    status  H-0007            registry rows + looks budget

Honesty rules implemented here (never bypass them, they are the point of the system):
- only research-view data (< vault_cutoff) is swept; dispatch ships research CSVs only;
- every evaluation reaches the trials ledger (batch row per sweep job, single rows for re-evaluations, baseline
  rows for random configs) BEFORE its numbers are returned;
- the trainval gate and the test gate are the ones written in the hypothesis BEFORE the sweep (hash-locked);
- what goes to the sealed test (specs + coins) is decided and recorded before the look is requested;
- a hypothesis is swept once and looks at sealed data once; a new idea is a new hypothesis id.

Runs on air (the brain) with Python >= 3.9. numpy + pandas + the trader core only.
CLI prints one JSON object; exit 0 ok, 1 error, 2 refused.
"""
from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from trader.core import config  # noqa: E402

HID_RE = re.compile(r"^H-[0-9]{4}$")
REQUIRED = ("id", "title", "rationale", "family", "base_spec", "grid", "coins", "trainval_criterion",
            "test_criterion", "max_trials")
SWEEP_CRIT_KEYS = ("cagr_tr", "cagr_va", "sh_tr", "sh_va", "entries", "mdd")
DEFAULT_TV = {"min_coins_passing": 1, "beat_random": True}
DEFAULT_TEST = {"portfolio_cagr_min": 0.0, "beat_bh": True, "beat_random_rate": True}
DEFAULT_SELECTION = {"top_specs": 1, "coins": "passing"}
INPROC_MAX_TRIALS = 400          # host "auto": small sweeps run in-process on this machine
DEFAULT_N_RANDOM_TV = 200        # random configs for the TRAIN/VAL baseline (morning_test style)
MAX_JOBS_PER_HYPOTHESIS = 3      # failed/killed jobs may be resubmitted; after that the hypothesis is abandoned


class SweepError(Exception):
    """Bad hypothesis / tool failure."""


class SweepRefused(SweepError):
    """The honesty rules refuse the request (already swept, budget spent, wrong status...)."""


# ----------------------------------------------------------------------------------------------- helpers
def _cfg() -> dict:
    return config.load_config()


def _hyp_path(hid: str) -> Path:
    return config.strategy_dir() / "hypotheses" / ("%s.json" % hid)


def _jsonable(x):
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


def _r(x, nd=4):
    return None if x is None or (isinstance(x, float) and not math.isfinite(x)) else round(float(x), nd)


def _set_path(obj, path: str, value) -> None:
    """Set a dotted path ("comps.0.p.up") inside nested dicts/lists. Integer parts index lists."""
    parts = path.split(".")
    cur = obj
    for i, p in enumerate(parts):
        last = i == len(parts) - 1
        if isinstance(cur, list):
            if not p.isdigit() or int(p) >= len(cur):
                raise SweepError("grid key %r: %r is not a valid index of a %d-item list" % (path, p, len(cur)))
            if last:
                cur[int(p)] = copy.deepcopy(value)
            else:
                cur = cur[int(p)]
        elif isinstance(cur, dict):
            if last:
                cur[p] = copy.deepcopy(value)
            else:
                if p not in cur:
                    raise SweepError("grid key %r: %r does not exist in base_spec" % (path, p))
                cur = cur[p]
        else:
            raise SweepError("grid key %r walks into a scalar at %r" % (path, p))


def _trial_cap(h: dict, cfg: dict) -> int:
    cap = int(cfg["research"]["max_trials_per_hypothesis"])
    mt = h.get("max_trials")
    return min(cap, int(mt)) if mt else cap


# ----------------------------------------------------------------------------------------------- expand
def expand(h: dict) -> List[dict]:
    """Grid -> specs: cartesian product of the grid's dotted-path value lists applied to base_spec, in grid key
    order. With comb "and" and no explicit "k" in the grid, k follows len(comps). Duplicate specs (same
    spec_hash) are dropped. Raises SweepError if a spec is invalid (catalog.validate_spec), a coin is outside
    the universe, or len(specs) x len(coins) exceeds min(max_trials, research.max_trials_per_hypothesis)."""
    from trader.core import catalog

    cfg = _cfg()
    base = h.get("base_spec")
    if not isinstance(base, dict):
        raise SweepError("hypothesis has no base_spec object")
    grid = h.get("grid") or {}
    if not isinstance(grid, dict):
        raise SweepError("grid must be an object {dotted.path: [values]}")
    keys = list(grid.keys())
    for k in keys:
        if not isinstance(grid[k], list) or not grid[k]:
            raise SweepError("grid[%r] must be a non-empty list" % k)
    specs, seen = [], set()
    for combo in itertools.product(*[grid[k] for k in keys]) if keys else [()]:
        s = copy.deepcopy(base)
        for k, v in zip(keys, combo):
            _set_path(s, k, v)
        if s.get("comb") == "and" and "k" not in grid and isinstance(s.get("comps"), list):
            s["k"] = len(s["comps"])
        probs = catalog.validate_spec(s)
        if probs:
            raise SweepError("grid point %s gives an invalid spec: %s" % (dict(zip(keys, combo)), "; ".join(probs)))
        sh = catalog.spec_hash(s)
        if sh in seen:
            continue
        seen.add(sh)
        specs.append(catalog.core_spec(s))
    coins = h.get("coins") or []
    if not coins:
        raise SweepError("hypothesis lists no coins")
    bad = [c for c in coins if c not in cfg["universe"]]
    if bad:
        raise SweepError("coins outside the universe: %s" % bad)
    n, cap = len(specs) * len(coins), _trial_cap(h, cfg)
    if n > cap:
        raise SweepError("%d specs x %d coins = %d trials > cap %d (min(max_trials, research.max_trials_per_"
                         "hypothesis)); narrow the grid" % (len(specs), len(coins), n, cap))
    return specs


# ----------------------------------------------------------------------------------------------- pre-registration
def next_id() -> str:
    """Next free H-NNNN (files in strategy/hypotheses and ids in the registry)."""
    from trader.core import trials

    used = set()
    hd = config.strategy_dir() / "hypotheses"
    if hd.exists():
        for f in hd.glob("H-*.json"):
            m = re.match(r"^H-([0-9]{4})", f.name)
            if m:
                used.add(int(m.group(1)))
    for r in config.jsonl_read(trials.hypotheses_path()):
        m = re.match(r"^H-([0-9]{4})$", str(r.get("hypothesis_id", "")))
        if m:
            used.add(int(m.group(1)))
    return "H-%04d" % ((max(used) if used else 0) + 1)


def prereg(path_or_id) -> dict:
    """Pre-register a hypothesis file written by the cycle (strategy/hypotheses/H-NNNN.json).

    Completes `created`/`author` if absent, validates the fields and the grid (expand), writes the sha256 `hash`
    and appends the "registered" row. Idempotent for an unchanged registered file. A file whose id is already
    registered with a different hash is refused (hypotheses are immutable: a change is a new id)."""
    from trader.core import trials

    p = Path(str(path_or_id)).expanduser()
    if not (p.suffix == ".json" or p.exists()):
        p = _hyp_path(str(path_or_id))
    if not p.exists():
        raise SweepError("hypothesis file not found: %s" % p)
    try:
        h = json.loads(p.read_text())
    except ValueError as e:
        raise SweepError("hypothesis %s is not valid JSON: %s" % (p, e))
    if not isinstance(h, dict):
        raise SweepError("hypothesis must be a JSON object")
    hid = h.get("id")
    if not isinstance(hid, str) or not HID_RE.match(hid):
        raise SweepError("id must look like H-0007, got %r" % hid)
    if p.parent.name == "hypotheses" and p.stem != hid:
        raise SweepError("file name %s does not match id %s" % (p.name, hid))
    missing = [k for k in REQUIRED if k not in h]
    if missing:
        raise SweepError("hypothesis %s misses fields %s" % (hid, missing))
    if len(str(h.get("rationale", "")).strip()) < 40:
        raise SweepError("rationale too short: say why, and cite playbook/trial evidence")
    for k in ("trainval_criterion", "test_criterion"):
        if not isinstance(h[k], dict):
            raise SweepError("%s must be an object" % k)
    prev = trials._registered_hash(hid)
    if h.get("hash") and h["hash"] == trials.hypothesis_hash(h):
        if prev is not None and prev != h["hash"]:
            raise SweepRefused("%s is registered with a different hash (immutable); use a new id" % hid)
    else:
        if prev is not None:
            raise SweepRefused("%s is already registered and its file changed since (hash mismatch); hypotheses "
                               "are immutable: restore it or write a new id (%s)" % (hid, next_id()))
        h.setdefault("created", config.iso())
        h.setdefault("author", "research-cycle@%s" % config.runner_id())
    specs = expand(h)  # validates grid + cap before anything is registered
    h["hash"] = trials.hypothesis_hash(h)
    target = _hyp_path(hid) if p.parent.name != "hypotheses" else p
    text = json.dumps(h, indent=1, sort_keys=False) + "\n"
    if not target.exists() or target.read_text() != text:
        if target.exists() and target != p:
            raise SweepRefused("%s already exists" % target)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(".tmp")
        tmp.write_text(text)
        os.replace(str(tmp), str(target))
    row = trials.register_hypothesis(target)
    return {"hypothesis_id": hid, "hash": h["hash"], "file": str(target), "n_specs": len(specs),
            "n_trials_planned": len(specs) * len(h["coins"]), "registered": row}


# ----------------------------------------------------------------------------------------------- run (TRAIN/VAL)
def _sweep_criterion(tv: dict) -> dict:
    from trader.fleet import sweep_runner
    crit = dict(sweep_runner.DEFAULT_CRITERION)
    for k in SWEEP_CRIT_KEYS:
        if k in tv:
            crit[k] = tv[k]
    return crit


def _schedule() -> dict:
    p = config.strategy_dir() / "schedule.json"
    try:
        return json.loads(p.read_text()) if p.exists() else {}
    except ValueError:
        return {}


def sweeps_last_24h() -> int:
    """Distinct sweep jobs whose batch rows were logged in the last 24 h."""
    from trader.core import trials
    import datetime as _dt
    since = config.iso(config.utcnow() - _dt.timedelta(hours=24))
    return len({r.get("job_id") for r in trials.read_trials(kind="batch", since=since) if r.get("job_id")})


def _check_daily_budget() -> None:
    mx = ((_schedule().get("research") or {}).get("max_sweeps_per_day"))
    if mx is not None and sweeps_last_24h() >= int(mx):
        raise SweepRefused("max_sweeps_per_day (%s, strategy/schedule.json) reached in the last 24 h" % mx)


def _jobs_of(hid: str) -> List[str]:
    from trader.core import trials
    out = []
    for r in trials.hypothesis_rows(hid):
        j = r.get("job_id")
        if r.get("status") == "sweeping" and j and j not in out:
            out.append(j)
    return out


def _make_job(h: dict, specs: List[dict], host: str, job_id: str, procs: Optional[int]) -> dict:
    job = {"job_id": job_id, "hypothesis_id": h["id"], "hypothesis_hash": h["hash"], "created": config.iso(),
           "host": host, "specs": specs, "coins": list(h["coins"]), "view": "research",
           "periods": ["train", "val"], "costs": None,
           "criterion": _sweep_criterion(h.get("trainval_criterion") or {}), "top_k": int(h.get("top_k", 200))}
    if procs:
        job["quota"] = {"max_procs": int(procs)}
    return job


def _wait_remote(job_id: str, deadline: float, poll_s: float) -> dict:
    from trader.fleet import dispatch
    while True:
        st = dispatch.status(job_id)
        if st["state"] in ("done", "failed", "killed"):
            return st
        if time.time() >= deadline:
            return st
        time.sleep(poll_s)


def _random_trainval_rate(h: dict, specs: List[dict], crit: dict, n_random: int, seed: int) -> dict:
    """Share of random configs (catalog.rand_cfg, same modes/tfs as the hypothesis, random coin of the
    hypothesis) passing the SAME sweep criterion on the research view (morning_test logic on TRAIN/VAL).
    Each random evaluation is logged kind "baseline" (excluded from the DSR N)."""
    from trader.core import backtest, catalog, trials, vault
    from trader.fleet import sweep_runner

    modes = tuple(sorted({s["mode"] for s in specs}))
    tfs = tuple(sorted({s["tf"] for s in specs}))
    coins = list(h["coins"])
    rng = random.Random(seed)
    cache: Dict[Tuple[str, str], object] = {}
    n = n_pass = skipped = 0
    for _ in range(int(n_random)):
        spec = catalog.rand_cfg(rng, modes=modes, tfs=tfs)
        coin = rng.choice(coins)
        key = (coin, spec["tf"])
        try:
            if key not in cache:
                cache[key] = vault.load(coin, spec["tf"], "research")
            df = cache[key]
            cs = backtest.cost_side(spec["mode"])
            wm = backtest.worker_metrics(spec, df, cost_side=cs)
            if wm is None:
                skipped += 1
                continue
            res = dict(backtest.evaluate_df(spec, df, "trainval", cost_side=cs))
        except FileNotFoundError:
            skipped += 1
            continue
        except Exception as e:  # one broken random config is not a trial; sealed access is never swallowed
            if e.__class__.__name__ == "VaultSealed":
                raise
            skipped += 1
            continue
        res.update(coin=coin, view="research", period="trainval")
        trials.log_trial(res, caller="research", kind="baseline")
        n += 1
        n_pass += int(sweep_runner.passes(wm, crit))
    return {"rate_random": (n_pass / n) if n else None, "n_random": n, "n_pass_random": n_pass,
            "n_skipped": skipped, "seed": seed, "modes": list(modes), "tfs": list(tfs)}


def _aggregate(result: dict, specs: List[dict], h: dict) -> List[dict]:
    """Per spec: coins passing the sweep criterion, median sh_tv of the passing coins, best row values."""
    from trader.core import catalog
    by = {}
    for row in result.get("rows") or []:
        sh = row.get("spec_hash")
        e = by.setdefault(sh, {"spec_hash": sh, "spec": row.get("spec"), "coins_pass": [], "sh_tv_pass": [],
                               "rows": 0, "best_sh_tv": None})
        e["rows"] += 1
        if row.get("pass"):
            e["coins_pass"].append(row["coin"])
            e["sh_tv_pass"].append(float(row["sh_tv"]))
        if e["best_sh_tv"] is None or row["sh_tv"] > e["best_sh_tv"]:
            e["best_sh_tv"] = float(row["sh_tv"])
    known = {catalog.spec_hash(s): s for s in specs}
    out = []
    for sh, e in by.items():
        if sh not in known:  # a row that is not one of the pre-registered specs is never selectable
            continue
        vals = sorted(e.pop("sh_tv_pass"))
        e["n_coins_pass"] = len(e["coins_pass"])
        e["median_sh_tv_pass"] = vals[len(vals) // 2] if vals else None
        e["spec"] = known[sh]
        out.append(e)
    out.sort(key=lambda e: (-e["n_coins_pass"], -(e["median_sh_tv_pass"] or -1e9), -(e["best_sh_tv"] or -1e9),
                            e["spec_hash"]))
    return out


def _portfolio_dsr(results: List[dict], bpy: int, ctx: dict) -> dict:
    from trader.core import score
    pr = score.portfolio_returns(results)
    m = score.metrics(pr.values, bpy) if len(pr) else None
    dsr = score.dsr_of_metrics(m, n_trials=ctx["n_trials"], sr_var_daily=ctx["sr_var_daily"]) if m else None
    return {"metrics": m, "dsr": dsr, "n_trials": ctx["n_trials"], "sr_var_daily": ctx["sr_var_daily"],
            "expected_max_sr_daily": score.expected_max_sr(ctx["n_trials"], ctx["sr_var_daily"])}


def _write_summary(hid: str, stage: str, obj: dict) -> Optional[str]:
    """Compact JSON summary for the retro: journal/retros/research/<H>-<stage>.json (agent-writable area)."""
    d = config.journal_dir() / "retros" / "research"
    d.mkdir(parents=True, exist_ok=True)
    p = d / ("%s-%s.json" % (hid, stage))
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(_jsonable(obj), indent=1, sort_keys=True, default=str) + "\n")
    os.replace(str(tmp), str(p))
    rel = os.path.relpath(str(p), str(config.TRADER_ROOT))
    return str(p) if rel.startswith("..") else rel


def run(hypothesis_id: str, *, host: str = "auto", wait: bool = True, procs: Optional[int] = None,
        timeout_min: float = 40.0, poll_s: float = 15.0, n_random: Optional[int] = None, seed: int = 1,
        write_summary: bool = True) -> dict:
    """Sweep a pre-registered hypothesis on TRAIN/VAL (research view) and apply its pre-registered gate.

    trials.register_hypothesis -> status "sweeping" (job_id) -> dispatch (in-process pool for host "inproc", or
    fleet/dispatch submit+collect for local/bigsur/m2/auto) -> batch row -> random-config baseline on the same
    criterion -> gate -> set_status passed_trainval | failed_trainval with the specs/coins fixed for the test.

    Resumable: if the latest status is "sweeping" with a job_id, that job is awaited/collected instead of a new
    one. With wait=False (remote hosts) returns {"state": "running", "job_id"} right after submission.
    Returns {"hypothesis_id","n_trials","passed","best":[...],"batch_trial_id", ...}.
    """
    from trader.core import trials
    from trader.fleet import dispatch

    h = trials.load_hypothesis(hypothesis_id)  # verifies the hash
    trials.register_hypothesis(hypothesis_id)  # idempotent; refuses a hash that differs from the registered one
    hid = h["id"]
    specs = expand(h)
    st = trials.hypothesis_status(hid) or {}
    status = st.get("status")
    if status not in ("registered", "sweeping"):
        raise SweepRefused("%s latest status is %r: a hypothesis is swept once (write a new hypothesis id)"
                           % (hid, status))
    n_planned = len(specs) * len(h["coins"])
    deadline = time.time() + float(timeout_min) * 60

    # ---- resume a job that an earlier cycle started
    result, job_id = None, None
    if status == "sweeping" and st.get("job_id"):
        job_id = st["job_id"]
        try:
            ent = dispatch._reg_read(job_id)
        except dispatch.DispatchError:
            ent = None
        if ent is not None and ent.get("local_result"):
            result = dispatch.collect(job_id)
        elif ent is not None and not ent.get("dry_run"):
            stt = _wait_remote(job_id, deadline, poll_s) if wait else dispatch.status(job_id)
            if stt["state"] == "done":
                result = dispatch.collect(job_id)
            elif stt["state"] in ("queued", "running"):
                return {"hypothesis_id": hid, "state": stt["state"], "job_id": job_id, "host": ent.get("host"),
                        "progress": stt.get("progress"), "passed": None,
                        "note": "sweep still running; run again next cycle to collect"}
            # failed / killed: its planned trials are already counted (provisional row); resubmit below
        if result is None:
            job_id = None

    # ---- submit a new job
    if result is None:
        if len(_jobs_of(hid)) >= MAX_JOBS_PER_HYPOTHESIS:
            trials.set_status(hid, "abandoned", summary="sweep jobs failed %d times" % MAX_JOBS_PER_HYPOTHESIS,
                              evidence=",".join(_jobs_of(hid)))
            raise SweepRefused("%s: %d sweep jobs failed; hypothesis abandoned" % (hid, MAX_JOBS_PER_HYPOTHESIS))
        _check_daily_budget()
        job_id = dispatch.new_job_id()
        eff = host
        if eff == "auto" and n_planned <= INPROC_MAX_TRIALS:
            eff = "inproc"
        n_procs = procs or max(1, min(4, (os.cpu_count() or 2) - 1))
        job = _make_job(h, specs, eff, job_id, n_procs if eff == "inproc" else procs)
        trials.set_status(hid, "sweeping", job_id=job_id, host=eff, trials=n_planned,
                          summary="sweep %d specs x %d coins" % (len(specs), len(h["coins"])))
        if eff == "inproc":
            job["host"] = "local"
            result = dispatch.run_local(job, procs=n_procs)
        else:
            try:
                rec = dispatch.submit_plan(job)
            except dispatch.DispatchError as e:
                if host == "auto" and "no sweep host reachable" in str(e):
                    job["host"] = "local"
                    result = dispatch.run_local(job, procs=n_procs)
                    eff = "inproc(fallback)"
                else:
                    raise
            else:
                if not wait:
                    return {"hypothesis_id": hid, "state": "running", "job_id": job_id, "host": rec["host"],
                            "passed": None, "n_trials_planned": n_planned}
                stt = _wait_remote(job_id, deadline, poll_s)
                if stt["state"] != "done":
                    return {"hypothesis_id": hid, "state": stt["state"], "job_id": job_id, "host": rec["host"],
                            "passed": None, "progress": stt.get("progress"),
                            "note": "not finished within %s min; run again next cycle" % timeout_min
                            if stt["state"] in ("queued", "running") else stt.get("reason")}
                result = dispatch.collect(job_id)
    return _finish_trainval(h, specs, result, job_id, n_random=n_random, seed=seed, write_summary=write_summary)


def _finish_trainval(h: dict, specs: List[dict], result: dict, job_id: str, *, n_random, seed,
                     write_summary) -> dict:
    from trader.core import backtest, trials

    hid = h["id"]
    tv = dict(DEFAULT_TV, **(h.get("trainval_criterion") or {}))
    sel_cfg = dict(DEFAULT_SELECTION, **(h.get("test_selection") or {}))
    crit = _sweep_criterion(tv)
    n_trials = int(result.get("n_trials") or 0)
    n_pass = int(result.get("n_pass") or 0)
    agg = _aggregate(result, specs, h)
    passing = [e for e in agg if e["n_coins_pass"] >= int(tv["min_coins_passing"])]
    grid_rate = (n_pass / n_trials) if n_trials else 0.0

    # random-config baseline on the same criterion and period (logged as baseline rows)
    nr = DEFAULT_N_RANDOM_TV if n_random is None else int(n_random)
    rnd = _random_trainval_rate(h, specs, crit, nr, seed) if nr > 0 else {"rate_random": None, "n_random": 0}
    reasons = []
    if n_trials == 0:
        reasons.append("no trial produced a row (data too short for every coin?)")
    if not passing:
        reasons.append("no spec passes on >= %s coins" % tv["min_coins_passing"])
    if tv.get("beat_random", True):
        rr = rnd.get("rate_random")
        if rr is None:
            reasons.append("random baseline unavailable (beat_random required)")
        elif not grid_rate > rr:
            reasons.append("grid pass rate %.4f <= random pass rate %.4f" % (grid_rate, rr))

    # specs + coins fixed for the sealed test BEFORE any look (pre-registered selection rule)
    selected = []
    for e in passing[:max(1, int(sel_cfg["top_specs"]))]:
        coins = list(e["coins_pass"]) if sel_cfg["coins"] == "passing" else list(h["coins"])
        coins = [c for c in h["coins"] if c in coins]  # hypothesis order
        selected.append({"spec_hash": e["spec_hash"], "spec": e["spec"], "coins": coins})

    # trainval re-evaluation of the selection (logged single rows): DSR with the global N, buy&hold
    tv_eval = None
    if selected and not reasons:
        res_all, bh_beat, pairs = [], 0, []
        for s in selected:
            for c in s["coins"]:
                r = backtest.evaluate(s["spec"], c, view="research", period="trainval", caller="research",
                                      hypothesis_id=hid)
                res_all.append(r)
                m, bh = r.get("metrics") or {}, r.get("bh_metrics") or {}
                beat = m.get("cagr") is not None and bh.get("cagr") is not None and m["cagr"] > bh["cagr"]
                bh_beat += int(beat)
                pairs.append({"spec_hash": s["spec_hash"], "coin": c, "trial_id": r.get("trial_id"),
                              "cagr": _r(m.get("cagr")), "sharpe": _r(m.get("sharpe")), "maxdd": _r(m.get("maxdd")),
                              "bh_cagr": _r(bh.get("cagr")), "beats_bh": bool(beat)})
        bpy = int(_cfg()["bars_per_year"][selected[0]["spec"]["tf"]])
        pdsr = _portfolio_dsr(res_all, bpy, trials.dsr_context())
        tv_eval = {"pairs": pairs, "n_beat_bh": bh_beat, "portfolio": {
            "cagr": _r((pdsr["metrics"] or {}).get("cagr")), "sharpe": _r((pdsr["metrics"] or {}).get("sharpe")),
            "maxdd": _r((pdsr["metrics"] or {}).get("maxdd")), "dsr": _r(pdsr["dsr"]),
            "dsr_n_trials": pdsr["n_trials"]}}
        if "dsr_min" in tv and (pdsr["dsr"] is None or pdsr["dsr"] < float(tv["dsr_min"])):
            reasons.append("trainval portfolio DSR %s < pre-registered %s" % (_r(pdsr["dsr"]), tv["dsr_min"]))

    passed = not reasons
    best = [{"spec_hash": e["spec_hash"], "spec": e["spec"], "n_coins_pass": e["n_coins_pass"],
             "coins_pass": e["coins_pass"], "median_sh_tv_pass": _r(e["median_sh_tv_pass"]),
             "best_sh_tv": _r(e["best_sh_tv"])} for e in agg[:5]]
    out = {"hypothesis_id": hid, "hypothesis_hash": h["hash"], "stage": "trainval", "job_id": job_id,
           "host": result.get("host"), "batch_trial_id": result.get("batch_trial_id"),
           "result_file": result.get("result_file"), "n_configs": result.get("n_configs"), "n_trials": n_trials,
           "n_pass": n_pass, "grid_pass_rate": _r(grid_rate, 6), "random": rnd, "criterion": crit,
           "trainval_criterion": tv, "n_specs_passing": len(passing), "best": best,
           "selected": selected if passed else [], "trainval_eval": tv_eval, "passed": passed,
           "reasons": reasons, "rows_truncated": result.get("rows_truncated", 0), "errors": result.get("errors"),
           "dsr_context": trials.dsr_context()}
    summary = ("%s: %d trials, %d pass (%.2f%% vs random %s), %d specs on >=%s coins -> %s"
               % (hid, n_trials, n_pass, 100 * grid_rate,
                  "n/a" if rnd.get("rate_random") is None else "%.2f%%" % (100 * rnd["rate_random"]),
                  len(passing), tv["min_coins_passing"], "PASS" if passed else "FAIL: " + "; ".join(reasons)))
    out["summary"] = summary
    if write_summary:
        out["summary_file"] = _write_summary(hid, "trainval", out)
    trials.set_status(hid, "passed_trainval" if passed else "failed_trainval", trials=n_trials,
                      summary=summary, evidence=out.get("result_file"), job_id=job_id,
                      batch_trial_id=result.get("batch_trial_id"), selected=out["selected"],
                      grid_pass_rate=out["grid_pass_rate"], random_pass_rate=rnd.get("rate_random"),
                      summary_file=out.get("summary_file"))
    return out


# ----------------------------------------------------------------------------------------------- open_vault (TEST)
def open_vault(hypothesis_id: str, reason: str, *, n_random: Optional[int] = None, seed: int = 1,
               write_summary: bool = True) -> dict:
    """ONE look at the sealed TEST period for the specs/coins fixed by run() (status passed_trainval).

    trials.request_look -> backtest.evaluate(view="full", period="test") per (spec, coin) (logged singles) ->
    equal-weight no-rebalance portfolio vs buy&hold of the same coins -> random-config baseline rate on the same
    window (score.random_baseline_rate, baseline rows) -> Deflated Sharpe with the global N
    (trials.dsr_context) -> pre-registered test_criterion -> test_pass | test_fail. Returns the report dict."""
    from trader.core import backtest, score, trials

    h = trials.load_hypothesis(hypothesis_id)
    hid = h["id"]
    st = trials.hypothesis_status(hid) or {}
    if st.get("status") != "passed_trainval":
        raise SweepRefused("%s latest status is %r; the vault opens only after passed_trainval"
                           % (hid, st.get("status")))
    selected = st.get("selected") or []
    if not selected:
        raise SweepRefused("%s: no specs/coins were fixed for the test by the sweep" % hid)
    from trader.core import catalog
    known = {catalog.spec_hash(s) for s in expand(h)}
    for s in selected:
        if s["spec_hash"] not in known or catalog.spec_hash(s["spec"]) != s["spec_hash"]:
            raise SweepRefused("%s: selected spec %s is not one of the pre-registered specs" % (hid, s["spec_hash"]))
    spec_hashes = [s["spec_hash"] for s in selected]
    coins = []
    for s in selected:
        for c in s["coins"]:
            if c not in coins:
                coins.append(c)
    look_id = trials.request_look(hid, spec_hashes, coins, reason, caller="research")

    tc = dict(DEFAULT_TEST, **(h.get("test_criterion") or {}))
    tc.setdefault("dsr_min", float(_cfg()["research"]["dsr_min"]))
    bcrit = dict(score.DEFAULT_CRITERION, **(h.get("baseline_criterion") or {}))
    results, pairs = [], []
    for s in selected:
        for c in s["coins"]:
            r = backtest.evaluate(s["spec"], c, view="full", period="test", look_id=look_id, caller="research",
                                  hypothesis_id=hid)
            results.append(r)
            m, bh = r.get("metrics") or {}, r.get("bh_metrics") or {}
            pairs.append({"spec_hash": s["spec_hash"], "coin": c, "trial_id": r.get("trial_id"),
                          "n_obs": m.get("n_obs"), "cagr": _r(m.get("cagr")), "total": _r(m.get("total")),
                          "sharpe": _r(m.get("sharpe")), "maxdd": _r(m.get("maxdd")),
                          "exposure": _r(m.get("exposure")), "entries": m.get("entries"),
                          "bh_cagr": _r(bh.get("cagr")), "bh_total": _r(bh.get("total")),
                          "passes_baseline_criterion": bool(score.passes(r.get("metrics"), r.get("bh_metrics"),
                                                                         bcrit))})
    tf = selected[0]["spec"]["tf"]
    bpy = int(_cfg()["bars_per_year"][tf])
    ctx = trials.dsr_context()
    pdsr = _portfolio_dsr(results, bpy, ctx)
    pm = pdsr["metrics"] or {}
    n_bars = int(pm.get("n_obs") or 0)
    bh_tot = [p["bh_total"] for p in pairs if p["bh_total"] is not None]
    bh_total = (sum(1 + x for x in bh_tot) / len(bh_tot) - 1) if bh_tot else None
    bh_cagr = ((1 + bh_total) ** (bpy / n_bars) - 1 if bh_total is not None and n_bars and 1 + bh_total > 0
               else (-1.0 if bh_total is not None else None))
    valid = [p for p in pairs if p["cagr"] is not None]
    rate_sel = (sum(p["passes_baseline_criterion"] for p in valid) / len(valid)) if valid else 0.0

    modes = tuple(sorted({s["spec"]["mode"] for s in selected}))
    nr = int(_cfg()["research"]["random_baseline_n"]) if n_random is None else int(n_random)
    rb = score.random_baseline_rate([], period="test", view="full", look_id=look_id, n_random=nr, seed=seed,
                                    criterion=bcrit, modes=modes, tfs=(tf,), coins=coins, caller="research")
    rate_rnd = rb.get("rate_random")

    reasons = []
    if pm.get("cagr") is None:
        reasons.append("test period too short for metrics")
    elif pm["cagr"] < float(tc["portfolio_cagr_min"]):
        reasons.append("portfolio CAGR %.4f < %s" % (pm["cagr"], tc["portfolio_cagr_min"]))
    if tc.get("beat_bh") and not (pm.get("cagr") is not None and bh_cagr is not None and pm["cagr"] > bh_cagr):
        reasons.append("does not beat buy&hold of the same coins (%s vs %s)" % (_r(pm.get("cagr")), _r(bh_cagr)))
    if tc.get("beat_random_rate") and not (rb.get("n_random") and rate_sel > (rate_rnd or 0.0)):
        reasons.append("pass rate %.4f does not beat random configs %.4f (n=%s)"
                       % (rate_sel, rate_rnd or 0.0, rb.get("n_random")))
    if tc.get("dsr_min") is not None and (pdsr["dsr"] is None or pdsr["dsr"] < float(tc["dsr_min"])):
        reasons.append("DSR %s < %s (N=%d)" % (_r(pdsr["dsr"]), tc["dsr_min"], ctx["n_trials"]))
    passed = not reasons
    out = {"hypothesis_id": hid, "hypothesis_hash": h["hash"], "stage": "test", "look_id": look_id,
           "reason": reason, "selected": selected, "pairs": pairs, "test_criterion": tc,
           "portfolio": {"n_bars": n_bars, "cagr": _r(pm.get("cagr")), "total": _r(pm.get("total")),
                         "sharpe": _r(pm.get("sharpe")), "maxdd": _r(pm.get("maxdd")), "dsr": _r(pdsr["dsr"]),
                         "expected_max_sr_daily": _r(pdsr["expected_max_sr_daily"], 6)},
           "buy_and_hold": {"total": _r(bh_total), "cagr": _r(bh_cagr)},
           "random_baseline": {"criterion": bcrit, "rate_selected": _r(rate_sel), "n_selected": len(valid),
                               "rate_random": _r(rate_rnd), "n_random": rb.get("n_random"),
                               "n_skipped": rb.get("n_skipped"), "seed": seed},
           "dsr_context": ctx, "passed": passed, "reasons": reasons}
    out["summary"] = ("%s TEST %s: portfolio CAGR %s vs B&H %s, pass rate %s vs random %s, DSR %s (N=%d) -> %s"
                      % (hid, look_id, _r(pm.get("cagr")), _r(bh_cagr), _r(rate_sel), _r(rate_rnd),
                         _r(pdsr["dsr"]), ctx["n_trials"], "PASS" if passed else "FAIL: " + "; ".join(reasons)))
    if write_summary:
        out["summary_file"] = _write_summary(hid, "test", out)
    trials.set_status(hid, "test_pass" if passed else "test_fail", look_id=look_id,
                      trials=len(results), summary=out["summary"], evidence=out.get("summary_file"),
                      portfolio=out["portfolio"], buy_and_hold=out["buy_and_hold"],
                      random_baseline=out["random_baseline"])
    return out


# ----------------------------------------------------------------------------------------------- status
def status(hypothesis_id: str) -> dict:
    from trader.core import trials
    rows = trials.hypothesis_rows(hypothesis_id)
    m = config.utcnow().strftime("%Y-%m")
    used, total = trials.looks_used(m), trials.looks_budget()
    sch = (_schedule().get("research") or {})
    return {"hypothesis_id": hypothesis_id, "latest": rows[-1] if rows else None, "history": rows,
            "looks": {"month": m, "used": used, "total": total, "remaining": max(total - used, 0)},
            "sweeps_last_24h": sweeps_last_24h(), "max_sweeps_per_day": sch.get("max_sweeps_per_day")}


# ----------------------------------------------------------------------------------------------- CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m trader.strategy.tools.sweep")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("next-id")
    p = sub.add_parser("prereg"); p.add_argument("hypothesis")
    e = sub.add_parser("expand"); e.add_argument("hypothesis"); e.add_argument("--specs", action="store_true")
    r = sub.add_parser("run"); r.add_argument("hypothesis")
    r.add_argument("--host", default="auto", help="auto | inproc | local | bigsur | m2")
    r.add_argument("--no-wait", action="store_true"); r.add_argument("--procs", type=int)
    r.add_argument("--timeout-min", type=float, default=40.0); r.add_argument("--poll-s", type=float, default=15.0)
    r.add_argument("--n-random", type=int, default=None)
    o = sub.add_parser("open"); o.add_argument("hypothesis"); o.add_argument("--reason", required=True)
    o.add_argument("--n-random", type=int, default=None)
    s = sub.add_parser("status"); s.add_argument("hypothesis")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "next-id":
            out = {"next_id": next_id()}
        elif a.cmd == "prereg":
            out = prereg(a.hypothesis)
        elif a.cmd == "expand":
            from trader.core import catalog, trials
            fp = _hyp_path(a.hypothesis) if HID_RE.match(a.hypothesis) else Path(a.hypothesis)
            h = json.loads(fp.read_text())
            if h.get("hash"):
                h = trials.load_hypothesis(fp)  # a hashed file must still match its hash
            specs = expand(h)
            out = {"hypothesis_id": h.get("id"), "n_specs": len(specs), "n_coins": len(h.get("coins") or []),
                   "n_trials_planned": len(specs) * len(h.get("coins") or []),
                   "cap": _trial_cap(h, _cfg())}
            if a.specs:
                out["specs"] = [dict(s, spec_hash=catalog.spec_hash(s)) for s in specs]
        elif a.cmd == "run":
            res = run(a.hypothesis, host=a.host, wait=not a.no_wait, procs=a.procs, timeout_min=a.timeout_min,
                      poll_s=a.poll_s, n_random=a.n_random)
            out = dict(res, ok=True)
            out.pop("dsr_context", None)
        elif a.cmd == "open":
            out = dict(open_vault(a.hypothesis, a.reason, n_random=a.n_random), ok=True)
        else:
            out = status(a.hypothesis)
    except SweepRefused as e:
        print(json.dumps({"ok": False, "refused": True, "error": str(e)}))
        return 2
    except Exception as e:  # CLI boundary: one JSON object
        refused = e.__class__.__name__ in ("LookBudgetExceeded", "HypothesisError", "VaultSealed", "QuotaError")
        print(json.dumps({"ok": False, "refused": refused, "error": "%s: %s" % (type(e).__name__, e)}))
        return 2 if refused else 1
    print(json.dumps(_jsonable(out), sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
