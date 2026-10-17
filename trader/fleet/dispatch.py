"""Sweep dispatch: ship a research sweep to a fleet host and bring back a compact result (CONTRACTS.md §8).

PROTECTED. Runs on air (the brain). A job is a list of strategy specs × coins evaluated on the research view.

Flow of ``submit(job)``:
1. validate (view == "research", specs × coins <= research.max_trials_per_hypothesis, coins in universe,
   quota within the host's limits) and require a pre-registered hypothesis (``trials`` registry: registered hash
   present, latest status ``registered`` or ``sweeping``; a job hypothesis_hash must equal the registered one);
2. pick the host (``auto`` = first reachable of bigsur, m2) and check its runner disk usage against
   ``fleet.hosts.<h>.max_disk_gb`` (bigsur 5 GB);
3. stage locally: ``job.json`` + ``data/<COIN>_<tf>.csv`` written from ``vault.load(view="research")`` (so the
   sweep host never receives a bar at or after vault_cutoff; the runner re-checks this);
4. rsync ``trader/`` code (no journal, no tests) to ``<remote_root>/code/trader`` and the staged job to
   ``<remote_root>/jobs/<job_id>/``, then start ``sweep_runner`` there with ``nohup`` over ssh.

``status`` reads the remote ``state.json`` (progress only: no pass counts); ``collect`` copies back
``result.json`` (refused if larger than ``fleet.max_result_kb``) into ``journal/sweeps/<job_id>.json`` and logs
the batch with ``trials.log_batch`` BEFORE returning anything (idempotent: a job is logged once, recorded as
``batch_trial_id`` in the registry and recovered from trials.jsonl after a crash); ``run_local`` does the same.
No sweep result ever reaches a caller without its trials-ledger row.

Trials count from SUBMISSION, not from collect: right before the runner is started, ``submit`` logs a provisional
batch row (``kind`` "batch", ``n_configs`` = len(specs), ``n_trials`` = len(specs) x len(coins), empty
``sr_moments``, ``n_pass`` 0, ``result_file`` = ``"pending:journal/sweeps/<job_id>.json"``) and records its id as
``provisional_trial_id`` in the registry. A local host's job dir (rows.jsonl, result.json) is readable on this
machine, so results can be seen without collect; with the provisional row those trials are already in the DSR N
even if collect never runs or the hypothesis is dropped. ``collect`` later appends the final batch row (real
moments) for the same ``job_id``; the final row supersedes the provisional one (per ``job_id`` the last batch row
wins; a ``pending:`` row is provisional). ``ingest`` = collect, returns the batch
trial_id. ``cleanup`` removes the remote job directory.

The host ``local`` (and any host whose ssh alias is ``local``, e.g. m2) runs the same commands through
``/bin/sh`` instead of ssh, so the whole remote path is testable on this machine. Every function that touches a
host accepts ``dry_run=True``: it returns the exact commands it would run and touches nothing.

Environment: ``PHIL_REMOTE_ROOT`` overrides ``fleet.remote_root`` (tests); ``PHIL_STATE_DIR`` (default
``~/.phil-crypto/jobs``) holds the local job registry.

CLI: ``python -m trader.fleet.dispatch submit job.json [--host H] [--dry-run] | status J | collect J | ingest J |
cleanup J | run-local job.json | sync HOST | hosts`` — prints one JSON object; exit 0 ok, 1 error, 2 refused.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import secrets
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable, Optional

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from trader.core import config  # noqa: E402
from trader.fleet import sweep_runner  # noqa: E402

SSH_OPTS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "-o", "ServerAliveInterval=30"]
AUTO_ORDER = ("bigsur", "m2")
CODE_EXCLUDES = ["journal/", "tests/", "__pycache__/", "*.pyc", ".loop.pid", "*.log", ".DS_Store"]
JOB_ID_RE = re.compile(r"^J-[0-9]{8}T[0-9]{4,6}-[0-9a-z]{2,8}$")
# wall-clock ceiling for a job when fleet.hosts.<h>.max_minutes is not set (protected config may override)
HOST_MAX_MINUTES_DEFAULT = 720

LOCAL_HOST = {"ssh": "local", "os": "local", "roles": ["sweep"], "python": sys.executable,
              "max_disk_gb": 20, "max_ram_mb": 8192, "max_procs": max(1, (os.cpu_count() or 2) - 1)}


class DispatchError(Exception):
    """Job refused or a host step failed."""


class QuotaError(DispatchError):
    """The job would exceed a host or job quota."""


# ── config / hosts ────────────────────────────────────────────────────────────
def _cfg() -> dict:
    return config.load_config()


def hosts(cfg: Optional[dict] = None) -> dict:
    cfg = cfg or _cfg()
    out = {k: dict(v) for k, v in cfg["fleet"]["hosts"].items()}
    out["local"] = dict(LOCAL_HOST)
    return out


def host_cfg(name: str, cfg: Optional[dict] = None) -> dict:
    h = hosts(cfg)
    if name not in h:
        raise DispatchError("unknown host %r (known: %s)" % (name, ", ".join(sorted(h))))
    return h[name]


def is_local(name: str, cfg: Optional[dict] = None) -> bool:
    return host_cfg(name, cfg).get("ssh") == "local"


def remote_root(cfg: Optional[dict] = None) -> str:
    """Remote root as written in config ('~/...'); PHIL_REMOTE_ROOT overrides (absolute path in tests)."""
    return os.environ.get("PHIL_REMOTE_ROOT") or (cfg or _cfg())["fleet"]["remote_root"]


def _sh_path(p: str) -> str:
    """A path for a POSIX shell command line: leading ~ → "$HOME", rest quoted."""
    if p.startswith("~/"):
        return '"$HOME"/' + shlex.quote(p[2:])
    if p == "~":
        return '"$HOME"'
    return shlex.quote(p)


def _local_path(p: str) -> Path:
    return Path(p).expanduser()


def _rsync_dest(host: str, p: str, cfg: Optional[dict] = None) -> str:
    """rsync destination spec: local path, or alias:path (home-relative when it starts with ~/)."""
    if is_local(host, cfg):
        return str(_local_path(p))
    alias = host_cfg(host, cfg)["ssh"]
    return "%s:%s" % (alias, p[2:] if p.startswith("~/") else p)


def _python(host: str, cfg: Optional[dict] = None) -> str:
    py = host_cfg(host, cfg).get("python") or "python3"
    if py.startswith("/usr/bin/env "):
        return py
    return _sh_path(py)


# ── command execution ─────────────────────────────────────────────────────────
def _shell_argv(host: str, cmd: str, cfg: Optional[dict] = None) -> list:
    if is_local(host, cfg):
        return ["/bin/sh", "-c", cmd]
    return ["ssh"] + SSH_OPTS + [host_cfg(host, cfg)["ssh"], cmd]


def _run(argv: list, *, timeout: int = 300, check: bool = True) -> subprocess.CompletedProcess:
    p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    if check and p.returncode != 0:
        raise DispatchError("command failed (%d): %s\n%s" % (p.returncode, " ".join(argv)[:300],
                                                             (p.stderr or p.stdout)[-800:]))
    return p


def _rsync_argv(src: str, dest: str, *, delete: bool = False, excludes=()) -> list:
    # flags understood by rsync 2.6.9 (macOS 11/14 system rsync), openrsync and rsync 3.x
    argv = ["rsync", "-a", "-z"]
    if delete:
        argv.append("--delete")
    for e in excludes:
        argv += ["--exclude", e]
    argv += ["-e", "ssh " + " ".join(SSH_OPTS), src, dest]
    return argv


def _copy(src: str, dest: str, host: str, *, delete=False, excludes=(), dry_run=False, plan=None,
          cfg=None) -> None:
    """Copy a directory (trailing slash = contents) or file to/from host; local hosts may lack rsync."""
    argv = _rsync_argv(src, dest, delete=delete, excludes=excludes)
    if plan is not None:
        plan.append(argv)
    if dry_run:
        return
    if is_local(host, cfg) and shutil.which("rsync") is None:
        s, d = Path(src.rstrip("/")), Path(dest.rstrip("/"))
        if s.is_dir():
            if delete and d.exists():
                shutil.rmtree(str(d))
            shutil.copytree(str(s), str(d), dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns(*[e.rstrip("/") for e in excludes]))
        else:
            d.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(s), str(d))
        return
    _run(argv, timeout=900)


def _exec(host: str, cmd: str, *, dry_run=False, plan=None, timeout=300, check=True, cfg=None):
    argv = _shell_argv(host, cmd, cfg)
    if plan is not None:
        plan.append(argv)
    if dry_run:
        return None
    return _run(argv, timeout=timeout, check=check)


def reachable(host: str, cfg: Optional[dict] = None) -> bool:
    try:
        return _exec(host, "true", timeout=20, check=False, cfg=cfg).returncode == 0
    except (subprocess.TimeoutExpired, OSError, DispatchError):
        return False


# ── local job registry ────────────────────────────────────────────────────────
def state_dir() -> Path:
    d = Path(os.environ.get("PHIL_STATE_DIR", "~/.phil-crypto/jobs")).expanduser()
    d.mkdir(parents=True, exist_ok=True)
    return d


def _reg_path(job_id: str) -> Path:
    if not JOB_ID_RE.match(job_id):
        raise DispatchError("bad job id %r" % job_id)
    return state_dir() / ("%s.json" % job_id)


def _reg_read(job_id: str) -> dict:
    p = _reg_path(job_id)
    if not p.exists():
        raise DispatchError("unknown job %s (no registry entry in %s)" % (job_id, state_dir()))
    return json.loads(p.read_text())


def _reg_write(job_id: str, entry: dict) -> None:
    p = _reg_path(job_id)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(entry, sort_keys=True, indent=1))
    os.replace(str(tmp), str(p))


def new_job_id() -> str:
    return "J-%s-%s" % (config.utcnow().strftime("%Y%m%dT%H%M%S"), secrets.token_hex(2))


# ── validation ────────────────────────────────────────────────────────────────
def _spec_problems(spec: dict, cfg: dict) -> list:
    try:
        from trader.core import catalog
        return list(catalog.validate_spec(spec))
    except ImportError:
        pass
    probs = []
    for k in sweep_runner.CORE_KEYS:
        if k not in spec:
            probs.append("missing %s" % k)
    if spec.get("tf") not in cfg["timeframes"]:
        probs.append("tf %r not in %s" % (spec.get("tf"), cfg["timeframes"]))
    if spec.get("mode") not in ("SPOT", "PERP"):
        probs.append("mode %r" % spec.get("mode"))
    comps = spec.get("comps")
    if not isinstance(comps, list) or not 1 <= len(comps) <= 3:
        probs.append("comps must be 1-3 items")
    return probs


def validate_job(job: dict, cfg: Optional[dict] = None) -> list:
    """[] if the job may be dispatched, else human-readable problems."""
    cfg = cfg or _cfg()
    probs = []
    for k in ("hypothesis_id", "specs", "coins"):
        if not job.get(k):
            probs.append("missing %s" % k)
    if probs:
        return probs
    if job.get("view", "research") != "research":
        probs.append("view must be 'research' (sweep hosts never receive sealed data)")
    if job.get("job_id") and not JOB_ID_RE.match(job["job_id"]):
        probs.append("bad job_id %r" % job["job_id"])
    bad = [c for c in job["coins"] if c not in cfg["universe"]]
    if bad:
        probs.append("coins not in universe: %s" % bad)
    n = len(job["specs"]) * len(job["coins"])
    cap = int(cfg["research"]["max_trials_per_hypothesis"])
    if n > cap:
        probs.append("%d specs x %d coins = %d trials > research.max_trials_per_hypothesis %d"
                     % (len(job["specs"]), len(job["coins"]), n, cap))
    for i, s in enumerate(job["specs"]):
        for p in _spec_problems(s, cfg):
            probs.append("spec %d: %s" % (i, p))
            break
    periods = job.get("periods", ["train", "val"])
    if any(p not in ("train", "val", "trainval") for p in periods):
        probs.append("periods must be research periods (train/val/trainval), got %s" % periods)
    return probs


def effective_quota(job: dict, host: str, cfg: Optional[dict] = None) -> dict:
    """Job quota validated and checked against the host's limits; raises QuotaError otherwise.

    Only DEFAULT_QUOTA keys are kept; each value must be a finite number with 0 < v <= host limit (a NaN, inf,
    string, bool or negative value would otherwise defeat every later ``>`` comparison). ``max_minutes`` is capped
    by ``fleet.hosts.<h>.max_minutes`` or, if the host has none, HOST_MAX_MINUTES_DEFAULT.
    """
    h = host_cfg(host, cfg)
    try:
        q = sweep_runner.sanitize_quota(job.get("quota"))
    except ValueError as e:
        raise QuotaError(str(e))
    limits = {"max_ram_mb": h.get("max_ram_mb"), "max_procs": h.get("max_procs"),
              "max_disk_mb": float(h["max_disk_gb"]) * 1024 if h.get("max_disk_gb") else None,
              "max_minutes": h.get("max_minutes") or HOST_MAX_MINUTES_DEFAULT}
    for k, lim in limits.items():
        if lim is None:
            continue
        try:
            lim_f = float(lim)
        except (TypeError, ValueError):
            raise QuotaError("host %s limit %s=%r is not a number" % (host, k, lim))
        if not math.isfinite(lim_f) or lim_f <= 0:
            raise QuotaError("host %s limit %s=%r must be finite and > 0" % (host, k, lim))
        if q[k] > lim_f:
            raise QuotaError("quota %s=%s exceeds host %s limit %s" % (k, q[k], host, lim))
    return q


def choose_host(job: dict, cfg: Optional[dict] = None, *, dry_run=False) -> str:
    want = job.get("host") or "auto"
    hs = hosts(cfg)
    if want != "auto":
        if want not in hs or "sweep" not in hs[want].get("roles", []):
            raise DispatchError("host %r cannot run sweeps" % want)
        return want
    for h in AUTO_ORDER:
        if h in hs and (dry_run or reachable(h, cfg)):
            return h
    raise DispatchError("no sweep host reachable (tried %s)" % ", ".join(AUTO_ORDER))


# ── staging (research-view data only) ─────────────────────────────────────────
def _vault_loader(coin: str, tf: str):
    from trader.core import vault
    return vault.load(coin, tf, "research")


def stage(job: dict, dest: Path, *, loader: Optional[Callable] = None, cfg: Optional[dict] = None) -> dict:
    """Write job.json and research-view CSVs into dest; returns {"files","bytes","coins_missing"}."""
    cfg = cfg or _cfg()
    loader = loader or _vault_loader
    cutoff = config_cutoff(cfg)
    data = dest / "data"
    data.mkdir(parents=True, exist_ok=True)
    tfs = sorted({s["tf"] for s in job["specs"]})
    files, nbytes, missing = 0, 0, []
    import pandas as pd
    for tf in tfs:
        for coin in job["coins"]:
            try:
                df = loader(coin, tf)
            except FileNotFoundError:
                missing.append("%s_%s" % (coin, tf))
                continue
            if df is None or len(df) == 0:
                missing.append("%s_%s" % (coin, tf))
                continue
            if not (df.index.max() < cutoff):  # defence in depth: the vault already guarantees this
                raise DispatchError("%s %s research view reaches %s >= cutoff" % (coin, tf, df.index.max()))
            cols = [c for c in ("open", "high", "low", "close", "volume", "turnover", "funding") if c in df.columns]
            out = df[cols].copy()
            if "funding" not in out.columns:
                out["funding"] = 0.0
            out.index = pd.DatetimeIndex(out.index).rename("time")
            fmt = "%Y-%m-%d" if tf == "1d" else "%Y-%m-%d %H:%M:%S"
            p = data / ("%s_%s.csv" % (coin, tf))
            out.to_csv(p, date_format=fmt)
            files += 1
            nbytes += p.stat().st_size
    (dest / "job.json").write_text(json.dumps(job, sort_keys=True, indent=1))
    return {"files": files, "bytes": nbytes, "coins_missing": missing}


def config_cutoff(cfg: dict):
    import pandas as pd
    return pd.Timestamp(cfg["vault"]["vault_cutoff"])


SWEEPABLE_STATUSES = ("registered", "sweeping")


def hypothesis_problems(job: dict) -> list:
    """[] if job["hypothesis_id"] is pre-registered and may still be swept, else problems (fails closed)."""
    hid = job.get("hypothesis_id")
    if not hid:
        return ["missing hypothesis_id"]
    try:
        from trader.core import trials
    except Exception as e:  # no ledger ⇒ no way to count the trials ⇒ refuse
        return ["trials ledger unavailable (%s: %s)" % (type(e).__name__, e)]
    reg = trials._registered_hash(hid)
    if reg is None:
        return ["hypothesis %s is not registered (register it with trials.register_hypothesis first)" % hid]
    probs = []
    if job.get("hypothesis_hash") and job["hypothesis_hash"] != reg:
        probs.append("hypothesis_hash %s != registered hash %s for %s"
                     % (str(job["hypothesis_hash"])[:12], reg[:12], hid))
    st = (trials.hypothesis_status(hid) or {}).get("status")
    if st not in SWEEPABLE_STATUSES:
        probs.append("hypothesis %s latest status is %r; sweeps only for %s" % (hid, st, list(SWEEPABLE_STATUSES)))
    return probs


def _prepare(job: dict, cfg: dict) -> dict:
    job = json.loads(json.dumps(job))
    job.setdefault("job_id", new_job_id())
    job.setdefault("created", config.iso())
    job.setdefault("view", "research")
    job.setdefault("periods", ["train", "val"])
    job.setdefault("costs", None)
    job.setdefault("criterion", dict(sweep_runner.DEFAULT_CRITERION))
    job.setdefault("top_k", 200)
    probs = validate_job(job, cfg)
    if not probs:
        probs = hypothesis_problems(job)
    if probs:
        raise DispatchError("job refused: " + "; ".join(probs))
    from trader.core import trials
    job["hypothesis_hash"] = trials._registered_hash(job["hypothesis_id"])
    return job


# ── public API ────────────────────────────────────────────────────────────────
def submit(job: dict, *, dry_run: bool = False, loader: Optional[Callable] = None,
           env: Optional[dict] = None) -> str:
    """Validate, stage, ship and start a sweep; returns job_id (dry_run: plan stored in the registry)."""
    res = submit_plan(job, dry_run=dry_run, loader=loader, env=env)
    return res["job_id"]


def submit_plan(job: dict, *, dry_run: bool = False, loader: Optional[Callable] = None,
                env: Optional[dict] = None) -> dict:
    """submit() returning the full record: {"job_id","host","job_dir","commands","staged","dry_run"}."""
    cfg = _cfg()
    job = _prepare(job, cfg)
    host = choose_host(job, cfg, dry_run=dry_run)
    job["quota"] = effective_quota(job, host, cfg)
    job["host_resolved"] = host
    root = remote_root(cfg)
    job_dir = "%s/jobs/%s" % (root, job["job_id"])
    code_dir = "%s/code" % root
    plan = []

    # host disk quota: runner dir usage must stay under max_disk_gb after this job
    hcfg = host_cfg(host, cfg)
    usage_kb = 0
    p = _exec(host, "mkdir -p %s/jobs %s && du -sk %s | cut -f1" % (_sh_path(root), _sh_path(code_dir),
                                                                   _sh_path(root)),
              dry_run=dry_run, plan=plan, timeout=60, cfg=cfg)
    if p is not None:
        try:
            usage_kb = int(p.stdout.strip().split()[-1])
        except (ValueError, IndexError):
            usage_kb = 0

    with tempfile.TemporaryDirectory(prefix="phil-stage-") as tmp:
        staged = stage(job, Path(tmp), loader=loader, cfg=cfg)
        if staged["files"] == 0:
            raise DispatchError("no research data staged for %s" % job["coins"])
        cap_kb = float(hcfg.get("max_disk_gb") or 1e9) * 1024 * 1024
        need_kb = usage_kb + staged["bytes"] / 1024 + float(job["quota"]["max_disk_mb"]) * 1024
        if need_kb > cap_kb:
            raise QuotaError("host %s runner dir would reach %.2f GB > max_disk_gb %s (run cleanup first)"
                             % (host, need_kb / 1024 / 1024, hcfg.get("max_disk_gb")))
        _copy(str(config.TRADER_ROOT) + "/", _rsync_dest(host, code_dir + "/trader/", cfg), host, delete=True,
              excludes=CODE_EXCLUDES, dry_run=dry_run, plan=plan, cfg=cfg)
        _copy(tmp + "/", _rsync_dest(host, job_dir + "/", cfg), host, dry_run=dry_run, plan=plan, cfg=cfg)

        envs = {"PYTHONPATH": code_dir, "TRADER_CONFIG": code_dir + "/trader/config/protected.json",
                "TRADER_JOURNAL_DIR": job_dir + "/journal", "TRADER_STRATEGY_DIR": job_dir + "/strategy",
                "PHIL_RUNNER": host, "NUMBA_CACHE_DIR": root + "/numba-cache"}
        envs.update(env or {})
        env_s = " ".join("%s=%s" % (k, _sh_path(v)) for k, v in sorted(envs.items()))
        start = ("cd %s && %s nohup %s -m trader.fleet.sweep_runner job.json > runner.log 2>&1 < /dev/null "
                 "& echo $! > %s/runner.pid; cat %s/runner.pid"
                 % (_sh_path(job_dir), env_s, _python(host, cfg), _sh_path(job_dir), _sh_path(job_dir)))
        prov_tid = None if dry_run else _log_provisional(job, host)   # counted before the runner can write rows
        p = _exec(host, start, dry_run=dry_run, plan=plan, timeout=60, cfg=cfg)

    entry = {"job_id": job["job_id"], "host": host, "job_dir": job_dir, "submitted": config.iso(),
             "hypothesis_id": job["hypothesis_id"], "hypothesis_hash": job.get("hypothesis_hash"),
             "periods": job.get("periods"), "costs": job.get("costs"), "view": job.get("view"),
             "n_specs": len(job["specs"]), "coins": job["coins"], "quota": job["quota"],
             "pid": (p.stdout.strip() if p is not None else None), "dry_run": dry_run,
             "staged": staged, "provisional_trial_id": prov_tid,
             "n_trials_planned": len(job["specs"]) * len(job["coins"]),
             "commands": ["".join(shlex.quote(a) for a in c) for c in plan]}
    _reg_write(job["job_id"], entry)
    return dict(entry, commands=plan)


def status(job_id: str) -> dict:
    """{"job_id","host","state","progress","elapsed_s", ...} from the host's state.json."""
    e = _reg_read(job_id)
    base = {"job_id": job_id, "host": e["host"]}
    if e.get("dry_run"):
        return dict(base, state="queued", progress=0.0, elapsed_s=0, dry_run=True)
    if e.get("local_result"):
        return dict(base, state="done", progress=1.0, elapsed_s=e.get("elapsed_s", 0))
    jd = _sh_path(e["job_dir"])
    cmd = ("cat %s/state.json 2>/dev/null || echo '{}'; echo; "
           "if [ -f %s/runner.pid ] && kill -0 $(cat %s/runner.pid) 2>/dev/null; then echo ALIVE; else echo DEAD; fi"
           % (jd, jd, jd))
    p = _exec(e["host"], cmd, timeout=60, cfg=None)
    lines = [ln for ln in p.stdout.splitlines() if ln.strip()]
    alive = bool(lines) and lines[-1] == "ALIVE"
    try:
        st = json.loads(lines[0]) if len(lines) > 1 else {}
    except ValueError:
        st = {}
    state = st.get("state") or ("queued" if alive else "failed")
    if state == "running" and not alive:
        state = "failed"
        st.setdefault("reason", "runner process gone without finishing")
    elapsed = st.get("elapsed_s")
    if elapsed is None:
        try:
            import datetime as _dt
            t = _dt.datetime.fromisoformat(e["submitted"].replace("Z", "+00:00"))
            elapsed = int((config.utcnow() - t).total_seconds())
        except Exception:
            elapsed = 0
    out = dict(base, state=state, progress=float(st.get("progress", 0.0)), elapsed_s=int(elapsed))
    # no n_pass here. A local host's job dir is readable directly, so honesty does not rely on collect: the
    # planned trials were already counted at submit (provisional batch row); collect adds the final moments.
    for k in ("n_trials", "errors", "reason"):
        if k in st:
            out[k] = st[k]
    return out


def _sweeps_dir() -> Path:
    d = config.journal_dir() / "sweeps"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _store_result(job_id: str, result: dict, raw: bytes) -> Path:
    max_bytes = int(_cfg()["fleet"]["max_result_kb"]) * 1024
    if len(raw) > max_bytes:
        raise QuotaError("result.json is %d bytes > fleet.max_result_kb" % len(raw))
    if result.get("job_id") != job_id:
        raise DispatchError("result.json job_id %r != %r" % (result.get("job_id"), job_id))
    dest = _sweeps_dir() / ("%s.json" % job_id)
    if dest.exists():  # add-only directory: never rewrite a committed result
        if dest.read_bytes() != raw:
            raise DispatchError("%s already exists with different content" % dest)
        return dest
    tmp = dest.with_suffix(".tmp")
    tmp.write_bytes(raw)
    os.replace(str(tmp), str(dest))
    return dest


class _JobLock:
    """Exclusive per-job lock (registry dir) so two collects cannot log the same batch twice."""

    def __init__(self, job_id: str):
        self.path = state_dir() / (".%s.lock" % job_id)
        self.fh = None

    def __enter__(self):
        self.fh = open(str(self.path), "a")
        try:
            import fcntl
            fcntl.flock(self.fh, fcntl.LOCK_EX)
        except ImportError:  # pragma: no cover
            pass
        return self

    def __exit__(self, *exc):
        try:
            import fcntl
            fcntl.flock(self.fh, fcntl.LOCK_UN)
        except ImportError:  # pragma: no cover
            pass
        self.fh.close()
        return False


def _rel_result(dest: Path) -> str:
    rel = os.path.relpath(str(dest), str(config.TRADER_ROOT))
    return str(dest) if rel.startswith("..") else rel


PENDING_PREFIX = "pending:"


def _log_provisional(job: dict, host: str) -> str:
    """Provisional batch row at submit time: the planned trials enter the ledger before any result exists.
    Raises if the ledger is unavailable, so a sweep never starts uncounted."""
    from trader.core import trials
    row = {"job_id": job["job_id"], "host": host, "view": job.get("view", "research"),
           "periods": job.get("periods"), "costs": job.get("costs"),
           "n_configs": len(job["specs"]), "n_trials": len(job["specs"]) * len(job["coins"]),
           "sr_moments": {}, "n_pass": 0,
           "result_file": "%sjournal/sweeps/%s.json" % (PENDING_PREFIX, job["job_id"])}
    return trials.log_batch(row, hypothesis_id=job["hypothesis_id"], caller="sweep")


def _is_provisional(row: dict, e: dict) -> bool:
    return row.get("trial_id") == e.get("provisional_trial_id") or \
        str(row.get("result_file") or "").startswith(PENDING_PREFIX)


def _log_batch_once(job_id: str, res: dict, rel: str) -> str:
    """trials.log_batch for this job exactly once; returns the batch trial_id (also stored in the registry).

    Idempotent through the registry's batch_trial_id and, after a crash between logging and the registry write,
    through the existing kind="batch" row with this job_id in trials.jsonl. Raises if the ledger is unavailable,
    so the caller never gets results that were not counted."""
    from trader.core import trials
    with _JobLock(job_id):
        e = _reg_read(job_id)
        tid = e.get("batch_trial_id")
        if not tid:
            prior = [r for r in trials.read_trials(kind="batch", job_id=job_id) if not _is_provisional(r, e)]
            if prior:
                tid = prior[0]["trial_id"]
            else:
                row = dict(res, result_file=rel)
                for k in ("periods", "costs", "view"):
                    if k not in row and e.get(k) is not None:
                        row[k] = e[k]
                tid = trials.log_batch(row, hypothesis_id=e["hypothesis_id"], caller="sweep")
            e = _reg_read(job_id)
            e["batch_trial_id"] = tid
            e["ingested"] = e.get("ingested") or config.iso()
            _reg_write(job_id, e)
    return tid


def collect(job_id: str) -> dict:
    """Copy result.json back into journal/sweeps/<job_id>.json, log the batch to the trials ledger (once), then
    return the result dict (+ "result_file", "batch_trial_id")."""
    e = _reg_read(job_id)
    dest = _sweeps_dir() / ("%s.json" % job_id)
    if e.get("local_result") and dest.exists():
        res = json.loads(dest.read_text())
    else:
        st = status(job_id)
        if st["state"] != "done":
            raise DispatchError("job %s is %s, not done" % (job_id, st["state"]))
        with tempfile.TemporaryDirectory(prefix="phil-collect-") as tmp:
            local = Path(tmp) / "result.json"
            src = e["job_dir"] + "/result.json"
            if is_local(e["host"]):
                shutil.copy2(str(_local_path(src)), str(local))
            else:
                _run(["rsync", "-a", "-e", "ssh " + " ".join(SSH_OPTS), _rsync_dest(e["host"], src), str(local)],
                     timeout=300)
            raw = local.read_bytes()
        res = json.loads(raw)
        _store_result(job_id, res, raw)
    rel = _rel_result(dest)
    e["collected"] = e.get("collected") or config.iso()
    e["result_file"] = rel
    _reg_write(job_id, e)
    tid = _log_batch_once(job_id, res, rel)          # counted before anyone sees the rows
    return dict(res, result_file=rel, batch_trial_id=tid)


def ingest(job_id: str) -> str:
    """Alias kept for the CLI/contract: collect (which logs the batch once); returns the batch trial_id."""
    return collect(job_id)["batch_trial_id"]


def cleanup(job_id: str, *, dry_run: bool = False) -> None:
    """Remove the remote job directory (keeps bigsur under its disk quota)."""
    e = _reg_read(job_id)
    if e.get("local_result") or e.get("dry_run"):
        return None
    jd = e["job_dir"]
    if "/jobs/J-" not in jd:  # never rm anything that is not a job dir
        raise DispatchError("refusing to remove %r" % jd)
    cmd = "if [ -f %s/runner.pid ]; then kill $(cat %s/runner.pid) 2>/dev/null; fi; rm -rf %s" % (
        _sh_path(jd), _sh_path(jd), _sh_path(jd))
    _exec(e["host"], cmd, dry_run=dry_run, timeout=120)
    if not dry_run:
        e["cleaned"] = config.iso()
        _reg_write(job_id, e)
    return None


def run_local(job: dict, *, loader: Optional[Callable] = None, procs: Optional[int] = None,
              evaluator_env: Optional[str] = None, keep_dir: Optional[str] = None) -> dict:
    """Same pipeline in-process (stage → sweep_runner.run → journal/sweeps copy); returns the result dict."""
    cfg = _cfg()
    job = _prepare(job, cfg)
    job["host_resolved"] = "inproc"
    job["quota"] = effective_quota(job, "local", cfg)
    tmp = tempfile.mkdtemp(prefix="phil-local-", dir=keep_dir)
    try:
        stage(job, Path(tmp), loader=loader, cfg=cfg)
        res = sweep_runner.run(Path(tmp) / "job.json", procs=procs or 1, evaluator_env=evaluator_env)
        raw = (Path(tmp) / "result.json").read_bytes()
        path = _store_result(job["job_id"], res, raw)
    finally:
        if keep_dir is None:
            shutil.rmtree(tmp, ignore_errors=True)
    rel = _rel_result(path)
    _reg_write(job["job_id"], {"job_id": job["job_id"], "host": "inproc", "job_dir": tmp, "local_result": True,
                               "hypothesis_id": job["hypothesis_id"], "hypothesis_hash": job.get("hypothesis_hash"),
                               "periods": job.get("periods"), "costs": job.get("costs"), "view": job.get("view"),
                               "submitted": config.iso(), "elapsed_s": res["elapsed_s"],
                               "collected": config.iso(), "result_file": rel, "dry_run": False})
    tid = _log_batch_once(job["job_id"], res, rel)   # counted before anyone sees the rows
    return dict(res, result_file=rel, batch_trial_id=tid)


def sync_code(host: str, *, dry_run: bool = False) -> list:
    """rsync trader/ code (no journal/tests) to <remote_root>/code/trader on host (berry's systemd units use it)."""
    cfg = _cfg()
    root = remote_root(cfg)
    plan = []
    _exec(host, "mkdir -p %s/code" % _sh_path(root), dry_run=dry_run, plan=plan, timeout=60, cfg=cfg)
    _copy(str(config.TRADER_ROOT) + "/", _rsync_dest(host, root + "/code/trader/", cfg), host, delete=True,
          excludes=CODE_EXCLUDES, dry_run=dry_run, plan=plan, cfg=cfg)
    return [" ".join(shlex.quote(a) for a in c) for c in plan]


def hosts_report() -> dict:
    cfg = _cfg()
    out = {}
    for name, h in hosts(cfg).items():
        out[name] = {"ssh": h.get("ssh"), "roles": h.get("roles"), "optional": bool(h.get("optional")),
                     "max_disk_gb": h.get("max_disk_gb"), "reachable": reachable(name, cfg)}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m trader.fleet.dispatch")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("submit"); s.add_argument("job"); s.add_argument("--host"); s.add_argument("--dry-run", action="store_true")
    for c in ("status", "collect", "ingest"):
        sub.add_parser(c).add_argument("job_id")
    c = sub.add_parser("cleanup"); c.add_argument("job_id"); c.add_argument("--dry-run", action="store_true")
    r = sub.add_parser("run-local"); r.add_argument("job"); r.add_argument("--procs", type=int, default=1)
    y = sub.add_parser("sync"); y.add_argument("host"); y.add_argument("--dry-run", action="store_true")
    sub.add_parser("hosts")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "submit":
            job = json.loads(Path(a.job).read_text())
            if a.host:
                job["host"] = a.host
            rec = submit_plan(job, dry_run=a.dry_run)
            out = {"ok": True, "job_id": rec["job_id"], "host": rec["host"], "dry_run": a.dry_run,
                   "commands": [" ".join(shlex.quote(x) for x in c) for c in rec["commands"]]}
        elif a.cmd == "status":
            out = dict(status(a.job_id), ok=True)
        elif a.cmd == "collect":
            res = collect(a.job_id)
            out = {"ok": True, "job_id": a.job_id, "result_file": res["result_file"], "n_trials": res["n_trials"],
                   "n_pass": res["n_pass"], "errors": res["errors"], "batch_trial_id": res["batch_trial_id"]}
        elif a.cmd == "ingest":
            out = {"ok": True, "job_id": a.job_id, "batch_trial_id": ingest(a.job_id)}
        elif a.cmd == "cleanup":
            cleanup(a.job_id, dry_run=a.dry_run)
            out = {"ok": True, "job_id": a.job_id}
        elif a.cmd == "run-local":
            res = run_local(json.loads(Path(a.job).read_text()), procs=a.procs)
            out = {"ok": True, "job_id": res["job_id"], "n_trials": res["n_trials"], "n_pass": res["n_pass"],
                   "batch_trial_id": res["batch_trial_id"]}
        elif a.cmd == "sync":
            out = {"ok": True, "host": a.host, "commands": sync_code(a.host, dry_run=a.dry_run)}
        else:
            out = {"ok": True, "hosts": hosts_report()}
    except QuotaError as e:
        print(json.dumps({"ok": False, "refused": True, "error": str(e)}))
        return 2
    except DispatchError as e:
        print(json.dumps({"ok": False, "refused": "refused" in str(e), "error": str(e)}))
        return 2 if str(e).startswith("job refused") else 1
    except Exception as e:  # CLI boundary: one JSON object, nonzero exit
        print(json.dumps({"ok": False, "error": "%s: %s" % (type(e).__name__, e)}))
        return 1
    print(json.dumps(out, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
