"""Single-writer cycle lease stored in git (CONTRACTS.md section 8). PROTECTED core.

Why: several runners (air's launchd tick, a manual operator run, a triggered cycle) may start a cycle in the
same minute. Each one commits and pushes the journal, so two concurrent cycles would fork the branch and
double-trade the paper ledger. The lease is one small file, ``trader/journal/lease.json``, on the shared
branch (``phil-crypto`` of ``origin``):

* held:  ``{"runner","cycle","started","ttl_s","pid"}``
* free:  ``{"runner": null, "released_by", "released"}``; a missing file is free too.

``acquire`` and ``release`` each create ONE commit that changes nothing but that file
(message ``cycle(lease): <runner> acquire|release``) and push it as a plain fast-forward. The remote applies a
push only if the branch still points where the pusher saw it, so of two runners racing for a free lease
exactly one push lands; the loser re-reads, sees the other holder and yields (exit 3). A runner that cannot
write the lease (remote unreachable, push refused, local HEAD behind or diverged) also yields (exit 4): it
could not publish its cycle anyway, and a runner on a stale ledger must not trade. A crashed cycle leaves a
lease that expires by itself after ``ttl_s`` (``fleet.lease_ttl_s``), so nothing wedges the fleet for longer.

The lease commit is built through a throw-away index (``GIT_INDEX_FILE``) on the freshly fetched REMOTE tip,
never on local HEAD, so a push publishes nothing but the lease change: ``acquire`` refuses (exit 4) unless local
HEAD == the remote tip (behind, diverged, or ahead with unpushed local commits are all refused; unpushed commits
are only published by loop.sh after its squash + guard + validate step). The working tree and the real index are
untouched until the push has landed; ``acquire`` then fast-forwards HEAD onto it, ``release`` leaves the local
branch alone (the next sync fast-forwards over it).

The lease is driven only by the runner shell, never by the cycle agent: loop.sh exports ``PHIL_LEASE=acquired``
to the headless claude process, and while that variable is set ``acquire`` and ``release`` refuse (exit 4,
nothing fetched or pushed). ``check`` stays available (read-only).

Without an ``origin`` remote there is no shared branch to protect: ``acquire`` returns
``{"acquired": true, "written": false}``.

CLI: ``python -m trader.core.lease check|acquire|release [--cycle K] [--ttl S] [--runner R]``
prints one JSON object; exit 0 acquired/released/checked, 3 held by another live lease, 4 not written.
Python >= 3.9, stdlib only. Environment: ``TRADER_LEASE_REPO`` (git work tree, default the repo root),
``PHIL_BRANCH`` (default ``phil-crypto``), ``PHIL_REMOTE`` (default ``origin``), ``PHIL_LEASE`` (set to
``acquired`` by loop.sh for the agent process: lease writes refused).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

if __package__ in (None, ""):  # allow `python trader/core/lease.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from trader.core import config  # noqa: E402

DEFAULT_TTL_S = 50 * 60          # fallback only; protected fleet.lease_ttl_s wins
EXIT_OK, EXIT_HELD, EXIT_NOT_WRITTEN = 0, 3, 4
RELEASE_ATTEMPTS = 3
GIT_TIMEOUT_S = 120


class LeaseError(Exception):
    """A git plumbing step failed in a way that is not a normal refusal."""


# ── environment ───────────────────────────────────────────────────────────────
def repo_dir() -> Path:
    v = os.environ.get("TRADER_LEASE_REPO")
    return Path(v).expanduser() if v else config.REPO_ROOT


def branch() -> str:
    return os.environ.get("PHIL_BRANCH", "phil-crypto")


def remote() -> str:
    return os.environ.get("PHIL_REMOTE", "origin")


def lease_relpath() -> str:
    """Path of lease.json relative to the git work tree (journal_dir honoured when inside the repo)."""
    target = config.journal_dir() / "lease.json"
    try:
        return target.resolve().relative_to(repo_dir().resolve()).as_posix()
    except ValueError:
        return "trader/journal/lease.json"


def default_ttl() -> int:
    try:
        return int(config.load_config()["fleet"]["lease_ttl_s"])
    except Exception:
        return DEFAULT_TTL_S


def _now() -> _dt.datetime:
    """Current UTC time; tests monkeypatch this."""
    return config.utcnow()


# ── git plumbing ──────────────────────────────────────────────────────────────
def _git(*args, check=True, input=None, env=None):
    proc = subprocess.run(["git", *args], cwd=str(repo_dir()), capture_output=True, text=True,
                          input=input, env=env, timeout=GIT_TIMEOUT_S)
    if check and proc.returncode != 0:
        raise LeaseError("git %s: %s" % (args[0], _last_line(proc, "failed")))
    return proc


def _last_line(proc, fallback):
    lines = [ln.strip() for ln in (proc.stderr or "").splitlines()
             if ln.strip() and not ln.startswith("hint:")]
    rejected = [ln for ln in lines if "rejected" in ln]
    return (rejected or lines or [fallback])[-1]


def _remote_ref() -> str:
    return "refs/remotes/%s/%s" % (remote(), branch())


def has_remote() -> bool:
    return _git("remote", "get-url", remote(), check=False).returncode == 0


def _fetch() -> str:
    """Sha of the remote branch after a fresh fetch. Raises LeaseError when unreachable."""
    p = _git("fetch", "--no-tags", "--quiet", remote(),
             "+refs/heads/%s:%s" % (branch(), _remote_ref()), check=False)
    if p.returncode != 0:
        raise LeaseError("fetch failed: " + _last_line(p, "fetch failed"))
    return _git("rev-parse", _remote_ref()).stdout.strip()


def _read_at(rev: str) -> dict:
    p = _git("show", "%s:%s" % (rev, lease_relpath()), check=False)
    if p.returncode != 0:
        return {}
    try:
        obj = json.loads(p.stdout)
    except ValueError:
        return {}
    return obj if isinstance(obj, dict) else {}


def _read_worktree() -> dict:
    path = repo_dir() / lease_relpath()
    try:
        obj = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    return obj if isinstance(obj, dict) else {}


def _is_ancestor(a: str, b: str) -> bool:
    return _git("merge-base", "--is-ancestor", a, b, check=False).returncode == 0


def _identity_env(runner: str) -> dict:
    env = dict(os.environ)
    if not _git("config", "user.email", check=False).stdout.strip():
        env.setdefault("GIT_AUTHOR_NAME", "phil-crypto %s" % runner)
        env.setdefault("GIT_AUTHOR_EMAIL", "%s@phil-crypto.local" % runner)
        env.setdefault("GIT_COMMITTER_NAME", env["GIT_AUTHOR_NAME"])
        env.setdefault("GIT_COMMITTER_EMAIL", env["GIT_AUTHOR_EMAIL"])
    return env


def _lease_commit(parent: str, payload: dict, message: str, runner: str) -> str:
    """Commit on `parent` whose only change is lease.json = payload (built in a throw-away index)."""
    blob = _git("hash-object", "-w", "--stdin",
                input=json.dumps(payload, sort_keys=True) + "\n").stdout.strip()
    with tempfile.TemporaryDirectory() as tmp:
        env = dict(os.environ, GIT_INDEX_FILE=os.path.join(tmp, "index"))
        _git("read-tree", parent, env=env)
        _git("update-index", "--add", "--cacheinfo", "100644,%s,%s" % (blob, lease_relpath()), env=env)
        tree = _git("write-tree", env=env).stdout.strip()
    return _git("commit-tree", tree, "-p", parent, "-m", message, env=_identity_env(runner)).stdout.strip()


def _push(commit: str):
    p = _git("push", "--quiet", remote(), "%s:refs/heads/%s" % (commit, branch()), check=False)
    return p.returncode == 0, _last_line(p, "push rejected")


# ── pure logic (unit-tested offline) ──────────────────────────────────────────
def _parse_ts(s) -> Optional[_dt.datetime]:
    try:
        return _dt.datetime.fromisoformat(str(s).replace("Z", "+00:00")).astimezone(_dt.timezone.utc)
    except (TypeError, ValueError):
        return None


def describe(payload: dict, me: str, now: Optional[_dt.datetime] = None) -> dict:
    """Interpret a lease payload for runner `me`.

    Returns {"held","mine","fresh","runner","cycle","age_s","ttl_s","expires"}. A lease is live when it names a
    runner and abs(age) < ttl_s (abs: two runners' clocks differ by a few seconds). An unparseable `started`
    counts as expired, so a corrupt file can never wedge the fleet.
    """
    payload = payload if isinstance(payload, dict) else {}
    runner = payload.get("runner")
    if not runner:
        return {"held": False, "mine": False, "fresh": False, "runner": None, "cycle": None,
                "age_s": None, "ttl_s": None, "expires": None}
    now = now or _now()
    started = _parse_ts(payload.get("started"))
    try:
        ttl = int(payload.get("ttl_s", DEFAULT_TTL_S))
    except (TypeError, ValueError):
        ttl = DEFAULT_TTL_S
    age = int((now - started).total_seconds()) if started else None
    fresh = age is not None and abs(age) < ttl
    expires = config.iso(started + _dt.timedelta(seconds=ttl)) if started else None
    return {"held": True, "mine": runner == me, "fresh": fresh, "runner": runner,
            "cycle": payload.get("cycle"), "age_s": age, "ttl_s": ttl, "expires": expires}


def held_by_other(d: dict) -> bool:
    return bool(d["held"] and d["fresh"] and not d["mine"])


def held_payload(runner: str, cycle: str, ttl_s: int, now: Optional[_dt.datetime] = None,
                 pid: Optional[int] = None) -> dict:
    return {"runner": runner, "cycle": cycle, "started": config.iso(now or _now()),
            "ttl_s": int(ttl_s), "pid": int(pid if pid is not None else os.getpid())}


def free_payload(runner: str, now: Optional[_dt.datetime] = None) -> dict:
    return {"runner": None, "released_by": runner, "released": config.iso(now or _now())}


# ── public API ────────────────────────────────────────────────────────────────
def check(*, runner: Optional[str] = None) -> dict:
    """Read-only view of the lease on the remote branch (working tree file when there is no remote)."""
    me = runner or config.runner_id()
    if not has_remote():
        return dict(describe(_read_worktree(), me), me=me, status="checked", exit=EXIT_OK,
                    reason="no remote")
    try:
        base = _fetch()
    except LeaseError as e:
        return dict(describe(_read_worktree(), me), me=me, status="checked", exit=EXIT_OK,
                    reason="unreachable: %s" % e)
    d = describe(_read_at(base), me)
    return dict(d, me=me, base=base, status="checked", exit=EXIT_OK)


def _refuse(me: str, reason: str) -> dict:
    return {"acquired": False, "written": False, "me": me, "status": "not_written", "reason": reason,
            "exit": EXIT_NOT_WRITTEN}


AGENT_ENV = "PHIL_LEASE"


def agent_context() -> bool:
    """True inside a cycle's agent process (loop.sh exports PHIL_LEASE=acquired to claude): no lease writes."""
    return bool(os.environ.get(AGENT_ENV, "").strip())


def _agent_refusal(me: str, what: str) -> dict:
    return {"acquired": False, "released": False, "written": False, "me": me, "status": "not_written",
            "reason": "%s refused: %s=%s is set (the lease is held by the runner shell, not the agent)"
                      % (what, AGENT_ENV, os.environ.get(AGENT_ENV)), "exit": EXIT_NOT_WRITTEN}


def acquire(cycle: str, *, runner: Optional[str] = None, ttl_s: Optional[int] = None) -> dict:
    """Take the lease for `cycle`. Result has "acquired", "written" and "exit" (0, 3 or 4)."""
    me = runner or config.runner_id()
    if agent_context():
        return _agent_refusal(me, "acquire")
    ttl = int(ttl_s) if ttl_s else default_ttl()
    if not has_remote():
        return {"acquired": True, "written": False, "me": me, "cycle": cycle, "status": "acquired",
                "reason": "no remote", "exit": EXIT_OK}
    try:
        base = _fetch()
    except LeaseError as e:
        return _refuse(me, "remote unreachable: %s" % e)
    d = describe(_read_at(base), me)
    if held_by_other(d):
        return dict(d, acquired=False, written=False, me=me, status="held", exit=EXIT_HELD,
                    reason="held by %s (%s) for %ss of %ss" % (d["runner"], d["cycle"], d["age_s"], d["ttl_s"]))
    head = _git("rev-parse", "HEAD").stdout.strip()
    if head != base:
        if _is_ancestor(head, base):
            return _refuse(me, "local HEAD is behind %s/%s %s; a runner on a stale ledger must not cycle"
                           % (remote(), branch(), base[:12]))
        if _is_ancestor(base, head):
            n = _git("rev-list", "--count", "%s..%s" % (base, head), check=False).stdout.strip() or "?"
            return _refuse(me, "local HEAD is ahead of %s/%s %s by %s unpushed commit(s); the lease never "
                           "publishes them (push or drop them through loop.sh/the operator first)"
                           % (remote(), branch(), base[:12], n))
        return _refuse(me, "local HEAD is diverged from %s/%s %s; a runner on a stale ledger must not cycle"
                       % (remote(), branch(), base[:12]))
    if _git("status", "--porcelain", "--", lease_relpath()).stdout.strip():
        return _refuse(me, "%s has local changes" % lease_relpath())
    payload = held_payload(me, cycle, ttl)
    try:
        commit = _lease_commit(base, payload, "cycle(lease): %s acquire" % me, me)
    except LeaseError as e:
        return _refuse(me, "could not build the lease commit: %s" % e)
    ok, err = _push(commit)
    if not ok:
        try:
            d2 = describe(_read_at(_fetch()), me)
        except LeaseError:
            d2 = describe({}, me)
        if held_by_other(d2):
            return dict(d2, acquired=False, written=False, me=me, status="held", exit=EXIT_HELD,
                        reason="lost the race: %s" % err)
        return _refuse(me, "push refused: %s" % err)
    result = {"acquired": True, "written": True, "me": me, "cycle": cycle, "sha": commit,
              "started": payload["started"], "ttl_s": ttl, "replaced": d["runner"],
              "status": "acquired", "exit": EXIT_OK}
    merged = _git("merge", "--ff-only", "--quiet", commit, check=False)
    if merged.returncode != 0:
        result["warning"] = "lease is on the remote but HEAD was not fast-forwarded: " + \
            _last_line(merged, "merge failed")
    return result


def release(*, runner: Optional[str] = None) -> dict:
    """Free the lease if this runner holds it. Exit 0 released (or nothing to release), 3 not ours, 4 not written."""
    me = runner or config.runner_id()
    if agent_context():
        return _agent_refusal(me, "release")
    if not has_remote():
        return {"released": False, "written": False, "me": me, "status": "released", "reason": "no remote",
                "exit": EXIT_OK}
    err = "push rejected"
    for _ in range(RELEASE_ATTEMPTS):
        try:
            base = _fetch()
        except LeaseError as e:
            return {"released": False, "written": False, "me": me, "status": "not_written",
                    "reason": "unreachable: %s" % e, "exit": EXIT_NOT_WRITTEN}
        d = describe(_read_at(base), me)
        if not d["held"]:
            return {"released": False, "written": False, "me": me, "status": "free",
                    "reason": "no lease held", "exit": EXIT_OK}
        if not d["mine"]:
            return {"released": False, "written": False, "me": me, "status": "not_ours",
                    "runner": d["runner"], "age_s": d["age_s"], "fresh": d["fresh"],
                    "reason": "held by %s, not ours" % d["runner"], "exit": EXIT_HELD}
        try:
            commit = _lease_commit(base, free_payload(me), "cycle(lease): %s release" % me, me)
        except LeaseError as e:
            return {"released": False, "written": False, "me": me, "status": "not_written",
                    "reason": "could not build the release commit: %s" % e, "exit": EXIT_NOT_WRITTEN}
        ok, err = _push(commit)
        if ok:
            return {"released": True, "written": True, "me": me, "sha": commit, "age_s": d["age_s"],
                    "status": "released", "exit": EXIT_OK}
        # the branch moved between fetch and push (another commit landed): rebuild on the new tip
    return {"released": False, "written": False, "me": me, "status": "not_written", "reason": err,
            "exit": EXIT_NOT_WRITTEN}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m trader.core.lease", description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=["check", "acquire", "release"])
    ap.add_argument("--cycle", default="research")
    ap.add_argument("--ttl", type=int, default=None, help="seconds (default protected fleet.lease_ttl_s)")
    ap.add_argument("--runner", default=None)
    a = ap.parse_args(argv)
    try:
        if a.cmd == "check":
            res = check(runner=a.runner)
        elif a.cmd == "acquire":
            res = acquire(a.cycle, runner=a.runner, ttl_s=a.ttl)
        else:
            res = release(runner=a.runner)
    except (LeaseError, subprocess.TimeoutExpired) as e:
        res = {"status": "not_written", "reason": str(e), "exit": EXIT_NOT_WRITTEN}
    print(json.dumps(res, sort_keys=True))
    return int(res.get("exit", EXIT_OK))


if __name__ == "__main__":
    sys.exit(main())
