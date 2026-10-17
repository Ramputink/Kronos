#!/usr/bin/env python3
"""Repo integrity guard for Phil-crypto (guard area, PROTECTED). Python >= 3.9, stdlib only, offline.

Enforces the agent/operator boundary of CONTRACTS.md section 7:

  - the agent-writable whitelist (everything else in the repo is operator-owned),
  - append-only journals (existing bytes of every JSONL / cycles.log identical to the parent),
  - add-only sweep results, immutable committed hypotheses, active.json ids never deleted or reused,
  - protected.json schema and hard ceilings (real caps, look budget, cutoff dates),
  - real_trading_enabled stays false unless an ``operator:`` commit set it,
  - journal well-formedness (every line parses, required keys, look budget, paper/real caps, cycles.log format),
  - strategy files (active.json caps, hypothesis hashes, schedule bounds),
  - every trader/**/*.py still compiles.

CLI (prints one JSON object, exit 0 ok / 1 errors / 2 usage):
  python trader/core/validate.py                              # config + journal + strategy + sources (+ real flag provenance)
  python trader/core/validate.py --diff BASE..HEAD            # + boundary check of every commit in the range (CI)
  python trader/core/validate.py --staged [--commit-msg-file F | --message M]   # + staged changes (pre-commit)
  python trader/core/validate.py --list-protected [--base REF] # whitelist + changed paths outside it (loop.sh)
  python trader/core/validate.py --enforce BASE               # loop.sh: undo every change outside the whitelist
  python trader/core/validate.py --cycle-stats BASE           # loop.sh: rows added since BASE (trials/looks/fills)

Trust model (see the regression tests in trader/tests/test_guard.py):
  - every git call runs hardened (no fsmonitor, no hooks, no excludes/attributes files, no sparse checkout,
    no replace objects), so state the agent's code may have planted in .git cannot hide a change;
  - work-tree changes are found by CONTENT (worktree bytes hashed against the base tree), not by `git diff`,
    so index flags (--skip-worktree / --assume-unchanged) cannot hide an edit;
  - untracked files under trader/, .githooks/ and .github/ are found by walking the file system, so .gitignore
    rules (trader/.gitignore has `data/`, the root has lib/, *.so ...) cannot hide a planted package such as
    trader/core/data/__init__.py that would shadow trader/core/data.py; __pycache__ there is always purged;
  - "operator" means an `operator:` subject AND, once .githooks/allowed_signers (or the out-of-repo anchor
    $PHIL_ALLOWED_SIGNERS / $PHIL_ALLOWED_SIGNERS_FILE) holds a key, a valid SSH signature from that key;
  - the trust anchor is never bootstrapped from the repo: a commit that changes .githooks/allowed_signers must
    carry a valid signature from a key that was ALREADY trusted (parent's file or out-of-repo anchor), so an
    unsigned subject-only `operator:` commit cannot plant the first key;
  - real_trading_enabled=true needs a signature that verifies against the OUT-OF-REPO anchor only
    ($PHIL_ALLOWED_SIGNERS / $PHIL_ALLOWED_SIGNERS_FILE); the in-repo file never counts for it;
  - every first-parent commit is checked against its first parent (merge commits included), and an unknown
    base (force push, new branch) checks the whole first-parent history since the guard existed.
"""
from __future__ import annotations

import datetime as _dt
import fnmatch
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from trader.core import config  # noqa: E402

# --------------------------------------------------------------------------------------------------
# Boundary definition (repo-relative paths, '/' separators). Patterns: '**' crosses directories,
# '*' does not.
# --------------------------------------------------------------------------------------------------
AGENT_WRITABLE = [
    "trader/strategy/**",
    "trader/journal/retros/**",
    "trader/journal/proposals.md",
    "trader/journal/*.jsonl",
    "trader/journal/cycles.log",
    "trader/journal/sweeps/**",
    "trader/journal/lease.json",
    "trader/journal/fleet-health.json",
]
APPEND_ONLY = [
    "trader/journal/*.jsonl",
    "trader/journal/cycles.log",
]
ADD_ONLY = [
    "trader/journal/sweeps/**",
]
# Inside the whitelist, these stay operator-only: anything Python can import or execute (the agent may run
# trader modules, so it must never be able to author one), and git control files that change how git sees
# the tree.
NOT_AGENT_WRITABLE = [
    "**/*.py", "**/*.pyc", "**/*.pyo", "**/*.pyd", "**/*.so", "**/*.pth", "**/*.sh",
    "**/.gitignore", "**/.gitattributes", "**/.gitmodules",
]
# Files whose rows only trusted code run by loop.sh (outside the claude session) may append: a session change
# to them is undone by --enforce. Commits may still append to them (the loop's own settlement commit does).
SESSION_PROTECTED = [
    "trader/journal/paper-ledger.jsonl",
    "trader/journal/real-ledger.jsonl",
]
HYPOTHESIS_FILES = "trader/strategy/hypotheses/H-*.json"
ACTIVE_FILE = "trader/strategy/active.json"
CONFIG_FILE = "trader/config/protected.json"
GUARD_FILE = "trader/core/validate.py"      # a commit whose first parent lacks it is pre-guard history
SIGNERS_FILE = ".githooks/allowed_signers"  # ssh allowed-signers for operator commits
# Local artefacts never considered a change. __pycache__ under the guarded roots is purged by --enforce
# (a planted .pyc with a matching source mtime/size would otherwise be imported instead of the source).
IGNORED = [
    "**/__pycache__/**",
    "trader/.loop.pid",
    "**/.DS_Store",
]
# Trees walked on disk for untracked files (ignore rules do not apply there).
GUARDED_ROOTS = ("trader", ".githooks", ".github")
IMPORTABLE_SUFFIXES = (".py", ".pyc", ".pyo", ".pyd", ".so", ".pth")
# Every git invocation: nothing configured in .git (or the user's config) may run code or hide paths.
GIT_HARDEN = [
    "--no-replace-objects",
    "-c", "core.fsmonitor=false", "-c", "core.hooksPath=/dev/null", "-c", "core.excludesFile=/dev/null",
    "-c", "core.attributesFile=/dev/null", "-c", "core.sparseCheckout=false", "-c", "core.untrackedCache=false",
    "-c", "gpg.ssh.program=ssh-keygen", "-c", "diff.noprefix=false",
]
WARNINGS: List[str] = []

OPERATOR_PREFIX = "operator:"
AGENT_KINDS = ("research", "forward", "deep_retro", "triggered", "research_sweep", "lease")
AGENT_MSG_RE = re.compile(r"^cycle\((%s)\): \S" % "|".join(AGENT_KINDS))
CYCLES_LOG_RE = re.compile(
    r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ (research|forward|deep_retro|triggered|research_sweep) "
    r"(ok|skipped|error|reverted)( \w+=\S+)*( \| .*)?$")
ISO_Z_RE = re.compile(r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ$")
STRATEGY_ID_RE = re.compile(r"^[a-z0-9_]{3,40}$")
ZERO_SHA_RE = re.compile(r"^0+$")
EMPTY_TREE = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"

# Hard ceilings that no protected.json edit may exceed (operator included: the operator edits the
# ceiling here, in a reviewed operator commit, not in the config).
HARD = {
    "real.per_order_eur": 100.0,
    "real.daily_eur": 250.0,
    "real.max_open_positions": 10,
    "vault.looks_per_month": 31,
    "vault.look_ttl_hours": 72,
    "screener.max_calls_per_day": 500,
    "research.max_trials_per_hypothesis": 100000,
}
EPS = 1e-6

TRIALS_COMMON = ("trial_id", "ts", "kind", "caller", "runner")
TRIALS_BY_KIND = {
    "single": ("spec_hash", "spec", "coin", "tf", "mode", "view", "period", "cost_side", "metrics"),
    "baseline": ("spec_hash", "spec", "coin", "tf", "mode", "view", "period", "cost_side", "metrics"),
    "batch": ("job_id", "hypothesis_id", "view", "n_configs", "n_trials", "sr_moments"),
}
LOOKS_REQ = ("look_id", "ts", "month", "hypothesis_id", "spec_hashes", "coins", "expires")
HYP_STATUSES = {"registered", "sweeping", "failed_trainval", "passed_trainval", "test_pass", "test_fail",
                "abandoned"}
REAL_STATUSES = {"dry_run", "refused", "submitted", "filled", "rejected"}


# --------------------------------------------------------------------------------------------------
# Path classification
# --------------------------------------------------------------------------------------------------
def _glob_re(pattern: str):
    out, i = "", 0
    while i < len(pattern):
        c = pattern[i]
        if pattern.startswith("**/", i):
            out += "(?:.*/)?"
            i += 3
            continue
        if pattern.startswith("**", i):
            out += ".*"
            i += 2
            continue
        if c == "*":
            out += "[^/]*"
        elif c == "?":
            out += "[^/]"
        else:
            out += re.escape(c)
        i += 1
    return re.compile("^" + out + "$")


_RE_CACHE: Dict[str, "re.Pattern"] = {}


def _match(path: str, patterns) -> bool:
    path = path.replace("\\", "/")
    if path.startswith("./"):
        path = path[2:]
    for p in patterns:
        rx = _RE_CACHE.get(p)
        if rx is None:
            rx = _RE_CACHE[p] = _glob_re(p)
        if rx.match(path):
            return True
    return False


def is_agent_writable(path: str) -> bool:
    """True if an agent commit may touch this repo-relative path."""
    return _match(path, AGENT_WRITABLE) and not _match(path, NOT_AGENT_WRITABLE)


def is_session_protected(path: str) -> bool:
    """Ledgers that only loop-run code may extend; agent-session changes to them are reverted."""
    return _match(path, SESSION_PROTECTED)


def is_append_only(path: str) -> bool:
    return _match(path, APPEND_ONLY)


def is_add_only(path: str) -> bool:
    return _match(path, ADD_ONLY)


def is_ignored(path: str) -> bool:
    return _match(path, IGNORED)


def is_hypothesis_file(path: str) -> bool:
    return fnmatch.fnmatchcase(path, HYPOTHESIS_FILES) and "/" not in path[len("trader/strategy/hypotheses/"):]


def is_operator_message(msg: Optional[str]) -> bool:
    return bool(msg) and msg.lstrip().startswith(OPERATOR_PREFIX)


# --------------------------------------------------------------------------------------------------
# git helpers
# --------------------------------------------------------------------------------------------------
def repo_root() -> Path:
    return config.REPO_ROOT


def _gitcmd(args, extra_cfg=()) -> List[str]:
    cfg = []
    for kv in extra_cfg:
        cfg += ["-c", kv]
    return ["git"] + GIT_HARDEN + cfg + ["-C", str(repo_root())] + list(args)


def _git(*args, check=True, binary=False) -> "str|bytes":
    p = subprocess.run(_gitcmd(args), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if check and p.returncode != 0:
        raise RuntimeError("git %s failed: %s" % (" ".join(args), p.stderr.decode(errors="replace").strip()))
    return p.stdout if binary else p.stdout.decode("utf-8", errors="replace")


def _git_ok(*args) -> bool:
    return subprocess.run(_gitcmd(args), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0


def in_git_repo() -> bool:
    return _git_ok("rev-parse", "--git-dir")


def _warn(msg: str) -> None:
    if msg not in WARNINGS:
        WARNINGS.append(msg)


def _blob(ref: Optional[str], path: str) -> Optional[bytes]:
    """Content of path at ref (None if missing). ref '' = index (staged), None = working tree."""
    if ref is None:
        f = repo_root() / path
        if f.is_symlink():
            return os.readlink(str(f)).encode()
        return f.read_bytes() if f.is_file() else None
    spec = ":%s" % path if ref == "" else "%s:%s" % (ref, path)
    p = subprocess.run(_gitcmd(["cat-file", "-p", spec]), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    return p.stdout if p.returncode == 0 else None


def _tree(ref: str) -> Dict[str, Tuple[str, str]]:
    """{path: (mode, blob sha)} of every blob/symlink in ref (submodules skipped)."""
    out = _git("ls-tree", "-r", "-z", "--full-tree", ref, binary=True).decode("utf-8", errors="surrogateescape")
    tree = {}
    for rec in out.split("\0"):
        if not rec:
            continue
        meta, path = rec.split("\t", 1)
        mode, typ, sha = meta.split()
        if typ == "blob":
            tree[path] = (mode, sha)
    return tree


def _wt_entry(path: str, algo: str = "sha1") -> Optional[Tuple[str, str]]:
    """(mode, blob sha) of the work-tree file as git would store it without filters; None if absent."""
    f = repo_root() / path
    try:
        st = os.lstat(str(f))
    except OSError:
        return None
    if stat.S_ISLNK(st.st_mode):
        data, mode = os.readlink(str(f)).encode(), "120000"
    elif stat.S_ISREG(st.st_mode):
        data = f.read_bytes()
        mode = "100755" if st.st_mode & stat.S_IXUSR else "100644"
    else:
        return None
    h = hashlib.new(algo)
    h.update(b"blob %d\0" % len(data))
    h.update(data)
    return mode, h.hexdigest()


def _name_status(args) -> List[Tuple[str, str]]:
    out = _git(*args)
    rows = []
    for line in out.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        rows.append((parts[0][:1], parts[-1]))
    return rows


def _rev_exists(ref: str) -> bool:
    return _git_ok("rev-parse", "--verify", "--quiet", ref + "^{commit}")


# --------------------------------------------------------------------------------------------------
# Change checks (shared by --diff, --staged and --enforce)
# --------------------------------------------------------------------------------------------------
def hypothesis_hash(h: dict) -> str:
    """Same formula as trials.hypothesis_hash (CONTRACTS section 5)."""
    body = {k: v for k, v in h.items() if k not in ("hash", "status")}
    return hashlib.sha256(config.canonical_json(body).encode()).hexdigest()


def _json_or_none(data: Optional[bytes]):
    if data is None:
        return None
    try:
        return json.loads(data.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return None


def _check_changes(changes: List[Tuple[str, str]], old_ref: Optional[str], new_ref: Optional[str],
                   is_agent: bool, label: str) -> List[str]:
    """changes = [(status A/M/D/T, path)]; old_ref/new_ref as in _blob."""
    errs = []
    for status, path in changes:
        if is_ignored(path):
            continue
        if is_agent and not is_agent_writable(path):
            errs.append("%s: agent change to protected path %s (%s)" % (label, path, status))
            if path == CONFIG_FILE:
                new = _json_or_none(_blob(new_ref, path)) if status != "D" else None
                if isinstance(new, dict) and (new.get("real") or {}).get("real_trading_enabled") is not False:
                    errs.append("%s: real_trading_enabled changed by a non-operator commit" % label)
            continue
        if not is_agent:
            continue
        if is_append_only(path) and status != "A":
            if status == "D":
                errs.append("%s: append-only file deleted: %s" % (label, path))
            else:
                old, new = _blob(old_ref, path) or b"", _blob(new_ref, path) or b""
                if not new.startswith(old):
                    errs.append("%s: append-only file rewritten (existing lines changed): %s" % (label, path))
        if is_add_only(path) and status != "A":
            errs.append("%s: add-only file modified or deleted: %s" % (label, path))
        if is_hypothesis_file(path):
            if status != "A":
                errs.append("%s: committed hypothesis is immutable (new id required): %s" % (label, path))
            else:
                h = _json_or_none(_blob(new_ref, path))
                if not isinstance(h, dict):
                    errs.append("%s: hypothesis is not valid JSON: %s" % (label, path))
                elif h.get("hash") != hypothesis_hash(h):
                    errs.append("%s: hypothesis hash mismatch: %s" % (label, path))
        if path == ACTIVE_FILE and status == "M":
            errs.extend("%s: %s" % (label, e) for e in _active_transition(
                _json_or_none(_blob(old_ref, path)), _json_or_none(_blob(new_ref, path))))
    return errs


def _active_transition(old, new) -> List[str]:
    """Entries of active.json are never deleted, ids never reused (retired stays retired)."""
    if not isinstance(old, dict) or not isinstance(new, dict):
        return []
    olds = {s.get("id"): s for s in old.get("strategies", []) if isinstance(s, dict)}
    news = {s.get("id"): s for s in new.get("strategies", []) if isinstance(s, dict)}
    errs = []
    for sid, s in olds.items():
        if sid not in news:
            errs.append("active.json: strategy entry %r deleted (retire it instead)" % sid)
        elif s.get("status") == "retired" and news[sid].get("status") != "retired":
            errs.append("active.json: retired strategy id %r reused" % sid)
    return errs


# --------------------------------------------------------------------------------------------------
# Operator identity: subject + SSH signature
# --------------------------------------------------------------------------------------------------
def anchor_signers() -> Optional[bytes]:
    """The out-of-repo trust anchor, or None when neither $PHIL_ALLOWED_SIGNERS nor $PHIL_ALLOWED_SIGNERS_FILE
    is set. A set-but-unreadable file yields b"" (no key: fails closed, never falls back to the repo)."""
    env = os.environ.get("PHIL_ALLOWED_SIGNERS")
    if env and env.strip():
        return env.encode()
    f = os.environ.get("PHIL_ALLOWED_SIGNERS_FILE")
    if f:
        try:
            return Path(f).expanduser().read_bytes()
        except OSError:
            return b""
    return None


def _signers_for(ref: Optional[str]) -> bytes:
    """Allowed signers trusted for a commit whose first parent is `ref`.

    An out-of-repo anchor wins (CI: repository variable PHIL_ALLOWED_SIGNERS; hosts: a file the agent cannot
    edit). Otherwise the committed .githooks/allowed_signers of the PARENT is used, never the commit's own
    version, so a commit cannot add the key that signs it."""
    anchor = anchor_signers()
    if anchor is not None:
        return anchor
    if not ref:
        return b""
    return _blob(ref, SIGNERS_FILE) or b""


def _has_keys(signers: bytes) -> bool:
    return any(l.strip() and not l.strip().startswith(b"#") for l in signers.splitlines())


def verify_signature(sha: str, signers: bytes) -> bool:
    """True iff `sha` carries an SSH signature that verifies against `signers` (allowed_signers format)."""
    if not _has_keys(signers):
        return False
    raw = _git("cat-file", "commit", sha, check=False)
    header = raw.split("\n\n", 1)[0]
    if "gpgsig -----BEGIN SSH SIGNATURE-----" not in header:
        return False
    fd, tmp = tempfile.mkstemp(prefix="phil-signers-")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(signers if signers.endswith(b"\n") else signers + b"\n")
        p = subprocess.run(_gitcmd(["verify-commit", sha], extra_cfg=("gpg.ssh.allowedSignersFile=" + tmp,)),
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return p.returncode == 0
    finally:
        os.unlink(tmp)


def operator_status(sha: str, subject: str, parent: Optional[str]) -> Tuple[bool, Optional[str]]:
    """(is_operator, problem). Unsigned `operator:` commits count only while no signer key is configured."""
    if not is_operator_message(subject):
        return False, None
    signers = _signers_for(parent)
    if not _has_keys(signers):
        _warn("operator signing not configured (%s has no key and PHIL_ALLOWED_SIGNERS is unset): "
              "'operator:' commits are recognised by subject only and can be forged" % SIGNERS_FILE)
        return True, None
    if verify_signature(sha, signers):
        return True, None
    return False, "'operator:' subject without a valid SSH signature from the allowed signers"


# --------------------------------------------------------------------------------------------------
# Commit ranges (CI) and staged changes (pre-commit)
# --------------------------------------------------------------------------------------------------
def _parents(sha: str) -> List[str]:
    return _git("rev-list", "--parents", "-n", "1", sha).split()[1:]


def check_diff(base: str, head: str) -> List[str]:
    """Boundary + append-only + immutability for every first-parent commit in base..head, each diffed
    against its FIRST parent, merge commits included (a merge brings the side branch's changes in through
    that diff). Unknown base (branch creation, force push): the whole first-parent history of head; commits
    whose first parent predates the guard (no trader/core/validate.py) are skipped."""
    errs = []
    if not _rev_exists(head):
        return ["--diff: unknown head %r" % head]
    if not base or ZERO_SHA_RE.match(base) or not _rev_exists(base):
        _warn("--diff: base %r unknown here; checking the whole first-parent history of %s" % (base, head))
        commits = _git("rev-list", "--first-parent", "--reverse", head).split()
    else:
        commits = _git("rev-list", "--first-parent", "--reverse", "%s..%s" % (base, head)).split()
    for sha in commits:
        parents = _parents(sha)
        if not parents or _blob(parents[0], GUARD_FILE) is None:
            continue  # pre-guard history
        parent = parents[0]
        subject = _git("log", "-1", "--format=%s", sha).strip()
        label = "commit %s (%r)" % (sha[:8], subject[:60])
        is_op, problem = operator_status(sha, subject, parent)
        if is_op:
            # The trust anchor must never be bootstrapped from the repo: changing the signers file needs a
            # signature from a key trusted BEFORE this commit (parent's file or the out-of-repo anchor).
            touched = _name_status(["diff-tree", "-r", "--no-commit-id", "--no-renames", "--name-status",
                                    parent, sha])
            if any(p == SIGNERS_FILE for _, p in touched):
                signers = _signers_for(parent)
                if not _has_keys(signers) or not verify_signature(sha, signers):
                    errs.append("%s: changes %s without a valid SSH signature from an already-trusted key "
                                "(%s); the first key must be signed against $PHIL_ALLOWED_SIGNERS"
                                % (label, SIGNERS_FILE, "none configured" if not _has_keys(signers)
                                   else "signature missing or not from a trusted key"))
            continue
        if problem:
            errs.append("%s: %s; checked as an agent commit" % (label, problem))
        elif not AGENT_MSG_RE.match(subject):
            errs.append("%s: agent commit subject must be 'cycle(<kind>): ...' or 'operator: ...'" % label)
        if len(parents) > 1:
            errs.append("%s: merge commit by a non-operator (agents never merge); its diff vs the first "
                        "parent is checked too" % label)
        changes = _name_status(["diff-tree", "-r", "--no-commit-id", "--no-renames", "--name-status",
                                parent, sha])
        errs.extend(_check_changes(changes, parent, sha, True, label))
    return errs


def check_staged(message: Optional[str]) -> List[str]:
    """pre-commit: staged changes vs HEAD (the first parent, also during a merge); agent rules unless the
    pending message starts with operator: (CI additionally requires the operator's signature)."""
    base = "HEAD" if _rev_exists("HEAD") else EMPTY_TREE
    changes = _name_status(["diff", "--cached", "--no-renames", "--name-status", base])
    agent = not is_operator_message(message)
    label = "staged (merge)" if _merge_in_progress() else "staged"
    errs = _check_changes(changes, base if base == "HEAD" else "__none__", "", agent, label)
    staged_cfg = _json_or_none(_blob("", CONFIG_FILE))
    if staged_cfg is not None and agent and (staged_cfg.get("real") or {}).get("real_trading_enabled") is True \
            and any(p == CONFIG_FILE for _, p in changes):
        errs.append("staged: real_trading_enabled=true may only be committed with an 'operator:' message")
    if agent and changes and message is not None and not AGENT_MSG_RE.match(message.strip()):
        errs.append("staged: agent commit message must be 'cycle(<kind>): ...' (or 'operator: ...')")
    return errs


def _merge_in_progress() -> bool:
    gd = Path(_git("rev-parse", "--git-dir").strip())
    if not gd.is_absolute():
        gd = repo_root() / gd
    return (gd / "MERGE_HEAD").exists()


# --------------------------------------------------------------------------------------------------
# Work-tree changes (loop.sh): content based, ignore rules and index flags do not apply
# --------------------------------------------------------------------------------------------------
def _skip_dirs() -> List[Path]:
    """Market-data stores (may live inside trader/ when TRADER_DATA_DIR points there): never walked.
    The config is read from HEAD, not the work tree (an edited protected.json must not hide a directory),
    and code / control directories are never skipped."""
    out = []
    cands = [os.environ.get("TRADER_DATA_DIR")]
    cfg = _json_or_none(_blob("HEAD", CONFIG_FILE)) if _rev_exists("HEAD") else None
    if isinstance(cfg, dict):
        cands += [_get(cfg, "data.dir"), _get(cfg, "data.seed_dir")]
    root = repo_root().resolve()
    never = [root / p for p in ("trader", "trader/core", "trader/config", "trader/fleet", "trader/tests",
                                "trader/strategy", "trader/journal", ".githooks", ".github")]
    for c in cands:
        if isinstance(c, str) and c:
            try:
                p = Path(c).expanduser().resolve()
            except OSError:
                continue
            if p == root or any(p == n or n in p.parents for n in never[1:]) or p == never[0]:
                continue
            out.append(p)
    return out


def _walk_files(top: Path, skip: List[Path]):
    """Yield every non-directory entry under top (symlinks not followed), pruning .git and data stores."""
    for dirpath, dirnames, filenames in os.walk(str(top)):
        keep = []
        for d in dirnames:
            full = Path(dirpath) / d
            if d == ".git":
                continue
            if full.is_symlink():
                filenames.append(d)  # a symlinked directory is an entry of its own
                continue
            try:
                rp = full.resolve()
            except OSError:
                rp = full
            if any(rp == s for s in skip):
                continue
            keep.append(d)
        dirnames[:] = keep
        for fn in filenames:
            yield Path(dirpath) / fn


def untracked_guarded(tree: Dict[str, Tuple[str, str]]) -> List[str]:
    """Files absent from `tree` under GUARDED_ROOTS (ignored ones included) plus untracked importable
    entries at the repo root (sys.path[0] for `python -m trader...`)."""
    root = repo_root()
    skip = _skip_dirs()
    tracked_dirs = {"/".join(p.split("/")[:i]) for p in tree for i in range(1, p.count("/") + 1)}
    out = set()
    for top in GUARDED_ROOTS:
        full = root / top
        if full.is_symlink():
            if top not in tree:
                out.add(top)
        elif full.is_dir():
            for f in _walk_files(full, skip):
                rel = f.relative_to(root).as_posix()
                if rel not in tree:
                    out.add(rel)
    try:
        names = os.listdir(str(root))
    except OSError:
        names = []
    for name in names:
        if name == ".git" or name in GUARDED_ROOTS or name in tree or name in tracked_dirs:
            continue
        full = root / name
        is_pkg = full.is_dir() and any((full / ("__init__" + s)).exists() for s in IMPORTABLE_SUFFIXES)
        if full.is_symlink() or full.is_file():
            if name.endswith(IMPORTABLE_SUFFIXES) or is_pkg:
                out.add(name)
        elif is_pkg:
            for f in _walk_files(full, skip):
                out.add(f.relative_to(root).as_posix())
    return sorted(out)


def changed_paths(base: str = "HEAD") -> List[str]:
    """Every path whose work-tree or index state differs from `base`, repo-wide:
    tracked files compared by content (index flags and fsmonitor cannot hide an edit), staged changes,
    untracked files (only per-directory .gitignore rules apply; info/exclude and core.excludesFile do not),
    and, under trader/, .githooks/, .github/ and the repo root, untracked files that ignore rules hide."""
    paths = set()
    tree: Dict[str, Tuple[str, str]] = {}
    if _rev_exists(base):
        tree = _tree(base)
        algo = "sha256" if any(len(v[1]) == 64 for v in list(tree.values())[:1]) else "sha1"
        for p, (mode, sha) in tree.items():
            if _wt_entry(p, algo) != (mode, sha):
                paths.add(p)
        for _, p in _name_status(["diff", "--cached", "--no-renames", "--name-status", base]):
            paths.add(p)
    for p in _git("ls-files", "-z", "--others", "--exclude-per-directory=.gitignore").split("\0"):
        if p:
            paths.add(p)
    paths.update(untracked_guarded(tree))
    return sorted(p for p in paths if not is_ignored(p))


def protected_changes(base: str = "HEAD") -> List[str]:
    return [p for p in changed_paths(base) if not is_agent_writable(p)]


def _reset_index(base: str) -> None:
    """Fresh index = base tree; no skip-worktree / assume-unchanged bit survives (they hide edits from git)."""
    _git("read-tree", base)
    flagged = []
    for line in _git("ls-files", "-v", "-z").split("\0"):
        if len(line) > 2 and (line[0].islower() or line[0] == "S"):
            flagged.append(line[2:])
    if flagged:
        _git("update-index", "--no-skip-worktree", "--no-assume-unchanged", "--", *flagged)


def purge_bytecode() -> List[str]:
    """Delete every __pycache__ directory under the guarded roots and at the repo root."""
    root = repo_root()
    gone = []
    if (root / "__pycache__").is_dir() and not (root / "__pycache__").is_symlink():
        shutil.rmtree(str(root / "__pycache__"), ignore_errors=True)
        gone.append("__pycache__")
    skip = _skip_dirs()
    for t in GUARDED_ROOTS:
        top = root / t
        if not top.is_dir() or top.is_symlink():
            continue
        for dirpath, dirnames, _ in os.walk(str(top)):
            for d in list(dirnames):
                full = Path(dirpath) / d
                if full.is_symlink() or any(full.resolve() == s for s in skip):
                    dirnames.remove(d)
                elif d == "__pycache__":
                    shutil.rmtree(str(full), ignore_errors=True)
                    gone.append(full.relative_to(root).as_posix())
                    dirnames.remove(d)
    return gone


def _remove(path: str) -> None:
    root = repo_root()
    f = root / path
    if f.is_symlink() or f.is_file():
        f.unlink()
    elif f.is_dir():
        shutil.rmtree(str(f), ignore_errors=True)
    d = f.parent
    while d != root and root in d.parents:
        try:
            d.rmdir()  # only if empty
        except OSError:
            break
        d = d.parent


def _restore(path: str, mode: str, sha: str) -> None:
    root = repo_root()
    f = root / path
    data = _git("cat-file", "blob", sha, binary=True)
    if f.is_symlink() or f.is_file():
        f.unlink()
    elif f.is_dir():
        shutil.rmtree(str(f), ignore_errors=True)
    f.parent.mkdir(parents=True, exist_ok=True)
    if mode == "120000":
        os.symlink(data.decode("utf-8", errors="surrogateescape"), str(f))
        return
    f.write_bytes(data)
    os.chmod(str(f), 0o755 if mode == "100755" else 0o644)


def enforce(base: str) -> dict:
    """loop.sh: restore every change since `base` that an agent session may not make. Mutates the worktree
    and resets the index to `base` (loop.sh re-stages the whitelist afterwards).

    - path outside the whitelist (or a code / git-control file inside it): restored to base or removed,
      ignored files under trader/, .githooks/, .github/ included,
    - session-protected ledgers (paper / real): restored to base (only loop-run code appends to them),
    - append-only file whose existing bytes changed / deleted: restored to base,
    - add-only file modified/deleted, committed hypothesis modified/deleted: restored to base,
    - every __pycache__ under the guarded roots: deleted.
    Repeats until stable (removing an untracked .gitignore can reveal files it hid).
    Returns {"reverted": [...], "restored": [...], "bytecode_purged": n}.
    """
    reverted, restored = [], []
    _reset_index(base)
    purged = purge_bytecode()
    tree = _tree(base)
    for _ in range(4):
        acted = False
        for path in changed_paths(base):
            entry = tree.get(path)
            bad_protected = not is_agent_writable(path)
            bad_session = not bad_protected and is_session_protected(path)
            bad_rule = False
            if not (bad_protected or bad_session) and entry is not None:
                cur = _blob(None, path)
                old = _git("cat-file", "blob", entry[1], binary=True)
                if is_append_only(path):
                    bad_rule = cur is None or not cur.startswith(old)
                elif is_add_only(path) or is_hypothesis_file(path):
                    bad_rule = cur != old
            if not (bad_protected or bad_session or bad_rule):
                continue
            if entry is not None:
                _restore(path, *entry)
            else:
                _remove(path)
            acted = True
            lst = reverted if (bad_protected or bad_session) else restored
            if path not in lst:
                lst.append(path)
        if not acted:
            break
    _reset_index(base)
    return {"reverted": reverted, "restored": restored, "bytecode_purged": len(purged)}


def cycle_stats(base: str) -> dict:
    """Rows appended since base: trials (single+batch+baseline), looks, paper fills."""
    def added(rel):
        old = _blob(base, rel) or b""
        cur = _blob(None, rel) or b""
        tail = cur[len(old):] if cur.startswith(old) else cur
        return [l for l in tail.decode("utf-8", errors="replace").splitlines() if l.strip()]
    fills = 0
    for line in added("trader/journal/paper-ledger.jsonl"):
        try:
            fills += json.loads(line).get("type") == "fill"
        except ValueError:
            pass
    return {"trials": len(added("trader/journal/trials.jsonl")), "looks": len(added("trader/journal/looks.jsonl")),
            "fills": int(fills)}


# --------------------------------------------------------------------------------------------------
# Config checks
# --------------------------------------------------------------------------------------------------
def _get(d, dotted, default=None):
    cur = d
    for k in dotted.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _date(s) -> Optional[_dt.date]:
    try:
        return _dt.datetime.strptime(s, "%Y-%m-%d").date()
    except (TypeError, ValueError):
        return None


def check_config(cfg: dict) -> List[str]:
    """Schema + hard ceilings of protected.json."""
    e = []
    if not isinstance(cfg, dict):
        return ["config: not a JSON object"]
    for k in ("costs", "vault", "forward", "data", "universe", "timeframes", "bars_per_year", "paper", "real",
              "promotion", "screener", "research", "schedule", "fleet"):
        if k not in cfg:
            e.append("config: missing key %r" % k)
    # costs >= 0
    def walk(node, prefix):
        if isinstance(node, dict):
            for k, v in node.items():
                walk(v, prefix + "." + k)
        elif _num(node) and node < 0:
            e.append("config: negative cost %s=%r" % (prefix, node))
    walk(cfg.get("costs", {}), "costs")
    for k in ("costs.spot.fee_side", "costs.spot.slippage_side"):
        if not _num(_get(cfg, k)):
            e.append("config: %s must be a number" % k)
    if _get(cfg, "costs.perp.available_in_eu") is not False:
        e.append("config: costs.perp.available_in_eu must be false (Bybit EU = spot only)")
    # vault
    cutoff, train_end = _date(_get(cfg, "vault.vault_cutoff")), _date(_get(cfg, "vault.train_end"))
    if cutoff is None:
        e.append("config: vault.vault_cutoff does not parse as YYYY-MM-DD")
    if train_end is None:
        e.append("config: vault.train_end does not parse as YYYY-MM-DD")
    if cutoff and train_end and not train_end < cutoff:
        e.append("config: vault.train_end must be before vault.vault_cutoff")
    lpm = _get(cfg, "vault.looks_per_month")
    if not isinstance(lpm, int) or isinstance(lpm, bool) or not 0 <= lpm <= HARD["vault.looks_per_month"]:
        e.append("config: vault.looks_per_month must be an int in [0, %d]" % HARD["vault.looks_per_month"])
    ttl = _get(cfg, "vault.look_ttl_hours")
    if not _num(ttl) or not 0 < ttl <= HARD["vault.look_ttl_hours"]:
        e.append("config: vault.look_ttl_hours must be in (0, %d]" % HARD["vault.look_ttl_hours"])
    wd = _get(cfg, "vault.warmup_days")
    if not isinstance(wd, int) or isinstance(wd, bool) or wd < 0:
        e.append("config: vault.warmup_days must be a non-negative int")
    fs = _get(cfg, "forward.forward_start")
    if fs is not None:
        d = _date(fs)
        if d is None:
            e.append("config: forward.forward_start must be null or YYYY-MM-DD")
        elif cutoff and d < cutoff:
            e.append("config: forward.forward_start must not be before vault_cutoff")
    # universe / timeframes
    uni = cfg.get("universe")
    if not isinstance(uni, list) or not uni or len(set(uni)) != len(uni) or \
            not all(isinstance(c, str) and re.match(r"^[A-Z0-9]{2,12}$", c) for c in uni):
        e.append("config: universe must be a non-empty list of unique uppercase symbols")
        uni = uni if isinstance(uni, list) else []
    tfs = cfg.get("timeframes") if isinstance(cfg.get("timeframes"), list) else []
    for tf in tfs:
        if not _num(_get(cfg, "bars_per_year." + tf)):
            e.append("config: bars_per_year missing for timeframe %r" % tf)
    # paper
    bank = _get(cfg, "paper.bankroll_usdt")
    if not _num(bank) or bank <= 0:
        e.append("config: paper.bankroll_usdt must be > 0")
    ms = _get(cfg, "paper.max_strategies")
    if not isinstance(ms, int) or isinstance(ms, bool) or ms < 1:
        e.append("config: paper.max_strategies must be an int >= 1")
    mas = _get(cfg, "paper.max_allocation_per_strategy_usdt")
    if not _num(mas) or mas <= 0 or (_num(bank) and mas > bank + EPS):
        e.append("config: paper.max_allocation_per_strategy_usdt must be in (0, bankroll]")
    modes = _get(cfg, "paper.allowed_modes")
    if not isinstance(modes, list) or not set(modes) <= {"SPOT"}:
        e.append("config: paper.allowed_modes must be a subset of ['SPOT'] (EU = spot only)")
    ptf = _get(cfg, "paper.allowed_timeframes")
    if not isinstance(ptf, list) or not set(ptf) <= set(tfs):
        e.append("config: paper.allowed_timeframes must be a subset of timeframes")
    if not _num(_get(cfg, "paper.min_order_usdt")) or _get(cfg, "paper.min_order_usdt") < 0:
        e.append("config: paper.min_order_usdt must be >= 0")
    # real
    rte = _get(cfg, "real.real_trading_enabled")
    if not isinstance(rte, bool):
        e.append("config: real.real_trading_enabled must be a JSON boolean")
    for key in ("real.per_order_eur", "real.daily_eur"):
        v = _get(cfg, key)
        if not _num(v) or v < 0 or v > HARD[key]:
            e.append("config: %s must be in [0, %s] (hard ceiling)" % (key, HARD[key]))
    if _num(_get(cfg, "real.per_order_eur")) and _num(_get(cfg, "real.daily_eur")) and \
            _get(cfg, "real.per_order_eur") > _get(cfg, "real.daily_eur"):
        e.append("config: real.per_order_eur must not exceed real.daily_eur")
    mop = _get(cfg, "real.max_open_positions")
    if not isinstance(mop, int) or isinstance(mop, bool) or not 0 <= mop <= HARD["real.max_open_positions"]:
        e.append("config: real.max_open_positions must be an int in [0, %d]" % HARD["real.max_open_positions"])
    if _get(cfg, "real.category") != "spot":
        e.append("config: real.category must be 'spot'")
    ac = _get(cfg, "real.allowed_coins")
    if not isinstance(ac, list) or not set(ac) <= set(uni):
        e.append("config: real.allowed_coins must be a subset of universe")
    if not isinstance(_get(cfg, "real.allowed_strategy_ids"), list):
        e.append("config: real.allowed_strategy_ids must be a list")
    if _get(cfg, "real.require_no_withdraw_permission") is not True:
        e.append("config: real.require_no_withdraw_permission must stay true")
    # promotion / screener / research
    pm = _get(cfg, "promotion.random_percentile_min")
    if not _num(pm) or not 0.5 <= pm <= 1:
        e.append("config: promotion.random_percentile_min must be in [0.5, 1]")
    mfd = _get(cfg, "promotion.min_forward_days")
    if not _num(mfd) or mfd < 30:
        e.append("config: promotion.min_forward_days must be >= 30")
    mc = _get(cfg, "screener.max_calls_per_day")
    if not isinstance(mc, int) or isinstance(mc, bool) or not 0 <= mc <= HARD["screener.max_calls_per_day"]:
        e.append("config: screener.max_calls_per_day must be an int in [0, %d]" % HARD["screener.max_calls_per_day"])
    mt = _get(cfg, "research.max_trials_per_hypothesis")
    if not isinstance(mt, int) or isinstance(mt, bool) or not 1 <= mt <= HARD["research.max_trials_per_hypothesis"]:
        e.append("config: research.max_trials_per_hypothesis must be an int in [1, %d]"
                 % HARD["research.max_trials_per_hypothesis"])
    reh = _get(cfg, "schedule.research_every_hours")
    if not _num(reh) or reh <= 0:
        e.append("config: schedule.research_every_hours must be > 0")
    for k in ("schedule.forward_cycle", "schedule.deep_retro"):
        if not re.match(r"^\d\d:\d\d$", str(_get(cfg, k))):
            e.append("config: %s must be HH:MM" % k)
    return e


def check_real_flag(cfg: dict) -> List[str]:
    """real_trading_enabled=true is only valid if the last commit that changed protected.json is an
    `operator:` commit carrying a valid SSH signature from the OUT-OF-REPO anchor ($PHIL_ALLOWED_SIGNERS or
    $PHIL_ALLOWED_SIGNERS_FILE; the in-repo signers file is ignored here), and the working copy equals HEAD's version byte for byte (content comparison:
    index flags cannot hide an uncommitted edit)."""
    if _get(cfg, "real.real_trading_enabled") is not True:
        return []
    if not in_git_repo():
        return ["real_trading_enabled is true but provenance cannot be checked (not a git repo)"]
    rel = CONFIG_FILE
    if not _rev_exists("HEAD") or _blob(None, rel) != _blob("HEAD", rel):
        return ["real_trading_enabled is true in an uncommitted protected.json"]
    sha = _git("log", "-1", "--format=%H", "HEAD", "--", rel, check=False).strip()
    subj = _git("log", "-1", "--format=%s", sha, check=False).strip() if sha else ""
    if not sha or not is_operator_message(subj):
        return ["real_trading_enabled is true but the last commit touching protected.json is not 'operator:' (%r)"
                % subj[:60]]
    # Out-of-repo anchor ONLY: keys committed to the repo (even through a chain of signed commits) are never
    # enough to go live, so no sequence of pushed commits can enable real trading on its own.
    signers = anchor_signers() or b""
    if not _has_keys(signers):
        return ["real_trading_enabled is true but operator signing is not configured: the commit that set it "
                "must be SSH-signed by a key from the out-of-repo anchor $PHIL_ALLOWED_SIGNERS or "
                "$PHIL_ALLOWED_SIGNERS_FILE (%s in the repo does not count)" % SIGNERS_FILE]
    if not verify_signature(sha, signers):
        return ["real_trading_enabled is true but commit %s that set it has no valid operator SSH signature"
                % sha[:8]]
    return []


# --------------------------------------------------------------------------------------------------
# Journal checks
# --------------------------------------------------------------------------------------------------
def _strict_jsonl(path: Path, errs: List[str]) -> List[dict]:
    rows = []
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as ex:
        errs.append("journal: cannot read %s: %s" % (path.name, ex))
        return rows
    for i, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except ValueError:
            errs.append("journal: %s line %d is not valid JSON" % (path.name, i))
            continue
        if not isinstance(obj, dict):
            errs.append("journal: %s line %d is not a JSON object" % (path.name, i))
            continue
        obj["__line"] = i
        rows.append(obj)
    if text and not text.endswith("\n"):
        errs.append("journal: %s does not end with a newline (torn last line?)" % path.name)
    return rows


def _iso_ok(v) -> bool:
    return isinstance(v, str) and bool(ISO_Z_RE.match(v))


def _check_trials(rows, errs):
    seen = set()
    for r in rows:
        where = "trials.jsonl line %d" % r["__line"]
        miss = [k for k in TRIALS_COMMON if k not in r]
        kind = r.get("kind")
        if kind not in TRIALS_BY_KIND:
            errs.append("journal: %s has unknown kind %r" % (where, kind))
        else:
            miss += [k for k in TRIALS_BY_KIND[kind] if k not in r]
        if miss:
            errs.append("journal: %s missing keys %s" % (where, miss))
        tid = r.get("trial_id")
        if tid in seen:
            errs.append("journal: %s duplicate trial_id %r" % (where, tid))
        seen.add(tid)
        if "ts" in r and not _iso_ok(r["ts"]):
            errs.append("journal: %s ts not ISO-Z" % where)
        if _num(r.get("cost_side")) and r["cost_side"] < 0:
            errs.append("journal: %s negative cost_side" % where)
        if r.get("period") == "test" and r.get("view") != "full":
            errs.append("journal: %s test period evaluated outside the full view" % where)
        if r.get("view") == "full" and r.get("period") == "test" and kind != "baseline" and not r.get("look_id"):
            errs.append("journal: %s sealed TEST evaluated without a look_id" % where)
        if r.get("view") == "research" and r.get("period") in ("test", "forward"):
            errs.append("journal: %s research view with sealed period %r" % (where, r.get("period")))


def _check_looks(rows, cfg, errs):
    lpm = _get(cfg, "vault.looks_per_month", 0) or 0
    seen, per_month, per_hyp = set(), {}, {}
    for r in rows:
        where = "looks.jsonl line %d" % r["__line"]
        miss = [k for k in LOOKS_REQ if k not in r]
        if miss:
            errs.append("journal: %s missing keys %s" % (where, miss))
        lid = r.get("look_id")
        if lid in seen:
            errs.append("journal: %s duplicate look_id %r" % (where, lid))
        seen.add(lid)
        m = r.get("month")
        per_month[m] = per_month.get(m, 0) + 1
        h = r.get("hypothesis_id")
        per_hyp[h] = per_hyp.get(h, 0) + 1
    for m, n in per_month.items():
        if n > lpm:
            errs.append("journal: looks budget exceeded in %s (%d > %d)" % (m, n, lpm))
    for h, n in per_hyp.items():
        if n > 1:
            errs.append("journal: hypothesis %s has %d looks (max 1)" % (h, n))


def _check_paper(rows, cfg, errs):
    uni = set(cfg.get("universe") or [])
    min_order = _get(cfg, "paper.min_order_usdt", 0) or 0
    cap = _get(cfg, "paper.max_allocation_per_strategy_usdt", 0) or 0
    max_strats = _get(cfg, "paper.max_strategies", 0) or 0
    ok_runs, init_by_strat, strats_by_run = {}, {}, {}
    for r in rows:
        where = "paper-ledger.jsonl line %d" % r["__line"]
        t = r.get("type")
        if t not in ("run", "fill", "skip", "mark"):
            errs.append("journal: %s unknown type %r" % (where, t))
            continue
        if t == "run" and r.get("status") == "ok":
            ab = r.get("asof_bar")
            if ab in ok_runs:
                errs.append("journal: %s second ok paper run for asof_bar %s" % (where, ab))
            ok_runs[ab] = r.get("run_id")
        if t == "fill":
            if r.get("coin") not in uni:
                errs.append("journal: %s fill coin %r outside universe" % (where, r.get("coin")))
            if r.get("side") not in ("buy", "sell"):
                errs.append("journal: %s fill side %r" % (where, r.get("side")))
            n = r.get("notional_usdt")
            if not _num(n) or n < min_order - EPS:
                errs.append("journal: %s fill notional %r below min_order_usdt %s" % (where, n, min_order))
            if not _num(r.get("fee_usdt")) or r.get("fee_usdt") < 0:
                errs.append("journal: %s fill fee_usdt invalid" % where)
            sid = r.get("strategy_id")
            strats_by_run.setdefault(r.get("run_id"), set()).add(sid)
            if r.get("reason") == "init" and _num(n):
                init_by_strat[sid] = init_by_strat.get(sid, 0.0) + n
    for sid, tot in init_by_strat.items():
        if tot > cap + 0.01:
            errs.append("journal: strategy %s initial fills %.2f exceed per-strategy cap %s" % (sid, tot, cap))
    for run, s in strats_by_run.items():
        if len(s) > max_strats:
            errs.append("journal: paper run %s traded %d strategies (> max %d)" % (run, len(s), max_strats))


def _check_real(rows, cfg, errs):
    per_order = _get(cfg, "real.per_order_eur", 0) or 0
    daily = _get(cfg, "real.daily_eur", 0) or 0
    allowed = set(_get(cfg, "real.allowed_coins", []) or [])
    spent = {}
    for r in rows:
        where = "real-ledger.jsonl line %d" % r["__line"]
        st = r.get("status")
        if st not in REAL_STATUSES:
            errs.append("journal: %s unknown status %r" % (where, st))
        if st in ("submitted", "filled"):
            n = r.get("notional_eur")
            if not _num(n) or n > per_order + EPS:
                errs.append("journal: %s order %r EUR above per_order_eur %s" % (where, n, per_order))
            if r.get("coin") not in allowed:
                errs.append("journal: %s real order on non-allowed coin %r" % (where, r.get("coin")))
            if _num(n) and st == "submitted":
                day = str(r.get("ts", ""))[:10]
                spent[day] = spent.get(day, 0.0) + n
    for day, tot in spent.items():
        if tot > daily + EPS:
            errs.append("journal: real spend %.2f EUR on %s above daily_eur %s" % (tot, day, daily))


def _check_forecasts(rows, errs):
    for r in rows:
        where = "forecasts.jsonl line %d" % r["__line"]
        t = r.get("type")
        if t == "forecast":
            p = r.get("p")
            if not isinstance(p, list) or not p or not all(_num(x) and 0 <= x <= 1 for x in p) \
                    or abs(sum(p) - 1) > 1e-3:
                errs.append("journal: %s p must be probabilities summing to 1" % where)
        elif t != "resolve":
            errs.append("journal: %s unknown type %r" % (where, t))


def check_journal(cfg: Optional[dict] = None) -> List[str]:
    cfg = cfg if cfg is not None else config.load_config()
    errs: List[str] = []
    jd = config.journal_dir()
    if not jd.exists():
        return errs
    for f in sorted(jd.glob("*.jsonl")):
        rows = _strict_jsonl(f, errs)
        if f.name == "trials.jsonl":
            _check_trials(rows, errs)
        elif f.name == "looks.jsonl":
            _check_looks(rows, cfg, errs)
        elif f.name == "hypotheses.jsonl":
            for r in rows:
                if r.get("status") not in HYP_STATUSES:
                    errs.append("journal: hypotheses.jsonl line %d unknown status %r" % (r["__line"], r.get("status")))
        elif f.name == "paper-ledger.jsonl":
            _check_paper(rows, cfg, errs)
        elif f.name == "real-ledger.jsonl":
            _check_real(rows, cfg, errs)
        elif f.name == "forecasts.jsonl":
            _check_forecasts(rows, errs)
    cl = jd / "cycles.log"
    if cl.exists():
        for i, line in enumerate(cl.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            if line.strip() and not CYCLES_LOG_RE.match(line):
                errs.append("journal: cycles.log line %d does not match the contract format" % i)
    for name in ("lease.json", "fleet-health.json"):
        f = jd / name
        if f.exists() and _json_or_none(f.read_bytes()) is None:
            errs.append("journal: %s is not valid JSON" % name)
    sw = jd / "sweeps"
    if sw.exists():
        for f in sorted(sw.glob("*.json")):
            if _json_or_none(f.read_bytes()) is None:
                errs.append("journal: sweeps/%s is not valid JSON" % f.name)
    return errs


# --------------------------------------------------------------------------------------------------
# Strategy checks
# --------------------------------------------------------------------------------------------------
def _basic_active(active, cfg) -> List[str]:
    e = []
    if not isinstance(active, dict) or not isinstance(active.get("strategies"), list):
        return ["active.json: must be an object with a 'strategies' list"]
    uni = set(cfg.get("universe") or [])
    modes = set(_get(cfg, "paper.allowed_modes", []) or [])
    tfs = set(_get(cfg, "paper.allowed_timeframes", []) or [])
    bank = _get(cfg, "paper.bankroll_usdt", 0) or 0
    cap = _get(cfg, "paper.max_allocation_per_strategy_usdt", 0) or 0
    maxc = _get(cfg, "paper.max_coins_per_strategy", 10 ** 6) or 10 ** 6
    ids, n_paper, total = set(), 0, 0.0
    for s in active["strategies"]:
        if not isinstance(s, dict):
            e.append("active.json: strategy entry is not an object")
            continue
        sid = s.get("id")
        if not isinstance(sid, str) or not STRATEGY_ID_RE.match(sid):
            e.append("active.json: bad strategy id %r" % (sid,))
        if sid in ids:
            e.append("active.json: duplicate strategy id %r" % (sid,))
        ids.add(sid)
        st = s.get("status")
        if st not in ("paper", "paused", "retired"):
            e.append("active.json: %s bad status %r" % (sid, st))
        spec = s.get("spec") or {}
        coins = s.get("coins") or []
        if st != "retired":
            if spec.get("mode") not in modes:
                e.append("active.json: %s mode %r not allowed for paper" % (sid, spec.get("mode")))
            if spec.get("tf") not in tfs:
                e.append("active.json: %s timeframe %r not allowed for paper" % (sid, spec.get("tf")))
            if not set(coins) <= uni:
                e.append("active.json: %s coins outside universe: %s" % (sid, sorted(set(coins) - uni)))
            if len(coins) > maxc:
                e.append("active.json: %s has %d coins (> %d)" % (sid, len(coins), maxc))
            alloc = s.get("allocation_usdt")
            if not _num(alloc) or alloc <= 0 or alloc > cap + EPS:
                e.append("active.json: %s allocation %r outside (0, %s]" % (sid, alloc, cap))
            elif st == "paper":
                total += alloc
        if st == "paper":
            n_paper += 1
    if n_paper > (_get(cfg, "paper.max_strategies", 0) or 0):
        e.append("active.json: %d paper strategies (> max_strategies)" % n_paper)
    if total > bank + EPS:
        e.append("active.json: total paper allocation %.2f exceeds bankroll %s" % (total, bank))
    return e


def check_strategy(cfg: Optional[dict] = None) -> Tuple[List[str], List[str]]:
    """Returns (errors, warnings)."""
    cfg = cfg if cfg is not None else config.load_config()
    errs, warns = [], []
    sd = config.strategy_dir()
    if not sd.exists():
        return errs, warns
    for f in sorted(sd.rglob("*.json")):
        if "__pycache__" in f.parts:
            continue
        if _json_or_none(f.read_bytes()) is None:
            errs.append("strategy: %s is not valid JSON" % f.relative_to(sd))
    active_f = sd / "active.json"
    active = _json_or_none(active_f.read_bytes()) if active_f.exists() else None
    if active is not None:
        errs.extend(_basic_active(active, cfg))
        try:
            from trader.core import paper  # noqa: WPS433 (lazy: pandas may be missing on a hook python)
            errs.extend("active.json: %s" % m for m in paper.validate_active(active)
                        if "active.json: %s" % m not in errs)
        except ImportError as ex:
            warns.append("paper.validate_active unavailable (%s); basic active.json checks only" % ex)
        except Exception as ex:  # a crash in validate_active must not hide as success
            errs.append("active.json: paper.validate_active raised %s: %s" % (type(ex).__name__, ex))
    hd = sd / "hypotheses"
    if hd.exists():
        for f in sorted(hd.glob("H-*.json")):
            h = _json_or_none(f.read_bytes())
            if not isinstance(h, dict):
                continue
            if not isinstance(h.get("id"), str) or not f.stem.startswith(h["id"]):
                errs.append("strategy: %s id %r does not match the file name" % (f.name, h.get("id")))
            if h.get("hash") != hypothesis_hash(h):
                errs.append("strategy: %s hash mismatch (hypotheses are immutable)" % f.name)
    sch_f = sd / "schedule.json"
    sch = _json_or_none(sch_f.read_bytes()) if sch_f.exists() else None
    if isinstance(sch, dict):
        eh = _get(sch, "research.every_hours")
        floor = _get(cfg, "schedule.research_every_hours", 0) or 0
        if eh is not None and (not _num(eh) or eh < floor):
            errs.append("strategy: schedule.json research.every_hours %r below protected %s" % (eh, floor))
        sc = _get(sch, "screener.coins")
        if isinstance(sc, list) and not set(sc) <= set(cfg.get("universe") or []):
            errs.append("strategy: schedule.json screener.coins outside universe")
    return errs, warns


# --------------------------------------------------------------------------------------------------
# Sources
# --------------------------------------------------------------------------------------------------
def check_sources() -> List[str]:
    errs = []
    for f in sorted(config.TRADER_ROOT.rglob("*.py")):
        if "__pycache__" in f.parts:
            continue
        try:
            compile(f.read_text(encoding="utf-8"), str(f), "exec")
        except (SyntaxError, UnicodeDecodeError, ValueError) as ex:
            errs.append("source: %s does not compile: %s" % (f.relative_to(config.TRADER_ROOT), ex))
    return errs


# --------------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------------
def _emit(obj, code):
    print(json.dumps(obj, indent=None, sort_keys=False))
    return code


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="validate.py", description="Phil-crypto integrity guard")
    ap.add_argument("--diff", help="BASE..HEAD commit range to boundary-check")
    ap.add_argument("--staged", action="store_true", help="check staged changes (pre-commit)")
    ap.add_argument("--commit-msg-file", help="pending commit message (operator: prefix = operator commit)")
    ap.add_argument("--message", help="pending commit message given inline")
    ap.add_argument("--list-protected", action="store_true", help="print whitelist and protected changes")
    ap.add_argument("--base", default="HEAD", help="base ref for --list-protected (default HEAD)")
    ap.add_argument("--enforce", metavar="BASE", help="revert every non-agent change since BASE (loop.sh)")
    ap.add_argument("--cycle-stats", metavar="BASE", help="rows appended since BASE")
    ap.add_argument("--quick", action="store_true", help="skip journal/strategy/source checks")
    a = ap.parse_args(argv)

    try:
        if a.list_protected:
            changed = protected_changes(a.base) if in_git_repo() else []
            return _emit({"ok": not changed, "errors": [], "agent_writable": AGENT_WRITABLE,
                          "append_only": APPEND_ONLY, "add_only": ADD_ONLY,
                          "protected_changed": changed}, 0)
        if a.enforce:
            res = enforce(a.enforce)
            res.update({"ok": True, "errors": []})
            return _emit(res, 0)
        if a.cycle_stats:
            res = cycle_stats(a.cycle_stats)
            res.update({"ok": True, "errors": []})
            return _emit(res, 0)
    except RuntimeError as ex:
        return _emit({"ok": False, "errors": [str(ex)]}, 1)

    errors: List[str] = []
    warnings: List[str] = []
    msg = a.message
    if a.commit_msg_file:
        try:
            msg = Path(a.commit_msg_file).read_text(encoding="utf-8", errors="replace")
            msg = "\n".join(l for l in msg.splitlines() if not l.startswith("#")).strip()
        except OSError as ex:
            errors.append("cannot read commit message file: %s" % ex)
    git = in_git_repo()

    # config: staged version in pre-commit mode, otherwise the working copy
    cfg = None
    if a.staged and git:
        cfg = _json_or_none(_blob("", CONFIG_FILE))
    if cfg is None:
        try:
            cfg = config.load_config()
        except (OSError, ValueError) as ex:
            errors.append("config: cannot load %s: %s" % (config.config_path(), ex))
            cfg = {}
    if cfg:
        errors += check_config(cfg)
        if git and not a.staged:
            errors += check_real_flag(cfg)
    try:
        if a.diff:
            if ".." not in a.diff:
                return _emit({"ok": False, "errors": ["--diff expects BASE..HEAD"]}, 2)
            base, head = a.diff.split("..", 1)
            errors += check_diff(base, head or "HEAD")
        if a.staged:
            if not git:
                errors.append("--staged outside a git repository")
            else:
                errors += check_staged(msg)
    except RuntimeError as ex:
        errors.append(str(ex))
    if not a.quick:
        errors += check_journal(cfg)
        se, sw = check_strategy(cfg)
        errors += se
        warnings += sw
        errors += check_sources()
    warnings += [w for w in WARNINGS if w not in warnings]
    out = {"ok": not errors, "errors": errors}
    if warnings:
        out["warnings"] = warnings
    return _emit(out, 0 if not errors else 1)


if __name__ == "__main__":
    sys.exit(main())
