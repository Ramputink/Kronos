"""Inbox merge on air (CONTRACTS.md §8 "Inbox"). PROTECTED. Python >= 3.9, stdlib only.

berry never writes git. It rsyncs into air's ``fleet.inbox_dir`` (``~/.phil-crypto/inbox``):
``fleet-health.json``, ``watch-triggers-<host>.jsonl`` (the host's whole trigger log, re-sent every time) and
``trigger-<ts>.json`` (one file per ``trigger_forward`` action).

``merge()`` (run by loop.sh before each cycle):
* copies ``fleet-health.json`` into ``journal/fleet-health.json`` when it differs;
* appends to ``journal/watch-triggers.jsonl`` every trigger row not already there (dedup on the canonical JSON
  of the row, so re-sent logs are idempotent);
* returns the pending ``trigger-*.json`` files, oldest first. They stay pending until ``consume(path)`` moves them
  to ``inbox/done/`` (loop.sh consumes after the triggered cycle has run, so a crashed cycle does not lose them).

Everything read from the inbox is untrusted (berry, or whatever can write there): ``merge`` only reads regular,
non-symlink files directly inside ``inbox_dir()`` (opened with O_NOFOLLOW, size-capped) and only copies documents that
match the ``fleet-health.json`` / trigger-row schemas, so no other local file can end up in the journal (and on GitHub).

``consume`` only ever moves a regular (non-symlink) file named ``trigger-*.json`` that sits directly inside
``inbox_dir()``; any other path is refused (it must never be usable to move data CSVs, secrets or journal files).

CLI: ``python -m trader.fleet.inbox merge [--consume]`` | ``consume FILE...`` — prints one JSON object, exit 0/1.
Environment: ``PHIL_INBOX`` overrides ``fleet.inbox_dir``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import sys
from pathlib import Path
from typing import Optional

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from trader.core import config  # noqa: E402


TRIGGER_RE = re.compile(r"^trigger-[A-Za-z0-9_.:+-]+\.json$")


class InboxError(Exception):
    """A path that is not a pending trigger file of this inbox."""


def inbox_dir(cfg: Optional[dict] = None) -> Path:
    v = os.environ.get("PHIL_INBOX") or (cfg or config.load_config())["fleet"]["inbox_dir"]
    return Path(v).expanduser()


def _key(row: dict) -> str:
    return hashlib.sha256(config.canonical_json(row).encode()).hexdigest()


# Size caps (bytes): anything larger is refused, never copied into the journal.
MAX_HEALTH_BYTES = 256 * 1024
MAX_TRIGGER_LOG_BYTES = 4 * 1024 * 1024
MAX_TRIGGER_FILE_BYTES = 64 * 1024

WATCH_LOG_RE = re.compile(r"^watch-triggers-[A-Za-z0-9_.-]+\.jsonl$")
ROW_REQUIRED = {"ts": str, "rule_id": str, "coin": str, "metric": str, "value": (int, float),
                "threshold": (int, float), "action": str, "host": str}
ROW_OPTIONAL = {"detail": str}
ROW_ACTIONS = {"note", "trigger_forward"}
MAX_STR = 512


def _read_inbox_file(name: str, cap: int) -> bytes:
    """Bytes of a regular, non-symlink file sitting directly in inbox_dir(), at most ``cap`` bytes.

    Opened with O_NOFOLLOW (+ O_NONBLOCK so a FIFO cannot hang the merge); anything else raises InboxError."""
    if "/" in name or name in ("", ".", ".."):
        raise InboxError("refusing %s: not a plain file name" % name)
    ib = inbox_dir().resolve()
    p = ib / name
    if p.is_symlink():
        raise InboxError("refusing %s: symlink" % name)
    if p.resolve().parent != ib:
        raise InboxError("refusing %s: not directly inside the inbox %s" % (name, ib))
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        fd = os.open(str(p), flags)
    except OSError as e:
        raise InboxError("refusing %s: %s" % (name, e))
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise InboxError("refusing %s: not a regular file" % name)
        if st.st_size > cap:
            raise InboxError("refusing %s: %d bytes > cap %d" % (name, st.st_size, cap))
        with os.fdopen(fd, "rb") as fh:
            fd = -1
            data = fh.read(cap + 1)
    finally:
        if fd >= 0:
            os.close(fd)
    if len(data) > cap:
        raise InboxError("refusing %s: larger than cap %d" % (name, cap))
    return data


def _is_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def valid_health(doc, host_names) -> Optional[str]:
    """None when ``doc`` has the journal/fleet-health.json shape (CONTRACTS.md §8), else the reason."""
    if not isinstance(doc, dict):
        return "not an object"
    extra = set(doc) - {"updated", "by", "hosts", "alerts"}
    if extra:
        return "unknown keys %s" % sorted(extra)
    if not isinstance(doc.get("updated"), str) or len(doc["updated"]) > 64:
        return "'updated' must be a timestamp string"
    if not isinstance(doc.get("by"), str) or len(doc["by"]) > 128:
        return "'by' must be a string"
    hosts = doc.get("hosts")
    if not isinstance(hosts, dict):
        return "'hosts' must be an object"
    unknown = set(hosts) - set(host_names)
    if unknown:
        return "unknown hosts %s" % sorted(unknown)
    for k, v in hosts.items():
        if not isinstance(v, dict):
            return "hosts.%s must be an object" % k
    alerts = doc.get("alerts", [])
    if not isinstance(alerts, list):
        return "'alerts' must be a list"
    for a in alerts:
        if not isinstance(a, dict) or set(a) - {"ts", "host", "level", "msg"}:
            return "bad alert %r" % (a,)
        if any(not isinstance(a.get(f), str) or len(a[f]) > MAX_STR for f in ("ts", "host", "level", "msg")):
            return "bad alert %r" % (a,)
    return None


def valid_trigger_row(row) -> Optional[str]:
    """None when ``row`` has the watch-triggers row shape (CONTRACTS.md §9 watchers), else the reason."""
    if not isinstance(row, dict):
        return "not an object"
    extra = set(row) - set(ROW_REQUIRED) - set(ROW_OPTIONAL)
    if extra:
        return "unknown keys %s" % sorted(extra)
    for k, t in ROW_REQUIRED.items():
        if k not in row:
            return "missing %s" % k
        v = row[k]
        if t is str:
            if not isinstance(v, str) or len(v) > MAX_STR:
                return "%s must be a short string" % k
        elif not _is_num(v):
            return "%s must be a number" % k
    for k, t in ROW_OPTIONAL.items():
        if k in row and (not isinstance(row[k], str) or len(row[k]) > MAX_STR):
            return "%s must be a short string" % k
    if row["action"] not in ROW_ACTIONS:
        return "unknown action %r" % row["action"]
    return None


def _host_names() -> list:
    try:
        return list(config.load_config()["fleet"]["hosts"])
    except Exception:  # config unreadable => no host is known => every health doc is refused
        return []


def merge(*, consume_all: bool = False) -> dict:
    ib = inbox_dir()
    ib.mkdir(parents=True, exist_ok=True)
    jd = config.journal_dir()
    out = {"health_copied": False, "new_trigger_rows": 0, "pending": [], "errors": []}

    src = ib / "fleet-health.json"
    if src.exists() or src.is_symlink():
        try:
            raw = _read_inbox_file("fleet-health.json", MAX_HEALTH_BYTES)
            why = valid_health(json.loads(raw), _host_names())  # never copy a torn or foreign file
            if why:
                raise InboxError("schema: %s" % why)
            dest = jd / "fleet-health.json"
            if not dest.exists() or dest.read_bytes() != raw:
                dest.parent.mkdir(parents=True, exist_ok=True)
                tmp = dest.with_suffix(".tmp")
                tmp.write_bytes(raw)
                os.replace(str(tmp), str(dest))
                out["health_copied"] = True
        except (OSError, ValueError, InboxError) as e:
            out["errors"].append("fleet-health.json: %s" % e)

    wt = jd / "watch-triggers.jsonl"
    seen = {_key(r) for r in config.jsonl_read(wt)}
    for f in sorted(ib.glob("watch-triggers-*.jsonl")):
        if not WATCH_LOG_RE.match(f.name):
            out["errors"].append("%s: refusing: bad name" % f.name)
            continue
        try:
            raw = _read_inbox_file(f.name, MAX_TRIGGER_LOG_BYTES)
        except InboxError as e:
            out["errors"].append("%s: %s" % (f.name, e))
            continue
        bad = 0
        for line in raw.decode("utf-8", "replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue  # torn last line: re-sent next time
            if valid_trigger_row(row):
                bad += 1
                continue
            k = _key(row)
            if k in seen:
                continue
            config.jsonl_append(wt, row)
            seen.add(k)
            out["new_trigger_rows"] += 1
        if bad:
            out["errors"].append("%s: %d row(s) refused (schema)" % (f.name, bad))

    pending = sorted(ib.glob("trigger-*.json"))
    good = []
    for p in pending:
        try:
            if not TRIGGER_RE.match(p.name):
                raise InboxError("refusing: bad name")
            why = valid_trigger_row(json.loads(_read_inbox_file(p.name, MAX_TRIGGER_FILE_BYTES)))
            if why:
                raise InboxError("schema: %s" % why)
            good.append(str(p))
        except (OSError, ValueError, InboxError) as e:
            out["errors"].append("%s: %s" % (p.name, e))
    out["pending"] = good
    if consume_all:
        out["consumed"] = [consume(p) for p in good]
    return out


def _trigger_path(path) -> Path:
    """The real path of a pending trigger file in inbox_dir(), or InboxError (fails closed)."""
    raw = Path(path).expanduser()
    if not raw.is_absolute():
        raw = Path.cwd() / raw
    ib = inbox_dir().resolve()
    if raw.is_symlink():
        raise InboxError("refusing %s: symlink" % path)
    p = raw.resolve()
    if p.parent != ib:
        raise InboxError("refusing %s: not directly inside the inbox %s" % (path, ib))
    if not TRIGGER_RE.match(p.name):
        raise InboxError("refusing %s: not a trigger-*.json file" % path)
    if not p.is_file():
        raise InboxError("refusing %s: not a regular file" % path)
    return p


def consume(path) -> str:
    """Move a pending trigger file (trigger-*.json directly in inbox_dir) to inbox/done/; returns its new path.
    Anything else raises InboxError and nothing is moved."""
    p = _trigger_path(path)
    done = p.parent / "done"
    done.mkdir(parents=True, exist_ok=True)
    dest = done / p.name
    shutil.move(str(p), str(dest))
    return str(dest)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m trader.fleet.inbox")
    ap.add_argument("cmd", choices=["merge", "consume"])
    ap.add_argument("files", nargs="*")
    ap.add_argument("--consume", action="store_true")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "merge":
            res = merge(consume_all=a.consume)
            res["ok"] = not res["errors"]
        else:
            paths = [_trigger_path(f) for f in a.files]       # validate all before moving any
            res = {"ok": True, "consumed": [consume(f) for f in paths]}
    except Exception as e:  # CLI boundary
        res = {"ok": False, "error": "%s: %s" % (type(e).__name__, e)}
    print(json.dumps(res, default=str))
    return 0 if res.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
