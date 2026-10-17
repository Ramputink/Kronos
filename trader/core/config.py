"""Shared paths, protected-config loader and JSONL helpers (Wave 0, PROTECTED).

Every core/fleet/strategy module resolves files through this module so that tests can redirect
everything with environment variables (see CONTRACTS.md section 1). Python >= 3.9, stdlib only.
"""
import datetime as _dt
import json
import os
import socket
from pathlib import Path

try:
    import fcntl  # POSIX only; all fleet hosts are POSIX
except ImportError:  # pragma: no cover
    fcntl = None

TRADER_ROOT = Path(os.environ.get("TRADER_ROOT", Path(__file__).resolve().parent.parent))
REPO_ROOT = TRADER_ROOT.parent


def _p(env, default):
    v = os.environ.get(env)
    return Path(v).expanduser() if v else Path(default)


def config_path():
    return _p("TRADER_CONFIG", TRADER_ROOT / "config" / "protected.json")


def journal_dir():
    return _p("TRADER_JOURNAL_DIR", TRADER_ROOT / "journal")


def strategy_dir():
    return _p("TRADER_STRATEGY_DIR", TRADER_ROOT / "strategy")


def load_config():
    """Fresh read of protected.json on every call (cheap; never cached so tests can swap it)."""
    with open(config_path()) as fh:
        return json.load(fh)


def data_dir(source=None):
    """<data root>/<source>; root = $TRADER_DATA_DIR or protected data.dir."""
    root = os.environ.get("TRADER_DATA_DIR") or load_config()["data"]["dir"]
    root = Path(root).expanduser()
    return root / source if source else root


def runner_id():
    return os.environ.get("PHIL_RUNNER") or socket.gethostname().split(".")[0].lower()


def utcnow():
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0)


def iso(ts=None):
    """ISO-8601 UTC with 'Z', second precision. Accepts datetime, pandas Timestamp or None (=now)."""
    if ts is None:
        ts = utcnow()
    if hasattr(ts, "to_pydatetime"):
        ts = ts.to_pydatetime()
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=_dt.timezone.utc)
    return ts.astimezone(_dt.timezone.utc).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")


def jsonl_append(path, obj):
    """Append one JSON object as one line. Exclusive flock so concurrent writers never interleave."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(obj, ensure_ascii=False, separators=(",", ":"), default=str) + "\n"
    with open(path, "a", encoding="utf-8") as fh:
        if fcntl:
            fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            fh.write(line)
            fh.flush()
            os.fsync(fh.fileno())
        finally:
            if fcntl:
                fcntl.flock(fh, fcntl.LOCK_UN)


def jsonl_read(path):
    """All parseable rows; a torn last line is skipped, never raised."""
    path = Path(path)
    if not path.exists():
        return []
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
    return out


def canonical_json(obj):
    """Canonical serialisation used for every hash in the system."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
