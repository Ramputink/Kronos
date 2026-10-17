"""Fleet health check (CONTRACTS.md §8). PROTECTED. Runs on berry every 15 min; Python >= 3.9, stdlib only.

``python -m trader.fleet.health run [--out PATH] [--journal] [--push] [--hosts air,bigsur] [--dry-run]``
probes every host in ``fleet.hosts`` (ssh with BatchMode; the host it runs on is probed locally) and writes
``fleet-health.json``:

* reachability (ssh), 1-minute load, free disk on $HOME;
* sweep hosts: size of ``fleet.remote_root`` vs ``max_disk_gb`` and running sweep jobs, venv python present;
* air: ``claude`` CLI present, lease state (``~/phil-crypto/trader/journal/lease.json``) and the last
  ``cycles.log`` line (status + age);
* berry: data freshness (newest CSV under ``data.dir``) and the watcher's last run (``~/.phil-crypto/watch-state.json``).

Alerts (``warn``/``crit``) are attached for unreachable non-optional hosts, low disk, runner disk near/over quota,
stale data, stale/errored cycles and expired-but-unreleased leases. ``last_seen`` of an unreachable host is kept from
the previous file so the operator sees how long it has been gone.

Default output is ``$PHIL_OUTBOX`` (``~/.phil-crypto/outbox``)/fleet-health.json; ``--journal`` writes to the
journal instead (manual run on air). ``--out`` may only name ``fleet-health.json`` directly inside the outbox or
the journal dir (anything else is refused, exit 2: it must never overwrite data CSVs or secrets). ``--push`` / ``python -m trader.fleet.health push`` rsyncs the outbox into
air's ``fleet.inbox_dir`` (``trigger-*.json`` files are moved, i.e. removed from the outbox after a successful
copy); ``fleet/inbox.py merge`` on air then copies it into ``journal/``.
Environment: ``PHIL_AIR_REPO`` (default ``~/phil-crypto``) = air's git checkout.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Optional

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from trader.core import config  # noqa: E402

SSH_OPTS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]
DATA_STALE_H = 26
CYCLE_STALE_H = 26
DISK_WARN_GB, DISK_CRIT_GB = 5.0, 2.0


def outbox_dir() -> Path:
    return Path(os.environ.get("PHIL_OUTBOX", "~/.phil-crypto/outbox")).expanduser()


def air_repo() -> str:
    return os.environ.get("PHIL_AIR_REPO", "~/phil-crypto")


def _home_path(p: str) -> str:
    """Path for a shell line: ~/x → "$HOME"/x."""
    if p.startswith("~/"):
        return '"$HOME"/' + shlex.quote(p[2:])
    return shlex.quote(p)


def probe_script(name: str, h: dict, cfg: dict) -> str:
    """POSIX sh that prints key=value lines; works on macOS 11/14 and Debian."""
    root = _home_path(cfg["fleet"]["remote_root"])
    lines = [
        "LC_ALL=C; LANG=C; export LC_ALL LANG",  # hosts run es_ES locales: decimal commas otherwise
        "echo now=$(date +%s)",
        "echo os=$(uname -s)",
        "if [ -r /proc/loadavg ]; then echo load=$(cut -d' ' -f1 /proc/loadavg); "
        "else echo load=$(/usr/sbin/sysctl -n vm.loadavg 2>/dev/null | tr -d '{}' | awk '{print $1}'); fi",
        "echo disk_free_kb=$(df -Pk \"$HOME\" | awk 'NR==2{print $4}')",
        "mt() { stat -c %Y \"$1\" 2>/dev/null || stat -f %m \"$1\" 2>/dev/null; }",
    ]
    roles = h.get("roles", [])
    py = h.get("python") or "python3"
    if not py.startswith("/usr/bin/env"):
        lines.append("if [ -x %s ]; then echo python=1; else echo python=0; fi" % _home_path(py))
    if "sweep" in roles:
        lines += [
            "if [ -d %s ]; then echo runner_kb=$(du -sk %s | cut -f1); else echo runner_kb=0; fi" % (root, root),
            "n=0; for f in %s/jobs/*/state.json; do [ -f \"$f\" ] && grep -q '\"state\": *\"running\"' \"$f\" "
            "&& n=$((n+1)); done; echo jobs_running=$n" % root,
        ]
    if "brain" in roles or "cycles" in roles:
        repo = _home_path(air_repo())
        lines += [
            "if command -v claude >/dev/null 2>&1 || [ -x \"$HOME/.local/bin/claude\" ] || "
            "[ -x /opt/homebrew/bin/claude ]; then echo claude=1; else echo claude=0; fi",
            "L=%s/trader/journal/lease.json; if [ -f \"$L\" ]; then echo lease_json=$(tr -d '\\n' < \"$L\"); fi" % repo,
            "C=%s/trader/journal/cycles.log; if [ -f \"$C\" ]; then echo last_cycle=$(tail -n 1 \"$C\"); fi" % repo,
        ]
    if "data" in roles or "watchers" in roles:
        data = _home_path(cfg["data"]["dir"])
        lines += [
            "newest=0; for f in %s/*/*.csv; do [ -f \"$f\" ] || continue; t=$(mt \"$f\"); "
            "[ \"$t\" -gt \"$newest\" ] && newest=$t; done; echo data_newest=$newest" % data,
            "W=\"$HOME/.phil-crypto/watch-state.json\"; if [ -f \"$W\" ]; then echo watch_mtime=$(mt \"$W\"); fi",
        ]
    return "; ".join(lines)


def _is_self(name: str, h: dict) -> bool:
    return h.get("ssh") == "local" or config.runner_id() == name


def run_probe(name: str, h: dict, cfg: dict, timeout: int = 45) -> Optional[dict]:
    """key=value dict, or None when the host could not be reached."""
    script = probe_script(name, h, cfg)
    argv = ["/bin/sh", "-c", script] if _is_self(name, h) else ["ssh"] + SSH_OPTS + [h["ssh"], script]
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    except (subprocess.TimeoutExpired, OSError):
        return None
    if p.returncode != 0 and "now=" not in p.stdout:
        return None
    kv = {}
    for ln in p.stdout.splitlines():
        if "=" in ln:
            k, v = ln.split("=", 1)
            kv[k.strip()] = v.strip()
    return kv if "now" in kv else None


def _f(x, default=None):
    try:
        return float(str(x).replace(",", ".")) if x is not None else default
    except (TypeError, ValueError):
        return default


def _ts(epoch) -> Optional[str]:
    e = _f(epoch)
    if not e:
        return None
    return config.iso(_dt.datetime.fromtimestamp(e, tz=_dt.timezone.utc))


def _age_h(iso_s: Optional[str], now: _dt.datetime) -> Optional[float]:
    try:
        t = _dt.datetime.fromisoformat(str(iso_s).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return (now - t).total_seconds() / 3600.0


CYCLE_RE = re.compile(r"^(\S+Z) (\S+) (ok|skipped|error|reverted)\b")


def evaluate_host(name: str, h: dict, kv: Optional[dict], cfg: dict, prev: dict, now: _dt.datetime):
    """(host entry, alerts) from a probe result. Pure function (unit-tested)."""
    alerts = []
    nowz = config.iso(now)

    def alert(level, msg):
        alerts.append({"ts": nowz, "host": name, "level": level, "msg": msg})

    if kv is None:
        entry = {"last_seen": (prev or {}).get("last_seen"), "ok": False, "checks": {"ssh": False}}
        if h.get("optional"):
            entry["optional"] = True
        else:
            alert("crit", "unreachable over ssh")
        return entry, alerts
    roles = h.get("roles", [])
    entry = {"last_seen": nowz, "ok": True, "checks": {"ssh": True}}
    if h.get("optional"):
        entry["optional"] = True
    load = _f(kv.get("load"))
    if load is not None:
        entry["load1"] = round(load, 2)
    free_kb = _f(kv.get("disk_free_kb"))
    if free_kb is not None:
        gb = round(free_kb / 1024 / 1024, 1)
        entry["disk_free_gb"] = gb
        if gb < DISK_CRIT_GB:
            alert("crit", "disk free %.1f GB" % gb)
        elif gb < DISK_WARN_GB:
            alert("warn", "disk free %.1f GB" % gb)
    if "python" in kv:
        entry["checks"]["python"] = kv["python"] == "1"
        if kv["python"] != "1":
            alert("warn", "venv python missing (%s); run fleet/setup_%s.sh" % (h.get("python"), name))
    if "sweep" in roles:
        rkb = _f(kv.get("runner_kb"), 0.0)
        rgb = round(rkb / 1024 / 1024, 2)
        entry["runner_disk_gb"] = rgb
        entry["jobs_running"] = int(_f(kv.get("jobs_running"), 0))
        cap = _f(h.get("max_disk_gb"))
        if cap:
            if rgb > cap:
                alert("crit", "runner disk %.1f/%g GB" % (rgb, cap))
            elif rgb > 0.8 * cap:
                alert("warn", "runner disk %.1f/%g GB" % (rgb, cap))
    if "claude" in kv:
        entry["checks"]["claude"] = kv["claude"] == "1"
        if kv["claude"] != "1":
            alert("crit", "claude CLI not found")
    if "brain" in roles or "cycles" in roles:
        lease = "free"
        if kv.get("lease_json"):
            try:
                lj = json.loads(kv["lease_json"])
            except ValueError:
                lj = {}
            if lj.get("runner"):
                age = _age_h(lj.get("started"), now)
                ttl_h = _f(lj.get("ttl_s"), 3000) / 3600.0
                if age is not None and age < ttl_h:
                    lease = "held:%s:%s" % (lj["runner"], lj.get("cycle"))
                else:
                    lease = "expired:%s" % lj["runner"]
                    alert("warn", "lease held by %s expired without release" % lj["runner"])
        entry["checks"]["lease"] = lease
        lc = kv.get("last_cycle")
        if lc:
            m = CYCLE_RE.match(lc)
            entry["last_cycle"] = lc[:200]
            if m:
                age = _age_h(m.group(1), now)
                if age is not None and age > CYCLE_STALE_H:
                    alert("warn", "last cycle %.0f h ago" % age)
                if m.group(3) in ("error", "reverted"):
                    alert("warn", "last cycle %s %s" % (m.group(2), m.group(3)))
        else:
            entry["last_cycle"] = None
    if "data" in roles or "watchers" in roles:
        dn = _f(kv.get("data_newest"), 0.0)
        fresh = bool(dn) and (now.timestamp() - dn) < DATA_STALE_H * 3600
        entry["checks"]["data_fresh"] = fresh
        if not fresh:
            alert("warn", "data not updated in %d h" % DATA_STALE_H)
        entry["checks"]["watch_last_run"] = _ts(kv.get("watch_mtime"))
    if any(a["level"] == "crit" for a in alerts):
        entry["ok"] = False
    return entry, alerts


def collect(*, only=None, cfg: Optional[dict] = None, prev: Optional[dict] = None, probe=run_probe) -> dict:
    """Probe hosts and build the fleet-health document."""
    cfg = cfg or config.load_config()
    prev = prev or {}
    now = config.utcnow()
    doc = {"updated": config.iso(now), "by": config.runner_id(), "hosts": {}, "alerts": []}
    for name, h in cfg["fleet"]["hosts"].items():
        if only and name not in only:
            continue
        kv = probe(name, h, cfg)
        entry, alerts = evaluate_host(name, h, kv, cfg, (prev.get("hosts") or {}).get(name, {}), now)
        doc["hosts"][name] = entry
        doc["alerts"].extend(alerts)
    return doc


OUT_NAME = "fleet-health.json"


def allowed_out(path) -> Path:
    """Resolve --out; only <outbox>/fleet-health.json or <journal>/fleet-health.json are accepted (ValueError)."""
    raw = Path(path).expanduser()
    if not raw.is_absolute():
        raw = Path.cwd() / raw
    if raw.is_symlink():
        raise ValueError("--out %s is a symlink" % path)
    p = raw.resolve()
    dirs = {outbox_dir().resolve(), config.journal_dir().resolve()}
    if p.name != OUT_NAME or p.parent not in dirs:
        raise ValueError("--out must be %s inside %s, got %s" % (OUT_NAME, " or ".join(sorted(map(str, dirs))), path))
    return p


def write(doc: dict, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    os.replace(str(tmp), str(path))
    return path


def push(*, dry_run: bool = False, cfg: Optional[dict] = None) -> dict:
    """rsync the outbox into air's inbox_dir; trigger-*.json are removed from the outbox once copied."""
    cfg = cfg or config.load_config()
    ob = outbox_dir()
    ob.mkdir(parents=True, exist_ok=True)
    air = cfg["fleet"]["hosts"][cfg["fleet"]["brain"]]["ssh"]
    inbox = cfg["fleet"]["inbox_dir"]
    dest_path = inbox[2:] if inbox.startswith("~/") else inbox
    dest = "%s:%s/" % (air, dest_path) if air != "local" else str(Path(inbox).expanduser()) + "/"
    e = ["-e", "ssh " + " ".join(SSH_OPTS)]
    mk = (["ssh"] + SSH_OPTS + [air, "mkdir -p " + _home_path(inbox)]) if air != "local" else \
        ["mkdir", "-p", str(Path(inbox).expanduser())]
    triggers = sorted(str(p) for p in ob.glob("trigger-*.json"))
    # -rt, not -a: no -l, so a symlink in the outbox is skipped instead of being recreated in air's inbox
    # (portable across rsync 2.6.9 / 3.x / openrsync; inbox.merge refuses symlinks anyway).
    cmds = [mk, ["rsync", "-rt"] + e + ["--exclude", "trigger-*.json", "--exclude", "*.tmp", str(ob) + "/", dest]]
    if triggers:
        cmds.append(["rsync", "-rt", "--remove-source-files"] + e + triggers + [dest])
    if dry_run:
        return {"ok": True, "dry_run": True, "commands": cmds}
    for c in cmds:
        p = subprocess.run(c, capture_output=True, text=True, timeout=120)
        if p.returncode != 0:
            return {"ok": False, "error": (p.stderr or p.stdout)[-500:], "command": " ".join(c)}
    return {"ok": True, "pushed_triggers": len(triggers)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m trader.fleet.health")
    ap.add_argument("cmd", choices=["run", "push"], nargs="?", default="run")
    ap.add_argument("--out", default=None)
    ap.add_argument("--journal", action="store_true", help="write journal/fleet-health.json directly")
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--hosts", default=None)
    ap.add_argument("--dry-run", action="store_true", help="print the document, write nothing")
    a = ap.parse_args(argv)
    if a.cmd == "push":
        res = push(dry_run=a.dry_run)
        print(json.dumps(res, default=str))
        return 0 if res.get("ok") else 1
    if a.out:
        try:
            out = allowed_out(a.out)
        except ValueError as e:
            print(json.dumps({"ok": False, "refused": True, "error": str(e)}))
            return 2
    else:
        out = (config.journal_dir() if a.journal else outbox_dir()) / OUT_NAME
    try:
        prev = json.loads(out.read_text()) if out.exists() else {}
    except ValueError:
        prev = {}
    doc = collect(only=set(a.hosts.split(",")) if a.hosts else None, prev=prev)
    if a.dry_run:
        print(json.dumps(doc, sort_keys=True))
        return 0
    write(doc, out)
    res = {"ok": True, "out": str(out), "alerts": len(doc["alerts"]),
           "hosts_ok": sorted(k for k, v in doc["hosts"].items() if v.get("ok"))}
    if a.push:
        res["push"] = push()
        res["ok"] = bool(res["push"].get("ok"))
    print(json.dumps(res, default=str))
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
