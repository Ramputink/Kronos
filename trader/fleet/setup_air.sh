#!/usr/bin/env bash
# PROTECTED (trader/fleet). Idempotent setup of air = the brain (MacBook Air, macOS 14.6 arm64, has `claude`).
# Run ON air (deploy wave: `ssh air 'bash -s' < trader/fleet/setup_air.sh`).
#   --dry-run       print every command, change nothing
#   --no-launchd    do not install/load the launchd jobs
#   --repo DIR      git checkout to use (default ~/phil-crypto; cloned if missing)
# What it does:
#   1. checks macOS, python3 >= 3.9, git, rsync and the claude CLI (warns if `claude` is missing or logged out)
#   2. clones the fork (PHIL_REPO_URL, branch phil-crypto) if missing; never resets an existing checkout;
#      sets core.hooksPath=.githooks (pre-commit guard)
#   3. venv ~/.phil-crypto/venv with binary wheels only (verified for cp39 macosx_11_0_arm64):
#        numpy 2.0.2, pandas 2.2.3, numba 0.60.0 (llvmlite 0.43.0), requests 2.32.3
#   4. dirs ~/.phil-crypto/{data,inbox/done,jobs,logs}; secrets.env is NOT created or touched
#   5. launchd: com.phil-crypto.tick (every 15 min), .forward (00:10 UTC), .deep_retro (06:00 UTC);
#      UTC times converted to local time now (re-run after a DST change; the tick checks UTC itself anyway)
# Real trading stays disabled: nothing here edits config/protected.json.
set -euo pipefail

DRY=0; LAUNCHD=1
REPO="${PHIL_REPO:-$HOME/phil-crypto}"
REPO_URL="${PHIL_REPO_URL:-https://github.com/Ramputink/Kronos.git}"
BRANCH="${PHIL_BRANCH:-phil-crypto}"
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY=1 ;;
    --no-launchd) LAUNCHD=0 ;;
    --repo) REPO="$2"; shift ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
  shift
done
run() { if [ "$DRY" = 1 ]; then echo "+ $*"; else "$@"; fi; }
say() { echo "[setup_air] $*"; }

PH="$HOME/.phil-crypto"
VENV="$PH/venv"
PINS="numpy==2.0.2 pandas==2.2.3 numba==0.60.0 llvmlite==0.43.0 requests==2.32.3"
JOBS="tick forward deep_retro"

if [ "$(uname -s)" != "Darwin" ] && [ "$DRY" = 0 ]; then say "not macOS; refusing"; exit 1; fi
PY="${PY:-$(command -v python3 || true)}"
[ -n "$PY" ] || { say "python3 not found (xcode-select --install)"; exit 1; }
"$PY" -c 'import sys; sys.exit(0 if sys.version_info[:2] >= (3, 9) else 1)' || { say "python >= 3.9 required"; exit 1; }
for t in git rsync; do command -v "$t" >/dev/null || { say "$t missing"; exit 1; }; done
CLAUDE="$(command -v claude || true)"
[ -n "$CLAUDE" ] || for c in "$HOME/.local/bin/claude" /opt/homebrew/bin/claude; do [ -x "$c" ] && CLAUDE="$c" && break; done
if [ -n "$CLAUDE" ]; then say "claude: $CLAUDE"; else say "WARNING: claude CLI not found; cycles cannot run"; fi

# 2. repo
if [ ! -d "$REPO/.git" ] && [ ! -f "$REPO/.git" ]; then
  run git clone --branch "$BRANCH" "$REPO_URL" "$REPO"
else
  say "repo present: $REPO (not reset; loop.sh does ff-only syncs)"
fi
run git -C "$REPO" config core.hooksPath .githooks

# 3. venv
run mkdir -p "$PH/data" "$PH/inbox/done" "$PH/jobs" "$PH/logs" "$PH/outbox"
if [ ! -x "$VENV/bin/python" ]; then run "$PY" -m venv "$VENV"; fi
run "$VENV/bin/python" -m pip install --quiet --upgrade "pip<26" wheel
# shellcheck disable=SC2086
run "$VENV/bin/python" -m pip install --quiet --only-binary=:all: $PINS
run "$VENV/bin/python" -c 'import numpy, pandas, numba; print("ok numpy", numpy.__version__, "numba", numba.__version__)'

# 5. launchd
utc_to_local() {  # "HH:MM" UTC -> "H M" local (today's offset)
  "$PY" - "$1" <<'EOF'
import sys, datetime as dt
h, m = map(int, sys.argv[1].split(":"))
now = dt.datetime.now(dt.timezone.utc)
t = now.replace(hour=h, minute=m, second=0, microsecond=0).astimezone()
print(t.hour, t.minute)
EOF
}
cfg_time() {  # key in protected.schedule, default
  "$PY" - "$REPO/trader/config/protected.json" "$1" "$2" <<'EOF'
import json, sys
try:
    print(json.load(open(sys.argv[1]))["schedule"][sys.argv[2]])
except Exception:
    print(sys.argv[3])
EOF
}
if [ "$LAUNCHD" = 1 ]; then
  SRC="$REPO/trader/fleet/launchd"
  DST="$HOME/Library/LaunchAgents"
  run mkdir -p "$DST"
  for j in $JOBS; do
    case "$j" in
      forward) read -r HH MM <<<"$(utc_to_local "$(cfg_time forward_cycle 00:10)")" ;;
      deep_retro) read -r HH MM <<<"$(utc_to_local "$(cfg_time deep_retro 06:00)")" ;;
      *) HH=0; MM=0 ;;
    esac
    f="com.phil-crypto.$j.plist"
    if [ "$DRY" = 1 ]; then echo "+ render $SRC/$f -> $DST/$f (local $HH:$MM)"; else
      [ -f "$SRC/$f" ] || { say "missing $SRC/$f"; exit 1; }
      sed -e "s#@HOME@#$HOME#g" -e "s#@REPO@#$REPO#g" -e "s#@VENV@#$VENV#g" -e "s#@RUNNER@#air#g" \
          -e "s#@HOUR@#$HH#g" -e "s#@MINUTE@#$MM#g" "$SRC/$f" > "$DST/$f.tmp"
      plutil -lint "$DST/$f.tmp" >/dev/null
      mv "$DST/$f.tmp" "$DST/$f"
    fi
    run launchctl bootout "gui/$(id -u)/com.phil-crypto.$j" 2>/dev/null || true
    run launchctl bootstrap "gui/$(id -u)" "$DST/$f" || say "launchctl bootstrap $j failed"
  done
fi
say "done. repo $REPO, venv $VENV. real_trading_enabled is untouched (operator-only)."
