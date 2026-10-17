#!/usr/bin/env bash
# PROTECTED (trader/fleet). Idempotent setup of berry = data + watchers + fleet health
# (Raspberry Pi, Debian 13, aarch64, python 3.13). Run ON berry (deploy wave: `ssh berry 'bash -s' < setup_berry.sh`).
#   --dry-run      print every command, change nothing
#   --no-systemd   do not install/enable the systemd user timers
# What it does:
#   1. checks python3 >= 3.9 with venv/ensurepip (Debian: `sudo apt install python3-venv` if missing)
#   2. venv ~/.phil-crypto/venv with binary wheels only (verified for cp313 manylinux aarch64):
#        numpy 2.2.6, pandas 2.2.3, requests 2.32.3   (no numba: berry never backtests)
#   3. dirs ~/.phil-crypto/{data,outbox,journal,logs,runner/code}
#      code lives in ~/.phil-crypto/runner/code/trader, pushed from air: python -m trader.fleet.dispatch sync berry
#   4. systemd --user units phil-{watch,health,data-update}.{service,timer} (+ linger so they run without login)
#   5. checks that `ssh air` works non-interactively (health/watch push into air's inbox over rsync)
# berry never writes git and never holds API secrets.
set -euo pipefail

DRY=0; SYSD=1
for a in "$@"; do
  case "$a" in
    --dry-run) DRY=1 ;;
    --no-systemd) SYSD=0 ;;
    *) echo "unknown option $a" >&2; exit 2 ;;
  esac
done
run() { if [ "$DRY" = 1 ]; then echo "+ $*"; else "$@"; fi; }
say() { echo "[setup_berry] $*"; }

PH="$HOME/.phil-crypto"
VENV="$PH/venv"
CODE="$PH/runner/code"
PINS="numpy==2.2.6 pandas==2.2.3 requests==2.32.3"
UNITS="phil-watch phil-health phil-data-update"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" 2>/dev/null && pwd || echo "")"

PY="$(command -v python3 || true)"
[ -n "$PY" ] || { say "python3 not found (sudo apt install python3 python3-venv)"; exit 1; }
"$PY" -c 'import sys; sys.exit(0 if sys.version_info[:2] >= (3, 9) else 1)' || { say "python >= 3.9 required"; exit 1; }
if ! "$PY" -c 'import venv, ensurepip' 2>/dev/null; then
  say "python3-venv missing: run 'sudo apt install python3-venv' and re-run"; [ "$DRY" = 1 ] || exit 1
fi
say "python: $PY ($("$PY" --version 2>&1))"

run mkdir -p "$PH/data/spot" "$PH/data/linear" "$PH/outbox" "$PH/journal" "$PH/logs" "$CODE"
if [ ! -x "$VENV/bin/python" ]; then
  run "$PY" -m venv "$VENV"
fi
run "$VENV/bin/python" -m pip install --quiet --upgrade pip wheel
# shellcheck disable=SC2086
run "$VENV/bin/python" -m pip install --quiet --only-binary=:all: $PINS
run "$VENV/bin/python" -c 'import numpy, pandas, requests; print("ok numpy", numpy.__version__, "pandas", pandas.__version__)'

if [ ! -f "$CODE/trader/core/config.py" ]; then
  say "code not present in $CODE/trader yet: from air run 'python -m trader.fleet.dispatch sync berry'"
fi

if [ "$SYSD" = 1 ]; then
  SRC="$SCRIPT_DIR/systemd"
  [ -f "$SRC/phil-health.service" ] || SRC="$CODE/trader/fleet/systemd"
  DST="$HOME/.config/systemd/user"
  run mkdir -p "$DST"
  if [ -f "$SRC/phil-health.service" ]; then
    for u in $UNITS; do
      for ext in service timer; do
        if [ "$DRY" = 1 ] || ! cmp -s "$SRC/$u.$ext" "$DST/$u.$ext"; then
          run cp "$SRC/$u.$ext" "$DST/$u.$ext"
        fi
      done
    done
    run systemctl --user daemon-reload
    for u in $UNITS; do run systemctl --user enable --now "$u.timer"; done
  else
    say "systemd unit files not found ($SRC); skipped"
  fi
  if [ "$(loginctl show-user "$USER" -p Linger 2>/dev/null | cut -d= -f2)" != "yes" ]; then
    run loginctl enable-linger "$USER" || say "could not enable linger: run 'sudo loginctl enable-linger $USER'"
  fi
fi

if [ "$DRY" = 0 ]; then
  if ssh -o BatchMode=yes -o ConnectTimeout=8 air true 2>/dev/null; then say "ssh air: ok"
  else say "WARNING: 'ssh air' fails non-interactively; add an ssh key + Host air in ~/.ssh/config"; fi
fi
say "done."
