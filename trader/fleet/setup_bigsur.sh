#!/usr/bin/env bash
# PROTECTED (trader/fleet). Idempotent setup of bigsur = CPU sweep pool (MBP 2014, macOS 11.7 Intel, python 3.9).
# Run ON bigsur (the deploy wave does: `ssh bigsur 'bash -s' < trader/fleet/setup_bigsur.sh`).
#   --dry-run   print every command, change nothing
#   --no-gc     do not install the daily sweep-gc launchd job
# What it does:
#   1. checks macOS + python3.9 + >= 6 GB free disk (the runner quota is fleet.hosts.bigsur.max_disk_gb = 5 GB)
#   2. venv ~/.phil-crypto/venv with binary wheels only (verified to resolve for cp39 macosx_10_9_x86_64):
#        numpy 2.0.2, pandas 2.2.3, numba 0.60.0 (llvmlite 0.43.0), requests 2.32.3
#   3. numba smoke test (njit compile + run)
#   4. runner dirs ~/.phil-crypto/{runner/jobs,runner/code,logs} (code + jobs arrive from air via fleet/dispatch.py)
#   5. launchd job com.phil-crypto.sweep-gc (daily 04:30 local): removes job dirs older than 7 days
# bigsur never holds sealed data, never runs claude and never writes git.
set -euo pipefail

DRY=0; GC=1
for a in "$@"; do
  case "$a" in
    --dry-run) DRY=1 ;;
    --no-gc) GC=0 ;;
    *) echo "unknown option $a" >&2; exit 2 ;;
  esac
done
run() { if [ "$DRY" = 1 ]; then echo "+ $*"; else "$@"; fi; }
say() { echo "[setup_bigsur] $*"; }

PH="$HOME/.phil-crypto"
VENV="$PH/venv"
PY39="${PY39:-}"
PINS="numpy==2.0.2 pandas==2.2.3 numba==0.60.0 llvmlite==0.43.0 requests==2.32.3"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" 2>/dev/null && pwd || echo "")"

if [ "$(uname -s)" != "Darwin" ] && [ "$DRY" = 0 ]; then say "not macOS; refusing"; exit 1; fi

# 1. python 3.9 (python.org build at /usr/local/bin/python3.9 on bigsur)
if [ -z "$PY39" ]; then
  for c in /usr/local/bin/python3.9 /Library/Frameworks/Python.framework/Versions/3.9/bin/python3.9 python3.9 python3; do
    if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(0 if sys.version_info[:2] >= (3, 9) else 1)' 2>/dev/null; then
      PY39="$(command -v "$c")"; break
    fi
  done
fi
[ -n "$PY39" ] || { say "python >= 3.9 not found"; exit 1; }
say "python: $PY39 ($("$PY39" --version 2>&1))"

free_kb="$(df -Pk "$HOME" | awk 'NR==2{print $4}')"
if [ "${free_kb:-0}" -lt $((6 * 1024 * 1024)) ]; then
  say "only $((free_kb / 1024 / 1024)) GB free on \$HOME; need >= 6 GB (5 GB runner quota + venv)"
  [ "$DRY" = 1 ] || exit 1
fi

# 2. venv + pinned binary wheels (idempotent: pip skips satisfied pins)
run mkdir -p "$PH/runner/jobs" "$PH/runner/code" "$PH/logs"
if [ ! -x "$VENV/bin/python" ]; then
  run "$PY39" -m venv "$VENV"
fi
run "$VENV/bin/python" -m pip install --quiet --upgrade "pip<26" wheel
# shellcheck disable=SC2086
run "$VENV/bin/python" -m pip install --quiet --only-binary=:all: $PINS

# 3. smoke test: imports + one njit compile (cache in the runner dir, counted in the quota)
run env NUMBA_CACHE_DIR="$PH/runner/numba-cache" "$VENV/bin/python" -c '
import numpy, pandas, numba
from numba import njit
@njit(cache=False)
def f(x):
    s = 0.0
    for v in x:
        s += v
    return s
assert f(numpy.arange(10.0)) == 45.0
print("ok numpy", numpy.__version__, "pandas", pandas.__version__, "numba", numba.__version__)
'

# 4. quota marker (read by humans; dispatch enforces fleet.hosts.bigsur.max_disk_gb)
if [ "$DRY" = 1 ]; then echo "+ write $PH/runner/QUOTA"; else
  printf 'max_disk_gb=5\nmanaged_by=trader/fleet/dispatch.py\n' > "$PH/runner/QUOTA"; fi

# 5. daily GC (launchd, local time)
if [ "$GC" = 1 ]; then
  SRC="$SCRIPT_DIR/launchd/com.phil-crypto.sweep-gc.plist"
  if [ ! -f "$SRC" ]; then SRC="$PH/runner/code/trader/fleet/launchd/com.phil-crypto.sweep-gc.plist"; fi
  DST="$HOME/Library/LaunchAgents/com.phil-crypto.sweep-gc.plist"
  if [ -f "$SRC" ]; then
    run mkdir -p "$HOME/Library/LaunchAgents"
    if [ "$DRY" = 1 ]; then echo "+ render $SRC -> $DST (04:30 local)"; else
      sed -e "s#@HOME@#$HOME#g" -e "s#@HOUR@#4#g" -e "s#@MINUTE@#30#g" "$SRC" > "$DST.tmp"
      plutil -lint "$DST.tmp" >/dev/null
      mv "$DST.tmp" "$DST"
    fi
    run launchctl bootout "gui/$(id -u)/com.phil-crypto.sweep-gc" 2>/dev/null || true
    run launchctl bootstrap "gui/$(id -u)" "$DST" || say "launchctl bootstrap failed (already loaded?)"
  else
    say "sweep-gc plist not found (sync code first: python -m trader.fleet.dispatch sync bigsur); skipped"
  fi
fi

du_kb="$(du -sk "$PH/runner" 2>/dev/null | cut -f1 || echo 0)"
say "done. runner dir $((du_kb / 1024)) MB of 5120 MB quota; venv $VENV"
