#!/usr/bin/env bash
# Phil-crypto cycle runner (guard area, PROTECTED). Pattern: github.com/bennyjo/phil loop.sh, rewritten for trader/.
#
# Usage: trader/loop.sh <research|forward|deep_retro|triggered|tick> [--dry-run]
#
#   tick       decide which cycle is due now (UTC) and run it, or exit quietly when nothing is due
#   --dry-run  read-only: print the decision, the claude command and the protected-path check; no lease,
#              no sync, no claude, no commit, no push
#
# One cycle = pid lock -> reattach branch + fetch + ff-only sync -> lease -> inbox merge -> (forward/triggered:
# paper settlement run HERE, outside the agent session, committed on its own) -> snapshot git control state and
# the guard code -> headless `claude -p "$(cat trader/CYCLE_<KIND>.md)"` (pinned model, scoped tools, turn +
# wall-clock limits) -> restore git control state -> squash any agent commits -> REVERT every change outside
# the agent whitelist with the SNAPSHOTTED guard (trader/core/validate.py --enforce) -> validate -> one commit
# "cycle(<kind>): <date> <summary>" -> final range gate -> push -> lease release -> cycles.log line.
#
# Threat model: the agent runs as this unix user. Its session loads NO user/local Claude settings and no MCP
# servers (--setting-sources project --strict-mcp-config), so only the allow rules below apply. It can only edit trader/strategy/** (no code files),
# trader/journal/retros/** and trader/journal/proposals.md (scoped Edit rules, dontAsk mode), and run trader
# modules. Everything it might still have changed in git's control files (.git/config, info/, hooks/, index
# flags, global/system gitconfig) is restored from in-memory snapshots before any git command runs, and
# every git command here runs hardened (no fsmonitor, no hooks, no excludes/attributes files, no replace
# objects). The durable fix is a separate unix user for claude: set PHIL_AGENT_USER (sudo -n -u) when the
# host has one (see open issues in trader/REAL.md / CONTRACTS section 7).
#
# The agent never commits or pushes: this shell does (keyring/credentials live here, see phil's loop.sh).
# Real trading is never started here; real.py refuses on its own while real_trading_enabled is false.
#
# Environment (all optional):
#   PHIL_PYTHON            python for trader modules (default: python3)
#   PHIL_CLAUDE_BIN        claude binary (default: claude)  -- tests point this at a fake
#   PHIL_AGENT_USER        run claude as this unix user via `sudo -n -H -u` (recommended isolation)
#   PHIL_PERMISSION_MODE   claude permission mode (default dontAsk: only the allow-rules below are usable)
#   PHIL_CYCLE_MODEL       model for every cycle (default claude-opus-5-5); PHIL_MODEL_<KIND> overrides per kind
#   PHIL_MAX_TURNS         --max-turns for claude (default 200; 0 = do not pass)
#   PHIL_MAX_BUDGET_USD    --max-budget-usd for claude (default: not passed, except triggered cycles)
#   PHIL_TRIGGERED_MAX_BUDGET_USD  budget cap of a triggered cycle when PHIL_MAX_BUDGET_USD is unset (default 5)
#   PHIL_TRIGGER_COOLDOWN_H  at most one triggered cycle per this many hours (default: protected
#                          schedule.triggered_cooldown_hours, else 4; floor 0.25). The listed trigger files are
#                          consumed (inbox/done) after the session whatever its outcome.
#   PHIL_WEBFETCH_DOMAINS  space-separated WebFetch domains for research cycles (other cycles get no web tools)
#   PHIL_CYCLE_TIMEOUT_MIN wall-clock limit (default: research = schedule.json research.max_minutes or 60,
#                          forward 45, deep_retro 45, triggered 20)
#   PHIL_ALLOWED_TOOLS     space-separated allowedTools override
#   PHIL_BRANCH            branch to cycle on (default phil-crypto); PHIL_PUSH_BRANCH where to push (default
#                          the same; point it at a staging branch that CI fast-forwards)
#   PHIL_NO_PUSH=1         commit locally only;  PHIL_NO_LEASE=1  skip the runner lease (single host / tests)
#   PHIL_NO_SETTLE=1       skip the loop-side paper settlement of forward/triggered cycles
#   PHIL_NOW               ISO UTC "now" for the tick decision (tests)
#   PHIL_HOME              state dir for logs/rejected patches (default ~/.phil-crypto)
set -uo pipefail

# The whole script is ONE brace group: bash parses it completely before executing anything, so an edit to
# this file while a cycle runs cannot change the commands of the running cycle.
{
TRADER_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$TRADER_DIR/.." && pwd)"
cd "$ROOT" || exit 1

KIND="${1:-}"
DRY=0
for a in "${@:2}"; do
  case "$a" in
    --dry-run) DRY=1;;
    *) echo "unknown option: $a" >&2; exit 2;;
  esac
done
case "$KIND" in
  research|forward|deep_retro|triggered|tick) ;;
  *) echo "usage: trader/loop.sh <research|forward|deep_retro|triggered|tick> [--dry-run]" >&2; exit 2;;
esac

PY="${PHIL_PYTHON:-python3}"
CLAUDE="${PHIL_CLAUDE_BIN:-claude}"
BRANCH="${PHIL_BRANCH:-phil-crypto}"
PUSH_BRANCH="${PHIL_PUSH_BRANCH:-$BRANCH}"
PHIL_HOME="${PHIL_HOME:-$HOME/.phil-crypto}"
LOGDIR="$PHIL_HOME/logs"
mkdir -p "$LOGDIR" 2>/dev/null || true
export PYTHONDONTWRITEBYTECODE=1 GIT_NO_REPLACE_OBJECTS=1
# Guard runs: isolated (-I: no script dir / cwd / PYTHON* env on sys.path), no bytecode written.
VALIDATE=("$PY" -I -B trader/core/validate.py)
CYCLES_LOG="trader/journal/cycles.log"
START_EPOCH="$(date -u +%s)"
# Every git call of this script: nothing in .git or the user's config may run code or hide a path.
GITX=(git --no-replace-objects -c core.fsmonitor=false -c core.hooksPath=/dev/null -c core.excludesFile=/dev/null
      -c core.attributesFile=/dev/null -c core.sparseCheckout=false -c core.untrackedCache=false)
g() { "${GITX[@]}" "$@"; }

log() { echo "[loop $(date -u +%FT%TZ)] $*" >&2; echo "$(date -u +%FT%TZ) $*" >> "$LOGDIR/loop.log" 2>/dev/null || true; }
jget() {  # jget <json> <python-expr on d>
  printf '%s' "$1" | "$PY" -I -c "import json,sys
d=json.load(sys.stdin)
v=$2
print(v if not isinstance(v,(list,dict)) else json.dumps(v))" 2>/dev/null
}
cycles_line() {  # cycles_line <kind> <status> <fields...> -- <note>
  local k="$1" st="$2"; shift 2
  local fields="" note=""
  while [ $# -gt 0 ]; do
    if [ "$1" = "--" ]; then shift; note="$*"; break; fi
    fields="$fields $1"; shift
  done
  note="$(printf '%s' "$note" | tr '\n\r' '  ' | cut -c1-200)"
  local line="$(date -u +%FT%TZ) $k $st$fields"
  [ -n "$note" ] && line="$line | $note"
  if [ "$DRY" -eq 1 ]; then echo "would append to cycles.log: $line"; return 0; fi
  mkdir -p "$(dirname "$CYCLES_LOG")"
  printf '%s\n' "$line" >> "$CYCLES_LOG"
}

redact_summary() {  # stdin -> one sanitised line of at most 100 chars
  LC_ALL=C tr -cd '\040-\176\n' | head -n 1 | "$PY" -I -S -c '
import re, sys
s = sys.stdin.read().strip()
s = re.sub(r"-----BEGIN[^-]*-----.*", "[redacted]", s)
s = re.sub(r"(?i)\b(ghp|gho|ghu|ghs|ghr|github_pat|sk-ant|sk|xox[abprs]|AKIA|ASIA|AIza|glpat|hf|npm|pypi)[-_][A-Za-z0-9_\-]{6,}", "[redacted]", s)
s = re.sub(r"(?i)\b(ssh-(rsa|ed25519|dss)|ecdsa-sha2-\S+)\s+\S+", "[redacted]", s)
s = re.sub(r"(?i)((api[_-]?key|secret|token|passw(or)?d|pwd|auth|bearer|credential|private[_-]?key)\w*\s*[:=]?\s*)\S+", r"\1[redacted]", s)
s = re.sub(r"[A-Za-z0-9+/=_.\-]{24,}", "[redacted]", s)
s = re.sub(r"\b(?=[A-Za-z0-9+/_\-]*\d)(?=[A-Za-z0-9+/_\-]*[A-Za-z])[A-Za-z0-9+/_\-]{16,}\b", "[redacted]", s)
s = re.sub(r"https?://\S+", "[url]", s)
print(re.sub(r"\s+", " ", s)[:100])' 2>/dev/null
}

# ---- in-memory snapshots (the agent can write any file this user can; it cannot touch this shell's memory)
snap_files() {  # snap_files <dir> <names...> -> base64 tar of the names that exist (empty if none)
  local dir="$1"; shift
  local have=()
  for n in "$@"; do [ -e "$dir/$n" ] || [ -L "$dir/$n" ] && have+=("$n"); done
  [ "${#have[@]}" -eq 0 ] && return 0
  tar -C "$dir" -cf - "${have[@]}" 2>/dev/null | base64
}
restore_files() {  # restore_files <dir> <snapshot> <names...>: delete the names, then unpack the snapshot
  local dir="$1" snap="$2"; shift 2
  for n in "$@"; do rm -rf "${dir:?}/$n"; done
  [ -n "$snap" ] && printf '%s\n' "$snap" | base64 -d | tar -C "$dir" -xf -
  return 0
}
file_snap() {  # file_snap <path> -> "1<base64>" if it exists, "0" otherwise
  if [ -f "$1" ]; then printf '1'; base64 < "$1"; else printf '0'; fi
}
file_restore() {  # file_restore <path> <file_snap output>: put back if changed (only files we may write)
  local p="$1" s="$2" cur
  cur="$(file_snap "$p")"
  [ "$cur" = "$s" ] && return 0
  log "WARNING: $p changed during the agent session - restoring it"
  if [ "${s:0:1}" = "1" ]; then printf '%s\n' "${s:1}" | base64 -d > "$p"; else rm -f "$p"; fi
}

# ---- 1. pid lock (one loop per checkout; a stale file is harmless, the pid is checked) --------------
LOCK="$TRADER_DIR/.loop.pid"
GUARD_DIR=""
cleanup() { rm -f "$LOCK"; [ -n "$GUARD_DIR" ] && rm -rf "$GUARD_DIR"; return 0; }
if [ "$DRY" -eq 0 ]; then
  if [ -f "$LOCK" ] && kill -0 "$(cat "$LOCK" 2>/dev/null)" 2>/dev/null; then
    log "loop.sh already running here (pid $(cat "$LOCK")); exiting"
    exit 0
  fi
  echo $$ > "$LOCK"
  trap cleanup EXIT
fi

export GIT_TERMINAL_PROMPT=0 GIT_ASKPASS=/usr/bin/true GIT_EDITOR=true

# ---- 2. reattach branch, fetch, fast-forward only --------------------------------------------------
HAS_ORIGIN=0
g remote get-url origin >/dev/null 2>&1 && HAS_ORIGIN=1
if [ "$DRY" -eq 0 ]; then
  if ! g symbolic-ref -q HEAD >/dev/null; then
    log "HEAD detached - reattaching $BRANCH to HEAD"
    g checkout -q -B "$BRANCH" HEAD
  fi
  CUR="$(g symbolic-ref --short HEAD)"
  if [ "$CUR" != "$BRANCH" ]; then
    log "checked out branch is $CUR, not $BRANCH - refusing to cycle"
    exit 1
  fi
  if [ "$HAS_ORIGIN" -eq 1 ] && [ -z "${PHIL_NO_PUSH:-}" ]; then
    if [ "$(g rev-parse --is-shallow-repository 2>/dev/null)" = "true" ]; then
      g fetch -q --unshallow origin 2>/dev/null || log "WARNING: could not unshallow"
    fi
    if g fetch -q origin "$BRANCH" 2>/dev/null; then
      if g merge-base --is-ancestor HEAD "origin/$BRANCH"; then
        g merge -q --ff-only "origin/$BRANCH" || log "WARNING: ff-only sync failed"
      elif ! g merge-base --is-ancestor "origin/$BRANCH" HEAD; then
        log "WARNING: local $BRANCH and origin/$BRANCH diverged - resolve manually; cycling on local state"
      fi
    else
      log "WARNING: could not fetch origin/$BRANCH; cycling on local state"
    fi
  fi
fi

# A dirty protected path before the run would be wiped by the revert step: refuse instead. The check sees
# ignored files under trader/, .githooks/, .github/ too (e.g. a planted trader/core/data/__init__.py).
PRE_CHECK="$("${VALIDATE[@]}" --list-protected --base HEAD)"
DIRTY="$(jget "$PRE_CHECK" 'd["protected_changed"]')"
if [ -z "$DIRTY" ]; then
  log "protected-path check failed to run: $PRE_CHECK"
  exit 1
fi
if [ "$DIRTY" != "[]" ]; then
  log "uncommitted changes in protected paths, refusing to run an agent over them: $DIRTY"
  [ "$DRY" -eq 1 ] && echo "dry-run: protected_changed=$DIRTY"
  exit 1
fi

# ---- 3. tick: which kind is due? -------------------------------------------------------------------
INBOX_DIR="${PHIL_INBOX:-}"   # same override as trader/fleet/inbox.py
[ -n "$INBOX_DIR" ] || INBOX_DIR="$("$PY" -c 'import sys;sys.path.insert(0,".");from trader.core import config;print(config.load_config()["fleet"]["inbox_dir"])' 2>/dev/null || echo "~/.phil-crypto/inbox")"
INBOX_DIR="${INBOX_DIR/#\~/$HOME}"
INBOX_DIR="${INBOX_DIR%/}"
TRIGGERS=""
if [ "$KIND" = "tick" ] || [ "$KIND" = "triggered" ]; then
  TRIGGERS="$(ls "$INBOX_DIR"/trigger-*.json 2>/dev/null | tr '\n' ' ')"
fi
if [ "$KIND" = "tick" ]; then
  KIND="$(PHIL_TRIGGERS="$TRIGGERS" "$PY" - <<'PY'
import datetime as dt, json, os, re, sys
sys.path.insert(0, ".")
from trader.core import config
cfg = config.load_config()
now = os.environ.get("PHIL_NOW")
now = dt.datetime.strptime(now, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc) if now else config.utcnow()
def at(hhmm):
    h, m = map(int, hhmm.split(":"))
    return now.replace(hour=h, minute=m, second=0, microsecond=0)
last = {}
p = config.journal_dir() / "cycles.log"
if p.exists():
    for line in p.read_text(errors="replace").splitlines():
        m = re.match(r"^(\S+Z) (\w+) (ok|error|reverted)\b", line)
        if not m:
            continue
        try:
            t = dt.datetime.strptime(m.group(1), "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc)
        except ValueError:
            continue
        last[m.group(2)] = max(last.get(m.group(2), t), t)
def ran_since(kind, t):
    return kind in last and last[kind] >= t
sched = cfg["schedule"]
try:
    s = json.loads((config.strategy_dir() / "schedule.json").read_text())
except (OSError, ValueError):
    s = {}
due = None
fwd = at(sched["forward_cycle"])
if now >= fwd and not ran_since("forward", fwd):
    due = "forward"
elif os.environ.get("PHIL_TRIGGERS", "").strip() and not ran_since(
        "triggered", now - dt.timedelta(hours=max(float(os.environ.get("PHIL_TRIGGER_COOLDOWN_H") or
                                                        sched.get("triggered_cooldown_hours", 4)), 0.25))):
    # at most one triggered cycle per cooldown window (also after an error/revert): a trigger that is not
    # consumed can never turn into an Opus session every tick, and research still runs in between
    due = "triggered"
elif now >= at(sched["deep_retro"]) and not ran_since("deep_retro", at(sched["deep_retro"])):
    due = "deep_retro"
else:
    r = s.get("research", {}) if isinstance(s, dict) else {}
    every = max(float(r.get("every_hours", sched["research_every_hours"])), float(sched["research_every_hours"]))
    if r.get("enabled", True) and not ran_since("research", now - dt.timedelta(hours=every)):
        due = "research"
print(due or "none")
PY
)"
  if [ "$KIND" = "none" ] || [ -z "$KIND" ]; then
    log "tick: nothing due"
    [ "$DRY" -eq 1 ] && echo "dry-run: kind=none"
    exit 0
  fi
  log "tick: $KIND is due"
fi
[ "$KIND" = "triggered" ] || TRIGGERS=""
[ "$DRY" -eq 1 ] && echo "dry-run: kind=$KIND"

# ---- cycle prompt and command ----------------------------------------------------------------------
KIND_UP="$(printf '%s' "$KIND" | tr '[:lower:]' '[:upper:]')"
CYCLE_FILE="trader/CYCLE_${KIND_UP}.md"
PREAMBLE=""
if [ "$KIND" = "triggered" ] && [ ! -f "$CYCLE_FILE" ]; then
  CYCLE_FILE="trader/CYCLE_FORWARD.md"
  PREAMBLE="TRIGGERED CYCLE (watcher trigger files: ${TRIGGERS:-none}). In this cycle you may ONLY settle, write the retro and pause/resume strategies in trader/strategy/active.json; never add new strategies or hypotheses.

"
fi
if [ ! -f "$CYCLE_FILE" ]; then
  log "missing $CYCLE_FILE"
  cycles_line "$KIND" error "runner=$(hostname -s)" -- "missing $CYCLE_FILE"
  exit 1
fi
PROMPT="${PREAMBLE}$(cat "$CYCLE_FILE")"

MODEL_VAR="PHIL_MODEL_${KIND_UP}"
MODEL="${!MODEL_VAR:-${PHIL_CYCLE_MODEL:-claude-opus-5-5}}"
MAX_TURNS="${PHIL_MAX_TURNS:-200}"
PERM_MODE="${PHIL_PERMISSION_MODE:-dontAsk}"
if [ -n "${PHIL_CYCLE_TIMEOUT_MIN:-}" ]; then
  TIMEOUT_MIN="$PHIL_CYCLE_TIMEOUT_MIN"
else
  case "$KIND" in
    research) TIMEOUT_MIN="$("$PY" -c 'import json,sys
try: print(int(json.load(open("trader/strategy/schedule.json"))["research"]["max_minutes"]))
except Exception: print(60)' 2>/dev/null || echo 60)";;
    forward|deep_retro) TIMEOUT_MIN=45;;
    *) TIMEOUT_MIN=20;;
  esac
fi
# Web: only the research cycle may look things up, and WebFetch only on a few documentation domains (a bare
# WebFetch would let a prompt-injected page make the agent put a secret in a URL to any host).
WEB_DOMAINS="${PHIL_WEBFETCH_DOMAINS:-bybit-exchange.github.io www.bybit.com arxiv.org docs.python.org numpy.org pandas.pydata.org numba.readthedocs.io}"
if [ -n "${PHIL_ALLOWED_TOOLS:-}" ]; then
  read -r -a TOOLS <<< "$PHIL_ALLOWED_TOOLS"
else
  # Reads: the repo's trader/ tree only (dontAsk denies every other read). Glob/Grep follow the Read rules.
  # Writes: only agent state, never code (the agent runs trader modules, so it must not author any).
  # Bash: only trader modules (operator code); no git, no arbitrary scripts.
  TOOLS=("Read(./trader/**)" "Glob" "Grep"
         "Edit(./trader/strategy/**)" "Edit(./trader/journal/retros/**)" "Edit(./trader/journal/proposals.md)"
         "Bash($PY -m trader.*)" "Bash(python3 -m trader.*)")
  if [ "$KIND" = "research" ]; then
    TOOLS+=("WebSearch")
    for d in $WEB_DOMAINS; do TOOLS+=("WebFetch(domain:$d)"); done
  fi
fi
# Deny rules win over allow rules and over acceptEdits: belt and braces if PHIL_PERMISSION_MODE is relaxed.
DENY=("Bash(git:*)" "Bash($PY -m trader.core.lease:*)" "Bash(python3 -m trader.core.lease:*)"
      "Edit(./**/*.py)" "Edit(./**/*.pyc)" "Edit(./**/*.so)" "Edit(./**/*.pth)" "Edit(./**/*.sh)"
      "Edit(./**/.gitignore)" "Edit(./**/.gitattributes)" "Edit(./.git/**)" "Edit(./.githooks/**)"
      "Edit(./.github/**)" "Edit(./trader/core/**)" "Edit(./trader/config/**)" "Edit(./trader/fleet/**)"
      "Edit(./trader/tests/**)" "Edit(./trader/journal/*.jsonl)" "Edit(./trader/journal/cycles.log)"
      "Edit(./trader/journal/*.json)" "Edit(./trader/journal/sweeps/**)" "Edit(~/.gitconfig)" "Edit(~/.config/**)")
# Belt and braces behind the settings isolation: no interpreter one-liners, no network or file-shuffling tools,
# no edits of claude/shell/login configuration (persistence the revert step would not see).
DENY+=("Bash(python3 -c:*)" "Bash(python -c:*)" "Bash($PY -c:*)" "Bash(node:*)" "Bash(npx:*)" "Bash(gh:*)"
       "Bash(curl:*)" "Bash(wget:*)" "Bash(nc:*)" "Bash(ssh:*)" "Bash(scp:*)" "Bash(sftp:*)" "Bash(rsync:*)"
       "Bash(sed:*)" "Bash(cp:*)" "Bash(mv:*)" "Bash(ln:*)" "Bash(chmod:*)" "Bash(printenv:*)" "Bash(env:*)"
       "Bash(sudo:*)" "Bash(osascript:*)" "Bash(launchctl:*)" "Bash(crontab:*)" "Bash(pip:*)" "Bash(pip3:*)"
       "Edit(./.claude/**)" "Edit(./.mcp.json)" "Edit(./CLAUDE.md)" "Edit(./**/CLAUDE.md)" "Edit(~/.claude/**)"
       "Edit(~/.claude.json)" "Edit(~/Library/**)" "Edit(~/.zshrc)" "Edit(~/.zprofile)" "Edit(~/.bashrc)"
       "Edit(~/.bash_profile)" "Edit(~/.profile)" "Edit(~/.ssh/**)" "Edit(~/.venvs/**)")
[ "$KIND" = "research" ] || DENY+=("WebFetch" "WebSearch")
# Reads: no credentials anywhere in the home directory (keys, tokens, agent plists, logs, git remotes with
# tokens), even if the permission mode is relaxed and the Read allow-rule above stops being the only door.
HOME_ABS="$(cd "$HOME" 2>/dev/null && pwd -P || printf '%s' "$HOME")"
for h in "~" "/$HOME" "/$HOME_ABS"; do
  for n in .ssh .gnupg .aws .azure .kube .docker .config .netrc .git-credentials .gitconfig .pypirc .npmrc \
           .phil-crypto .claude .claude.json Library .zsh_history .bash_history .python_history .env .secrets; do
    DENY+=("Read($h/$n)" "Read($h/$n/**)")
  done
done
DENY+=("Read(./.git/**)" "Read(./**/.env)" "Read(./**/*.pem)" "Read(./**/*.key)" "Read(./**/secrets*)"
       "Read(//etc/**)" "Read(//private/**)" "Read(//var/**)" "Read(//tmp/**)")
# Reads: no secrets, no raw market data (sealed periods live there; the vault is the only door).
while IFS= read -r p; do
  [ -z "$p" ] && continue
  DENY+=("Read(/$p)" "Read(/$p/**)")
done < <("$PY" -c 'import os,sys;sys.path.insert(0,".")
from trader.core import config
c = config.load_config()
for v in (os.environ.get("TRADER_DATA_DIR"), c.get("data",{}).get("dir"), c.get("data",{}).get("seed_dir"),
          c.get("real",{}).get("secrets_file"), c.get("fleet",{}).get("inbox_dir")):
    if v:
        a = os.path.abspath(os.path.expanduser(v))
        print(a)
        if os.path.realpath(a) != a: print(os.path.realpath(a))' 2>/dev/null)
# A triggered cycle is a short reaction (settle, pause/resume): it gets a hard spend cap by default.
MAX_BUDGET="${PHIL_MAX_BUDGET_USD:-}"
[ -z "$MAX_BUDGET" ] && [ "$KIND" = "triggered" ] && MAX_BUDGET="${PHIL_TRIGGERED_MAX_BUDGET_USD:-5}"
CMD=()
[ -n "${PHIL_AGENT_USER:-}" ] && CMD+=(sudo -n -H -u "$PHIL_AGENT_USER" --)
# Settings isolation: Claude Code MERGES --allowedTools with the allow rules of ~/.claude/settings*.json, so
# without --setting-sources the operator's own profile (Bash(python3:*), Bash(gh:*), Read(~/**), ...) would be
# usable by the agent. Load only the project source (the repo has no .claude/settings*.json; the agent may not
# create one, see DENY and step 7a) and no MCP server at all (no user github/chrome/... servers).
ISOLATION=(--setting-sources project --strict-mcp-config --mcp-config '{"mcpServers":{}}')
CMD+=("$CLAUDE" -p "$PROMPT" --model "$MODEL" --permission-mode "$PERM_MODE" "${ISOLATION[@]}"
      --allowedTools "${TOOLS[@]}" --disallowedTools "${DENY[@]}")
[ "$MAX_TURNS" != "0" ] && CMD+=(--max-turns "$MAX_TURNS")
[ -n "$MAX_BUDGET" ] && [ "$MAX_BUDGET" != "0" ] && CMD+=(--max-budget-usd "$MAX_BUDGET")

REAL_ENABLED="$("$PY" -c 'import sys;sys.path.insert(0,".");from trader.core import config;print(str(config.load_config()["real"]["real_trading_enabled"]).lower())' 2>/dev/null || echo false)"

if [ "$DRY" -eq 1 ]; then
  echo "dry-run: cycle_file=$CYCLE_FILE model=$MODEL max_turns=$MAX_TURNS max_budget_usd=${MAX_BUDGET:-none} timeout_min=$TIMEOUT_MIN real_enabled=$REAL_ENABLED"
  echo "dry-run: triggers=${TRIGGERS:-none}"
  echo "dry-run: command: $CLAUDE -p <$CYCLE_FILE> --model $MODEL --permission-mode $PERM_MODE ${ISOLATION[*]} ..."
  echo "dry-run: protected_changed=[]"
  exit 0
fi

RUNNER="$("$PY" -c 'import sys;sys.path.insert(0,".");from trader.core import config;print(config.runner_id())' 2>/dev/null || hostname -s)"

# ---- 4. runner lease -------------------------------------------------------------------------------
LEASE_HELD=0
if [ -z "${PHIL_NO_LEASE:-}" ]; then
  if [ -f trader/core/lease.py ]; then
    LEASE_OUT="$("$PY" -m trader.core.lease acquire --cycle "$KIND" 2>&1)"
    LRC=$?
    case "$LRC" in
      0) LEASE_HELD=1;;
      3|4) log "lease not taken ($LRC): $LEASE_OUT"
           cycles_line "$KIND" skipped "runner=$RUNNER" -- "lease not taken (exit $LRC)"
           exit 0;;
      *) log "lease acquire failed ($LRC): $LEASE_OUT"
         cycles_line "$KIND" error "runner=$RUNNER" -- "lease acquire failed (exit $LRC)"
         exit 1;;
    esac
  else
    log "WARNING: trader/core/lease.py not present - running without a runner lease"
  fi
fi
release_lease() {
  if [ "$LEASE_HELD" -eq 1 ]; then
    "$PY" -m trader.core.lease release >/dev/null 2>&1 || log "WARNING: lease release failed (it expires on its own)"
    LEASE_HELD=0
  fi
}
trap 'release_lease; cleanup' EXIT

# ---- 5. inbox merge (berry -> air copies; triggers are moved to inbox/done by inbox.py) -------------
if [ -f trader/fleet/inbox.py ]; then
  "$PY" -m trader.fleet.inbox merge >/dev/null 2>>"$LOGDIR/loop.log" || log "WARNING: inbox merge failed"
fi

# ---- 5b. paper settlement OUTSIDE the agent session (forward / triggered) ---------------------------
# paper-ledger.jsonl / real-ledger.jsonl are session-protected: rows the agent adds during its session are
# reverted by --enforce, so the only rows that reach a commit are written here by trusted code.
BASE0="$(g rev-parse HEAD)"
if { [ "$KIND" = "forward" ] || [ "$KIND" = "triggered" ]; } && [ -z "${PHIL_NO_SETTLE:-}" ]; then
  if [ -f trader/core/paper.py ]; then
    if [ -f trader/core/data.py ]; then
      "$PY" -m trader.core.data update --source spot >>"$LOGDIR/loop.log" 2>&1 || log "WARNING: data update failed"
    fi
    "$PY" -m trader.core.paper run >>"$LOGDIR/loop.log" 2>&1 || log "WARNING: paper run failed"
    for f in trader/journal/paper-ledger.jsonl trader/journal/real-ledger.jsonl; do
      [ -f "$f" ] && g add -- "$f"
    done
    if ! g diff --cached --quiet; then
      SMSG="cycle($KIND): $(date -u +%F) paper settlement (loop)"
      SVAL="$("${VALIDATE[@]}" --staged --message "$SMSG")"
      if [ "$(jget "$SVAL" 'd["ok"]')" = "True" ]; then
        g commit -q --no-verify -m "$SMSG" || log "WARNING: settlement commit failed"
      else
        log "settlement rows failed validation, discarded: $SVAL"
        g reset -q -- trader/journal 2>/dev/null
        g checkout -q HEAD -- trader/journal/paper-ledger.jsonl trader/journal/real-ledger.jsonl 2>/dev/null || true
      fi
    fi
  else
    log "WARNING: trader/core/paper.py missing - no loop-side settlement"
  fi
fi

# ---- 6. snapshots, then headless claude with a wall-clock limit ------------------------------------
PRE="$(g rev-parse HEAD)"
GIT_COMMON="$(g rev-parse --path-format=absolute --git-common-dir 2>/dev/null || g rev-parse --git-common-dir)"
case "$GIT_COMMON" in /*) ;; *) GIT_COMMON="$ROOT/$GIT_COMMON";; esac
GIT_DIR_ABS="$(g rev-parse --path-format=absolute --git-dir 2>/dev/null || g rev-parse --git-dir)"
case "$GIT_DIR_ABS" in /*) ;; *) GIT_DIR_ABS="$ROOT/$GIT_DIR_ABS";; esac
COMMON_NAMES=(config info hooks)
SNAP_COMMON="$(snap_files "$GIT_COMMON" "${COMMON_NAMES[@]}")"
SNAP_WT=""
[ "$GIT_DIR_ABS" != "$GIT_COMMON" ] && SNAP_WT="$(snap_files "$GIT_DIR_ABS" config.worktree info)"
GLOBAL_CFGS=("$HOME/.gitconfig" "${XDG_CONFIG_HOME:-$HOME/.config}/git/config")
SYS_CFG="$(g config --system --show-origin --list 2>/dev/null | head -n 1 | sed -n 's/^file:\([^	]*\)	.*/\1/p')"
[ -n "$SYS_CFG" ] && [ -w "$SYS_CFG" ] && GLOBAL_CFGS+=("$SYS_CFG")
GLOBAL_SNAPS=()
for f in "${GLOBAL_CFGS[@]}"; do GLOBAL_SNAPS+=("$(file_snap "$f")"); done
# Claude Code settings the next session would load (user + project + local sources): a planted allow rule,
# hook or MCP server must not outlive this session.
CLAUDE_CFGS=("$HOME/.claude/settings.json" "$HOME/.claude/settings.local.json" "$ROOT/.claude/settings.json"
             "$ROOT/.claude/settings.local.json" "$ROOT/.mcp.json")
CLAUDE_SNAPS=()
for f in "${CLAUDE_CFGS[@]}"; do CLAUDE_SNAPS+=("$(file_snap "$f")"); done
# The guard itself, as committed at PRE (PRE_CHECK proved the work tree equals it): enforce runs from this
# copy, so nothing the agent changed under trader/ is imported by the code that undoes the change.
GUARD_SNAP="$(snap_files "$ROOT" trader/__init__.py trader/core/__init__.py trader/core/config.py trader/core/validate.py)"

TS="$(date -u +%Y%m%dT%H%M%SZ)"
CLOG="$LOGDIR/${TS}-${KIND}.log"
log "running $KIND with $MODEL (mode=$PERM_MODE, max_turns=$MAX_TURNS, timeout=${TIMEOUT_MIN}m), log $CLOG"
PHIL_CYCLE_KIND="$KIND" PHIL_LEASE=acquired PHIL_PUSH_BY_LOOP=1 PHIL_REAL_ENABLED="$REAL_ENABLED" \
PHIL_TRIGGER_FILES="$TRIGGERS" \
  "${CMD[@]}" > "$CLOG" 2>&1 < /dev/null &
CPID=$!
TIMED_OUT=0
SECS=$((TIMEOUT_MIN * 60))
WAITED=0
while kill -0 "$CPID" 2>/dev/null; do
  if [ "$WAITED" -ge "$SECS" ]; then
    TIMED_OUT=1
    log "cycle exceeded ${TIMEOUT_MIN}m - terminating claude (pid $CPID)"
    kill -TERM "$CPID" 2>/dev/null
    for _ in 1 2 3 4 5 6 7 8 9 10; do kill -0 "$CPID" 2>/dev/null || break; sleep 1; done
    kill -KILL "$CPID" 2>/dev/null
    break
  fi
  sleep 2
  WAITED=$((WAITED + 2))
done
wait "$CPID" 2>/dev/null
CRC=$?
[ "$TIMED_OUT" -eq 1 ] && CRC=124

# ---- 7. enforce the boundary -----------------------------------------------------------------------
# 7a. git control state back to the pre-session snapshot BEFORE any git command runs (a planted fsmonitor,
#     hook, exclude, sparse-checkout or alias must never execute or hide anything).
restore_files "$GIT_COMMON" "$SNAP_COMMON" "${COMMON_NAMES[@]}"
[ "$GIT_DIR_ABS" != "$GIT_COMMON" ] && restore_files "$GIT_DIR_ABS" "$SNAP_WT" config.worktree info
rm -rf "$GIT_COMMON/refs/replace" 2>/dev/null
i=0
for f in "${GLOBAL_CFGS[@]}"; do file_restore "$f" "${GLOBAL_SNAPS[$i]}"; i=$((i + 1)); done
i=0
for f in "${CLAUDE_CFGS[@]}"; do file_restore "$f" "${CLAUDE_SNAPS[$i]}"; i=$((i + 1)); done
# 7b. Squash any commit the agent made (it should not commit at all; this also neutralises a forged
#     "operator:" subject).
if [ "$(g rev-parse -q --verify "refs/heads/$BRANCH")" != "$PRE" ] || \
   [ "$(g symbolic-ref -q HEAD)" != "refs/heads/$BRANCH" ]; then
  log "agent moved HEAD or $BRANCH (commits/checkout) - resetting the ref to $PRE, keeping the work tree"
fi
g update-ref "refs/heads/$BRANCH" "$PRE"
g symbolic-ref HEAD "refs/heads/$BRANCH"
# 7c. revert everything outside the whitelist with the snapshotted guard (python -I -S: no site-packages).
GUARD_DIR="$(mktemp -d "${TMPDIR:-/tmp}/phil-guard.XXXXXX")"
restore_files "$GUARD_DIR" "$GUARD_SNAP"
ENF="$(TRADER_ROOT="$TRADER_DIR" "$PY" -I -S -B "$GUARD_DIR/trader/core/validate.py" --enforce "$PRE")"
if [ "$(jget "$ENF" 'd["ok"]')" != "True" ]; then
  log "enforce failed ($ENF) - dropping the whole session"
  g reset -q --hard "$PRE"
  g clean -q -fdx -- trader/core trader/config trader/fleet trader/tests .githooks .github 2>/dev/null
  g clean -q -fd -- trader 2>/dev/null
  ENF='{"ok": true, "reverted": ["<whole session>"], "restored": []}'
fi
# 7d. consume the watcher triggers this cycle was started for - whatever the outcome (ok, error, timeout,
#     revert): the session ran, and a trigger left pending would start another one every tick. Only the files
#     listed before the session are touched; trigger files that arrived meanwhile wait for the next window.
#     inbox.py runs from the tree enforce just restored (protected paths equal PRE again).
if [ -n "$TRIGGERS" ]; then
  for t in $TRIGGERS; do
    { [ -e "$t" ] || [ -L "$t" ]; } || continue
    if [ -f trader/fleet/inbox.py ] && "$PY" -m trader.fleet.inbox consume "$t" >>"$LOGDIR/loop.log" 2>&1; then
      continue
    fi
    if [ -f "$t" ] && [ ! -L "$t" ] && [ "$(dirname "$t")" = "$INBOX_DIR" ]; then
      mkdir -p "$INBOX_DIR/done" && mv -f -- "$t" "$INBOX_DIR/done/" && continue
    fi
    # not a plain trigger file any more (symlink, directory, ...): park it where the listing cannot see it
    mkdir -p "$PHIL_HOME/quarantine" && mv -f -- "$t" "$PHIL_HOME/quarantine/${TS}-$(basename "$t")" \
      && log "WARNING: trigger $t could not be consumed - moved to $PHIL_HOME/quarantine" \
      || log "WARNING: trigger $t could not be consumed or quarantined"
  done
fi
REVERTED="$(jget "$ENF" 'd.get("reverted", [])')"
RESTORED="$(jget "$ENF" 'd.get("restored", [])')"
STATUS=ok
NOTE=""
if [ -n "$REVERTED" ] && [ "$REVERTED" != "[]" ]; then
  STATUS=reverted
  NOTE="reverted protected edits: $REVERTED"
  log "WARNING: agent touched protected paths - reverted: $REVERTED"
fi
if [ -n "$RESTORED" ] && [ "$RESTORED" != "[]" ]; then
  STATUS=reverted
  NOTE="$NOTE restored rewritten journal/hypothesis files: $RESTORED"
  log "WARNING: append-only/immutable files rewritten - restored: $RESTORED"
fi
if [ "$CRC" -ne 0 ]; then
  STATUS=error
  NOTE="claude exit $CRC$([ "$TIMED_OUT" -eq 1 ] && echo ' (timeout)'); $NOTE"
fi
# The summary is agent text that lands in cycles.log and the pushed commit message: printable ASCII only,
# and anything shaped like a credential (key blocks, known token prefixes, key=value secrets, long
# high-entropy words) is redacted before it can leave this host.
SUMMARY="$(grep -v '^[[:space:]]*$' "$CLOG" 2>/dev/null | tail -n 1 | redact_summary)"
[ -z "$SUMMARY" ] && SUMMARY="no output"

# ---- 8. stage whitelist, validate, commit ----------------------------------------------------------
# From here on trader/ equals PRE outside the whitelist, so the in-tree guard is the committed one.
STATS="$("${VALIDATE[@]}" --cycle-stats "$PRE")"
DUR=$(( $(date -u +%s) - START_EPOCH ))
F_TRIALS="trials=+$(jget "$STATS" 'd.get("trials",0)')"
F_LOOKS="looks=+$(jget "$STATS" 'd.get("looks",0)')"
F_FILLS="fills=$(jget "$STATS" 'd.get("fills",0)')"
cycles_line "$KIND" "$STATUS" "runner=$RUNNER" "dur=${DUR}s" "$F_TRIALS" "$F_LOOKS" "$F_FILLS" \
  "base=$(g rev-parse --short "$PRE")" -- "${NOTE:+$NOTE; }$SUMMARY"
g add -A -- trader/strategy trader/journal
MSG="cycle($KIND): $(date -u +%F) $SUMMARY"
VAL="$("${VALIDATE[@]}" --staged --message "$MSG")"
if [ "$(jget "$VAL" 'd["ok"]')" != "True" ]; then
  log "validation failed after enforce: $VAL - discarding the agent's strategy edits and retrying"
  mkdir -p "$PHIL_HOME/rejected"
  g diff --no-ext-diff --no-textconv --cached "$PRE" > "$PHIL_HOME/rejected/${TS}-${KIND}.patch" 2>/dev/null || true
  g reset -q -- trader/strategy
  g checkout -q "$PRE" -- trader/strategy 2>/dev/null || true
  g clean -q -fd -- trader/strategy 2>/dev/null || true
  g add -A -- trader/strategy trader/journal
  VAL="$("${VALIDATE[@]}" --staged --message "$MSG")"
  if [ "$(jget "$VAL" 'd["ok"]')" != "True" ]; then
    log "validation still failing - dropping the whole cycle (patch kept in $PHIL_HOME/rejected): $VAL"
    g reset -q --hard "$PRE"
    g clean -q -fd -- trader/strategy trader/journal 2>/dev/null || true
    cycles_line "$KIND" error "runner=$RUNNER" "dur=${DUR}s" "base=$(g rev-parse --short "$PRE")" \
      -- "validation failed, cycle dropped; see $PHIL_HOME/rejected/${TS}-${KIND}.patch"
    g add -- "$CYCLES_LOG"
    g commit -q --no-verify -m "cycle($KIND): $(date -u +%F) dropped (validation failed)" || true
    STATUS=error
  else
    STATUS=reverted
  fi
fi
if ! g diff --cached --quiet; then
  # validated just above; --no-verify (and hooksPath=/dev/null) so no hook of any kind runs here
  g commit -q --no-verify -m "$MSG" || log "WARNING: commit failed"
fi

# ---- 8b. final gate: the exact commits about to be pushed, as CI will see them ---------------------
gate() {  # gate <base>: 0 if every commit in base..HEAD passes the boundary check
  local out
  out="$("${VALIDATE[@]}" --diff "$1..HEAD" --quick)"
  [ "$(jget "$out" 'd["ok"]')" = "True" ] && return 0
  log "final gate failed for $1..HEAD: $out"
  return 1
}
PUSH_OK=1
if ! gate "$BASE0"; then
  mkdir -p "$PHIL_HOME/rejected"
  g diff --no-ext-diff --no-textconv "$BASE0" HEAD > "$PHIL_HOME/rejected/${TS}-${KIND}-gate.patch" 2>/dev/null || true
  g reset -q --hard "$BASE0"
  cycles_line "$KIND" error "runner=$RUNNER" "base=$(g rev-parse --short "$BASE0")" \
    -- "final boundary gate failed, cycle dropped; see $PHIL_HOME/rejected/${TS}-${KIND}-gate.patch"
  g add -- "$CYCLES_LOG"
  g commit -q --no-verify -m "cycle($KIND): $(date -u +%F) dropped (boundary gate)" || true
  STATUS=error
  PUSH_OK=0
fi

# ---- 9. push (from this shell; CI re-checks the boundary on the pushed range) ----------------------
if [ "$PUSH_OK" -eq 1 ] && [ "$HAS_ORIGIN" -eq 1 ] && [ -z "${PHIL_NO_PUSH:-}" ]; then
  if ! g push -q origin "HEAD:$PUSH_BRANCH" 2>>"$LOGDIR/loop.log"; then
    log "push rejected - rebasing onto origin/$PUSH_BRANCH and retrying"
    if g pull -q --rebase origin "$PUSH_BRANCH" 2>>"$LOGDIR/loop.log"; then
      if gate "origin/$PUSH_BRANCH"; then
        g push -q origin "HEAD:$PUSH_BRANCH" 2>>"$LOGDIR/loop.log" || log "WARNING: push still failing - resolve manually"
      else
        log "WARNING: rebased range fails the boundary gate - not pushing, resolve manually"
      fi
    else
      g rebase --abort 2>/dev/null || true
      log "WARNING: rebase failed - commits are local only, resolve manually"
    fi
  fi
fi

# ---- 10. lease release -----------------------------------------------------------------------------
release_lease
log "cycle $KIND finished: $STATUS"
[ "$STATUS" = "error" ] && exit 1
exit 0
}
