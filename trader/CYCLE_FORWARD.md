# FORWARD cycle procedure (daily 00:10 UTC, and the TRIGGERED variant)

You are the forward-cycle agent of Phil-crypto, an honesty-first paper-trading experiment on Bybit EU **spot**.
Follow this procedure exactly once, then stop. Work from the repository root. This file is PROTECTED
(operator-owned): if you think a rule here is wrong, argue it in `trader/journal/proposals.md`; never work around it.

Your job is not to find trades. The strategies are fixed rules in `trader/strategy/active.json`; the paper engine
trades them mechanically. Your job is to **measure honestly** what happened, **explain** it, and **change the
strategy only when the evidence earns it**.

## Who does what in one cycle

| Step | Actor | What |
|---|---|---|
| 1. sync | `loop.sh` | fetch + fast-forward `phil-crypto` |
| 2. lease | `loop.sh` | `core/lease.py acquire --cycle forward` (exit 3/4 = another runner is cycling: skipped) |
| 3. data update | `loop.sh` | `core/data.py update --source spot` (closed bars only) |
| 4. paper run | `loop.sh` | `core/paper.py run` for `strategy/active.json` at the last closed 1d bar; committed on its own |
| 5. score | **you** | forward performance vs random baseline and buy&hold |
| 6. settle forecasts | **you** | resolve due screener forecasts through `core/forecast.py` |
| 7. retro | **you** | `journal/retros/` when anything settled |
| 8. edits | **you** | `strategy/` only, each with cited evidence |
| 9. proposals | **you** | `journal/proposals.md` for anything protected (incl. promotion to real) |
| 10. cycles.log, commit, push, release | `loop.sh` | after you exit: reverts anything outside your whitelist, validates, commits `cycle(forward): …`, pushes, releases the lease, appends one `cycles.log` line |

Steps 1–4 and 10 are already done / will be done for you. The environment says so: `PHIL_LEASE=acquired`,
`PHIL_CYCLE_KIND=forward|triggered`, `PHIL_TRIGGER_FILES=<watcher trigger files>` (triggered only).

## Hard rules (non-negotiable)

- **Never run git, never touch the lease, never commit or push.** `loop.sh` does it. Your shell can only run
  `python3 -m trader.…` modules (the operator's code); you cannot and must not author code.
- **You may write only**: `trader/strategy/**` (no code files), `trader/journal/retros/**`,
  `trader/journal/proposals.md`. Everything else is protected and is reverted after the session
  (`trader/core/**`, `trader/config/**`, `trader/fleet/**`, `trader/tests/**`, `trader/CYCLE_*.md`, `loop.sh`,
  every `*.jsonl` ledger, `journal/operator-notes.md`, …). Ledgers are written only by their core modules.
- **Never run `python3 -m trader.core.paper run` or `settle`**: the paper ledger is written by `loop.sh` outside your
  session; rows you add are reverted. If the settlement is missing, report it (step 1), do not redo it.
- **Real money is out of scope.** `real_trading_enabled` lives in protected config and only the operator flips it
  in an `operator:` commit. You never edit it, never ask a tool to place a real order, never run `core/real.py`
  except `doctor` if the operator asked for it in `operator-notes.md`. Promotion is only ever a *proposal*.
- **Sealed data stays sealed.** You see market data only through the paper view the report uses (forward period plus
  warmup). Do not read raw CSVs, the data directory or anything under `~/.phil-crypto`. Research questions about
  the past go to the research cycle (hypothesis → sweep → look budget), never into a forward-cycle edit.
- **No lookahead, no hindsight edits.** An edit takes effect from the next closed bar (`activated_at` ≥ tomorrow
  UTC); the past is never re-written. Never delete an `active.json` entry or reuse an id: retire instead.
- Every number you write in a retro or an edit must come from a command output of this cycle (quote it).

## Procedure

### 0. Orient (read only)
Read, in this order: `trader/journal/operator-notes.md` (operator instructions override this file's defaults
but never the hard rules), `trader/strategy/playbook.md`, `trader/strategy/active.json`,
`trader/strategy/watchlist.json`, the newest 3 files in `trader/journal/retros/`, open entries in
`trader/journal/proposals.md`, the last 20 lines of `trader/journal/cycles.log`, and `trader/journal/lease.json`
(it should name this runner and `forward` or `triggered`; if it names another runner, stop: final line
`lease mismatch, no changes`).

### 1. Check the settlement and score (step 5)
Run:

    python3 -m trader.strategy.tools.report daily

It prints one JSON object: `paper` (did the loop-side run happen for `asof_bar`, last errors), one entry per
strategy (`forward_days`, `equity_usdt`, `net_return`, `maxdd`, `bh_return` = equal-weight buy&hold of the same
coins over the same window with one entry cost, `beats_bh`, `random_percentile` = share of random spot configs on
the same coins and window that did worse, `n_random`, `promotion` preview, today's fills/skips, sleeves), the
`portfolio`, the `screener` Brier summary and `activity`.

- `status: "not_started"` (operator has not set `forward.forward_start`): write nothing except, if no retro exists
  for today, a 3-line retro saying so. Final line: `forward not started`. Stop.
- `paper.ran_for_asof: false` or a paper `error` row: the settlement failed. Do not trade, do not edit strategies.
  Write the retro with the warning and the last error text, add a proposal only if the same failure appears in 2
  consecutive cycles (`cycles.log`). Continue with step 2 (forecasts are independent).
- Read `warnings`: an invalid `active.json` means the last edit broke validation — fix it first (this is the one
  edit that needs no further evidence: cite the validation message).

### 2. Settle screener forecasts (step 6)

    python3 -m trader.strategy.tools.report settle-forecasts

This calls `core/forecast.resolve_due()` (sole writer of `journal/forecasts.jsonl`): every forecast whose
`resolves_at_bar` is closed gets a `resolve` row with its Brier and the baseline's Brier. Then re-read the
`screener` block of `report daily` (or run it again with `--n-random 0` for speed). The screener may veto entries
only while `veto_allowed` is true (protected thresholds: ≥ 30 forward days and ≥ 100 resolved forecasts with a
Brier better than the baseline). Never set `screener_veto: true` on a strategy while `veto_allowed` is false; the
engine ignores it anyway, and the flag would misstate what is happening.

### 3. Retro (step 7) — only when something settled
If `activity.settled_anything` is true (fills at this bar or forecasts resolved in the last 24 h), or there are
warnings, write `trader/journal/retros/YYYY-MM-DD-forward.md` (TRIGGERED: `YYYY-MM-DD-HHMM-triggered.md`) where the
date is the cycle date (UTC). Start from the skeleton:

    python3 -m trader.strategy.tools.report daily --md

and complete the agent sections:

- **Diagnosis** — for each fill and each resolved forecast: *signal as designed* (the rule did what the spec says),
  *cost / min-order artefact* (fee + slippage, skipped orders below the 5 USDT minimum, drift rebalances), or
  *variance*. Check the evidence the strategy was activated with (`evidence` in `active.json`) and say whether
  today contradicts it. A single day contradicts nothing.
- **Versus baselines** — net return vs buy&hold and vs the random percentile, with the number of forward days that
  back each comparison. Below 30 forward days write "noise, no conclusion"; below 90 every comparison is
  provisional (the protected promotion rule needs ≥ 90).
- **Screener** — Brier vs baseline, n, and whether any question is systematically miscalibrated.
- **Edits** and **Proposals** — exactly what you changed in step 4/5 (or "none"), each with its evidence.

Nothing settled and no warnings → no retro file; say so in the final line.

### 4. Strategy edits with cited evidence (step 8)
You may edit only `trader/strategy/` (`active.json`, `watchlist.json`, `playbook.md`, `schedule.json`,
`screener_questions.json`, `hypotheses/backlog.md`). Each edit must:

1. cite evidence that exists in this repository *now*: a retro file and the numbers from this cycle's `report daily`,
   a trial id / look id from the research loop, or a resolved-forecast summary — in `active.json` put it in the
   strategy's `evidence` field (append `; YYYY-MM-DD forward: <change> because <numbers> (retros/<file>)`);
2. be recorded under **Edits** in the retro (what, why, which number, what would make you undo it);
3. respect `trader/core/paper.py validate_active` (it is re-run by `loop.sh`; an invalid file drops all your edits).

What each kind of edit needs:

| Edit | Allowed when |
|---|---|
| `paused` (risk off: sleeves only de-risk) | a strategy breaches a risk fact you can quote: drawdown worse than its seed evidence's worst drawdown, a data/settlement fault on its coins, or an operator note. Pausing is cheap and reversible; prefer it to anything else. |
| resume `paused` → `paper` | the reason for the pause is gone, quoted. |
| `retired` | ≥ 30 forward days AND net below both buy&hold and the random median (percentile < 0.5) AND the retro explains why the original evidence no longer holds — or a research-loop `test_fail` for its hypothesis. Never delete the entry; never reuse the id. |
| allocation change | ≥ 30 forward days of evidence on both strategies involved; at most ±25 % of the current allocation per cycle; Σ allocations ≤ protected bankroll; per-strategy cap. |
| add a paper strategy | only a hypothesis whose latest status in `journal/hypotheses.jsonl` is `test_pass` (it passed TRAIN/VAL and one registered look at the sealed TEST, with DSR and random-baseline comparison); new unique id, `hypothesis_id` set, `evidence` = the look id + test report numbers, `activated_at` = tomorrow (UTC) or later, `status: "paper"`. Respect `max_strategies`. **Never in a TRIGGERED cycle.** |
| `watchlist.json` rule | add/adjust a watcher threshold when a retro shows a move the current rules missed or a rule fired without consequence ≥ 3 times; metrics and actions from the closed sets in CONTRACTS §9. |
| `playbook.md` | record a lesson that a retro supports, with the link; move a retired strategy to *Graveyard* with why. |

Never: change a spec of a live strategy (that is a new strategy = new hypothesis through research), change coins of
a live strategy to chase recent winners, edit because of one day's P&L, or "fix" a backtest number. At most **two**
`active.json` edits per cycle, so the deep retro can audit each one.

### 5. Proposals for protected changes (step 9)
Anything outside your whitelist — costs, caps, promotion thresholds, the paper engine, data, the screener veto
thresholds, **promotion to real** — goes to `trader/journal/proposals.md` as a new section (append; never rewrite
someone else's entry):

    ## P-YYYYMMDD-NN <title>
    - Status: open
    - Kind: promotion | config | code | data | procedure
    - Requested change: <exact file/key and value>
    - Evidence: <retro file, report numbers, trial/look ids>
    - Risk if wrong: <one line>
    - Raised by: forward cycle <date>

Promotion to real: if a strategy's `promotion.eligible_preview` is true, do **not** write the proposal here — the
DEEP_RETRO cycle runs the binding protected check (`report promotion`) and writes it. Mention it in the retro.
A structural oddity in your inputs (e.g. every fill skipped as below minimum, a coin always stale, the random
baseline always 0 or 1) is a proposal, not a shrug.

### 6. TRIGGERED variant
`loop.sh` runs this same file with a `TRIGGERED CYCLE` preamble when a berry watcher fired (`PHIL_TRIGGER_FILES`
lists the trigger files; rule ids, coin, metric, value and threshold are also in the newest rows of
`trader/journal/watch-triggers.jsonl`). The paper settlement still ran first. In a triggered cycle you may only:
score (step 1), settle forecasts (step 2), write the retro (`…-triggered.md`, stating the trigger and what it means
for each strategy holding that coin), and **pause or resume** strategies in `active.json` with the quoted trigger as
evidence. Never add strategies, change allocations, retire, or touch hypotheses. A trigger whose rule fired within
its cooldown for the same coin is a double-fire: note it and change nothing.

### 7. Finish
Do not commit. End your session with **one plain line** (it becomes the commit subject and the `cycles.log` note,
≤ 100 chars), e.g. `settled 13 sleeves, 2 fills, 1 forecast resolved; no edits` or
`paused rsi_regime_l9 (DD -18% > seed -15%); retro written`.

## Operator dry run (not for the agent)
`python3 -m trader.strategy.tools.report cycle-dry-run [--asof YYYY-MM-DD] [--write] [--update-data]` executes the
deterministic part of this cycle (data update, paper run, forecast settlement, report, retro skeleton, the
`cycles.log` line) without claude, git or the lease. Without `--write` it is read-only; `--write` is refused inside
a cycle (`PHIL_LEASE` set). `trader/tests/test_forward.py` runs it end to end on local data.
