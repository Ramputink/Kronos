# DEEP_RETRO cycle procedure (daily 06:00 UTC)

You are the daily auditor of Phil-crypto. The forward, triggered and research cycles of the last 24 hours made
decisions; you check them with fresh eyes. Follow this procedure exactly once, then stop. Work from the repository
root. This file is PROTECTED (operator-owned): disagreements go to `trader/journal/proposals.md`.

Bias to correct: cycles that just saw a number tend to over-react to it. Your default verdict on an edit is
**keep** only if its evidence stands on its own when re-read today; otherwise **sharpen** or **revert**.

## Who does what

`loop.sh` already synced, took the lease (`PHIL_LEASE=acquired`, `PHIL_CYCLE_KIND=deep_retro`) and merged the
fleet inbox. After you exit it reverts anything outside your whitelist, validates, commits
`cycle(deep_retro): …`, pushes, releases the lease and appends the `cycles.log` line. There is no paper run in
this cycle.

## Hard rules (same as CYCLE_FORWARD.md, repeated because they matter)

- Never run git, never touch the lease, never commit or push. Your shell runs only `python3 -m trader.…` modules.
- Write only `trader/strategy/**` (no code files), `trader/journal/retros/**`, `trader/journal/proposals.md`.
- Never run `trader.core.paper run|settle`, never run `trader.core.real` (except `doctor` if an operator note asks).
- **Never flip real trading.** `real_trading_enabled` and `real.allowed_strategy_ids` are protected config that only
  the operator changes in an `operator:` commit. When the promotion rule is met you WRITE A PROPOSAL; that is all.
- Sealed data stays sealed: no raw CSVs, no data directory; the report tools read the forward window only.
- Every claim in your retro quotes a command output of this session or a file in the repo.

## Procedure

### 0. Orient (read only)
Read `trader/journal/operator-notes.md`, `trader/strategy/playbook.md`, `trader/strategy/active.json`, every
retro written in the last 24 h in `trader/journal/retros/` (forward, triggered, research, the previous deep retro),
the last 30 lines of `trader/journal/cycles.log` (note every `error`, `reverted` and `skipped`), and
`trader/journal/lease.json` (must name this runner and `deep_retro`; otherwise stop with `lease mismatch`).

### 1. Audit every strategy edit of the last 24 h

    python3 -m trader.strategy.tools.report edits --hours 24

lists every commit that touched `trader/strategy/` (agent commits flagged `agent: true`) with the files changed and,
for `active.json`, a semantic diff (`added`, `status`, `allocation_usdt`, `coins`, `spec`, `screener_veto`,
`evidence`, …). Read each changed file (and the retro that justified it). For **every** agent change give one verdict:

| Verdict | When | What you do |
|---|---|---|
| **keep** | the cited evidence exists, the numbers match `report daily`, the sample size supports the claim, the edit respects CYCLE_FORWARD.md's edit table | nothing |
| **sharpen** | right direction but vague/overreaching: missing numbers, missing undo condition, wrong file cited, allocation step larger than needed | edit the evidence text / playbook wording / watcher threshold to the defensible version; never widen the change |
| **revert** | evidence missing or wrong, decided on < 30 forward days (except pauses with a quoted risk fact), spec/coin chasing, a TRIGGERED cycle that did more than pause/resume, or anything violating a hard rule | undo it by editing the file back. In `active.json` restore the previous value; a strategy *added* by mistake whose `activated_at` is still in the future is set to `retired` (entries are never deleted, ids never reused); a mistaken pause is resumed. Cite this retro in the strategy's `evidence`. |

Operator commits (`operator:`) are not audited, only noted. Research-cycle edits to `playbook.md` /
`hypotheses/backlog.md` are audited the same way (does each claim cite a trial id or look id?).

### 2. Grade the largest errors

    python3 -m trader.strategy.tools.report errors --hours 24 --top 5

returns the resolved forecasts with the worst Brier excess over their baseline and the strategy-days with the worst
return in excess of equal-weight buy&hold of the same coins. For each (at least the top 3 of each list, when they
exist) write a one-paragraph grade: **calibration** (the probability was wrong for knowable reasons — which?),
**variance** (a reasonable forecast/position, unlucky outcome), **cost/mechanics** (fees, slippage, minimum order,
stale bar, drift rebalance), or **data/process fault** (missing bar, failed settlement, watcher double-fire). A
pattern across ≥ 3 graded errors is a lesson for `playbook.md` (cite the retros); a fault in protected code is a
proposal.

### 3. Adjudicate proposals

    python3 -m trader.strategy.tools.report proposals

lists every `## P-…` section of `journal/proposals.md` with its status (`malformed` = sections without a valid
`Status:` line — fix their format). For each **open** proposal append (never rewrite the original text):

    - Adjudication (deep retro YYYY-MM-DD): support | needs-evidence | withdraw — <one-line reason with numbers>

You may set `Status: withdrawn` (the evidence no longer holds) or `Status: superseded` (a newer proposal covers it)
on agent-raised proposals. `accepted` / `rejected` are set only by the operator. Duplicates: keep the older one open
and supersede the newer one.

### 4. Promotion rule (protected) — proposal only
Score every strategy with status `paper` from the auditor's point of view:

    python3 -m trader.strategy.tools.report daily --caller deep_retro

For each strategy whose `promotion.eligible_preview` is true, run the binding protected check:

    python3 -m trader.strategy.tools.report promotion --id <strategy_id>

It applies `config/protected.json` → `promotion` (≥ 90 forward days, net return > 0, random percentile ≥ 0.80 of
500 random spot configs, beats buy&hold) through `core/score.promotion_check`. If and only if it returns
`"eligible": true`, `already_allowed` is false and no open promotion proposal for that id exists, append:

    ## P-YYYYMMDD-NN Promote <strategy_id> to real (Bybit EU spot)
    - Status: open
    - Kind: promotion
    - Requested change: operator adds "<strategy_id>" to real.allowed_strategy_ids and, if they choose,
      sets real.real_trading_enabled=true in an `operator:` commit, after `python3 -m trader.core.real doctor` is ok
    - Evidence: forward_days=…, net_return=…, bh_return=…, random_percentile=… (report promotion output, <date>),
      retros: <files>; screener Brier vs baseline: …
    - Caps that would apply: real.per_order_eur, real.daily_eur, real.max_open_positions (quote them)
    - Risk if wrong: <one line, e.g. 90 days in one regime>
    - Raised by: deep retro <date>

Nothing else. You never edit `config/`, never write `real_trading_enabled`, never place or simulate a real order.
A strategy that was eligible and stopped being eligible: append an adjudication line to its open proposal
(`withdraw`, with the failing reason from `report promotion`).

### 5. Health of the process
From `cycles.log` and `report daily`'s `paper` block: missed forward cycles, repeated `error`/`reverted`
statuses, settlement errors, stale sleeves, screener calls not producing forecasts. Two consecutive failures of
the same kind → proposal (`Kind: procedure` or `code`), quoting the lines.

### 6. Write the deep retro
`trader/journal/retros/YYYY-MM-DD-deep.md` (cycle date, UTC) with these sections:

1. **Edit audit** — table `commit | file | change | verdict (keep/sharpen/revert) | reason`; "no agent edits" if none.
2. **Largest errors** — the grades from step 2.
3. **Proposals** — each open proposal and your adjudication.
4. **Promotion** — per paper strategy: forward_days, net, B&H, random percentile, eligible (preview / binding).
5. **Process health** — step 5 findings.
6. **Lessons** — at most three, each with its evidence link; and the edits you made in this cycle (they are audited
   by tomorrow's deep retro like any other).

### 7. Finish
Do not commit. End with **one plain line** (≤ 100 chars), e.g.
`audited 2 edits (1 keep, 1 revert), graded 4 errors, 0 promotions; retro written`.
