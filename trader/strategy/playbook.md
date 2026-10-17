# Phil-crypto playbook

AGENT-EDITABLE (`trader/strategy/**`). Version v0, 2026-10-16. The operator seeded it from `research/` and it is
**unproven forward**. Every edit cites its evidence: a trial id (`T-…`/`B-…`), a look id (`L-…`), a paper-ledger
run id (`P-…`), a retro file, or a path under `research/`. A claim with no id is an opinion, and it goes under
Open questions, not Findings.

How to use this file: run `grep -n '^## ' trader/strategy/playbook.md` for the section map. Read Rules of honesty
every cycle. Keep the file under ~600 lines. When a section grows, move settled narrative into
`journal/retros/` and leave a one-line pointer.

## Thesis

1. We cannot predict crypto direction. Kronos zero-shot, LLM forecasts and 2 M brute-force configs all failed to
   show a directional edge out of sample (see the Graveyard section).
2. What has survived is **risk-shaping, not prediction**:
   - few trades (daily bars, hysteresis);
   - going to cash in downtrends (RSI regime, EMA200, ADX);
   - sizing by volatility (vol-target 0.5).
   In a bear market this protects capital. In a bull market it lags buy & hold.
3. Executable venue: Bybit EU, **spot only, long-only** (Spain → no perps). Cost model: 0.25 % taker + 5 bp
   slippage per side (`protected.costs.spot`, 0.0030 on turnover). Any edge that does not survive 0.30 % per side
   does not exist for us.
4. Honest expectation for the seed strategies (INFORME_TEST.md §3): 10–25 %/yr in a good scenario, with
   −25 to −35 % drawdowns and possible negative years (2022: −17 % to −25 %). A target above 15 % a year is not
   supported by any robust out-of-sample evidence so far.

## Findings (evidence-backed)

| # | Finding | Evidence | Strength |
|---|---|---|---|
| F1 | RSI(14) daily regime (enter > 55, exit < 50) + vol-target 0.5, spot long-only, is the best simple rule found. TEST 2025-07→2026-10 (bear): ETH +26.8 %/yr (DD −32 %), SOL +12.1 %, BNB +38 %, 9-coin portfolio +12.1 % (DD −25 %). At 0.25 % fee: ETH +23.0 %, portfolio +9.1 %. Deflated Sharpe 0.55 (N = 960). | `research/kronos_lab/indicators/INFORME.md`, `indicators/final_test_output.txt`; spec hash `cd9ff45458116162a26b8d3c8fb059ec` | Moderate: one bear TEST period, DSR only just above 0.5 |
| F2 | Same rule at Bybit EU costs, 15-month TEST totals: ETH only +30.6 %, BTC+ETH+SOL+BNB +22.1 %, 9 large caps +10.5 %, vs BTC buy & hold −22.6 % and ETH buy & hold +0.2 %. Worst drawdowns −25 to −35 %. | `research/kronos_lab/overnight/INFORME_TEST.md` §2 | Moderate. The 4-coin and ETH+BNB sets were chosen after seeing the TEST (optimistic). The 9-coin set is clean. |
| F3 | Breakout 50d + EMA200 with a BTC > EMA200 gate (goal15 C4, cross-sectional top-10 by 60d momentum, 15-day hold): OOS-A 2024-06→2026-06 SPOT +33.5 %/yr, Sharpe 0.92, DSR 0.75 (N = 12), maxDD −39 %. Full period 2021-26 +20.3 %/yr vs BTC +19.4 %, maxDD −54 %. **Failed the strict forward check**: +15 % vs BTC +29 % over 4 months. | `research/kronos_lab/goal15/PREREGISTRATION.md`, `goal15/evaluate_output.txt`, `research/README.md` §3 | Weak to moderate. Large DD; not expressible in the per-coin spec (see H-0004 and the backlog). |
| F4 | Daily beats 4h (costs). Spot long-only beats PERP with shorts (funding plus losing shorts). | `indicators/INFORME.md` "Lo que no funcionó" | Moderate |
| F5 | ADX gate and vol-targeting help as **risk filters**. Per-coin best configs across 29 coins (240 configs each) traded together gave −0.4 %/yr vs buy & hold −11.1 % (maxDD −16 % vs −63 %): they protect capital but do not generate returns. | `research/kronos_lab/top10/INFORME.md` | Moderate |
| F6 | ETH and BNB are the only coins where ~68 % of overnight candidates confirmed in the TEST. This matches F1. It may be a regime artefact of a single period. | `overnight/INFORME_TEST.md` §1 | Weak (post hoc) |

## Rules of honesty (non-negotiable; the core also enforces most of them)

1. **Pre-register before you look.** Write a hypothesis `strategy/hypotheses/H-NNNN.json` (hash-sealed, immutable
   once committed; edits = a new id) **before** running any sweep. State the expected sign and why.
2. **Research sees only data before `vault_cutoff`** (2025-07-01). TRAIN = [start + 210 d, 2024-01-01),
   VAL = [2024-01-01, 2025-07-01). The TEST is sealed. Opening it costs one registered look
   (`trials.request_look`, 8 per month, one per hypothesis, only after `passed_trainval`).
3. **The TEST window up to 2026-10-13 is already burnt** for the RSI-regime, breakout and indicator-combination
   families. It was opened by hand on 2026-10-15 (`INFORME_TEST.md`). A look at it gives no fresh evidence for
   those families. Spend looks on genuinely new families, or wait until the operator moves `vault_cutoff` past
   2026-10-13 and fresh sealed data accumulates.
4. **Every evaluation counts.** It goes through `backtest.evaluate` (logged to `journal/trials.jsonl`) or a sweep
   batch row. The Deflated Sharpe uses the **total** trials count (`trials.dsr_context`), not just "the ones that
   mattered". No off-ledger backtests.
5. **Always compare against two baselines**: the random spot config baseline (`score.random_baseline_rate` /
   `random_percentile`) and buy & hold of the same coins over the same window. A strategy that does not beat
   random has no edge, even if it is positive.
6. **Small grids.** Brute force overfits: the overnight run had 2.155 M trials and 7.89 M candidates, and candidates
   confirmed at 6.3 % vs 9.4 % for random configs. Cap each hypothesis at tens to hundreds of specs, never thousands
   "just in case".
7. **No lookahead.** Signals are decided at the close of bar t and held during bar t+1. Costs of a change at t are
   charged in t+1. Never feed a forecast, screener or feature any data stamped after the decision bar.
8. **Report everything, including failures.** A failed hypothesis gets a Graveyard line with its ids.
9. **Promotion to real** needs ≥ 90 forward paper days, net > 0, random percentile ≥ 0.80 and beating buy & hold
   (`protected.promotion`). Even then the cycle only writes a proposal. Only the operator flips
   `real_trading_enabled`.
10. **The screener (Haiku) is a forecaster on probation.** It may veto entries only once its Brier score beats the
    baseline over ≥ 30 forward days and ≥ 100 resolved forecasts. LLM calibration is not edge (Graveyard G4).

## Active strategies (paper)

Capital target 1000 EUR ≈ 1160 USDT. Weights 0.5 / 0.3 / 0.2 (see `active.json` → `capital`).

| id | rule | coins | alloc | hypothesis | status | what would make me retire it |
|---|---|---|---|---|---|---|
| `rsi_regime_l4` | RSI(14) 55/50 regime + vt 0.5, 1d spot | BTC ETH SOL BNB | 580 USDT (0.5) | H-0001 | paper from 2026-10-20 | after ≥ 90 forward days: random percentile < 0.5 **and** worse than b&h; or DD > −40 % |
| `rsi_regime_l9` | same | 9 large caps (engine.UNIV) | 348 USDT (0.3) | H-0002 | paper from 2026-10-20 | same |
| `breakout_c4_basket` | breakout 50d + EMA200 + BTC gate, top-10 | basket | 232 USDT (0.2) **reserved in cash** | backlog B-01 / H-0004 proxy | **not active**: not expressible in the spec format | n/a |

Known paper-vs-backtest differences: orders below 5 USDT are skipped (the vol-target trims small rebalances), fills
are at the close ± 5 bp, and sleeves are never rebalanced against each other.

## Graveyard (what failed, and why; do not resurrect without a new reason)

- **G1. Kronos direction** (`research/README.md` §1, `kronos_bybit_lab_report.md`): following Kronos-small's 24h
  signal net of fees gave BTC −21 %, ETH −47 %, SOL −62 % vs buy & hold +4.5 / +12 / +20 % (May–Oct 2026). As a
  filter on Bollinger, RSI, Fibonacci and similar it is equal to or worse than a random filter. It is useful only
  as a volatility estimator, and that did not translate into money.
- **G2. MarketReasercher 1h strategies** (`research/README.md` §2): 15 evolved strategies averaged −1.2 %. With a
  Kronos filter −0.4 %, the same as a random filter (−0.3 %). Fees ate the gross edge. Lesson: intraday plus
  0.30 %/side is a dead end for us.
- **G3. Brute force** (`overnight/INFORME_TEST.md`): 2.048 M configs × coins → 7.89 M TRAIN/VAL candidates. TEST:
  6.3 % confirmed vs 9.4 % random. The all-candidates portfolio made +1.0 %/yr and ex-ante portfolios +1.7 to
  +5.5 % over 15 months. TRAIN/VAL rank did not predict the TEST (RUNE ranked best: +0.3 %).
- **G4. LLM calibration ≠ edge**: Phil itself (the structure we copy) ran at ROI −3.2 % with a Brier worse than
  the market. A well-calibrated forecaster can still have zero trading value. The screener must beat a base-rate
  Brier **and** improve paper P&L when it vetoes, or it stays advisory.
- **G5. 5-indicator votes, Donchian 100/50, EMA10/50 + ADX** (`indicators/INFORME.md`): good in train, weak median,
  SOL negative in VAL. TEST results +1.7 % and +4.1 %.
- **G6. 4h timeframe and PERP shorts**: worse than daily spot (costs, funding). PERP is not executable in the EU anyway.
- **G7. Per-coin "best strategy" picking** (`top10/INFORME.md`): 13 of 29 coins beat buy & hold, but choosing them
  afterwards is hindsight. Ex ante, all 29 together made −0.4 %/yr.

## Open questions

- Does the RSI regime edge survive a **bull** regime, or is it purely "cash in a bear market"? The only TEST was bearish.
- Is the vol-target, rather than the signal, doing most of the work? (backlog B-03, B-04)
- Can the C4 cross-sectional ranking and BTC gate be expressed? It needs a portfolio-level spec (open contract issue).
- Will Haiku's Brier beat `base_rate_365d` on 7-day direction? Prior: no (G4). Measure it, do not assume it.
- Can a vault reset be done honestly? Proposed: the operator moves `vault_cutoff` to 2026-10-13 and opens new looks
  only on data that nobody has seen.

## Changelog

- 2026-10-16 v0: seeded by the operator build from `research/` (README, indicators/INFORME.md,
  overnight/INFORME_TEST.md, top10/INFORME.md, goal15/).
