# Hypothesis backlog

AGENT-EDITABLE. These are candidates for pre-registration and are not registered yet. To work one, copy it into
`strategy/hypotheses/H-NNNN.json` (the format is in CONTRACTS.md §9; its `hash` is
`trials.hypothesis_hash`), commit it **before** sweeping, then mark the line here `→ H-NNNN`. Never edit a
committed H-file; a change means a new id.

Conventions for every item:
- Costs are protected spot costs (0.30 %/side).
- Research view only (TRAIN + VAL < 2025-07-01).
- Compare against the random spot baseline and buy & hold.
- Default `trainval_criterion` = H-0003's. Default `test_criterion` = {portfolio_cagr_min 0.10, beat_bh,
  beat_random_rate, dsr_min 0.5}.
- `L9` = BTC ETH SOL XRP BNB DOGE ADA LINK AVAX. `L4` = BTC ETH SOL BNB. `REST20` = the other 20 universe coins.
- Base RSI spec `RSI` = `{"tf":"1d","mode":"SPOT","comps":[{"t":"rsi_reg","p":{"n":14,"up":55,"gap":5}}],"comb":"and","k":1,"adx":0,"vt":0.5,"stop":0}`.

**Look discipline:** the TEST window up to 2026-10-13 is already burnt for the RSI, breakout and indicator families
(playbook, Rules of honesty #3). For those items, passing TRAIN/VAL makes them **candidates for forward paper**,
not for a look. Spend a look only on items marked *fresh family*, or after the operator moves `vault_cutoff`.

Priority: P1 = do next, P2 = soon, P3 = when idle.

| id | P | hypothesis | spec / grid (dotted paths into the spec) | coins | ≈ trials | expected sign and why | expressible? |
|---|---|---|---|---|---|---|---|
| B-01 | P1 | **Breakout C4 basket**: 50d breakout AND close > EMA200, top-10 by 60d momentum, 15-day hold, all to cash when BTC < EMA200 | needs a portfolio-level spec (cross-sectional rank + BTC gate) | 29-coin universe | 1 (fixed, as pre-registered in goal15) | **+** vs b&h basket in bear and range regimes (goal15 OOS-A SPOT +33.5 %/yr, DSR 0.75); **−** vs BTC in strong BTC-led bulls (4-month forward: +15 % vs BTC +29 %) | **No.** Open contract issue. Per-coin proxy registered as **H-0004** |
| B-02 | P1 | ADX gate on the RSI regime cuts whipsaws net of 0.30 % costs | `RSI`, grid `comps.0.p.up` [50,55,60] × `adx` [0,15,20,25] | L9 | 108 | **+ Sharpe, − entries**, CAGR about flat: fewer trades means less cost drag (playbook F5) | yes → **H-0003** |
| B-03 | P1 | Vol-target level: is 0.5 a lucky pick? | `RSI`, grid `vt` [0,0.3,0.5,0.8] | L9 | 36 | **− maxDD monotone in lower vt**; CAGR peaks around 0.5–0.8. If vt=0 has a similar Sharpe, the sizing (not the signal) is not the edge | yes |
| B-04 | P1 | Is the signal or the sizing doing the work? Control: an always-on trend proxy with the same vt | `{"comps":[{"t":"ma","p":{"kind":"sma","n":20}}]}` vs `RSI`, both vt 0.5 (2 specs) | L9 | 18 | RSI regime **>** SMA20 by ≥ 2 pp CAGR; if not, F1 is mostly "vol-scaled trend exposure", not RSI specifically | yes (approximate control) |
| B-05 | P2 | Slower RSI with a wider gap gives fewer, better trades | `RSI`, grid `comps.0.p.n` [14,21] × `comps.0.p.gap` [5,10] | L9 | 36 | **+ net return at 0.30 % costs** (fewer entries); risk: later exits in crashes | yes |
| B-06 | P2 | ATR trailing stop on top of the RSI regime | `RSI`, grid `stop` [0,2,3,4] | L9 | 36 | **≤ 0**: crypto daily noise triggers stops that the regime exit would have held, adding turnover. Expected to fail; it is a cheap check | yes |
| B-07 | P1 | EMA200 trend filter AND RSI regime (no long below EMA200) | comps `[rsi_reg(14,55,5), ma(ema,200)]`, comb and, k 2; grid `vt` [0.3,0.5] | L9 | 18 | **+ Sharpe, − DD** in bears (cuts bear-market rallies); slight CAGR loss in early bull recoveries | yes |
| B-08 | P3 | Supertrend(10,3) + vt (UNI's top10 winner) generalises to large caps | comps `[st(10,3)]`, grid `vt` [0,0.5] × `adx` [0,20] | L9 | 36 | **≈ 0 vs random**. top10 per-coin picks are hindsight (playbook G7). This tests that and is expected to fail | yes |
| B-09 | P3 | RSI(2) mean reversion only in an uptrend (BNB top10 result) | comps `[rsi_rev(n=2,lo=10,hi=70,tr=200)]`, grid `comps.0.p.lo` [5,10] × `adx` [0,25] | L4 | 16 | **− after costs** for most coins (many short trades at 0.30 %/side); possibly + on BNB only, which would be hindsight | yes |
| B-10 | P2 | Haiku screener veto improves paper P&L | forward A/B: twin paper sleeves with and without `screener_veto` (no backtest possible: the forecasts must be live) | L4 | 0 backtest; ≥ 30 days and ≥ 100 resolved forecasts | **≈ 0**: LLM calibration ≠ edge (playbook G4). Accept only if Brier skill > 0 **and** the vetoed sleeve beats its twin | forward-only (*fresh family*, no vault needed) |
| B-11 | P3 | Regime rotation: cash from coins whose RSI regime is off is redistributed to coins whose regime is on (equal weight across "on" coins) | needs a portfolio-level spec | L9 | 1–4 | **+ exposure in bulls, ≈ same DD**; risk of concentration in one coin | **No.** Same contract issue as B-01 |
| B-12 | P2 | The RSI regime generalises beyond large caps | `RSI` (1 spec) | REST20 | 20 | **− vs L9, + vs b&h**: weaker than on large caps (thinner books, alt bear market, top10 showed −0.4 % ex ante), below the 0.10 CAGR bar. If it passes, the edge is general and not large-cap specific | yes |
| B-13 | P3 | Majority vote of three slow trend rules | comps `[rsi_reg(14,55,5), ma(ema,200), donch(50,20)]`, comb vote, grid `k` [2,3] | L9 | 18 | **≈ RSI alone** (G5: votes looked good in train only). Expected to add nothing; useful to close the question | yes |
| B-14 | P3 | Control: the same RSI regime on 4h | `RSI` with `tf` 4h, grid `comps.0.p.n` [14,84] (84 × 4h ≈ 14 days) | L4 | 8 | **−** vs daily (costs, F4). Research-only (paper allows only 1d) | yes (research only) |
| B-15 | P2 | Higher entry bar (60/50) gives fewer entries and the same capture | `RSI`, grid `comps.0.p.up` [55,60,65] × `comps.0.p.gap` [5,10] | L4 | 24 | **+ net Sharpe** if cost drag dominates; **−** if late entries miss most of the trend | yes |

## Registered from this backlog

- H-0001: provenance of `rsi_regime_l4` (no look; burnt TEST)
- H-0002: provenance of `rsi_regime_l9` (no look; burnt TEST)
- H-0003 ← B-02
- H-0004 ← per-coin proxy of B-01

## Rejected before registration (and why)

- Kronos as a direction or filter signal: Graveyard G1/G2. Do not register unless a new model version comes with a
  new reason.
- Brute-force random sweeps over the full catalog: Graveyard G3.
- PERP or short strategies: not executable in the EU (G6).
