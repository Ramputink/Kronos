# r2_vol_gate_economics: does Kronos path vol make money once it is turned into a vol forecast?

**Short answer: no.** Adding Kronos path vol to a free HAR/Parkinson forecaster gives no economic gain that holds up, in any of the three uses we fixed in advance.

## How the test was set up
- Code: `run.py`. Raw output: `out.txt`, `results.json`. Runtime is about 5 s.
- Data: 300 origins per symbol, 12h apart, 2026-05-15 to 2026-10-13. Each origin has 6 Kronos paths of 24h.
- Target: log realised 24h vol, computed from hourly log returns over rows o+1..o+24.
- Baseline forecaster (BASE): walk-forward OLS on log Parkinson 24h and 168h vol, log realised 24h and 168h vol, and the share of weekend hours in the next 24h.
- Kronos forecaster (KRON): the same model plus log Kronos vol, defined as the mean over the 6 paths of sqrt(sum of squared simulated hourly log returns).
- Shuffled forecaster (SHUF): BASE plus a randomly permuted Kronos feature, averaged over 20 permutations.
- Training uses an expanding window and only origins whose target had ended (O_j+24 <= O_t). The first 60 origins are training only, which leaves 240 for evaluation.
- Walk-forward R² on log vol (BASE -> KRON):

  | Symbol | BASE | KRON |
  |---|---|---|
  | BTC | 0.425 | 0.445 |
  | ETH | 0.271 | 0.274 |
  | SOL | 0.295 | 0.271 |

  This repeats the earlier result: a small gain on BTC, about zero on ETH, and worse on SOL.
- Fees: 0.055% per side.
- Two evaluation periods: the full walk-forward and the last third of the origins (index 200 onwards).
- Confidence intervals: 90% paired block bootstrap of the difference KRON minus BASE. Blocks are 5 windows for the 24h strategies and 10 segments for the 12h ones. Sharpe is annualised. Max drawdown and net return are additive sums of per-period returns.

### The three uses, all fixed before running
- **(a) Vol-targeted long.** Non-overlapping 24h windows. Weight = clip(expanding median of past realised vol / forecast, 0, 3).
- **(b) Regime gate.** Each 12h segment checks whether the forecast is below the expanding median of the forecaster's own past forecasts. If it is, the segment runs Bollinger(20,2) mean reversion: enter outside a band, exit at the middle band. Otherwise it runs Donchian48 stop-and-reverse (variant "don") or stays flat (variant "flat"). The two no-gate baselines run BB alone and Donchian alone. Positions update hourly using data up to that hour, and fees are charged on every position change.
- **(c) Stop width.** Entry on the sign of the past 24h return. Stop at 1.5 × forecast 24h vol from the entry price, filled at the open if price gaps past it. Otherwise exit at +24h. This is compared against no stop and against a trailing rv24 stop.

## Results: net Sharpe, and KRON−BASE Sharpe difference [90% CI]

| Use | Sym | Full KRON / BASE / SHUF | Full ΔSh [CI] | Last 1/3 KRON / BASE | Last 1/3 ΔSh [CI] |
|---|---|---|---|---|---|
| (a) voltarget | BTC | 1.29 / 1.31 / 1.31 | -0.02 [-0.32, 0.27] | 1.47 / 1.37 | +0.10 [-0.23, 0.39] |
| (a) | ETH | 2.05 / 2.03 / 2.03 | +0.02 [-0.07, 0.11] | 0.13 / 0.03 | +0.09 [-0.12, 0.32] |
| (a) | SOL | 1.92 / 1.95 / 1.95 | +0.01 [-0.12, 0.15] | 1.12 / 1.13 | +0.02 [-0.12, 0.20] |
| (b) gate→Don | BTC | -1.41 / -0.07 / -0.39 | -1.32 [-3.29, 0.33] | -3.28 / -2.25 | -1.01 [-5.5, 2.7] |
| (b) | ETH | -0.99 / -0.07 / -0.78 | **-1.01 [-2.14, -0.09]** | -3.24 / -1.81 | -1.67 [-4.5, 0.5] |
| (b) | SOL | -0.43 / -1.44 / -0.90 | **+1.08 [+0.04, +2.30]** | -1.31 / -2.88 | +1.77 [-0.2, 3.9] |
| (b) gate→flat | BTC | -1.81 / -0.52 / -0.91 | -1.31 [-3.05, 0.20] | -3.76 / -2.21 | -1.71 [-4.4, 1.1] |
| (b) | ETH | -0.93 / -0.29 / -0.77 | **-0.74 [-1.55, -0.12]** | -1.31 / +0.07 | -1.42 [-3.2, 0.2] |
| (b) | SOL | -1.58 / -1.92 / -1.64 | +0.50 [-0.30, 1.48] | -0.89 / -1.97 | +1.02 [-1.2, 3.5] |
| (c) stop 1.5σ | BTC | 0.08 / -0.04 / -0.03 | +0.11 [-0.00, 0.24] | -0.16 / -0.12 | -0.05 [-0.18, 0.09] |
| (c) | ETH | -0.11 / -0.10 / -0.06 | -0.01 [-0.07, 0.06] | -1.64 / -1.72 | +0.07 [-0.03, 0.19] |
| (c) | SOL | 0.34 / 0.32 / 0.29 | +0.02 [-0.17, 0.21] | -0.32 / -0.28 | -0.07 [-0.36, 0.16] |

Differences in max drawdown and net return follow the Sharpe differences and are tiny for (a) and (c). For example, (a) ETH full: net +0.450 vs +0.437, Δ +0.014 [-0.005, +0.035]. All numbers are in `out.txt`.

## Reference points, full period
**Buy and hold:** BTC +28%, ETH +46%, SOL +53%.

**Use (a), vol-targeted long (net):**

| Symbol | Constant weight 1 | With either vol target |
|---|---|---|
| BTC | +12.7% | about +16% |
| ETH | +28.8% | about +44% |
| SOL | +35.3% | about +41% |

Vol targeting helps in Sharpe terms, but that comes from the HAR forecast and is the same with or without Kronos.

**Use (b), regime gate:** without the gate, BB mean reversion has a full-period Sharpe of -2.7 (BTC), -1.6 (ETH) and -2.6 (SOL). Donchian alone is also negative. Every version of (b) loses money net of fees.

**Use (c), stops:** stop-out rates are the same for KRON, BASE and SHUF, about 12-18%. The 24h-momentum entry has no edge.

## How to read this
1. **(a) Vol targeting.** KRON and BASE are statistically identical on all 3 symbols in both periods. Every confidence interval includes 0, and every point estimate is within ±0.1 Sharpe. The vol-targeting benefit is entirely in the free HAR forecaster.
2. **(b) Regime gate.** The sign of the Kronos effect is inconsistent:
   - It hurts on BTC and ETH, which are the two symbols where Kronos vol had shown a little extra information. On ETH the CI excludes 0 on the negative side.
   - It helps on SOL, where Kronos vol does not add forecasting skill. There the CI excludes 0 on the positive side for the full period.

   A median-threshold gate is a knife edge: small changes in the forecast flip the regime and change which hourly trades happen. So these swings are noise in the path, not forecasting skill. The ordering KRON vs SHUF also changes from symbol to symbol. In any case, the underlying strategies lose money.
3. **(c) Stop width.** Same stop-out rate and net P&L for KRON and BASE. The BTC full-period +0.11 Sharpe (CI touching 0) reverses in the last third.

## Caveats
- The sample is short: about 5 months, 120 non-overlapping 24h windows (50 in the last third), and a single market regime that was mostly a bull run.
- Bootstrap CIs on Sharpe are wide.
- Six comparisons per symbol times 3 symbols means a couple of "significant" CIs are expected by chance. The two that exclude 0 in (b) point in opposite directions.
- Hourly BB and Donchian positions use data up to each hour inside the segment, as live execution would. The gate itself uses only information available at the origin.
- Fills are at close with no slippage. Funding costs are ignored.
- The Kronos vol feature is the average of only 6 sampled paths, so it is noisy. More samples might sharpen it, but that is untested.
