# Contrarian: is fading Kronos a real edge?

**Verdict: no robust edge.** Fading Kronos shows a positive net return over the full sample at 12h and 24h horizons. None of the bootstrap confidence intervals exclude zero, and the result does not hold up in the holdout. The negative correlation the earlier test found comes mostly from one regime. It is also mostly explained by a structural bias in Kronos: its forecasts pull back toward the mean of the 256-candle context window.

## Method (script `fade.py`, raw output `out.txt`, numbers `results.json`)
- Cache: Kronos-small, 1h candles, 256-candle context, 6 sampled paths, 300 origins per symbol spaced 12h apart. Symbols: BTC, ETH and SOL.
- Prediction: median of the 6 paths' close at horizon h, divided by the last close, minus 1. Horizons h = 1, 4, 12 and 24.
- Strategies, all entered at the origin close and held for h hours, with 0.11% round-trip fee:
  - **fade** = -sign(pred)
  - **follow** = +sign(pred)
  - **zscore_revert** = -sign(z), where z = (last close - context mean) / context std. This is the indicator-only mean-reversion baseline.
  - **rsi_revert**: RSI14 above or below 50
  - **buy&hold**
  - **shuffled-signal fade**: 500 permutations
- Statistics:
  - Block bootstrap with blocks of 6, because the windows overlap.
  - Holdout = last third of origins (n = 100).
  - Partial check: correlation of realised return with the Kronos prediction after regressing out z and the past-24h return.

## Key numbers (24h horizon, net bp per trade, full sample n=300 / holdout n=100)
| sym | corr(pred, real) full [95% CI] | corr first 2/3 | corr holdout | fade net full [CI] | fade net holdout | z-revert net full | buy&hold net |
|---|---|---|---|---|---|---|---|
| BTC | -0.19 [-0.34, +0.07] | -0.25 | +0.06 | +6.8 [-23, +41] | +4.3 | -19.9 | -5.1 |
| ETH | -0.13 [-0.29, +0.07] | -0.17 | +0.04 | +13.2 [-28, +66] | -19.4 | -25.3 | +1.7 |
| SOL | -0.17 [-0.34, +0.03] | -0.24 | -0.01 | +32.9 [-5, +87] | +14.4 | -23.0 | +7.2 |

At the 12h horizon, fade nets +6.2, -9.7 and -2.1 bp (BTC, ETH, SOL); holdout +15.0, -28.9 and -17.8 bp.

The shuffled-signal fade has a 95th percentile of about +10 to +20 bp at 24h. Only SOL's full-sample fade falls outside that range, and only in-sample.

At 1h, Kronos correlates *positively* with the realised return: ETH +0.22, and after removing z and the past-24h return it is +0.24 with CI [+0.04, +0.39]. But the gross edge is about 2 bp, far below the 11 bp fee.

## Why the correlation is negative (mechanism)
- `KronosPredictor.predict()` (model/kronos.py:544-556) z-normalises each column using the context window's mean and std, clips to ±5, and then de-normalises the output.
- The sampled paths are strongly **shrunk toward the context-window mean**. Regressing the predicted 24h close (in context-z units) on the current z gives a slope of **0.39 to 0.48**. The same regression on the realised close gives **1.05 to 1.17**: real prices did not revert to the window mean over 24h.
- The prediction is therefore mostly a reversal signal:
  - corr(pred24, z) is about -0.70.
  - corr(pred24, past-24h return) is about -0.65 to -0.71.
- The bias already shows at step 1. The predicted first open differs from the last close by an amount that correlates -0.38 to -0.49 with z. That points to tokenizer reconstruction or quantisation pulling toward the normalised mean.
- So "fade Kronos" is mostly a momentum bet on the context window. It won in the first two-thirds of the sample, when trends persisted, and stopped working in the holdout. After z and the past-24h return are regressed out, the remaining correlation at 24h is -0.10 to -0.14, and every CI includes 0.

## Caveats
- About 5 months of data, 300 overlapping windows per symbol, one regime, one model (small), zero-shot. No parameters were tuned.
- The fade_AND_z row averages over all origins, counting flat ones as 0, so its "hit" column is not a per-trade hit rate.
- Only the median of 6 paths was used; the dispersion across paths was not examined here.

## Practical takeaway
- Do not trade Kronos direction, either following it or fading it.
- If anything, treat Kronos as a mean-reversion prior that is biased toward the window mean. Correct for it, for example by using pred minus the reversion implied by z, or use only the distributional outputs (range or volatility).
- The 1h residual signal is statistically interesting but cannot pay taker fees.
