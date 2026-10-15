# levels_vwap: Kronos x support/resistance, pivots, VWAP, Ichimoku

Script: levels.py. Outputs: prob_eval.csv, trade_eval.csv, rows_<SYM>.csv. Data: cached Kronos-small paths (6 paths, 24h), 300 origins/symbol every 12h (about 5 months), BTC/ETH/SOL 1h.

## Method (no lookahead: levels use rows <= o only)
Levels at each origin: classic daily pivot (P/R1/S1 from the trailing 24h), VWAP over 24h and 168h, a 48h Donchian high and low, ATR14, and an Ichimoku cloud (9/26/52, shifted 26).
- **A. Probability skill.** Events within the next 24h:
  - touch: the high or low reaches the level;
  - break: the +24h close is beyond the level (for VWAP, the opposite side).
  - Kronos probability = the share of the 6 paths where the event happens.
  - Baseline = logistic regression on distance-to-level/ATR. A combined model uses distance plus the Kronos probability.
  - Fit on the first 2/3 of origins and test on the last 1/3 (n_test=100). Compared by AUC and log loss.
- **B. Trading.** Hold 24h from each origin, with a 0.11% round-trip fee.
  - Indicator signals: VWAP24 reversion, VWAP168 trend, Ichimoku above/below cloud, pivot side, Donchian fade.
  - Each signal alone, then with a Kronos filter (keep a trade only when the sign of the Kronos mean-path return agrees), then its complement (Kronos disagrees).
  - Also: a random filter with the same trade count (1000 draws), Kronos alone, and buy & hold.

## Results
A. Kronos path probabilities are always worse than plain distance/ATR:
- Full-sample AUC: Kronos 0.46-0.72 vs distance 0.59-0.86.
- Test log loss: adding Kronos to distance never meaningfully improves it (changes of about ±0.005, more often worse).
- Kronos touch probabilities have some ranking ability (AUC about 0.6-0.7), but that comes from the paths starting near the level. They add nothing beyond distance.
- Kronos break probabilities are close to random (AUC about 0.46-0.58).

B. The Kronos "agree" filter makes every indicator worse, on every symbol:
| Symbol | Strategy | Indicator alone, net bp/trade | With Kronos agree, net bp/trade | Random filter, same n |
|---|---|---|---|---|
| BTC | VWAP168 trend | -3.1 | -29.9 | -2.7 |
| ETH | VWAP168 trend | +0.2 | -51.0 | -0.2 |
| SOL | VWAP168 trend | +10.9 | -52.9 | +11.3 |
| SOL | Ichimoku | -18.3 | -107.8 | -17.9 |

- The Kronos-agree subset ranks at the 0.1-6th percentile of the random filters.
- Kronos alone: -31, -48 and -62 bp net per trade (BTC, ETH, SOL).
- Buy & hold: -5, +2 and +7 bp per 24h window.

**The contrarian effect does not hold up.** In-sample, keeping only trades where Kronos disagrees (or betting against Kronos) looks profitable. Split by time, corr(Kronos, realised) is -0.17 to -0.25 in the first 2/3 but +0.02 to +0.07 in the last 1/3. VWAP168 trend plus Kronos-disagree goes from +27/+59/+53 bp to -32/-45/+29 bp. It is a regime artifact, not an edge.

## Conclusion
No edge. Kronos-small paths do not predict rejection or breakout at pivots, VWAP, Donchian or Ichimoku levels better than distance-to-level/ATR. Using them as a confirmation filter hurts. The inverse signal is unstable across time.

## Caveats
- About 300 overlapping windows per symbol over about 5 months, so standard errors are large.
- Only 6 paths per origin, so probabilities are coarse (steps of 1/6).
- Zero-shot Kronos-small with 256 candles of context. No fine-tuning.
- `pivot_revert` (close outside R1/S1) never fires by construction, so ignore it. Donchian fade has n<=21.
