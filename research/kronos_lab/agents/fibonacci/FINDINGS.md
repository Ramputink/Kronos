# Fibonacci + Kronos (key: fibonacci)

Script: fib.py, raw numbers: results.json. Data: cached Kronos-small paths (6 samples x 24h, origins every 12h, 300 per symbol, about 5 months) for BTC, ETH and SOL, 1h candles.

## Method
- Swing = max high and min low of the last 72 candles (rows <= o). "Up" if the high came after the low.
- Fib-alone signal: up-swing with close inside the 38.2-61.8% retracement -> LONG, target = swing high, stop = 78.6% retracement. Down-swing is the mirror case (SHORT). Entry at close[o], exit at intrabar target or stop (stop is checked first, which is conservative), otherwise at 24h. Fee is 0.11% round trip.
- Kronos confirmation: (A) the median path's 24h close is in the trade direction, (B) more of the 6 paths touch the target than the stop.
- Baselines: random direction at the same times (24h hold), Kronos-alone (sign of median path), long every window, buy&hold, and a shuffled-confirmation test (random subsets of the fib trades, same size as A, 2000 draws).
- Magnet test: distance of the predicted and realised 24h high/low to the nearest fib level, in swing-range units. We also checked whether snapping Kronos' predicted extreme to the nearest fib level reduces its error.

## Results (net mean per trade %, n)
| | BTC | ETH | SOL |
|---|---|---|---|
| Fib alone | -0.40 (63) | -0.13 (87) | -0.10 (63) |
| Fib + Kronos median-dir | **-0.82** (25) | -0.13 (42) | **-0.49** (30) |
| Fib trades Kronos rejected | -0.12 (38) | -0.13 (45) | +0.24 (33) |
| Fib + Kronos target>stop | -0.53 (6) | -0.51 (6) | -1.07 (5) |
| Random dir same times | -0.32 | -0.15 | +0.26 |
| Kronos alone 24h (n=300) | -0.29 | -0.35 | -0.55 |
| Long every window | -0.05 | +0.02 | +0.07 |
| Buy&hold period | +5.9% | +13.4% | +20.2% |
| Shuffled-conf p (P random subset >= confirmed) | 0.99 | 0.51 | 0.89 |
| Fib alone, all hourly origins (overlapping) | -0.37 (782) | -0.21 (996) | -0.29 (905) |

Holdout, last third: fib alone -0.10 / -0.28 / +0.26; fib + Kronos median-dir -0.41 / -0.01 / +0.05, with n of only 9-31. These are noise-level numbers.

Magnet test: Kronos' predicted extremes sit closer to fib levels (about 0.048 range units) than realised extremes do (0.066-0.085). The realised extremes are no closer to the true fib grid than to a randomly shifted grid (0.066 vs 0.065, 0.085 vs 0.085, 0.068 vs 0.067), so fib levels show no magnet effect on realised extremes. Kronos' smaller distance is most likely an artifact: its paths have a compressed range and stay inside the swing, where the levels are dense. Snapping Kronos' high/low to the nearest fib level does not reduce the forecast error (for example BTC high 0.254 raw vs 0.262 snapped).

## Conclusion
- Fib retracement entries (38.2-61.8 zone, target at the swing extreme, stop at 78.6) have no edge on 1h crypto over this period. They lose after fees on all three coins, including the larger hourly-origin sample.
- Kronos does NOT add value as a fib confirmation. On BTC and SOL it makes results worse: the trades it rejects do better than the ones it confirms. That matches the earlier finding of a negative predicted-vs-realised correlation. On ETH it makes no difference (p=0.51).
- Fib levels are not magnets for realised 24h highs and lows, and fib snapping does not improve Kronos' range forecast.

## Caveats
- Small n (25-87 trades per variant). Windows overlap (24h horizon, 12h step), so t-stats are optimistic, especially for the hourly-origin sample.
- One fib parameterisation only (72h swing, 38.2-61.8 zone), chosen up front and not tuned. Other swing detectors such as ZigZag or pivots were not tried.
- Zero-shot Kronos-small with 6 samples. The fact that Kronos-confirmed trades do worse could reflect a contrarian signal, but the earlier negative correlation is weak and the sample is too small to trade on.
