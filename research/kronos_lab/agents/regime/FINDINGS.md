# Regime analysis: Kronos-small zero-shot, 1h candles, 24h horizon

## Method
- Data: cached 6-path Kronos forecasts at 300 origins per symbol (every 12h, so 24h windows overlap by half). BTCUSDT and ETHUSDT only; SOL was not cached in time. n=600 pooled.
- Signal: mean over the 6 paths of close[t+24]/close[t]-1. Trade = sign(signal), held 24h, round-trip fee 0.11%.
- Regime features use only rows <= o: ADX14 (>25 = trend), 48h efficiency ratio (>0.2), variance ratio k=8 over 168h (>1 = persistent), 24h realised vol vs its trailing 30-day median (hi/lo), weekend (Fri 12:00 UTC to Sun), and path agreement (>=5 of 6 paths share a sign).
- Thresholds were set before looking at results; none were tuned.
- Baselines: 24h momentum sign (indicator alone, net of fees), buy and hold, and a permutation p-value for the hit rate (2000 shuffles).
- Holdout: first 2/3 vs last 1/3 of each symbol.
- Script: regime.py. Outputs: run_btc_eth.txt, windows.csv, results.txt.

## Results
| | n | hit | corr(pred, real) | Kronos net %/trade | mom net % | B&H %/24h |
|---|---|---|---|---|---|---|
| BTC all | 300 | 0.520 | -0.18 | -0.31 | +0.05 | +0.06 |
| ETH all | 300 | 0.470 | -0.13 | -0.48 | +0.14 | +0.13 |

- **No regime cell is significant.** The best pooled cell (hi vol, hit 0.534, n=294) has permutation p=0.15. No cell has p < 0.10.
- **Every regime effect flips sign out of sample.** ADX>25: hit 0.43 in train, 0.64 in test. Hi vol: corr -0.28 in train, +0.09 in test. Path agreement >=5/6: hit 0.47 in train, 0.59 in test. Momentum flips the opposite way each time.
- **Root cause:** Kronos's 24h mean forecast is about -0.7 correlated with the past 24h return. Its sign disagrees with 24h momentum 65-68% of the time. In practice it is a short-horizon mean-reversion signal, so it "works" exactly when reversal works (the last third of the sample) and fails in trending stretches. Its regime-conditional "accuracy" is mostly that of a reversal rule. After removing momentum from both forecast and outcome, the leftover corr(pred, real) is still negative (-0.13 BTC, -0.11 ETH). So Kronos adds nothing beyond a reversal indicator.
- **Weekend:** corr was positive in both symbols (+0.09, +0.15) versus negative on weekdays (-0.26, -0.23). Kronos net PnL was still negative there (-0.13%, -0.27%), and n=110 per symbol. This is a weak hypothesis only.
- **Calibration:** the realised close fell inside the [min, max] of the 6 paths only 61-62% of the time. An exchangeable draw would land there 5/7 = 71% of the time, so the path ensemble is somewhat overconfident (too narrow). It is better on weekends (68-69%), where vol is lower.
- **Path spread vs realised |return|:** corr was about 0 (0.03 BTC, 0.01 ETH). The spread across the 6 paths did not predict realised magnitude here. Note that this is only 6 samples per origin.

## Caveats
- Small samples: about 300 overlapping windows per symbol, with an effective sample about half that.
- Only 2 symbols and about 5 months of data, which is one market regime path.
- 6 samples is too few for good uncertainty estimates.
- Zero-shot Kronos-small only.

## Verdict
There is no stable regime where Kronos's direction adds value over the indicator alone (or over a plain 24h reversal rule). Direction should not be gated on regime. If Kronos is used at all, it would be as a widened volatility or range input, and even there this test found no signal from the 6-path spread.
