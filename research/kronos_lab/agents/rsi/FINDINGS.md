# RSI x Kronos (key: rsi)

## Method
- Data: cache CSV + Kronos paths (6 paths, 24h, ctx 256), 300 origins every 12h per symbol (~5 months), BTC/ETH/SOL.
- At origin o: RSI (Wilder, n=7/14/21) on closes <= o. Kronos signal = consensus of 6 paths at h=24 (>=4/6 up -> long, <=2/6 -> short, else flat).
- Strategies (24h hold, entry close[o], exit close[o+24]); fee 0.11% round trip:
  RSI mean-reversion (long < lo, short > hi), thresholds 30/70, 35/65, 40/60;
  RSI + Kronos agree (take the trade only if Kronos consensus is in the same direction);
  RSI + Kronos disagree; RSI(14) divergence over 48h (two 24h halves), alone and with Kronos agreeing;
  Kronos alone; Kronos inverted; always long; random long/short (2000 reps).
- Filter test: p = P(random subset of RSI trades, same size as the Kronos-filtered set, has mean >= filtered mean), 5000 reps.
- Holdout: last third of origins for RSI14 35/65 (no tuning was done, so this is only a stability check).
- Scripts: rsi_exp.py; outputs in run_all.txt, results.json, kronos_vs_rsi_corr.txt, kronos_residual.txt.

## Key numbers (RSI14, net per trade, mean)
| sym | RSI 35/65 alone | +Kronos agree | p vs random subset | RSI 40/60 alone | +Kronos agree | p |
|---|---|---|---|---|---|---|
| BTC | -0.34% (n=63) | -0.36% (n=52) | 0.56 | -0.25% (n=108) | -0.30% (n=77) | 0.65 |
| ETH | -0.13% (n=64) | -0.27% (n=53) | 0.83 | -0.19% (n=115) | -0.48% (n=84) | 0.98 |
| SOL | -0.17% (n=60) | -0.55% (n=49) | 0.98 | -0.26% (n=130) | -0.72% (n=93) | 0.99 |
Across all 9 RSI configs x 3 symbols plus divergence, the Kronos-agree filter never beat a random subset of the same size (p from 0.28 to 0.99). Divergence: BTC -0.44% (n=69), ETH -1.19% (n=59), SOL +0.41% (n=69); adding Kronos made it worse or no better in every case.
Benchmarks (net per 24h window): always long BTC -0.05%, ETH +0.02%, SOL +0.07%. Buy and hold over the period: BTC +5.9%, ETH +13.4%, SOL +20.2%.

## Why: Kronos is a mean-reversion forecaster, so it repeats what RSI already says
- corr(Kronos mean 24h forecast, past 24h return) = -0.71 BTC, -0.70 ETH, -0.65 SOL; corr with RSI14 = -0.57/-0.56/-0.56.
  When RSI is extreme, Kronos almost always agrees (it predicts a reversal), so the "confirmation" adds no new information.
- In this period 24h returns showed slight momentum (corr(past24h, next24h) = +0.13/+0.08/+0.02), so RSI mean-reversion lost money and the inverted (momentum) version made a small profit.
- After regressing out past return and RSI, the leftover part of Kronos's forecast is still negatively correlated with the realised move (-0.13/-0.10/-0.19).
  With ~150 effectively independent windows the standard error is about 0.08, so this is weak but has the same sign on all 3 symbols.
- Kronos inverted (fade the consensus) made net +0.08%/+0.20%/+0.40% per trade (n=254-272). This matches the earlier finding of a negative correlation and is NOT a validated edge.
- "RSI extreme but Kronos says continuation" (disagree) was positive on all 3 symbols (for example SOL 40/60: +1.22% net, n=28), but n is tiny (2-28) and I picked this cut after seeing the data.

## Verdict
No evidence that Kronos adds value as a filter or confirmation for RSI overbought/oversold or for RSI divergence. If anything it removes the few good trades.
RSI mean-reversion alone also lost money after fees over these 5 months. That looks like a regime effect, not a stable property: on the holdout third RSI14 35/65 was positive, with n of only 18-23.

## Caveats
Small sample (300 overlapping windows per symbol; 15-130 trades per config), one 5-month regime, zero-shot Kronos-small, 6 samples per path, and the 24h holding period was fixed. The 24h windows overlap by 12h, so p-values are optimistic.
