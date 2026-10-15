# Pitchfork + Kronos (key: pitchfork)

## Method
- Script: `pitchfork.py`, raw output: `results.json`.
- Pivots are fractal highs and lows with a half-window k (k=6 and k=12, both fixed in advance and not tuned). Consecutive pivots of the same type are collapsed so highs and lows alternate. A pivot at bar i is used only once i+k <= o, so there is no lookahead.
- Andrews' Pitchfork is drawn from the last 3 confirmed pivots. The median line runs from P1 through the midpoint of P2 and P3, and the tines are parallel lines through P2 and P3. z = (close - median) / half the tine width.
- Signals at each origin o, each held 24h (300 origins per symbol, spaced 12h apart, so holds overlap):
  - reversion: if |z| > 0.5, trade back toward the median line.
  - breakout: if |z| > 1 (price outside a tine), trade in the direction of the move.
  - fork_slope: trade in the direction of the median-line slope.
- Kronos filters:
  - `+kronos_dir`: keep the trade only when the mean of the 6 paths at h=24 points the same way.
  - `+kronos_touch`: at least half of the paths cross the median line within 24h.
  - `+kronos_toward`: the mean path ends closer to the median line than price is now.
  - `kronos_alone`: trade the sign of the mean path.
- Fee is 0.11% per round trip. The t-stat uses n/2 to allow for the overlapping holds. The holdout is the last third of origins. The random baseline shuffles the reversion positions 500 times.

## Results (mean per trade, net of fees)
| config | reversion | rev + Kronos dir | breakout | brk + Kronos dir | fork_slope | slope + Kronos dir | Kronos alone |
|---|---|---|---|---|---|---|---|
| BTC k6  | +0.05 (210) | -0.14 (115) | -0.33 | -0.70 | +0.11 | -0.08 | -0.31 (t -1.8) |
| BTC k12 | -0.25 | -0.52 | +0.12 | -0.33 | +0.07 | -0.14 | -0.31 |
| ETH k6  | +0.12 | -0.34 | -0.30 | -0.67 | +0.20 | -0.19 | -0.48 (t -2.1) |
| ETH k12 | -0.28 | -0.80 | +0.40 | -0.07 | +0.03 | -0.43 | -0.48 |
| SOL k6  | -0.22 | -0.55 | +0.03 | -0.03 | +0.05 | -0.56 | -0.62 (t -2.5) |
| SOL k12 | -0.02 | -0.60 | -0.21 | -0.82 | +0.31 | -0.25 | -0.62 |

Buy and hold, mean over the 24h windows: BTC +0.06%, ETH +0.13%, SOL +0.18%. Whole-period returns: BTC +5.9%, ETH +13.4%, SOL +20.2%.

## Key findings
1. **The pitchfork alone has no edge.**
   - Reversion and breakout swap sign when k changes, and every |t| is below 1.4.
   - The shuffle baseline never rejects chance: p = 0.11 to 0.85 for gross reversion.
   - fork_slope is mildly positive in 6 of 6 configs but has t <= 1.2. It is roughly a trend or long tilt in a rising market and does not beat buy and hold per window by any significant margin.
2. **Adding Kronos as a confirmation filter makes things worse in 18 of 18 comparisons.**
   - Within the reversion trades, the subset where Kronos agrees averages -0.03% to -0.69% gross.
   - The subset where Kronos disagrees averages +0.22% to +0.74% gross, in 6 of 6 configs.
   - The `kronos_touch` filter, "paths reach the median line", correlates negatively with reversion PnL (-0.09 to -0.21).
3. These results repeat the earlier finding that the 24h direction of Kronos-small is anti-correlated with what actually happens. Kronos alone loses -0.31% to -0.62% net per trade, with t between -1.8 and -2.5 across all 3 symbols.
   - Using Kronos as a *contrarian* filter would have helped in this sample.
   - The 5-month sample is too short to trust that. It needs out-of-sample confirmation on a different period before anyone relies on it, and it is most likely tied to a mean-reversion regime: Kronos extrapolates recent momentum, and the market reverted.

## Caveats
- n is about 300 overlapping windows per symbol (about 150 effectively independent), and all three symbols are highly correlated with each other.
- k=6 and k=12 were the only parameters tried, and the |z| thresholds were set before looking at results.
- Pivot detection needs k bars of confirmation, so the fork is always somewhat stale.
- Only zero-shot Kronos-small was tested, with 6 sampled paths per origin.
