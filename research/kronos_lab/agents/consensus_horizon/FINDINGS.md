# consensus_horizon — path consensus, horizon curve, IC per step

## Method
Scripts: run.py (24h consensus/slope strategies + IC per step), short.py (short-hold 1-6h), partial.py (Kronos IC net of 1h reversal).
Data: cached zero-shot Kronos-small paths (6 paths x 24h, 256h context), origins every 12h, W=300 per symbol (2026-05-15 -> 2026-10-13).
No lookahead: signals use close[o] and paths for o only. Fees 0.11% round trip. Spearman IC across origins.
Strategies: hold 24h (or h hours) in the direction of the signal; mean per-trade return gross/net. 12h spacing with 24h holds => overlapping trades (t-stats optimistic by ~sqrt2).

## Results (BTC / ETH; SOL see bottom)
### 24h horizon: consensus and slope carry NO edge (if anything, negative)
| strategy | BTC n | BTC net bp/trade | ETH n | ETH net bp/trade |
|---|---|---|---|---|
| buy&hold (long every origin) | 300 | -5.1 | 300 | +1.7 |
| Kronos median sign | 300 | -28.8 (t -2.4) | 300 | -35.2 (t -2.1) |
| Kronos median-path slope sign | 300 | -29.9 | 300 | -41.0 |
| Kronos consensus >=5/6 | 191 | -34.8 | 181 | -48.1 |
| Kronos consensus 6/6 | 124 | -41.3 | 118 | -33.9 |
| MOM24 alone (indicator) | 300 | +5.3 | 300 | +14.1 |
| MOM24 filtered by K consensus | 50 | -51.6 | 44 | -36.8 |
| SMA48 alone | 300 | -3.5 | 299 | -2.0 |
| SMA48 filtered by K sign | 90 | -28.1 | 95 | -33.8 |
Random shuffled-consensus 5-95% band gross: BTC [-18,+21] bp, ETH [-32,+24] bp — Kronos consensus (-24/-37 gross) sits at/below the lower edge.
- Higher agreement does NOT improve accuracy (6/6 no better than >=5/6). 6/6-agreement is frequent (~40% of origins), i.e. the 6 samples are highly correlated, not independent opinions.
- IC(median 24h pred, r24): BTC -0.00, ETH -0.06; slope -0.00/-0.06; frac paths up -0.01/-0.06.
- Contrarian looks tempting (+12.8 / +26.1 bp net) but is NOT stable: first 2/3 IC -0.05/-0.11, last 1/3 IC +0.12/+0.05 (sign flips). Do not trade it.
- Adding Kronos as a filter to MOM24 or SMA48 makes them WORSE on both symbols.

### Horizon curve: signal lives only in steps 1-3
IC(median pred at step k, realised return at step k):
- BTC: k1 +0.09, k2 +0.06, k3 +0.10, k4 +0.05, then ~-0.05 for k6-k18, ~0 at k24
- ETH: k1 +0.16, k2 +0.16, k3 +0.10, k4 +0.05, decays to ~-0.06 by k16-24
ETH k1/k2 IC stable across halves (k1: +0.16/+0.16; k2: +0.09/+0.28). n=300 so SE ~0.058: ETH ~2.7 sigma, BTC ~1.6 sigma.
Partial IC controlling for the 1h reversal (-last 1h return, itself IC +0.12-0.19): BTC k1 +0.056, k3 +0.068 (Kronos mostly re-encodes reversal, corr 0.3); ETH k1 +0.14, k2 +0.14 (genuinely incremental, corr only 0.16).

### But it is not tradable after fees
Short holds (h=1..6h) on the Kronos sign: gross +1.8..+7.0 bp vs 11 bp fee => all net negative (best: ETH h=2 sign, gross +7.0, net -4.0 bp, hit 0.58). Typical |r_1h| is only 22-30 bp, so a 0.1-0.16 IC cannot pay taker fees.

### Side note (volatility)
IC(path dispersion, |r24|) +0.13 BTC / +0.06 ETH; IC(|median 24h|, |r24|) +0.11 / +0.16: weak magnitude/vol information — more promising than direction (other agents' angle).

## Conclusion
- Path consensus, slope of median path and 24h direction: no edge, no added value over simple indicators; filters reduce performance.
- The only real directional information is in the first 1-3 hourly steps (IC ~0.1-0.16, partly independent of 1h reversal on ETH). Too small to beat 0.11% taker fees with origins every 12h. Could matter only with maker fees (~0.02%/side), hourly re-forecasting (more n), or as one input to a short-horizon ensemble / execution-timing tool.

## Caveats
Single ~5-month regime; 300 overlapping windows per symbol; 6 samples only (consensus is coarse); zero-shot Kronos-small; no slippage/funding modelled; many configurations looked at => treat individual |t|<2.5 as noise.

## SOL addendum (W=300; outputs in out.txt, short_out.txt, partial_out.txt)
- 24h, net bp per trade: buy&hold +7.2; K median sign -54.9 (t -3.1); slope sign -41.5; consensus>=5/6 -44.5 (n=192); 6/6 -48.8 (n=128); MOM24 alone +12.4; MOM24 filtered by K consensus -23.4 (n=47); SMA48 alone -3.3; SMA48 filtered by K sign -72.8 (n=88). Shuffled baseline 5-95% gross band [-33,+27].
- IC(med24, r24) -0.09, slope -0.10. Contrarian IC -0.12 in first 2/3 -> -0.03 in last 1/3 (fades); contrarian net +22.5 bp, t 0.96.
- IC per step: k1 +0.07, k2 +0.04, k3 +0.06, then -0.06 to -0.13 for k8-k24.
- Partial IC k1 after removing 1h reversal: +0.045, so mostly the reversal again. Short holds of 1-6h all net <= -5 bp.
- Magnitude: IC(dispersion,|r24|) +0.12, IC(|med24|,|r24|) +0.19.
Conclusion holds across all 3 symbols: no edge from consensus/slope/24h direction; the small useful signal at steps 1-3 cannot pay taker fees; weak information on volatility/magnitude.
