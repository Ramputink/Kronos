# r2_recalibrated_paths: post-hoc recalibration of Kronos 24h paths

Script: recal.py. Raw output: results.json.
Data: cached Kronos-small paths, 6 per origin, 300 origins every 12h for each of BTC, ETH and SOL.
Split: first 200 origins are train, last 100 are test (2026-08-23 to 2026-10-12).

## Method (all fits on train only)
- Target: the realised 24h log return y.
- Drift correction: OLS of y on [Kronos mean-path 24h log return, z-score of log close over the 256h context, past 24h log return].
- Ablation: the same OLS without the Kronos term (noKronos_reg).
- Spread: members are re-centred on the debiased mean, then scaled by s, so the train coverage of the 6-member min/max equals 5/7 (the nominal 71%).
- Baselines:
  - N(0, trailing-7d vol) (gauss_vol7), plus a train-scaled version (gauss_vol7_cal).
  - N(0, Parkinson-24h vol × a weekend/weekday variance multiplier fitted on train) (gauss_park_wkend).
- Metrics:
  - Fair ensemble CRPS for the ensembles; closed-form CRPS for the Gaussians.
  - Coverage of the 5–95% interval and the mean of the 5% and 95% pinball losses. The Kronos quantiles come from mu ± 1.645·s·sd.
  - Spearman IC.
  - One trade per window: sign of the forecast, 24h hold, 0.11% round-trip fee.
- CIs are block bootstraps (blocks of 10 origins, 2000 resamples).

## Fitted coefficients (train)
| sym | const | kmean | z | r24 | spread s | Kronos mean drift (bp, train) | realised mean (bp) |
|---|---|---|---|---|---|---|---|
| BTC | 0.0000 | -0.095 | 0.0009 | +0.089 | 2.48 | -20 | -1 |
| ETH | 0.0011 | -0.049 | 0.0022 | +0.016 | 2.13 | -55 | +12 |
| SOL | -0.0001 | -0.164 | 0.0015 | -0.054 | 2.20 | -65 | +8 |

- **Stable across symbols:**
  - The Kronos-mean coefficient is negative on all three: the debiased forecast fades Kronos.
  - The spread scale is about 2.1–2.5×. Kronos paths are about 2.3× too narrow.
  - The bearish drift is confirmed: -20 to -65 bp per 24h against a realised mean near 0.
- **Not stable:** the r24 coefficient changes sign across symbols.
- Kronos-mean slope on z: -0.014 to -0.020 per z unit, against +0.003 to +0.004 realised (train).

## Test results (n=100 per symbol)
Density. CRPS is in bp (lower is better); cov90 is the 5–95% coverage (nominal 0.90); pinball is in bp.
| sym | raw Kronos ens | recal Kronos ens | gauss_vol7 | gauss_vol7_cal | gauss_park_wkend |
|---|---|---|---|---|---|
| BTC | 135 / 0.53 / 47 | 101 / 0.86 / 28 | 100 / 0.91 / 24 | 106 / 0.93 / 26 | 108 / 0.91 / 27 |
| ETH | 182 / 0.51 / 63 | 135 / 0.78 / 39 | 123 / 0.89 / 27 | 143 / 0.99 / 35 | 138 / 0.94 / 33 |
| SOL | 272 / 0.41 / 93 | 179 / 0.77 / 44 | 179 / 0.87 / 36 | 182 / 0.93 / 37 | 182 / 0.91 / 38 |

- Recalibration cuts Kronos CRPS by 25–35% and pinball loss by about half.
- The recalibrated ensemble ties plain trailing-7d vol: CRPS diff (recal − vol7) is +0.5 [-6, +11] bp on BTC, +11.9 [+1, +20] on ETH (significantly worse), and 0.0 [-14, +14] on SOL.
- Against the weekend-adjusted Parkinson baseline it is 3–7 bp better, but every CI includes 0.
- The 5–95% interval is still under-covered (0.77–0.86), and the pinball loss is worse than every Gaussian baseline on all three symbols.
- The min/max coverage after recalibration is 0.68–0.82 on test against 0.71 nominal (raw: 0.33–0.44).

Directional IC (Spearman, test, with 95% block-bootstrap CI)
| sym | debiased Kronos | raw Kronos mean | same regression without Kronos | 24h momentum | 4h reversal |
|---|---|---|---|---|---|
| BTC | -0.25 [-0.50,-0.09] | +0.12 | -0.27 | -0.18 | +0.08 |
| ETH | -0.29 [-0.57,-0.10] | +0.06 | -0.32 | -0.20 | +0.17 [0.03,0.36] |
| SOL | +0.06 [-0.20,0.23] | -0.00 | -0.09 | -0.18 | -0.01 |

- The debiased IC is almost identical to the regression without Kronos.
- The Kronos term adds nothing. The train-fitted z/r24 relation flipped sign in the test period, so both regressions end up anti-predictive on BTC and ETH.
- The raw Kronos-mean IC is small and positive here, opposite to the train sign, so fading it hurt too.

Trades (mean per 24h trade, bp; net includes 11 bp fee)
| sym | debiased Kronos gross/net | without Kronos net | momentum net | 4h reversal net | buy&hold net | shuffled sign: mean / p95 / p-value |
|---|---|---|---|---|---|---|
| BTC | -27 / -38 [-79,-3] | -38 | -12 | +16 | +2 | -10 / +20 / 0.93 |
| ETH | -24 / -35 [-80,+11] | -52 | -18 | +10 | -8 | -10 / +20 / 0.91 |
| SOL | +33 / +22 [-54,+67] | +6 | -27 | +8 | +13 | -9 / +46 / 0.17 |

## Conclusion
- The cheap recalibration fixes Kronos's calibration problems: drift, spread and coverage. Once fixed, the Kronos ensemble is only as good as a Gaussian with trailing 7-day vol, and is significantly worse on ETH.
- The recalibrated paths carry no out-of-sample directional value:
  - The debiased mean's IC equals the IC of the indicator-only regression.
  - The trade loses net on BTC and ETH.
  - The SOL gain is not distinguishable from a shuffled sign (p=0.17) or from buy&hold.
- **Answer to the key question: no.** After recalibration, nothing in Kronos beats the trivial baselines out of sample.
- The only robust, stable facts are:
  - Kronos paths are about 2.3× under-dispersed.
  - Kronos paths carry a bearish drift of -20 to -65 bp per day.
  - The Kronos mean is weakly contrarian in train only.

## Caveats
- n=100 test windows per symbol, all in one 7-week period. Windows overlap (24h hold, 12h step), so the effective n is about 50.
- The three symbols are highly correlated, so the agreement across them is not three independent confirmations.
- 6 paths per origin is too few for empirical 5/95% quantiles, so Gaussian quantiles were used.
- The spread scale targets min/max coverage, not the 90% interval, which explains the residual under-coverage at 5/95.
- Kronos-small only, with 256h context and zero-shot. Fine-tuning or larger models were not tested.
