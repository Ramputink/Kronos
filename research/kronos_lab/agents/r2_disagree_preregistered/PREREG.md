# Pre-registration (written before running any holdout numbers)
- Sample: per symbol (BTC, ETH, SOL), origins with o+24 < len(csv); holdout = last third (index >= 2T//3). Pooled.
- Trade: side in {+1,-1} at origin close, exit close o+24. Net = side*ret - 0.0011.
- Rules (defaults copied from agent scripts): donchian48 state, sign(EMA12-EMA48), sign(MACD-signal9),
  RSI14 (Wilder ewm) <35 long >65 short, Bollinger(20,2,ddof0) relaxed %B<.2/>.8 MR and BO (BO = -MR),
  pitchfork k6 reversion (|z|>0.5 fade toward median).
- PRIMARY Kronos direction: kdir = sign(median over 6 paths of close[h24]) - c0) (trend_ma/bollinger default).
  agree: kdir == side; disagree: kdir == -side.
- Momentum substitute: mom = sign(c[o]/c[o-24]-1); variant 4 keeps trade if mom == side.
- Primary test: pooled over rules and symbols, OLS net_ret ~ 1 + kdis + mom_agree + is_long,
  SEs clustered by (symbol, origin-pair). Partial t of kdis. Bar: |t|>2 AND (IC>0.15 or ann. Sharpe>2.8).
- Secondary only: consensus-based kdir (>=4/6 paths), mean-based kdir.
- Random same-size filter: 2000 draws from rule-alone trades, percentile of filtered mean.
- CI: circular block bootstrap of mean net, block = 4 consecutive origins, 2000 reps.
