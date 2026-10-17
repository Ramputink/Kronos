# REAL trading — operator manual (PROTECTED)

`core/real.py` is **built and disabled**. Nothing in the loop can turn it on; only the operator can, via an
`operator:` commit. Bybit EU = **spot only**.

## Gates (all must hold, else `RealTradingDisabled`)
1. `config/protected.json` → `real.real_trading_enabled` is the JSON boolean `true`.
2. The strategy id is in `real.allowed_strategy_ids` (empty by default) and the coin in `real.allowed_coins`.
3. Env `BYBIT_API_KEY` / `BYBIT_API_SECRET` are set (source them from `~/.phil-crypto/secrets.env`, mode 600, outside
   the repo; the key must have **no withdraw permission** and spot trading only).

## Hard caps (`RealCapExceeded`)
`per_order_eur` (20), `daily_eur` (50, counts today's `submitted|filled` buys), `max_open_positions` (4),
`min_order_usdt` (5), spot only, buy/sell only, unique `client_order_id` (≤36 chars). Live orders are IOC limits at
last price ± `max_slippage_bps`, never naked market orders. `validate.py` caps the config at 100 / 250 EUR / 10
positions, and `real.py` re-checks those ceilings itself on every order (a config above them is refused outright).

## Where the gate is read from (env redirects are ignored)
`real.py` reads the gate and caps only from the fixed file `trader/config/protected.json` next to its own code,
and keeps its ledger at the fixed `trader/journal/real-ledger.jsonl`. `TRADER_CONFIG`, `TRADER_JOURNAL_DIR` and
`TRADER_ROOT` do not apply to it. The one exception is for tests: with `PHIL_CI=1`, *dry runs* may use the
redirected config and journal. A live order (`dry_run=False`, `mirror --live`) never does. Before a live order,
`validate.check_real_flag` must also pass on the fixed repo: the flag was set by an `operator:` commit and the
working copy equals HEAD. `GIT_DIR` and similar variables are removed while that check runs.

Every attempt — refusals and dry runs included — is a row in `journal/real-ledger.jsonl`. Secrets are scrubbed
from every message and never written.

## Promotion checklist (operator)
1. `score.promotion_check(<id>)` eligible (≥90 forward paper days, net > 0, beats random ≥ p80, beats buy&hold).
2. `python -m trader.core.real doctor` → every check ok (without `--offline`: API reachable, key valid, no withdraw).
3. Dry run: `python -m trader.core.real mirror <paper run_id>` (dry by default) and read the ledger rows.
4. `operator:` commit adding the id to `allowed_strategy_ids` and setting `real_trading_enabled: true`.
5. To stop: set `real_trading_enabled: false` (operator commit). Open positions are not auto-closed.

## Mirroring (twins follow the paper WEIGHT, not the paper notional)
`mirror_paper(run_id)` places one twin order per paper fill of an allowed strategy. Each real sleeve
(strategy, coin) has a fixed size `real.sleeve_eur` (falls back to `per_order_eur` when the key is absent), and
every twin makes the same weight move as its paper fill (`prev_weight` → `target_weight`):
- buy: `(target_weight - prev_weight) × sleeve_eur`, then capped by `per_order_eur` (the cap limits the scaled
  order; it never replaces the scaling);
- partial sell: the fraction `(prev_weight - target_weight) / prev_weight` of the real holding;
- `target_weight == 0` (signal exit or retire): close, i.e. sell the whole real holding of that coin.
So buy → rebalance → exit leaves the real position flat when paper is flat. A twin below `min_order_usdt` is
refused and logged (paper has the same rule). Fills without weights are refused, never sized from notional.
Refusals are returned, not raised.

## Sells are sized from the coin actually held
Bybit spot takes the buy fee in the base coin, so a position holds slightly less coin than its notional bought.
Live sells therefore read the free base-coin balance (`get_wallet_balance`, `walletBalance - locked`) and sell
`min(target qty, free balance)` floored to `basePrecision`; a close sells the whole free balance (only this
strategy's ledger qty when another strategy also holds the coin). `per_order_eur` caps buys only; a sell can only
reduce a position. A sell below the exchange `minOrderAmt`/`minOrderQty` is refused with the residual recorded
(`residual_qty`, dust). A partial sell that would leave less than `minOrderAmt` is enlarged to the whole balance
when this strategy is the only holder, and refused otherwise. Filled buy rows record `qty` net of the base-coin
fee; sell rows that close a position carry `closed: true`.
