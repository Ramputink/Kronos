# Phil-crypto (`trader/`)

A crypto trading research and paper-trading system for Bybit that improves itself and puts honesty first.
Its structure follows github.com/bennyjo/phil: a protected core, a strategy the agent may edit only with
evidence, daily cycles, watchers, and a real-money path that is built but switched off.

Read `CONTRACTS.md` before touching anything. It defines every interface and file schema, and who owns which file.

## Layout
- `core/` and `config/protected.json` are **PROTECTED**. They hold the data and the sealed vault, the backtest
  catalog (a faithful port of `research/kronos_lab`), the trials ledger with its look budget, scoring (Deflated
  Sharpe, a random-config baseline, buy&hold), the paper engine, the screener forecasts, `real.py` (disabled),
  the lease and the validator.
- `strategy/` is **agent-editable**: the playbook, pre-registered hypotheses, `active.json`, the watchlist,
  the screener questions, the schedule and the tools.
- `journal/` is the ledgers. The JSONL files are append-only and only core code writes them. It also holds
  retros, `cycles.log`, the lease and fleet health.
- `fleet/` has the setup scripts, launchd/systemd units, sweep dispatch, the health check and the watchers.
- `CYCLE_*.md` and `loop.sh` are the operator's procedures for headless `claude -p` cycles.

## Honesty rules
1. Research can only see bars before `vault_cutoff` (2025-07-01). Opening the sealed TEST spends one registered
   look, and the budget is 8 per month, at most 1 per pre-registered hypothesis.
2. Every evaluation is written to `journal/trials.jsonl` and counts toward the Deflated Sharpe N.
3. Every result is compared with random configurations and with buy&hold, after Bybit EU spot costs
   (0.25 % fee + 5 bp slippage per side).
4. No lookahead: a position is decided at a bar's close and held during the next bar.
5. Real trading is spot-only (Bybit EU) with tiny caps, and `real_trading_enabled` stays false until the
   operator flips it in an `operator:` commit.

## Fleet
- `air`: the brain (cycles, screener, git writer).
- `bigsur`: the CPU sweep pool, which only ever receives research-view data.
- `berry`: data, watchers every 15 min and health checks.
- M2: optional GPU.

## Tests
Each test is a plain script and takes less than 3 minutes:
`/Users/matveypro/.venvs/kronos/bin/python trader/tests/test_<area>.py`.
