# Phil-crypto — CONTRACTS (Wave 0)

This file is the single interface agreement between all build areas. It is **PROTECTED** (operator-owned).
If you build an area, implement exactly what your section says; if you consume another area, code only
against what is written here. If something you need is missing, do **not** edit another area's files:
note it in your `open_issues` and the integrator will amend this file (commit `operator: contracts ...`).

Conventions used everywhere:

- Root: the package is `trader` (repo root on `sys.path`). Import as `from trader.core import backtest`.
  Scripts and tests start with `sys.path.insert(0, str(Path(__file__).resolve().parents[N]))` so that
  `import trader` works when run as `python trader/tests/test_x.py` from the repo root.
- Python: **everything in `core/catalog.py`, `core/backtest.py`, `core/vault.py`, `core/data.py`,
  `core/config.py`, `core/score.py`, `core/trials.py`, `fleet/sweep_runner.py`, `fleet/watch.py`, `fleet/health.py`
  must run on Python 3.9** (bigsur, berry). No `match`, no `X | Y` annotations evaluated at runtime (signatures in
  this file use `X|None` only as documentation; write `Optional[X]` or add `from __future__ import annotations`).
  Dependencies allowed: numpy, pandas, numba, requests (`fleet/watch.py` and `fleet/health.py`: stdlib + requests
  only). `core/real.py` may additionally import `pybit` lazily (inside functions). Nothing else.
- Time: all timestamps are UTC. JSON timestamps are ISO-8601 strings `"2026-10-20T00:10:03Z"` (use
  `config.iso()`). Bar timestamps are the **bar open time** (Bybit convention), naive-UTC in pandas, ISO with `Z`
  in JSON. Dates alone are `"YYYY-MM-DD"`.
- Numbers: returns, CAGR, drawdowns, weights and costs are **decimals** (0.268 = 26.8 %), never percents.
- JSONL: one JSON object per line, append-only, written only through `config.jsonl_append()` (flock'd) and
  read through `config.jsonl_read()`. A module that "owns" a JSONL file is the only writer of that file.
- Hashes: `spec_hash` = `md5(json.dumps(core_spec, sort_keys=True).encode()).hexdigest()` (byte-identical to
  `cfg_hash` in `research/kronos_lab/overnight/worker.py`, default separators). Every other hash
  (hypotheses, looks, files) = `sha256(config.canonical_json(obj).encode()).hexdigest()`.
- Runner identity: `config.runner_id()` = `$PHIL_RUNNER` or short hostname (`air`, `bigsur`, `berry`, `m2`).
- Errors: raise the named exception classes below; never `sys.exit` inside library functions. CLIs print one
  JSON object to stdout and use the exit codes listed.

---

## 0. Ownership map (who may create/edit what)

| Area (wave) | Owned files |
|---|---|
| contracts (W0) | `CONTRACTS.md`, `README.md`, `config/protected.json`, `core/config.py`, `__init__.py` files, `.gitkeep` placeholders |
| data (W1) | `core/data.py`, `core/vault.py`, `tests/test_data.py` |
| backtest (W1) | `core/catalog.py`, `core/backtest.py`, `tests/test_backtest.py` |
| trials (W1) | `core/trials.py`, `core/score.py`, `tests/test_trials.py` |
| paper (W1) | `core/paper.py`, `core/real.py`, `tests/test_paper.py`, `REAL.md` |
| guard (W1) | `loop.sh`, `core/validate.py`, `../.githooks/pre-commit`, `../.github/workflows/trader-ci.yml`, `tests/test_guard.py`, `../.gitattributes` (trader lines only) |
| fleet (W1) | `core/lease.py`, `fleet/dispatch.py`, `fleet/sweep_runner.py`, `fleet/health.py`, `fleet/inbox.py`, `fleet/setup_air.sh`, `fleet/setup_bigsur.sh`, `fleet/setup_berry.sh`, `fleet/launchd/*.plist`, `fleet/systemd/*.service`, `fleet/systemd/*.timer`, `tests/test_fleet.py` |
| research loop (W2) | `CYCLE_RESEARCH.md`, `strategy/tools/sweep.py`, `tests/test_research.py` |
| forward loop (W2) | `CYCLE_FORWARD.md`, `CYCLE_DEEP_RETRO.md`, `strategy/tools/report.py`, `tests/test_forward.py` |
| screener (W2) | `core/forecast.py`, `strategy/screener.py`, `strategy/screener_questions.json`, `tests/test_screener.py` |
| watchers (W2) | `fleet/watch.py`, `strategy/watchlist.json`, `tests/test_watch.py` |
| seed (W2) | `strategy/playbook.md`, `strategy/active.json`, `strategy/schedule.json`, `strategy/hypotheses/backlog.md`, `strategy/hypotheses/H-0001..H-0004*.json` (optional seeds), `tests/test_seed.py` |

All paths in this document are relative to `trader/` unless they start with `../` (repo root).

Run-time file **writers** (who appends at run time, not who authors the code):

| File (under `journal/`) | Sole writer |
|---|---|
| `trials.jsonl`, `looks.jsonl`, `hypotheses.jsonl` | `core/trials.py` |
| `paper-ledger.jsonl` | `core/paper.py` |
| `real-ledger.jsonl` | `core/real.py` |
| `forecasts.jsonl` | `core/forecast.py` |
| `lease.json` | `core/lease.py` |
| `cycles.log` | `loop.sh` |
| `fleet-health.json`, `watch-triggers.jsonl` | `fleet/inbox.py` on air (copies what berry produced: see §7) |
| `sweeps/<job_id>.json` | `fleet/dispatch.py` (compact result copies, each ≤ `fleet.max_result_kb`) |
| `retros/*.md`, `proposals.md` | the cycle agent (Claude) |
| `operator-notes.md` | operator only (agent read-only) |

---

## 1. Shared helpers — `core/config.py` (exists, W0)

```python
TRADER_ROOT: Path; REPO_ROOT: Path
config_path() -> Path        # $TRADER_CONFIG or trader/config/protected.json
journal_dir() -> Path        # $TRADER_JOURNAL_DIR or trader/journal
strategy_dir() -> Path       # $TRADER_STRATEGY_DIR or trader/strategy
data_dir(source=None) -> Path  # ($TRADER_DATA_DIR or protected data.dir)/<source>
load_config() -> dict        # fresh read of protected.json
runner_id() -> str
utcnow() -> datetime (tz-aware UTC); iso(ts=None) -> "YYYY-MM-DDTHH:MM:SSZ"
jsonl_append(path, obj) -> None; jsonl_read(path) -> list[dict]
canonical_json(obj) -> str
```

Tests MUST set `TRADER_JOURNAL_DIR`, `TRADER_STRATEGY_DIR` (when writing) and `TRADER_DATA_DIR` to temp dirs
and never write the real `journal/`. Test data: make a temp dir with a symlink
`<tmp>/linear -> /Users/matveypro/kronos_lab/indicators/data` (and `<tmp>/spot -> same` if a spot source is
needed) and set `TRADER_DATA_DIR=<tmp>`.

---

## 2. `config/protected.json` (exists, W0) — keys every area may rely on

| Key | Meaning |
|---|---|
| `costs.spot.fee_side` + `costs.spot.slippage_side` | per-side cost on turnover for SPOT (0.0025 + 0.0005 = **0.0030**) |
| `costs.perp.*`, `costs.perp.status="not_available_in_EU"` | research comparison only; PERP specs may be backtested in research, never paper/real |
| `costs.legacy_research.spot_side/perp_side` | 0.0015 / 0.00105 — only to reproduce old research numbers |
| `vault.vault_cutoff` `"2025-07-01"`, `vault.train_end` `"2024-01-01"`, `vault.warmup_days` 210, `vault.looks_per_month` 8, `vault.look_ttl_hours` 6 | sealed-test parameters |
| `forward.forward_start` | `null` until the operator starts paper; then `"YYYY-MM-DD"` |
| `data.dir`, `data.research_source` (`linear`), `data.paper_source` (`spot`), `data.seed_dir`, `data.api` | data layout |
| `universe` (29 coins), `timeframes` `["1d","4h"]`, `bars_per_year` `{1d:365, 4h:2190}` | |
| `paper.*` | bankroll 2000 USDT, ≤6 strategies, ≤1000 USDT/strategy, ≤12 coins/strategy, SPOT + 1d only, min order 5 USDT |
| `real.*` | `real_trading_enabled:false`, per-order 20 EUR, daily 50 EUR, allowed coins/strategy ids, secrets file |
| `promotion.*` | ≥90 forward days, net>0, beats random (percentile ≥0.80 of 500 random spot configs), beats buy&hold |
| `screener.*` | model `claude-haiku-5-5` via `claude` CLI, ≤60 calls/day, veto only after ≥30 days & ≥100 resolved forecasts with better Brier than baseline |
| `research.*` | `max_trials_per_hypothesis` 5000, `random_baseline_n` 2000, `dsr_min` 0.5 |
| `schedule.*` | forward 00:10 UTC daily, deep retro 06:00 UTC, research every 6 h, watchers + health every 15 min |
| `fleet.*` | hosts, ssh aliases, quotas (bigsur ≤5 GB), `remote_root`, `inbox_dir`, `max_result_kb` 1024, `lease_ttl_s` |

Periods (derived, used by every area; `start` = first bar of the series):

| name | range (bar open time) | allowed view |
|---|---|---|
| `train` | `[start + warmup_days, train_end)` | research |
| `val` | `[train_end, vault_cutoff)` | research |
| `trainval` | `[start + warmup_days, vault_cutoff)` | research |
| `test` | `[vault_cutoff, forward_start or data end]` | full (needs a look) |
| `forward` | `[max(forward_start, activated_at), data end]` | paper (forward/deep-retro callers only) |
| `all` | `[start + warmup_days, data end]` of the loaded view | any |

Indicators are always computed on the whole loaded view and **then** masked to the period (worker.py behaviour);
because the research view physically stops at the cutoff, nothing at or after the cutoff can influence it.

---

## 3. Data area — `core/data.py`, `core/vault.py`

### Files on disk (outside git)
`<data root>/<source>/<COIN>_<tf>.csv` with header `time,open,high,low,close,volume,turnover`
(time = bar open, `YYYY-MM-DD` for 1d or `YYYY-MM-DD HH:MM:SS` for 4h, naive UTC, sorted, unique, **closed bars
only** — the forming bar is never written). `<data root>/linear/<COIN>_funding.csv` header `time,fundingRate`.
`source ∈ {"linear","spot"}`; symbol = `f"{COIN}USDT"`. This is byte-compatible with
`/Users/matveypro/kronos_lab/indicators/data` (which is a `linear` source). POL has 1d only and starts 2024-09.

### `core/data.py`
```python
class DataError(Exception)
fetch_klines(symbol: str, interval: str, category: str, start_ms: int|None=None, end_ms: int|None=None) -> pd.DataFrame
    # Bybit v5 /market/kline pagination (port of indicators/fetch.py); interval "D" or "240";
    # returns closed bars only, index=time (naive UTC), columns open,high,low,close,volume,turnover (float)
fetch_funding(symbol: str, start_ms: int|None=None) -> pd.Series   # linear only; name "fundingRate"
update(coins: list[str]|None=None, tfs: list[str]|None=None, source: str="spot", funding: bool=True) -> dict
    # incremental: reads existing CSV, fetches from last bar (overlap 2 bars), dedups, appends, atomic rewrite
    # (write tmp + os.replace). Returns {"source","updated":{"BTC_1d": {"rows_added":n,"last":"..."}}, "errors":{...}}
seed_from(path: str|None=None, source: str="linear") -> int   # copy (not symlink) lab CSVs into data root; returns files copied
load_raw(coin, tf, source) -> tuple[pd.DataFrame, pd.Series]  # INTERNAL: full history + per-bar funding; only vault.py may call it
load = vault.load   # re-export: the public loader is vault.load
```
CLI: `python -m trader.core.data update --source spot [--coins BTC,ETH] [--tfs 1d]` → prints the dict; exit 0/1.

### `core/vault.py`
```python
class VaultSealed(Exception)        # access to sealed data without a valid look
cutoff() -> pd.Timestamp            # vault.vault_cutoff
period_mask(index: pd.DatetimeIndex, period: str, *, activated_at: str|None=None) -> np.ndarray[bool]
load(coin: str, tf: str, view: str="research", *, source: str|None=None, look_id: str|None=None,
     spec_hash: str|None=None, asof: str|pd.Timestamp|None=None) -> pd.DataFrame
```
`load` returns a DataFrame indexed by bar open time with columns `open,high,low,close,volume,turnover,funding`
(`funding` = per-bar sum of funding events, worker.py rule `(t - 1ms).floor(tf)`, 0.0 for spot) and
`df.attrs = {"coin","tf","view","source","cutoff","data_end"}`.

- `view="research"` (default source `data.research_source`): rows with index `< vault_cutoff`. Never anything else.
- `view="full"` (default source research_source): all closed rows; **requires** `look_id` that
  `trials.check_look(look_id, spec_hash)` accepts, else `VaultSealed`.
- `view="paper"` (default source `data.paper_source`): all closed rows with index `<= asof` (default now);
  intended only for `core/paper.py`, `core/forecast.py`, `strategy/screener.py`, `fleet/watch.py` and forward
  scoring. Any `backtest.evaluate` with `view="paper"` is logged with its caller (validate flags research callers).
- Invariant tested: for every coin/tf, `load(..., "research").index.max() < cutoff()`.

---

## 4. Backtest area — `core/catalog.py`, `core/backtest.py`

### Strategy spec (JSON) — same shape as worker.py `cfg`
```json
{"tf": "1d", "mode": "SPOT",
 "comps": [{"t": "rsi_reg", "p": {"n": 14, "up": 55, "gap": 5}}],
 "comb": "and", "k": 1, "adx": 0, "vt": 0.5, "stop": 0}
```
- `tf ∈ {"1d","4h"}`, `mode ∈ {"SPOT","PERP"}` (PERP research-only), `comps`: 1–3 items, `t` ∈ catalog keys,
  `p` = params from the catalog grid (off-grid values allowed if the function accepts them),
  `comb ∈ {"and","vote"}`, `k` = votes needed (`and` ⇒ `k=len(comps)`), `adx` ∈ {0,15,20,25,30,...} (0=off),
  `vt` annual vol target (0=off; SPOT cap 1.0, PERP cap 2.0), `stop` ATR trailing-stop multiple (0=off).
- Core keys are exactly these 9; extra keys (`name`, `notes`, ...) are allowed and ignored by hashing.
- `spec_hash(spec)` = md5 of the 9 core keys (see conventions) — equals worker.py `cfg_hash` for worker configs.

### `core/catalog.py` (faithful port of worker.py; same numbers)
```python
nb_supertrend(h, l, c, a, mult) ; nb_hyst(li, lo, si, so) ; nb_trailing(pos, c, a, k)   # numba njit(cache=True)
ema, sma, rmax, rmin, atr(d, n=14), rsi(c, n), adx(d, n=14), sgn
CATALOG: dict[str, tuple[fn, dict[str, list]]]   # the 15 components: ema_x ma donch st macd rsi_reg roc bb_break kelt aroon cci ichi obv rsi_rev bb_rev
FILTERS = {"adx": [0,15,20,25,30], "vt": [0,0.3,0.5,0.8], "stop": [0,2,3,4]}
rand_comp(rng) ; rand_cfg(rng, modes=("SPOT","PERP"), tfs=("1d","1d","4h")) ; mutate(cfg, rng)
    # with default modes/tfs the RNG call sequence is identical to worker.py (same seed ⇒ same configs);
    # modes=("SPOT",) is the spot-only random baseline
validate_spec(spec) -> list[str]    # [] if valid, else human-readable problems
spec_hash(spec) -> str
make_d(df) -> dict                  # {"c","h","l","v","fund","idx"} numpy arrays from a vault DataFrame
base_signals(df) -> dict[str, pd.Series]   # port of indicators/engine.py (legacy names e.g. "rsi55_50")
```

### `core/backtest.py`
```python
class InsufficientData(Exception)
target_positions(spec, d) -> np.ndarray
    # position decided at the CLOSE of each bar using data <= that bar (before the 1-bar shift):
    # votes → pos; SPOT clip ≥0; ADX gate; nb_trailing stop; vol-target (rolling 30-bar std × sqrt(bpy), cap)
bar_returns(pos, d, mode, cost_side) -> tuple[np.ndarray, np.ndarray]
    # wx = pos shifted 1 bar (held during bar t); r = wx*ret - turnover(t-1)*cost_side; PERP also - wx*fund
cost_side(mode, costs=None) -> float
    # costs=None → protected costs[mode] fee+slippage (SPOT 0.0030); "legacy" → legacy_research; float → as given
evaluate(spec, coin, view="research", period="trainval", *, costs=None, look_id=None,
         caller: str, hypothesis_id: str|None=None, activated_at: str|None=None, log: bool=True) -> dict
evaluate_many(spec, coins, view="research", period="trainval", **kw) -> list[dict]   # one evaluate per coin
evaluate_df(spec, df: pd.DataFrame, period: str="all", *, cost_side: float, activated_at=None) -> dict
    # PURE core used by evaluate, sweep_runner and tests: no vault access, no logging, same return dict
    # (trial_id None). df = a vault.load-shaped frame. Period masks via vault.period_mask (cutoff from config).
    # Wave-1 tests may build df straight from the lab CSVs, so backtest tests do not depend on trials/vault.
legacy_bar_returns(df, fund, sig, cost, tf, adx_thr, vt, adx_s)   # verbatim port of engine.bar_returns (vt fixed 0.5)
```
`evaluate` returns:
```python
{"spec_hash": str, "coin": str, "tf": str, "mode": str, "view": str, "period": str, "cost_side": float,
 "returns":   pd.Series,   # net per-bar strategy return, restricted to the period
 "positions": pd.Series,   # wx: exposure held during each bar (same index)
 "target":    pd.Series,   # pos decided at each bar's close (same index) — paper uses the last value
 "metrics":   dict|None,   # score.metrics(returns, bpy, positions); None if period has < 30 bars
 "bh_metrics": dict|None,  # score.buy_and_hold(close over the same period) with one entry cost
 "trial_id":  str|None}    # set when log=True
```
- `caller ∈ {"research","forward","deep_retro","sweep","screener","test","operator"}`.
- `log=True` (default) appends one `single` row to `trials.jsonl` via `trials.log_trial`. Only tests and
  `fleet/sweep_runner.py` (which reports through a batch row instead) may pass `log=False`.
- `view="full"` calls `vault.load(..., look_id=, spec_hash=)`; period `"test"` requires it.
- No lookahead: target at bar t uses only rows ≤ t; it is held during bar t+1; costs of a change at t are
  charged in bar t+1 (identical to worker.py and engine.py).

**Regression fixtures (test_backtest.py must assert):**
1. Spec `{"tf":"1d","mode":"SPOT","comps":[{"t":"rsi_reg","p":{"n":14,"up":55,"gap":5}}],"comb":"and","k":1,"adx":0,"vt":0.5,"stop":0}`
   on ETH full history (via `evaluate_df` on the lab CSV, so no look is spent), cost_side = legacy spot 0.0015,
   period test `[2025-07-01, 2026-10-13]` (470 bars) → CAGR **0.268 ± 0.001** (verified in W0 with the worker.py
   kernels: 0.26818); `spec_hash` = `cd9ff45458116162a26b8d3c8fb059ec`; legacy path
   `legacy_bar_returns(..., base_signals(df)["rsi55_50"], ...)` gives the same. (Source: indicators/final_test_output.txt.)
2. For 20 random worker configs (`random.Random(1)`, `rand_cfg`) on research view, the trainval metrics equal a direct
   call of the original `research/kronos_lab/overnight/worker.evaluate` with `COST` patched to `legacy` (≤1e-9).

---

## 5. Trials area — `core/trials.py`, `core/score.py`

### `journal/trials.jsonl` rows
Single evaluation (written by `backtest.evaluate`):
```json
{"trial_id": "T-20261020T001003-3fa9c1", "ts": "...Z", "kind": "single", "caller": "research", "runner": "air",
 "hypothesis_id": "H-0007", "hypothesis_hash": "sha256…", "spec_hash": "md5…", "spec": {...},
 "coin": "ETH", "tf": "1d", "mode": "SPOT", "view": "research", "period": "trainval", "look_id": null,
 "cost_side": 0.003, "data_end": "2025-06-30T00:00:00Z",
 "metrics": {"cagr":0.21,"total":0.9,"sharpe":1.1,"maxdd":-0.3,"n_obs":1100,"skew":0.1,"kurt":6.2,
             "sr_bar":0.057,"bpy":365,"exposure":0.45,"entries":17}}
```
Batch (written by `trials.log_batch`, one per sweep job; never millions of lines):
```json
{"trial_id": "B-…", "ts": "...Z", "kind": "batch", "caller": "sweep", "runner": "bigsur", "job_id": "J-…",
 "hypothesis_id": "H-0007", "hypothesis_hash": "…", "view": "research", "periods": ["train","val"],
 "cost_side": 0.003, "n_configs": 1200, "n_trials": 34800,
 "sr_moments": {"1d": [n, sum_sr_ann, sumsq_sr_ann], "4h": [n, sum, sumsq]},
 "n_pass": 41, "result_file": "journal/sweeps/J-….json"}
```
Baseline evaluations: `"kind": "baseline"` (same fields as single, `spec` = the random spec); **excluded** from DSR N.

`sr_ann` everywhere = annualised Sharpe of the trainval (or evaluated) segment, as worker.py `sh_tv`.

### `journal/looks.jsonl` rows
```json
{"look_id": "L-2026-10-0003", "ts": "...Z", "month": "2026-10", "runner": "air", "caller": "research",
 "hypothesis_id": "H-0007", "hypothesis_hash": "…", "spec_hashes": ["…"], "coins": ["ETH","BNB"],
 "reason": "passed TRAIN/VAL criteria …", "expires": "...Z", "budget_used": 3, "budget_total": 8}
```

### `journal/hypotheses.jsonl` (status registry, core-written)
```json
{"ts":"...Z","hypothesis_id":"H-0007","hypothesis_hash":"…","status":"registered|sweeping|failed_trainval|passed_trainval|test_pass|test_fail|abandoned",
 "trials":34800,"look_id":null,"summary":"one line","evidence":"journal/sweeps/J-….json"}
```

### `core/trials.py`
```python
class LookBudgetExceeded(Exception); class HypothesisError(Exception)
hypothesis_hash(h: dict) -> str            # sha256(canonical_json(h minus {"hash","status"}))
load_hypothesis(path_or_id) -> dict         # strategy/hypotheses/<id>.json; raises HypothesisError if hash mismatch
register_hypothesis(path) -> dict           # verifies hash, appends status "registered" row; idempotent
set_status(hypothesis_id, status, **fields) -> dict
log_trial(result: dict, *, caller, hypothesis_id=None, kind="single", look_id=None) -> str   # returns trial_id
log_batch(job_result: dict, *, hypothesis_id, caller="sweep") -> str
looks_used(month: str|None=None) -> int     # month "YYYY-MM", default current UTC month
request_look(hypothesis_id: str, spec_hashes: list[str], coins: list[str], reason: str, *, caller="research") -> str
    # requires: hypothesis registered, hash verified, latest status == "passed_trainval", no previous look for
    # this hypothesis, looks_used(month) < vault.looks_per_month; else raises. Appends a looks.jsonl row.
check_look(look_id: str, spec_hash: str|None) -> dict   # raises VaultSealed (import from vault) if unknown/expired/spec not covered
read_trials(**filters) -> list[dict]       # filters: kind, caller, hypothesis_id, since (ISO)
dsr_context(*, tf=None, since=None) -> dict   # {"n_trials": N, "sr_var_daily": v} pooled over single+batch rows (not baseline)
```
CLI: `python -m trader.core.trials looks` → `{"month","used","total","remaining"}`;
`python -m trader.core.trials stats` → counts per kind/caller and the current DSR context.

### `core/score.py`
```python
metrics(returns, bpy: int, positions=None) -> dict
    # {"cagr","total","sharpe","maxdd","n_obs","skew","kurt","sr_bar","bpy","exposure","entries"}
    # cagr = prod(1+r)^(bpy/n) - 1 (-1 if equity ≤ 0); sharpe = mean/std*sqrt(bpy) (0 if std==0);
    # kurt is RAW (non-excess) kurtosis; entries counts flat→long/short transitions of positions (worker rule)
buy_and_hold(close: pd.Series, bpy: int, entry_cost: float|None=None) -> dict   # metrics of holding, minus one entry cost (default spot side)
deflated_sharpe(sr_daily: float, n_obs_days: int, skew: float, kurt: float, *, n_trials: int, sr_var_daily: float) -> float
    # Bailey & López de Prado, exactly goal15/evaluate.py: sr0 = sqrt(V)*((1-γ)Φ⁻¹(1-1/N)+γΦ⁻¹(1-1/(N e)));
    # z = (sr-sr0)*sqrt(n-1)/sqrt(1 - skew*sr + (kurt-1)/4*sr²); returns Φ(z). Daily-equivalent SR = sr_ann/sqrt(365),
    # n_obs_days = n_obs*365/bpy. Uses scipy-free norm_cdf/norm_ppf (port of scipy_free_stats.py, same code).
norm_cdf(x) ; norm_ppf(p)
random_baseline_rate(selected: list[tuple[dict, str]], *, period: str, view: str, look_id: str|None=None,
                     n_random: int|None=None, seed: int=1, criterion: dict|None=None,
                     modes=("SPOT",), tfs=None, coins=None, caller: str) -> dict
    # morning_test.py logic: evaluate each selected (spec, coin) and n_random random (rand_cfg(rng, modes, tfs),
    # rng.choice(coins)) pairs on the same period; default criterion {"cagr_min":0.20,"beat_bh":True};
    # returns {"rate_selected","rate_random","n_selected","n_random","criterion","beats_random": rate_selected > rate_random}
    # random evaluations are logged with kind="baseline".
random_percentile(strategy_returns: pd.Series, coins, *, period, view, n_random, seed=1, caller) -> float
    # share of random spot configs (same coins, equal-weight, same window) whose total return is below the strategy's
portfolio(results: list[dict]) -> pd.Series
    # equal-weight, NO rebalancing between sleeves (portfolio_test.py / budget_options.py): mean of per-sleeve equity curves
promotion_check(strategy_id: str, *, asof=None) -> dict
    # {"eligible": bool, "forward_days": int, "net_return": float, "random_percentile": float,
    #  "bh_return": float, "reasons": [str]} using paper-ledger marks and protected.promotion
brier(p: list[float], outcome: int) -> float      # multi-class Brier: sum_i (p_i - 1[i==outcome])²
```
**Regression test (test_trials.py):** DSR reproduces `indicators/final_test_output.txt` style numbers
(e.g. with N=960 and the daily SR std 0.0226 the expected max SR0 = 0.0732 ± 0.0005), the look budget refuses the
9th look of a month, a second look for the same hypothesis is refused, and `random_baseline_rate` mechanics on a
small sample (≤200 evaluations) run in < 3 min. Reproducing 6.3 % vs 9.4 % exactly is out of scope (596 k candidates).

---

## 6. Paper area — `core/paper.py`, `core/real.py`

### `strategy/active.json` (agent-editable; validated by `paper.validate_active`)
```json
{"version": 1, "updated": "2026-10-20T00:30:00Z",
 "strategies": [
  {"id": "rsi_regime_l4", "status": "paper",
   "spec": {"tf":"1d","mode":"SPOT","comps":[{"t":"rsi_reg","p":{"n":14,"up":55,"gap":5}}],"comb":"and","k":1,"adx":0,"vt":0.5,"stop":0},
   "coins": ["BTC","ETH","SOL","BNB"], "allocation_usdt": 800,
   "activated_at": "2026-10-21", "hypothesis_id": "H-0001",
   "evidence": "research/README.md §6; TEST BTC+ETH+SOL+BNB +22.1% (Bybit EU costs)",
   "screener_veto": false}
 ]}
```
`status ∈ {"paper","paused","retired"}`; `id` matches `^[a-z0-9_]{3,40}$` and is never reused; `activated_at`
is the first UTC date the strategy trades (≥ the date of the commit that added it). Constraints from
`protected.paper`: mode/tf allowed, ≤ max_strategies with status paper, Σ allocation ≤ bankroll, per-strategy cap,
coins ⊂ universe. Retiring = set `status: "retired"` (paper closes its sleeves next run); deleting entries is not allowed.

### `journal/paper-ledger.jsonl` rows (sole writer `core/paper.py`)
```json
{"type":"run","run_id":"P-20261021T001000","ts":"…Z","asof_bar":"2026-10-20T00:00:00Z","status":"ok|skipped|error","n_sleeves":4,"equity_usdt":2003.1,"notes":""}
{"type":"fill","run_id":"…","ts":"…Z","asof_bar":"…","strategy_id":"rsi_regime_l4","sleeve_id":"rsi_regime_l4:ETH","coin":"ETH",
 "side":"buy|sell","qty":0.081,"price":2496.85,"fill_price":2498.10,"notional_usdt":202.3,"fee_usdt":0.51,"slippage_bps":5,
 "prev_weight":0.0,"target_weight":0.81,"reason":"signal|init|retire|rebalance"}
{"type":"skip","run_id":"…","sleeve_id":"…","coin":"ETH","reason":"below_min_order|no_data|veto","notional_usdt":3.2}
{"type":"mark","run_id":"…","ts":"…Z","asof_bar":"…","strategy_id":"…","sleeve_id":"…","coin":"ETH","price":2496.85,
 "qty":0.081,"cash_usdt":0.0,"equity_usdt":202.3,"weight":0.81}
```
Sleeve = (strategy, coin); initial sleeve cash = `allocation_usdt / len(coins)`; no rebalancing between sleeves
(matches `portfolio()`); fills at the close of `asof_bar` ± slippage, fee on notional (Bybit EU spot); orders below
`paper.min_order_usdt` are skipped (logged `skip`) — a known, documented difference from the backtest.

### `core/paper.py`
```python
class PaperError(Exception)
validate_active(active: dict) -> list[str]
run_paper(active_path: str|None=None, *, asof: str|None=None, dry_run: bool=False) -> dict
    # daily at protected.paper.run_at_utc (00:10): asof = last closed 1d bar (yesterday's open time);
    # idempotent per asof_bar (a second run for the same bar returns status "skipped");
    # uses vault.load(view="paper") + backtest.target_positions (same code as backtest ⇒ same signals);
    # returns {"run_id","asof_bar","fills":n,"skips":n,"equity_usdt":x,"per_strategy":{id: equity}}
settle(asof=None) -> dict          # mark-to-market only (no trading); writes mark rows; used by forward cycle
equity(strategy_id=None, *, since=None) -> pd.Series    # daily equity (USDT) from mark rows (all strategies if None)
positions() -> dict                # current {sleeve_id: {"coin","qty","cash_usdt","weight"}}
```
CLI: `python -m trader.core.paper run [--asof YYYY-MM-DD] [--dry-run]`, `... settle`, `... equity [--id X]`.

### `core/real.py` (built, **disabled**)
```python
class RealTradingDisabled(Exception); class RealCapExceeded(Exception)
doctor() -> dict
    # {"ok": bool, "enabled": bool, "checks": [{"name","ok","detail"}]} — checks: config flag, secrets file
    # perms 600 & outside repo, API reachable, key valid, key has NO withdraw permission (/v5/user/query-api),
    # account is spot/UTA, balances, today's spend vs daily cap. Read-only; works when disabled.
place_order(coin: str, side: str, notional_eur: float, *, client_order_id: str, strategy_id: str,
            dry_run: bool=True) -> dict
    # raises RealTradingDisabled unless real_trading_enabled AND strategy_id in allowed_strategy_ids AND coin in
    # allowed_coins; enforces per_order_eur, daily_eur, max_open_positions, min_order_usdt, spot only;
    # writes a real-ledger row for every attempt (dry_run included). Never logs secrets.
mirror_paper(run_id: str, *, dry_run: bool=True) -> list[dict]   # twin orders for allowed strategies' fills in run_id
```
`journal/real-ledger.jsonl` row: `{"ts","client_order_id","strategy_id","coin","side","notional_eur","notional_usdt","status":"dry_run|refused|submitted|filled|rejected","reason","exchange_order_id","fill_price","fee"}`.

---

## 7. Guard area — `loop.sh`, `core/validate.py`, `../.githooks/pre-commit`, `../.github/workflows/trader-ci.yml`

### Agent-writable whitelist (everything else in the repo is PROTECTED)
```
trader/strategy/**                        (except nothing)
trader/journal/retros/**
trader/journal/proposals.md
trader/journal/*.jsonl                    APPEND-ONLY (existing lines byte-identical to the parent commit)
trader/journal/cycles.log                 APPEND-ONLY
trader/journal/sweeps/**                  add-only (new files)
trader/journal/lease.json, trader/journal/fleet-health.json   (rewritten by core/fleet code)
```
Protected (non-exhaustive, for clarity): `trader/core/**`, `trader/config/**`, `trader/fleet/**`, `trader/tests/**`,
`trader/loop.sh`, `trader/CYCLE_*.md`, `trader/REAL.md`, `trader/CONTRACTS.md`, `trader/README.md`,
`trader/journal/operator-notes.md`, `.githooks/**`, `.github/**`, and everything outside `trader/`.
Additionally immutable for everyone but `operator:` commits: a file `strategy/hypotheses/H-*.json` once committed
(its `hash` must keep matching; edits = new hypothesis id).

### Commit-prefix rule
- `operator: <msg>` — human/integrator commits; may touch anything.
- Agent commits: `cycle(<kind>): <msg>` with kind ∈ `research|forward|deep_retro|triggered|research_sweep|lease`; may
  only touch the whitelist. Any non-`operator:` commit is treated as an agent commit.

### `core/validate.py`
```python
AGENT_WRITABLE: list[str]   # glob patterns above
APPEND_ONLY: list[str]
is_agent_writable(path: str) -> bool        # repo-relative path
check_config(cfg: dict) -> list[str]        # schema + hard ceilings: real.per_order_eur ≤ 100, daily_eur ≤ 250,
                                            # real_trading_enabled is bool, costs ≥ 0, looks_per_month ≤ 31, cutoff parses
check_journal() -> list[str]                # all JSONL parse, paper fills within caps, trials rows have required keys
check_strategy() -> list[str]               # active.json valid (paper.validate_active), hypotheses hashes match
check_diff(base: str, head: str) -> list[str]   # per commit in base..head: boundary + append-only + hypothesis immutability
main(argv) -> int
```
CLI: `python trader/core/validate.py [--diff BASE..HEAD] [--list-protected] [--commit-msg-file F]` → prints
`{"ok": bool, "errors": [...]}`; exit 0 ok / 1 errors.

### `loop.sh`
`trader/loop.sh <kind> [--dry-run]` with kind ∈ `research|forward|deep_retro|triggered|tick`.
Steps: pid lock `trader/.loop.pid` → reattach branch & `git fetch` + ff-only sync of `phil-crypto` →
`python -m trader.core.lease acquire` (exit 3/4 ⇒ log `skipped` and exit 0) → for `tick`: decide which kind is due
from `protected.schedule` + `strategy/schedule.json` + pending triggers in `~/.phil-crypto/inbox` → `fleet/inbox.py merge`
→ run `claude -p "$(cat trader/CYCLE_<KIND>.md)" --model <cycle model> --permission-mode acceptEdits` (headless,
timeout) → **revert any change outside the agent whitelist** (`git checkout HEAD -- <path>` / remove untracked) →
`validate.py` → commit `cycle(<kind>): <date> <summary>` → push → lease release → append one `cycles.log` line.
Real mode is never started by loop.sh while `real_trading_enabled` is false.

### `journal/cycles.log` line format (one line per cycle, space-separated, free note after ` | `)
```
2026-10-21T00:10:03Z forward ok runner=air dur=312s trials=+0 looks=+0 fills=4 commit=a1b2c3d | settled 4 sleeves, no edits
```
status ∈ `ok|skipped|error|reverted` (reverted = protected-path edits were undone). Regex:
`^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ (research|forward|deep_retro|triggered|research_sweep) (ok|skipped|error|reverted)( \w+=\S+)*( \| .*)?$`

### CI (`../.github/workflows/trader-ci.yml`) and pre-commit
Both run `validate.py` (CI with `--diff ${{ github.event.before }}..${{ github.sha }}`, pre-commit on the staged
tree with the pending message) and `python -m py_compile` on `trader/**/*.py`; CI also runs the fast tests
(`trader/tests/test_*.py`, each < 3 min, no network). Activate locally with `git config core.hooksPath .githooks`.

---

## 8. Fleet area — `core/lease.py`, `fleet/*`

### `journal/lease.json`
held: `{"runner":"air","cycle":"forward","started":"…Z","ttl_s":5400,"pid":1234}`;
free: `{"runner":null,"released_by":"air","released":"…Z"}`; missing file = free.
```python
check() -> dict ; acquire(cycle: str, *, runner=None, ttl_s=None) -> dict ; release(*, runner=None) -> dict
```
CLI `python -m trader.core.lease check|acquire|release [--cycle K] [--ttl S]`, exit 0 acquired/released,
3 held by another live lease, 4 not written (push failed / diverged). Same commit-and-push semantics as the Phil
pattern: each acquire/release is one commit `cycle(lease): <runner> <acquire|release>` touching only
`journal/lease.json` (so `lease` is also an allowed agent-commit kind; it is whitelisted by the guard).
Without an `origin` remote: `{"acquired": true, "written": false}`.

### Sweep jobs — `fleet/dispatch.py`
Job file (`<remote_root>/jobs/<job_id>/job.json`):
```json
{"job_id":"J-20261021T0200-ab12","hypothesis_id":"H-0007","hypothesis_hash":"…","created":"…Z",
 "host":"bigsur|m2|local|auto","specs":[{...},{...}],"coins":["BTC","ETH"],"view":"research",
 "periods":["train","val"],"costs":null,
 "criterion":{"cagr_tr":0.20,"cagr_va":0.20,"sh_tr":0.8,"sh_va":0.8,"entries":8,"mdd":-0.45},
 "top_k":200,"quota":{"max_disk_mb":512,"max_ram_mb":4096,"max_minutes":120,"max_procs":6}}
```
`criterion` defaults to worker.py `is_candidate`. `view` MUST be `"research"`: dispatch ships **only research-view
CSVs** (truncated at the cutoff, written by `vault.load`) to the remote host, so sweep hosts never hold sealed data.
```python
submit(job: dict) -> str                    # validates (≤ research.max_trials_per_hypothesis specs×coins), picks host,
                                            # rsyncs trader/ code + research CSVs + job.json, starts sweep_runner via ssh nohup
status(job_id) -> dict                      # {"job_id","host","state":"queued|running|done|failed|killed","progress":0..1,"elapsed_s"}
collect(job_id) -> dict                     # fetches result.json (≤ fleet.max_result_kb), copies to journal/sweeps/<job_id>.json
ingest(job_id) -> str                       # collect + trials.log_batch → returns batch trial_id
run_local(job: dict) -> dict                # same as remote but in-process (tests, small jobs)
cleanup(job_id) -> None                     # removes remote job dir (keeps bigsur under its disk quota)
```
CLI: `python -m trader.fleet.dispatch submit job.json | status J | collect J | ingest J | cleanup J | hosts`.

`fleet/sweep_runner.py job.json` (remote, py3.9) writes `result.json`:
```json
{"job_id":"…","host":"bigsur","n_configs":1200,"n_trials":34800,"elapsed_s":812,
 "sr_moments":{"1d":[n,sum,sumsq],"4h":[n,sum,sumsq]},"n_pass":41,
 "rows":[{"spec_hash":"…","spec":{...},"coin":"ETH","cagr_tr":…,"sh_tr":…,"cagr_va":…,"sh_va":…,"sh_tv":…,"mdd":…,"entries":…,"expo":…,"pass":true}],
 "errors":0}
```
`rows` = all passing rows plus the best `top_k` by `sh_tv`, truncated so the file stays ≤ `max_result_kb`.
Runner enforces quota (RAM via resource limits/streaming, wall clock, disk) and exits nonzero on breach.

### Inbox (berry → air, no git writes on berry)
berry `rsync`s into air `fleet.inbox_dir`: `fleet-health.json`, `watch-triggers-<host>.jsonl`, `trigger-<ts>.json`.
`fleet/inbox.py merge` (run by loop.sh on air) copies `fleet-health.json` into `journal/`, appends new trigger rows to
`journal/watch-triggers.jsonl`, and returns the list of pending trigger files (consumed → moved to `inbox/done/`).

### `journal/fleet-health.json`
```json
{"updated":"…Z","by":"berry",
 "hosts":{"air":{"last_seen":"…Z","ok":true,"load1":0.8,"disk_free_gb":60.1,"checks":{"ssh":true,"claude":true,"lease":"free"}},
          "bigsur":{"last_seen":"…Z","ok":true,"disk_free_gb":16.2,"runner_disk_gb":0.4,"jobs_running":1,"checks":{"ssh":true,"python":true}},
          "berry":{"last_seen":"…Z","ok":true,"disk_free_gb":43.0,"checks":{"data_fresh":true,"watch_last_run":"…Z"}},
          "m2":{"last_seen":null,"ok":false,"optional":true}},
 "alerts":[{"ts":"…Z","host":"bigsur","level":"warn|crit","msg":"runner disk 4.6/5 GB"}]}
```
`fleet/health.py` (berry, every 15 min) produces it; `fleet/setup_*.sh` are idempotent and install
launchd (`fleet/launchd/com.phil-crypto.<job>.plist`: `tick` every 15 min, `forward` 00:10 UTC, `deep_retro` 06:00 UTC on air)
and systemd user timers (`fleet/systemd/phil-<job>.{service,timer}`: `watch`, `health`, `data-update` on berry).
launchd schedules are in local time: setup_air.sh must convert from UTC (or rely on `tick`, which checks UTC itself).

---

## 9. Wave 2 areas

### Hypothesis pre-registration — `strategy/hypotheses/H-NNNN.json` (agent-written, immutable once committed)
```json
{"id": "H-0007", "created": "2026-10-21T02:00:00Z", "author": "research-cycle@air",
 "title": "RSI regime + ADX gate on large caps", "rationale": "…why, with links to playbook evidence…",
 "family": "rsi_reg+adx",
 "base_spec": {"tf":"1d","mode":"SPOT","comps":[{"t":"rsi_reg","p":{"n":14,"up":55,"gap":5}}],"comb":"and","k":1,"adx":20,"vt":0.5,"stop":0},
 "grid": {"comps.0.p.up": [50,55,60], "adx": [0,15,20,25]},
 "coins": ["BTC","ETH","SOL","BNB","XRP","ADA","LINK","AVAX","DOGE"],
 "trainval_criterion": {"min_coins_passing": 4, "cagr_tr": 0.10, "cagr_va": 0.10, "sh_tr": 0.6, "sh_va": 0.6, "mdd": -0.45, "entries": 8},
 "test_criterion": {"portfolio_cagr_min": 0.10, "beat_bh": true, "beat_random_rate": true, "dsr_min": 0.5},
 "max_trials": 432,
 "hash": "sha256 of canonical JSON of every field except hash/status"}
```
Grid keys are dotted paths into the spec; expanded specs = cartesian product applied to `base_spec`;
`len(specs) × len(coins) ≤ min(max_trials, research.max_trials_per_hypothesis)`.

### Research loop — `CYCLE_RESEARCH.md`, `strategy/tools/sweep.py`
```python
expand(h: dict) -> list[dict]                      # grid → specs (validated by catalog.validate_spec)
run(hypothesis_id: str, *, host="auto", wait=True) -> dict
    # trials.register_hypothesis → status sweeping → dispatch.submit/ingest → apply trainval_criterion
    # → set_status passed_trainval|failed_trainval → returns {"hypothesis_id","n_trials","passed": bool,"best":[...],"batch_trial_id"}
open_vault(hypothesis_id: str, reason: str) -> dict
    # only if passed_trainval: trials.request_look → backtest.evaluate(view="full", period="test") on the chosen specs
    # → score.random_baseline_rate + buy&hold + DSR (trials.dsr_context) → test_pass|test_fail → returns the report dict
```
CLI: `python -m trader.strategy.tools.sweep run H-0007 | open H-0007 --reason "…"`. The cycle writes a
`journal/retros/YYYY-MM-DD-research.md` and a playbook entry citing trial ids / look ids for every claim.

### Forward loop — `CYCLE_FORWARD.md` (00:10 UTC), `CYCLE_DEEP_RETRO.md` (06:00 UTC), `strategy/tools/report.py`
```python
daily_report(asof=None) -> dict
    # per strategy: forward equity, net return, vs buy&hold of its coins, vs random percentile, maxdd,
    # days since activation, promotion_check, screener Brier vs baseline; writes nothing (the cycle writes the retro)
```
Forward cycle: `data.update(source="spot")` → `paper.run_paper()` → `forecast.resolve_due()` → `report.daily_report()`
→ retro `journal/retros/YYYY-MM-DD-forward.md` → evidence-backed edits to `strategy/` only. Deep retro: audits the
last 24 h of agent edits, verdict per edit `keep|sharpen|revert` in `journal/retros/YYYY-MM-DD-deep.md`.

### Screener — `core/forecast.py`, `strategy/screener.py`, `strategy/screener_questions.json`
`strategy/screener_questions.json`:
```json
{"version":1,"questions":[
 {"id":"q_up_7d","kind":"binary","tf":"1d","horizon_bars":7,
  "text":"Will {coin} close higher in 7 days than at the last close?",
  "resolver":{"type":"close_gt"},"baseline":"base_rate_365d"},
 {"id":"q_ret_bucket_7d","kind":"choice","tf":"1d","horizon_bars":7,"options":["< -5%","-5%..+5%","> +5%"],
  "resolver":{"type":"ret_bucket","edges":[-0.05,0.05]},"baseline":"base_rate_365d"},
 {"id":"q_score_trend_14d","kind":"score","tf":"1d","horizon_bars":14,
  "text":"Probability (0-100) that {coin} gains more than 3% over 14 days",
  "resolver":{"type":"ret_gt","x":0.03},"baseline":"base_rate_365d"}]}
```
Resolver types (implemented in `core/forecast.py`, closed set): `close_gt`, `ret_gt` (x), `ret_lt` (x),
`ret_bucket` (edges), `vol_gt` (x, realised annual vol over horizon). `score` = probability 0–100 → p/100, binary Brier.
Baselines: `base_rate_365d` (frequency of the outcome in the previous 365 days of paper-view data before asof),
`uniform`.

`journal/forecasts.jsonl` (sole writer `core/forecast.py`):
```json
{"type":"forecast","forecast_id":"F-20261021-ETH-q_up_7d","ts":"…Z","asof_bar":"2026-10-20T00:00:00Z","question_id":"q_up_7d",
 "coin":"ETH","kind":"binary","options":["no","yes"],"p":[0.42,0.58],"baseline_p":[0.47,0.53],"model":"claude-haiku-5-5",
 "prompt_hash":"sha256…","horizon_bars":7,"resolves_at_bar":"2026-10-27T00:00:00Z"}
{"type":"resolve","forecast_id":"…","ts":"…Z","outcome":1,"brier":0.3528,"brier_baseline":0.4418}
```
```python
log_forecast(question: dict, coin: str, asof_bar, p: list[float], *, model: str, prompt_hash: str) -> str   # computes baseline_p
resolve_due(asof=None) -> list[dict]          # appends resolve rows for forecasts whose resolves_at_bar ≤ last closed bar
brier_summary(*, since=None, question_id=None) -> dict   # {"n","brier","brier_baseline","skill": 1 - brier/brier_baseline,"days"}
veto_allowed() -> bool                         # protected.screener thresholds
```
`strategy/screener.py run [--coins …] [--dry-run]`: builds prompts from questions + features (paper view, closed bars
only), calls Haiku through `claude -p --model <screener.model> --output-format json` (injectable `call_model`
function for tests; tests never call the model), parses strict JSON, respects `max_calls_per_day`, logs via
`forecast.log_forecast`. One forecast per (question, coin, asof_bar) — duplicates are refused.

### Watchers — `fleet/watch.py`, `strategy/watchlist.json` (berry, every 15 min, pure Python + requests)
```json
{"version":1,"rules":[
 {"id":"btc_drop_6pct_24h","coin":"BTC","metric":"ret_24h","op":"<","value":-0.06,"cooldown_min":360,"action":"trigger_forward"},
 {"id":"eth_vol_spike","coin":"ETH","metric":"vol_ratio_24h_30d","op":">","value":2.5,"cooldown_min":720,"action":"note"}]}
```
Metrics (closed set): `ret_1h`, `ret_4h`, `ret_24h`, `price`, `vol_ratio_24h_30d`, `dist_from_ema200_1d`.
Actions: `note` (row only) | `trigger_forward` (row + trigger file ⇒ loop.sh runs a `triggered` cycle; a triggered cycle
may only settle, retro and pause/resume strategies — never add new ones).
`watch-triggers` row: `{"ts","rule_id","coin","metric","value","threshold","action","host":"berry"}`;
trigger file `trigger-<ts>.json` = the same object. State (cooldowns) in `~/.phil-crypto/watch-state.json` on berry.

### Seed — `strategy/playbook.md`, `strategy/active.json`, `strategy/schedule.json`, `strategy/hypotheses/backlog.md`
`strategy/schedule.json` (agent-editable, bounded by protected.schedule):
```json
{"version":1,"research":{"enabled":true,"every_hours":6,"max_minutes":60,"max_sweeps_per_day":4},
 "screener":{"enabled":true,"at_utc":"00:20","coins":["BTC","ETH","SOL","BNB"]},
 "data_update":{"spot_every_minutes":60}}
```
`active.json` seed: RSI regime (spec of §4 fixture, but with protected spot costs) on BTC/ETH/SOL/BNB and on the
9 large caps (`engine.UNIV`), plus the breakout C4 as a hypothesis in `backlog.md` (not active until it passes via the
research loop, because C4 needs cross-sectional weights the spec format does not express — see open issue).
`playbook.md` sections: Findings (with evidence links into `research/`), Rules of honesty, Active strategies,
Graveyard (what failed and why), Open questions.

---

## 10. Test commands (each < 3 min, no network unless the test name ends in `_net`)
```
/Users/matveypro/.venvs/kronos/bin/python trader/tests/test_data.py
/Users/matveypro/.venvs/kronos/bin/python trader/tests/test_backtest.py
/Users/matveypro/.venvs/kronos/bin/python trader/tests/test_trials.py
/Users/matveypro/.venvs/kronos/bin/python trader/tests/test_paper.py
/Users/matveypro/.venvs/kronos/bin/python trader/tests/test_guard.py
/Users/matveypro/.venvs/kronos/bin/python trader/tests/test_fleet.py
(+ W2: test_research.py, test_forward.py, test_screener.py, test_watch.py, test_seed.py)
```
Tests are plain scripts with `assert`, end by printing `OK <name>`, and exit nonzero on failure.
