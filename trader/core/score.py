"""Scoring: metrics, buy&hold, Deflated Sharpe, random-config baseline, portfolio, promotion, Brier.

PROTECTED CORE (trials area, Wave 1). Interface: trader/CONTRACTS.md section 5.
Python 3.9; numpy + pandas only (catalog/backtest/vault/paper are imported lazily inside functions so this
module stays importable on every host and has no import cycle with them).

Ports (numbers identical to the research code):
- metrics(): research/kronos_lab/overnight/worker.py seg_stats/evaluate (numpy std ddof=0, CAGR rule,
  maxdd on the equity curve, entries rule) + goal15/evaluate.py stats (pandas skew, raw kurtosis = kurt()+3).
- deflated_sharpe(), norm_cdf(), norm_ppf(): goal15/evaluate.py + goal15/scipy_free_stats.py (same code).
- random_baseline_rate(): overnight/morning_test.py (random.Random(seed); rand_cfg(rng) then rng.choice(coins);
  confirmed = CAGR > 0.20 and CAGR > buy&hold).
- portfolio(): overnight/portfolio_test.py / budget_options.py (equal weight, no rebalancing between sleeves).
"""
from __future__ import annotations

import contextlib
import math
import random
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from trader.core import config

EULER_GAMMA = 0.5772156649
DEFAULT_CRITERION = {"cagr_min": 0.20, "beat_bh": True}


# ----------------------------------------------------------------------------------------------- stats helpers
def norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def norm_ppf(p):
    """Inverse normal CDF by bisection (sufficient precision for DSR). Same code as scipy_free_stats.py."""
    lo, hi = -10.0, 10.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if norm_cdf(mid) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _f(x):
    """JSON-safe float (nan/inf -> None)."""
    if x is None:
        return None
    x = float(x)
    return x if math.isfinite(x) else None


def _spot_side():
    c = config.load_config()["costs"]["spot"]
    return float(c["fee_side"]) + float(c["slippage_side"])


# ----------------------------------------------------------------------------------------------- metrics
def metrics(returns, bpy: int, positions=None, *, min_obs: int = 30) -> Optional[dict]:
    """Per-segment metrics; None if fewer than `min_obs` bars (worker.py seg_stats rule: n < 30 -> None).

    cagr = prod(1+r)^(bpy/n) - 1 (-1 if equity <= 0); sharpe = mean/std*sqrt(bpy) with numpy std (ddof=0, as
    worker.py) and 0 if std == 0; maxdd = min(eq/cummax(eq) - 1); skew = pandas skew; kurt = RAW kurtosis
    (pandas excess kurt + 3, as goal15); sr_bar = per-bar mean/std; exposure = mean |positions|;
    entries = bars where the held position becomes non-zero with a new sign (worker.py rule).
    """
    r = np.asarray(returns, dtype=float)
    r = np.nan_to_num(r, nan=0.0)
    n = len(r)
    if n < max(int(min_obs), 2):
        return None
    g = float(np.prod(1 + r))
    cagr = g ** (bpy / n) - 1 if g > 0 else -1.0
    sd = float(r.std())
    mean = float(r.mean())
    sharpe = mean / sd * math.sqrt(bpy) if sd > 0 else 0.0
    sr_bar = mean / sd if sd > 0 else 0.0
    eq = np.cumprod(1 + r)
    with np.errstate(divide="ignore", invalid="ignore"):
        dd = eq / np.maximum.accumulate(eq) - 1
    maxdd = float(np.nanmin(dd)) if np.isfinite(dd).any() else -1.0
    s = pd.Series(r)
    skew = s.skew()
    kurt = s.kurt() + 3
    exposure = entries = None
    if positions is not None:
        w = np.nan_to_num(np.asarray(positions, dtype=float), nan=0.0)
        exposure = float(np.abs(w).mean()) if len(w) else 0.0
        entries = int(((w[1:] != 0) & (np.sign(w[1:]) != np.sign(w[:-1]))).sum()) if len(w) > 1 else 0
    return {"cagr": _f(cagr), "total": _f(g - 1), "sharpe": _f(sharpe), "maxdd": _f(maxdd), "n_obs": int(n),
            "skew": _f(skew) if _f(skew) is not None else 0.0, "kurt": _f(kurt) if _f(kurt) is not None else 3.0,
            "sr_bar": _f(sr_bar), "bpy": int(bpy), "exposure": exposure, "entries": entries}


def buy_and_hold(close: pd.Series, bpy: int, entry_cost: Optional[float] = None) -> Optional[dict]:
    """Metrics of holding `close` over its whole span, minus one entry cost (default spot fee+slippage side).

    Buy at the first close, hold: returns are close-to-close from the 2nd bar on (len(close)-1 bars, as
    morning_test.py annualises), the entry cost is charged on the first of them. backtest.evaluate_df passes the
    close series including the bar before the period, so the bars line up with the strategy returns.
    """
    if entry_cost is None:
        entry_cost = _spot_side()
    c = pd.Series(close, dtype=float).dropna()
    if len(c) < 2:
        return None
    r = c.pct_change().iloc[1:].values.copy()
    r[0] -= float(entry_cost)
    return metrics(r, bpy, np.ones(len(r)))


# ----------------------------------------------------------------------------------------------- DSR
def expected_max_sr(n_trials: int, sr_var_daily: float) -> float:
    """SR0 = sqrt(V)*((1-g)*Phi^-1(1-1/N) + g*Phi^-1(1-1/(N e))) — the expected max daily SR under the null."""
    n = int(n_trials)
    if n < 2 or not sr_var_daily or sr_var_daily <= 0:
        return 0.0
    em = EULER_GAMMA
    return math.sqrt(sr_var_daily) * ((1 - em) * norm_ppf(1 - 1 / n) + em * norm_ppf(1 - 1 / (n * math.e)))


def deflated_sharpe(sr_daily: float, n_obs_days: int, skew: float, kurt: float, *, n_trials: int,
                    sr_var_daily: float) -> float:
    """Bailey & Lopez de Prado DSR, exactly goal15/evaluate.py (kurt is RAW kurtosis).

    z = (sr - sr0) * sqrt(n-1) / sqrt(1 - skew*sr + (kurt-1)/4*sr^2); returns Phi(z).
    Callers convert: sr_daily = sr_ann/sqrt(365); n_obs_days = n_obs*365/bpy.
    """
    sr0 = expected_max_sr(n_trials, sr_var_daily)
    n = float(n_obs_days)
    if n <= 1:
        return 0.0
    den = 1 - skew * sr_daily + (kurt - 1) / 4 * sr_daily ** 2
    if not math.isfinite(den) or den <= 1e-12:
        den = 1e-12
    z = (sr_daily - sr0) * math.sqrt(n - 1) / math.sqrt(den)
    return norm_cdf(z)


def dsr_of_metrics(m: dict, *, n_trials: int, sr_var_daily: float) -> Optional[float]:
    """Convenience: DSR from a metrics() dict (annualised sharpe -> daily-equivalent, n_obs -> days)."""
    if not m or m.get("sharpe") is None:
        return None
    bpy = m.get("bpy") or 365
    sr_d = float(m["sharpe"]) / math.sqrt(365)
    n_days = int(round(m["n_obs"] * 365 / bpy))
    return deflated_sharpe(sr_d, n_days, float(m.get("skew") or 0.0), float(m.get("kurt") or 3.0),
                           n_trials=n_trials, sr_var_daily=sr_var_daily)


# ----------------------------------------------------------------------------------------------- baselines
def passes(m: Optional[dict], bh: Optional[dict], criterion: dict) -> bool:
    """morning_test 'confirmed' rule generalised: cagr > cagr_min and (if beat_bh) cagr > buy&hold cagr."""
    if not m or m.get("cagr") is None:
        return False
    if m["cagr"] <= criterion.get("cagr_min", -1e18):
        return False
    if "sharpe_min" in criterion and (m.get("sharpe") or 0.0) <= criterion["sharpe_min"]:
        return False
    if "mdd_min" in criterion and (m.get("maxdd") or -1.0) <= criterion["mdd_min"]:
        return False
    if criterion.get("beat_bh"):
        if not bh or bh.get("cagr") is None or m["cagr"] <= bh["cagr"]:
            return False
    return True


def _default_universe():
    return list(config.load_config()["universe"])


def random_baseline_rate(selected: List[Tuple[dict, str]], *, period: str, view: str,
                         look_id: Optional[str] = None, n_random: Optional[int] = None, seed: int = 1,
                         criterion: Optional[dict] = None, modes=("SPOT",), tfs=None, coins=None,
                         caller: str) -> dict:
    """Share of selected (spec, coin) pairs meeting `criterion` vs the share of random (spec, coin) pairs.

    Selected pairs go through backtest.evaluate (logged as kind "single"). Random pairs are drawn exactly like
    morning_test.py (rng = random.Random(seed); spec = rand_cfg(rng, modes, tfs); coin = rng.choice(coins)), are
    evaluated on the same view/period with the pure backtest.evaluate_df on vault-loaded data (a granted look
    through the private trials._baseline_scope, restricted to the coins the look covers) and are logged with
    kind "baseline" (excluded from DSR N). On view="full" the random coin pool is the look's coins.
    """
    from trader.core import backtest, trials

    crit = dict(DEFAULT_CRITERION if criterion is None else criterion)
    if n_random is None:
        n_random = int(config.load_config()["research"]["random_baseline_n"])
    pool = list(coins) if coins else _default_universe()

    sel_pass = sel_n = 0
    for spec, coin in selected:
        res = backtest.evaluate(spec, coin, view=view, period=period, look_id=look_id, caller=caller)
        if res.get("metrics") is None:
            continue
        sel_n += 1
        sel_pass += passes(res["metrics"], res.get("bh_metrics"), crit)

    if view == "full":
        # sealed data: random pairs may only use coins the look covers (check_look refuses the rest)
        covered = set(trials.look_coins(look_id)) if look_id else set()
        pool = [c for c in pool if c in covered]
        if not pool:
            raise _vault_sealed()("random baseline on view='full' needs a look covering at least one pool coin")
        scope = trials._baseline_scope(look_id, caller=caller)
    else:
        scope = contextlib.nullcontext()
    with scope:
        rnd_pass, rnd_n, skipped = _random_pairs(pool, n_random, seed, modes, tfs, view, period, look_id, crit,
                                                 caller)

    rs = sel_pass / sel_n if sel_n else 0.0
    rr = rnd_pass / rnd_n if rnd_n else 0.0
    return {"rate_selected": rs, "rate_random": rr, "n_selected": sel_n, "n_random": rnd_n,
            "n_skipped": skipped, "criterion": crit, "beats_random": bool(sel_n and rs > rr)}


def _vault_sealed():
    try:
        from trader.core.vault import VaultSealed
        return VaultSealed
    except Exception:
        from trader.core import trials
        return trials._vault_sealed()


def _random_pairs(pool, n_random, seed, modes, tfs, view, period, look_id, crit, caller):
    from trader.core import backtest, catalog, trials, vault

    rng = random.Random(seed)
    cache: Dict[Tuple[str, str], pd.DataFrame] = {}
    rnd_pass = rnd_n = skipped = 0
    for _ in range(int(n_random)):
        kw = {"modes": tuple(modes)}
        if tfs is not None:
            kw["tfs"] = tuple(tfs)
        spec = catalog.rand_cfg(rng, **kw)
        coin = rng.choice(pool)
        key = (coin, spec["tf"])
        try:
            if key not in cache:
                cache[key] = vault.load(coin, spec["tf"], view, look_id=look_id, spec_hash=None)
            res = backtest.evaluate_df(spec, cache[key], period,
                                       cost_side=backtest.cost_side(spec["mode"]))
        except Exception as e:  # missing coin/tf file or too little data: not a trial
            if e.__class__.__name__ == "VaultSealed":
                raise
            skipped += 1
            continue
        if res.get("metrics") is None:
            skipped += 1
            continue
        res = dict(res)
        res.setdefault("spec", spec)
        res["coin"] = coin
        res.setdefault("tf", spec["tf"])
        res.setdefault("mode", spec["mode"])
        res.setdefault("view", view)
        res.setdefault("period", period)
        trials.log_trial(res, caller=caller, kind="baseline", look_id=look_id)
        rnd_n += 1
        rnd_pass += passes(res["metrics"], res.get("bh_metrics"), crit)
    return rnd_pass, rnd_n, skipped


def _window(s: pd.Series, start, end) -> pd.Series:
    return s[(s.index >= start) & (s.index <= end)]


def random_percentile(strategy_returns: pd.Series, coins, *, period: str, view: str, n_random: int,
                      seed: int = 1, caller: str, look_id: Optional[str] = None, tfs=("1d",),
                      activated_at: Optional[str] = None, data: Optional[Dict[str, pd.DataFrame]] = None,
                      log: bool = True) -> float:
    """Share of random SPOT configs (same coins, equal-weight no-rebalance, same window) whose total return is
    strictly below the strategy's. One summary row of kind "baseline" is logged (not one per config, so a
    daily report does not flood the ledger). `data` ({coin: df}) may be passed to skip vault loading.
    """
    from trader.core import backtest, catalog

    sr = pd.Series(strategy_returns).dropna()
    if len(sr) == 0:
        return 0.0
    start, end = sr.index.min(), sr.index.max()
    strat_total = float(np.prod(1 + sr.values) - 1)
    coins = list(coins)
    if data is None:
        from trader.core import vault
        if view == "full":
            from trader.core import trials
            with trials._baseline_scope(look_id, caller=caller):
                data = {c: vault.load(c, tfs[0], view, look_id=look_id, spec_hash=None, caller=caller) for c in coins}
        else:
            data = {c: vault.load(c, tfs[0], view, look_id=look_id, spec_hash=None, caller=caller) for c in coins}
    rng = random.Random(seed)
    below = n = 0
    for _ in range(int(n_random)):
        spec = catalog.rand_cfg(rng, modes=("SPOT",), tfs=tuple(tfs))
        sleeves = []
        for c in coins:
            df = data.get(c)
            if df is None:
                continue
            res = backtest.evaluate_df(spec, df, "all", cost_side=backtest.cost_side("SPOT"),
                                       activated_at=activated_at)
            r = res.get("returns")
            if r is None:
                continue
            sleeves.append({"returns": _window(r, start, end)})
        if not sleeves:
            continue
        eq = portfolio(sleeves)
        if len(eq) == 0:
            continue
        n += 1
        below += float(eq.iloc[-1] - 1) < strat_total
    pct = below / n if n else 0.0
    if log:
        from trader.core import trials
        trials.log_baseline_summary({"what": "random_percentile", "coins": coins, "period": period,
                                     "view": view, "n_random": n, "seed": seed, "percentile": pct,
                                     "strategy_total": strat_total, "window": [config.iso(start), config.iso(end)]},
                                    caller=caller, look_id=look_id)
    return pct


# ----------------------------------------------------------------------------------------------- portfolio
def portfolio(results: List[dict]) -> pd.Series:
    """Equal-weight portfolio EQUITY curve (starts near 1.0), NO rebalancing between sleeves
    (portfolio_test.py: eq = mean over sleeves of cumprod(1 + r)). Missing bars of a sleeve earn 0.
    Use portfolio_returns() for the per-bar returns of that curve.
    """
    rs = [pd.Series(x["returns"], dtype=float) for x in results if x.get("returns") is not None]
    if not rs:
        return pd.Series(dtype=float)
    R = pd.concat(rs, axis=1).fillna(0.0)
    return (1 + R).cumprod().mean(axis=1)


def portfolio_returns(results: List[dict]) -> pd.Series:
    eq = portfolio(results)
    if len(eq) == 0:
        return eq
    return eq.pct_change().fillna(eq.iloc[0] - 1.0)


# ----------------------------------------------------------------------------------------------- promotion
PAPER_LEDGER = "paper-ledger.jsonl"


def _bar(x) -> pd.Timestamp:
    t = pd.Timestamp(x)
    return t.tz_convert("UTC").tz_localize(None) if t.tzinfo is not None else t


def sleeve_contributions(strategy_id: str, rows: Optional[list] = None) -> Dict[pd.Timestamp, float]:
    """Net capital flows into the strategy's sleeves per bar: {asof_bar: USDT} (deposits > 0, withdrawals < 0).

    Read from the paper ledger (journal/paper-ledger.jsonl, read-only here), never from active.json:
    - mark rows that record flows (``contributed_usdt`` present): the flow is the mark's ``flow_usdt`` (0 if absent);
    - legacy mark rows (no ``contributed_usdt``): one init deposit on the sleeve's first mark = that mark's cash +
      buys (notional + fee) - sells (notional - fee) of the same sleeve in the same run, nothing afterwards.
    """
    if rows is None:
        rows = config.jsonl_read(config.journal_dir() / PAPER_LEDGER)
    fills: Dict[Tuple[str, str], float] = {}
    for f in rows:
        if f.get("type") == "fill" and f.get("strategy_id", str(f.get("sleeve_id", "")).split(":")[0]) == strategy_id:
            n, fee = float(f.get("notional_usdt") or 0.0), float(f.get("fee_usdt") or 0.0)
            k = (f.get("sleeve_id"), f.get("run_id"))
            fills[k] = fills.get(k, 0.0) + ((n + fee) if f.get("side") == "buy" else -(n - fee))
    seen = set()
    flows: Dict[pd.Timestamp, float] = {}
    for m in rows:
        if m.get("type") != "mark" or m.get("strategy_id") != strategy_id:
            continue
        sid = m.get("sleeve_id")
        if m.get("contributed_usdt") is not None:
            fl = float(m.get("flow_usdt") or 0.0)
        elif sid not in seen:
            fl = float(m.get("cash_usdt") or 0.0) + fills.get((sid, m.get("run_id")), 0.0)
        else:
            fl = 0.0
        seen.add(sid)
        if fl:
            t = _bar(m["asof_bar"])
            flows[t] = flows.get(t, 0.0) + fl
    return flows


def flow_adjusted_returns(eq: pd.Series, flows: Dict[pd.Timestamp, float]) -> pd.Series:
    """Time-weighted per-bar returns of an equity curve with capital flows removed.

    Paper moves capital at the close of a bar (deposit before that close's trade, withdrawal after it), so
    r_t = (E_t - F_t) / E_{t-1} - 1; on a bar with no capital at risk before it (E_{t-1} <= 0) r_t = E_t / F_t - 1
    (the first bar: the init deposit and its entry costs). Chaining prod(1 + r) gives the return on capital actually
    invested, independent of the allocation written in active.json. Bars with no capital at all earn 0.
    """
    eq = pd.Series(eq, dtype=float).dropna().sort_index()
    out = np.zeros(len(eq))
    prev = 0.0
    for i, (t, e) in enumerate(eq.items()):
        f = float(flows.get(_bar(t), 0.0))
        if prev > 0:
            out[i] = (e - f) / prev - 1.0
        elif f > 0:
            out[i] = e / f - 1.0
        prev = float(e)
    return pd.Series(out, index=eq.index)


def promotion_check(strategy_id: str, *, asof=None) -> dict:
    """Eligibility for real-money proposal, from paper-ledger marks and protected.promotion.

    Needs core/paper.py (equity), the paper ledger's capital records and strategy/active.json. Any missing piece
    -> eligible False with a reason. net_return and the random-baseline input are TIME-WEIGHTED, flow-adjusted
    returns on the capital actually put into the sleeves (sleeve_contributions: deposits/withdrawals recorded on
    the paper ledger); active.json allocation_usdt is never used for scoring, so editing it cannot inflate or
    deflate the result. Extra keys: contributed_usdt (net capital in, as of the last mark), flows_usdt (in window).
    """
    import json

    cfg = config.load_config()
    P = cfg["promotion"]
    out = {"eligible": False, "forward_days": 0, "net_return": None, "random_percentile": None,
           "bh_return": None, "reasons": []}
    try:
        active = json.loads((config.strategy_dir() / "active.json").read_text())
        strat = next(s for s in active.get("strategies", []) if s.get("id") == strategy_id)
    except Exception as e:
        out["reasons"].append(f"strategy not found in active.json: {e.__class__.__name__}")
        return out
    try:
        from trader.core import paper
        eq = paper.equity(strategy_id)
    except Exception as e:
        out["reasons"].append(f"no paper equity: {e.__class__.__name__}: {e}")
        return out
    eq = pd.Series(eq, dtype=float).dropna()
    eq.index = pd.DatetimeIndex([_bar(t) for t in eq.index])
    eq = eq.sort_index()
    try:
        flows = sleeve_contributions(strategy_id)
    except Exception as e:
        out["reasons"].append(f"paper ledger unreadable: {e.__class__.__name__}: {e}")
        return out
    if not flows:
        out["reasons"].append("no sleeve capital records in the paper ledger (cannot score flow-adjusted)")
        return out
    # time-weighted over the whole history (so the first in-window bar is measured against the previous mark)
    rets_all = flow_adjusted_returns(eq, flows)
    act = pd.Timestamp(strat.get("activated_at") or cfg["forward"].get("forward_start") or "1970-01-01")
    fs = cfg["forward"].get("forward_start")
    # activated_at is the first CLOSE date the strategy trades; its first mark is the bar opened the day before
    # (bar open time, paper.py: close_date = asof_bar + 1 day). forward_start is already a bar open time.
    start = act - pd.Timedelta(days=1)
    if fs:
        act = max(act, pd.Timestamp(fs))
        start = max(start, pd.Timestamp(fs))
    keep = eq.index >= start
    if asof is not None:
        keep &= eq.index <= _bar(asof)
    eq = eq[keep]
    if len(eq) == 0:
        out["reasons"].append("no forward marks yet")
        return out
    # Score from capital actually put into the sleeves (paper ledger), NOT active.json allocation_usdt: an
    # allocation edit must never inflate/deflate net_return, the random-baseline input or the buy&hold test.
    rets = rets_all[keep]
    out["forward_days"] = int(len(eq))
    out["net_return"] = float(np.prod(1 + rets.values) - 1)
    out["contributed_usdt"] = float(sum(v for t, v in flows.items() if t <= eq.index.max()))
    out["flows_usdt"] = float(sum(v for t, v in flows.items() if eq.index.min() <= t <= eq.index.max()))
    coins = list(strat.get("coins", []))
    try:
        from trader.core import vault
        data = {c: vault.load(c, "1d", "paper", asof=asof, caller="paper") for c in coins}
        bhs = []
        for c in coins:
            cl = data[c]["close"]
            cl = cl[(cl.index >= eq.index.min()) & (cl.index <= eq.index.max())]  # buy at the first mark close
            if len(cl) >= 2:
                bhs.append(float(cl.iloc[-1] / cl.iloc[0]) * (1 - _spot_side()) - 1)
        out["bh_return"] = float(np.mean(bhs)) if bhs else None
        # same anchoring as strategy/tools/report.score_strategy: B&H buys at the first mark's close (above); the
        # random configs are compared on the bars AFTER the first mark (rets[1:]): rets[0] is the entry bar of the
        # strategy (entry costs only, no market move yet), so it must not widen the random window by one bar
        out["random_percentile"] = random_percentile(rets.iloc[1:], coins, period="forward", view="paper",
                                                     n_random=int(P["random_baseline_n"]), caller="forward",
                                                     data=data, activated_at=str(act.date()))
    except Exception as e:
        out["reasons"].append(f"baseline unavailable: {e.__class__.__name__}: {e}")
    R = out["reasons"]
    if out["forward_days"] < int(P["min_forward_days"]):
        R.append(f"forward_days {out['forward_days']} < {P['min_forward_days']}")
    if P.get("require_net_positive") and not out["net_return"] > 0:
        R.append(f"net_return {out['net_return']:.4f} <= 0")
    if P.get("require_beats_random_baseline"):
        rp = out["random_percentile"]
        if rp is None or rp < float(P["random_percentile_min"]):
            R.append(f"random_percentile {rp} < {P['random_percentile_min']}")
    if P.get("require_beats_buy_and_hold"):
        if out["bh_return"] is None or not out["net_return"] > out["bh_return"]:
            R.append(f"net_return {out['net_return']} does not beat buy&hold {out['bh_return']}")
    out["eligible"] = not R
    return out


# ----------------------------------------------------------------------------------------------- forecasts
def brier(p: Sequence[float], outcome: int) -> float:
    """Multi-class Brier: sum_i (p_i - 1[i == outcome])^2."""
    return float(sum((float(pi) - (1.0 if i == int(outcome) else 0.0)) ** 2 for i, pi in enumerate(p)))
