"""Backtest engine (Wave 1, PROTECTED core). See trader/CONTRACTS.md section 4.

Faithful port of research/kronos_lab/overnight/worker.py `evaluate` (positions, ADX gate, ATR trailing stop,
vol-target, per-side turnover costs, PERP funding) and indicators/engine.py `bar_returns`.

No lookahead: the target position at bar t uses only rows <= t, it is held during bar t+1 and the cost of a
change decided at t is charged in bar t+1 (identical to worker.py and engine.py).

`evaluate_df` is pure (no vault, no logging). `evaluate` loads through `core.vault` and logs every evaluation
to the trials ledger through `core.trials` (both imported lazily so this module also runs on sweep hosts that
only ship catalog/backtest/config). Python >= 3.9; numpy, pandas, numba only.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from trader.core import catalog as C
from trader.core import config as cfgmod

BPY = C.BPY
CALLERS = ("research", "forward", "deep_retro", "sweep", "screener", "test", "operator")
PERIODS = ("train", "val", "trainval", "test", "forward", "all")
MIN_BARS = 30
# engine.py COSTS (legacy research costs; SPOT side 0.0015, PERP side 0.00105)
LEGACY_COSTS = {'SPOT': dict(side=0.0010 + 0.0005, funding=False, lo=0.0, cap=1.0),
                'PERP': dict(side=0.00055 + 0.0005, funding=True, lo=-1.0, cap=2.0)}


class InsufficientData(Exception):
    """The loaded frame is too short (or lacks columns) to run a backtest."""


# ── costs ────────────────────────────────────────────────────────────────────
def cost_side(mode, costs=None) -> float:
    """Per-side cost on turnover.

    costs=None -> protected costs[mode] fee_side + slippage_side (SPOT 0.0030);
    "legacy" -> protected legacy_research (SPOT 0.0015 / PERP 0.00105); a number -> as given;
    a dict {"SPOT": x, "PERP": y} -> dict[mode]."""
    mode = str(mode).upper()
    if mode not in C.MODES:
        raise ValueError("unknown mode %r" % mode)
    if costs is None:
        c = cfgmod.load_config()["costs"][mode.lower()]
        return float(c["fee_side"]) + float(c["slippage_side"])
    if isinstance(costs, str):
        if costs != "legacy":
            raise ValueError("costs must be None, 'legacy', a number or a {mode: number} dict")
        leg = cfgmod.load_config()["costs"]["legacy_research"]
        return float(leg["spot_side"] if mode == "SPOT" else leg["perp_side"])
    if isinstance(costs, dict):
        return float(costs[mode])
    v = float(costs)
    if not (v >= 0 and math.isfinite(v)):
        raise ValueError("cost_side must be a finite number >= 0")
    return v


# ── periods ──────────────────────────────────────────────────────────────────
def _local_period_mask(index, period, activated_at=None):
    """CONTRACTS §2 period table (same rule as vault.period_mask; used only if core.vault is absent)."""
    cfg = cfgmod.load_config()
    v = cfg["vault"]
    idx = pd.DatetimeIndex(index)
    if len(idx) == 0:
        return np.zeros(0, dtype=bool)
    start = idx[0] + pd.Timedelta(days=int(v["warmup_days"]))
    cutoff, train_end = pd.Timestamp(v["vault_cutoff"]), pd.Timestamp(v["train_end"])
    fs = cfg.get("forward", {}).get("forward_start")
    if period == "train":
        m = (idx >= start) & (idx < train_end)
    elif period == "val":
        m = (idx >= train_end) & (idx < cutoff)
    elif period == "trainval":
        m = (idx >= start) & (idx < cutoff)
    elif period == "test":
        m = idx >= cutoff
        if fs:
            m &= idx < pd.Timestamp(fs)
    elif period == "forward":
        if not fs:
            return np.zeros(len(idx), dtype=bool)
        lo = pd.Timestamp(fs)
        if activated_at:
            a = pd.Timestamp(activated_at)
            a = a.tz_convert(None) if a.tzinfo is not None else a
            lo = max(lo, a)
        m = idx >= lo
    elif period == "all":
        m = idx >= start
    else:
        raise ValueError("unknown period %r (expected one of %s)" % (period, list(PERIODS)))
    return np.asarray(m, dtype=bool)


def period_mask(index, period, activated_at=None) -> np.ndarray:
    """vault.period_mask when core.vault exists, else the identical local rule."""
    if period not in PERIODS:
        raise ValueError("unknown period %r (expected one of %s)" % (period, list(PERIODS)))
    try:
        from trader.core import vault  # noqa: WPS433 (lazy: sweep hosts may not ship vault)
        fn = getattr(vault, "period_mask", None)
    except ImportError:
        fn = None
    if fn is None:
        return _local_period_mask(index, period, activated_at)
    return np.asarray(fn(index, period, activated_at=activated_at), dtype=bool)


# ── metrics (score.metrics when available) ───────────────────────────────────
def _local_metrics(returns, bpy, positions=None):
    r = np.asarray(returns, dtype=np.float64)
    n = len(r)
    if n < MIN_BARS:
        return None
    g = float(np.prod(1 + r))
    cagr = g ** (bpy / n) - 1 if g > 0 else -1.0
    sd = r.std()
    sr_bar = float(r.mean() / sd) if sd > 0 else 0.0
    eq = np.cumprod(1 + r)
    mdd = float((eq / np.maximum.accumulate(eq) - 1).min())
    s = pd.Series(r)
    skew = float(s.skew()) if n > 2 else 0.0
    kurt = float(s.kurt()) + 3.0 if n > 3 else 3.0
    out = {"cagr": float(cagr), "total": g - 1.0, "sharpe": sr_bar * math.sqrt(bpy), "maxdd": mdd, "n_obs": n,
           "skew": 0.0 if math.isnan(skew) else skew, "kurt": 3.0 if math.isnan(kurt) else kurt,
           "sr_bar": sr_bar, "bpy": int(bpy), "exposure": None, "entries": None}
    if positions is not None:
        w = np.asarray(positions, dtype=np.float64)
        out["exposure"] = float(np.abs(w).mean())
        out["entries"] = int(((w[1:] != 0) & (np.sign(w[1:]) != np.sign(w[:-1]))).sum())
    return out


def _metrics(returns, bpy, positions=None):
    if len(returns) < MIN_BARS:
        return None
    try:
        from trader.core import score
        fn = getattr(score, "metrics", None)
    except ImportError:
        fn = None
    if fn is None:
        return _local_metrics(returns, bpy, positions)
    return fn(returns, bpy, positions)


def _bh_metrics(close, bpy, entry_cost):
    try:
        from trader.core import score
        fn = getattr(score, "buy_and_hold", None)
    except ImportError:
        fn = None
    if fn is not None:
        return fn(close, bpy, entry_cost)
    r = close.pct_change().iloc[1:].copy()
    if len(r) < MIN_BARS:
        return None
    r.iloc[0] -= entry_cost
    return _local_metrics(r.values, bpy, np.ones(len(r)))


# ── core engine (worker.py evaluate, one coin) ───────────────────────────────
def _comp_state(d, comp):
    cache = d.setdefault("_cache", {})
    ck = ("comp", comp['t'], tuple(sorted(comp['p'].items())))
    if ck not in cache:
        cache[ck] = C.CATALOG[comp['t']][0](d, **comp['p']).astype(np.float64)
    return cache[ck]


def _aux(d, name, bpy=None):
    cache = d.setdefault("_cache", {})
    ck = ("aux", name, bpy)
    if ck not in cache:
        if name == 'adx':
            cache[ck] = C.adx(d)
        elif name == 'atr':
            cache[ck] = C.atr(d)
        elif name == 'vol':
            r = np.diff(d['c'], prepend=np.nan) / np.r_[np.nan, d['c'][:-1]]
            cache[ck] = pd.Series(r).rolling(30).std().values * math.sqrt(bpy)
    return cache[ck]


def target_positions(spec, d) -> np.ndarray:
    """Position decided at the CLOSE of each bar from data <= that bar (before the 1-bar shift)."""
    S = np.vstack([_comp_state(d, c) for c in spec['comps']])
    bull, bear = (S > 0).sum(0), (S < 0).sum(0)
    pos = np.where(bull >= spec['k'], 1.0, np.where(bear >= spec['k'], -1.0, 0.0))
    if spec['mode'] == 'SPOT':
        pos = np.clip(pos, 0, None)
    if spec['adx']:
        pos = pos * (np.nan_to_num(_aux(d, 'adx')) > spec['adx'])
    if spec['stop']:
        pos = C.nb_trailing(pos, d['c'], _aux(d, 'atr'), float(spec['stop']))
    if spec['vt']:
        cap = C.VT_CAP[spec['mode']]
        with np.errstate(divide='ignore', invalid='ignore'):
            pos = pos * np.nan_to_num(np.minimum(cap, spec['vt'] / _aux(d, 'vol', BPY[spec['tf']])))
    return pos


def bar_returns(pos, d, mode, cost_side):
    """(r, wx): wx = pos shifted one bar (held during bar t); r = wx*ret - turnover(t-1)*cost_side; PERP - wx*fund."""
    c = d['c']
    ret = np.r_[0.0, c[1:] / c[:-1] - 1]
    wx = np.r_[0.0, pos[:-1]]
    turn = np.abs(np.diff(pos, prepend=0.0))
    r = wx * ret - np.r_[0.0, turn[:-1]] * cost_side
    if mode == 'PERP':
        r = r - wx * d['fund']
    return r, wx


def _check_frame(df):
    need = ("open", "high", "low", "close", "volume")
    missing = [k for k in need if k not in df.columns]
    if missing:
        raise InsufficientData("frame lacks columns %s" % missing)
    if len(df) < 2:
        raise InsufficientData("frame has %d rows" % len(df))


def _check_spec(spec):
    errs = C.validate_spec(spec)
    if errs:
        raise ValueError("invalid spec: " + "; ".join(errs))


def evaluate_df(spec, df, period="all", *, cost_side, activated_at=None, d=None) -> dict:
    """PURE core: no vault access, no logging. df = vault.load-shaped frame (index = bar open time).

    Indicators are computed on the whole frame and then masked to `period`. `d` (from catalog.make_d(df)) may be
    passed to reuse cached component states across specs on the same frame."""
    _check_spec(spec)
    _check_frame(df)
    if d is None:
        d = C.make_d(df)
    tf, mode = spec['tf'], spec['mode']
    bpy = BPY[tf]
    pos = target_positions(spec, d)
    r, wx = bar_returns(pos, d, mode, cost_side)
    mask = period_mask(df.index, period, activated_at)
    idx = df.index[mask]
    returns = pd.Series(r[mask], index=idx, name="returns")
    positions = pd.Series(wx[mask], index=idx, name="positions")
    target = pd.Series(pos[mask], index=idx, name="target")
    metrics = _metrics(returns, bpy, positions) if len(returns) >= MIN_BARS else None
    bh = None
    if len(idx):
        first = int(np.argmax(mask))
        sel = mask.copy()
        if first > 0:
            sel[first - 1] = True  # hold from the close before the period: same bars as the strategy returns
        close = df['close'][sel]
        if len(close) > MIN_BARS:
            bh = _bh_metrics(close, bpy, cost_side)
    attrs = getattr(df, "attrs", {}) or {}
    data_end = attrs.get("data_end")
    if data_end is None and len(df):
        data_end = cfgmod.iso(pd.Timestamp(df.index[-1]))
    return {"spec_hash": C.spec_hash(spec), "spec": C.core_spec(spec), "coin": attrs.get("coin"), "tf": tf,
            "mode": mode, "view": attrs.get("view"), "period": period, "cost_side": float(cost_side),
            "returns": returns, "positions": positions, "target": target, "metrics": metrics,
            "bh_metrics": bh, "trial_id": None, "data_end": data_end}


def _seg_stats(r, bpy):
    """worker.py seg_stats: (cagr, annualised Sharpe with population std) or None if < 30 bars."""
    n = len(r)
    if n < 30:
        return None
    g = np.prod(1 + r)
    cagr = g ** (bpy / n) - 1 if g > 0 else -1.0
    sd = r.std()
    return cagr, (r.mean() / sd * math.sqrt(bpy)) if sd > 0 else 0.0


def worker_metrics(spec, df, *, cost_side, d=None) -> Optional[dict]:
    """One coin of worker.py `evaluate` on a research-view frame: {cagr_tr, sh_tr, cagr_va, sh_va, sh_tv, mdd,
    entries, expo} (no `coin` key) or None when train/val are too short (worker rule). Used by sweep_runner."""
    _check_spec(spec)
    _check_frame(df)
    if d is None:
        d = C.make_d(df)
    bpy = BPY[spec['tf']]
    pos = target_positions(spec, d)
    r, wx = bar_returns(pos, d, spec['mode'], cost_side)
    m_tr = period_mask(df.index, "train")
    m_va = period_mask(df.index, "val")
    m_ev = period_mask(df.index, "trainval")
    rt, rv, re = r[m_tr], r[m_va], r[m_ev]
    st, sv, se = _seg_stats(rt, bpy), _seg_stats(rv, bpy), _seg_stats(re, bpy)
    if st is None or sv is None or len(rt) < 180 * bpy / 365:
        return None
    eq = np.cumprod(1 + re)
    mdd = float((eq / np.maximum.accumulate(eq) - 1).min())
    w = wx[m_ev]
    entries = int(((w[1:] != 0) & (np.sign(w[1:]) != np.sign(w[:-1]))).sum())
    return dict(cagr_tr=st[0], sh_tr=st[1], cagr_va=sv[0], sh_va=sv[1], sh_tv=se[1],
                mdd=mdd, entries=entries, expo=float(np.abs(w).mean()))


def is_candidate(m) -> bool:
    """worker.py is_candidate (default sweep criterion)."""
    return (m['cagr_tr'] > 0.20 and m['cagr_va'] > 0.20 and m['sh_tr'] > 0.8 and m['sh_va'] > 0.8
            and m['entries'] >= 8 and m['mdd'] > -0.45)


# ── public, logged entry points ──────────────────────────────────────────────
def evaluate(spec, coin, view="research", period="trainval", *, costs=None, look_id=None, caller,
             hypothesis_id=None, activated_at=None, log=True) -> dict:
    """Load through core.vault, run evaluate_df, log one `single` row to trials.jsonl (log=True)."""
    if caller not in CALLERS:
        raise ValueError("caller must be one of %s" % (CALLERS,))
    # CONTRACTS §4: only tests / sweep_runner may skip the trials ledger, and never on the sealed
    # 'full' view (an unlogged sealed look would vanish from DSR N and leave no audit trail).
    if not log and (caller not in ("test", "sweep") or view == "full"):
        raise ValueError("log=False is only allowed for caller 'test'/'sweep' and never with view='full'")
    if view not in ("research", "full", "paper"):
        raise ValueError("view must be 'research', 'full' or 'paper'")
    if period not in PERIODS:
        raise ValueError("unknown period %r" % period)
    if period == "test" and view != "full":
        raise ValueError("period 'test' requires view='full' and a valid look_id")
    if period == "forward" and view == "research":
        raise ValueError("period 'forward' is not visible in the research view")
    _check_spec(spec)
    from trader.core import vault
    sh = C.spec_hash(spec)
    if view == "full":
        df = vault.load(coin, spec['tf'], "full", look_id=look_id, spec_hash=sh)
    else:
        df = vault.load(coin, spec['tf'], view)
    res = evaluate_df(spec, df, period, cost_side=cost_side(spec['mode'], costs), activated_at=activated_at)
    res["coin"], res["view"] = coin, view
    res["look_id"] = look_id
    if log:
        from trader.core import trials
        res["trial_id"] = trials.log_trial(res, caller=caller, hypothesis_id=hypothesis_id, kind="single",
                                           look_id=look_id)
    return res


def evaluate_many(spec, coins, view="research", period="trainval", **kw) -> List[dict]:
    """One `evaluate` per coin (each logged)."""
    return [evaluate(spec, c, view, period, **kw) for c in coins]


# ── engine.py legacy path ────────────────────────────────────────────────────
def legacy_bar_returns(df, fund, sig, cost, tf, adx_thr, vt, adx_s):
    """Verbatim engine.bar_returns (vol target fixed at 0.5). `cost` = LEGACY_COSTS[mode]-shaped dict
    (or a mode name "SPOT"/"PERP"). Returns (r, wx) as pandas Series."""
    if isinstance(cost, str):
        cost = LEGACY_COSTS[cost]
    pos = sig.clip(lower=cost['lo'], upper=1.0)
    if adx_thr:
        pos = pos * (adx_s > adx_thr)
    ret = df.close.pct_change().fillna(0.0)
    if vt:
        bpy = 365 * (6 if tf == '4h' else 1)
        vol = ret.rolling(30).std() * math.sqrt(bpy)
        pos = pos * (0.5 / vol).clip(upper=cost['cap']).fillna(0.0)
    wx = pos.shift(1).fillna(0.0)
    turn = (pos - pos.shift(1).fillna(0.0)).abs()
    r = wx * ret - (turn * cost['side']).shift(1).fillna(0.0)
    if cost['funding']:
        r = r - wx * fund
    return r, wx
