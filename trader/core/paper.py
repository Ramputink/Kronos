"""Paper-trading engine (Wave 1, PROTECTED core). Interface: trader/CONTRACTS.md section 6.

One daily run per closed 1d bar (``asof_bar`` = bar open time). For every active strategy and coin there is a
*sleeve* (strategy, coin) that starts with ``allocation_usdt / len(coins)`` cash and is never rebalanced against
other sleeves (same accounting as ``score.portfolio`` / research portfolio_test.py).

At the close of ``asof_bar`` each sleeve:
  1. is marked to market at that close (qty and cash carried over from its previous ``mark`` row);
  2. gets its target weight = ``backtest.target_positions(spec, catalog.make_d(df))[-1]`` where ``df`` is
     ``vault.load(coin, "1d", view="paper", asof=asof_bar)`` -> the very same code the backtest uses, on bars
     ``<= asof_bar`` only (no lookahead);
  3. trades the difference between target value (weight x equity) and current value at the close +/- slippage,
     paying the protected spot fee on the notional. Orders under ``paper.min_order_usdt`` are skipped and logged;
  4. writes a ``mark`` row with the post-trade state (the state of record for the next run).

Ledger state lives only in ``journal/paper-ledger.jsonl`` (append-only, sole writer: this module). All rows of a
run are computed in memory first and appended at the end, so a run either writes everything or (crash) almost
nothing; a bar that has fills/marks but no ``run`` row is reported as an error instead of being traded twice.

Honesty seal: paper only runs on the forward period. ``run_paper``/``settle`` (dry runs included) refuse while
``forward.forward_start`` is null and for any ``asof < forward_start``; sleeves with ``activated_at <
forward_start`` (or ``<= vault_cutoff`` before paper starts) are invalid; market data is read through
``vault.load(view="paper", caller="forward")``, i.e. only ``[forward_start - warmup_days, asof]``.

Capital flows (review fix: allocation edits used to leave sleeve capital unchanged, so every scorer that divides
by the current ``allocation_usdt`` reported phantom P&L). Each sleeve tracks its *contributed capital*
``contributed_usdt`` (on every mark row). On each run, before trading, a live sleeve (status paper/paused, started)
gets a flow ``allocation_usdt / len(coins) - contributed``: a deposit adds cash; a withdrawal takes cash and, when
cash is short, first sells coin at the close (fill reason ``withdraw``, costs paid, never below the min order).
A new sleeve (init, or a coin added to a live strategy) starts at 0 and is funded by a deposit flow. A coin removed
from a live strategy is sold (reason ``retire``) and its whole cash is withdrawn. Flows are recorded on the mark row
(``flow_usdt``) so ``Σ contributed == allocation_usdt`` for a live strategy whenever the sleeves could be funded,
and ``returns()`` / ``net_return()`` give flow-adjusted (time-weighted) returns. A fully retired strategy keeps its
cash (no withdrawal), as before.

Status semantics (``active.json``):
  * ``paper``   - trades normally (from the first close whose date is >= ``activated_at``).
  * ``paused``  - risk-reducing only: weights may go down (exits), never up. Positions are still marked.
  * ``retired`` - every sleeve is sold to cash on the next run (reason ``retire``), then no longer traded/marked.
  * ``screener_veto: true`` blocks weight increases (skip reason ``veto``) only while
    ``forecast.veto_allowed()`` is true; otherwise the flag is ignored and a note is written on the run row.

Differences from the backtest, documented on purpose: (a) the paper sleeve is rebalanced to the exact target
weight at each close, so it also pays cost on drift (the backtest charges only |delta target|); (b) orders below
the minimum are skipped; (c) the cost of a trade at close t is booked at t (the backtest books it in bar t+1).
With zero costs and no minimum order, paper equity equals the backtest equity exactly (tests/test_paper.py).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

if __package__ in (None, ""):  # allow `python trader/core/paper.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from trader.core import config

LEDGER_NAME = "paper-ledger.jsonl"
ID_RE = re.compile(r"^[a-z0-9_]{3,40}$")
STATUSES = ("paper", "paused", "retired")
EPS_QTY = 1e-12


class PaperError(Exception):
    pass


# ── helpers ──────────────────────────────────────────────────────────────────
def ledger_path() -> Path:
    return config.journal_dir() / LEDGER_NAME


def _active_path(active_path=None) -> Path:
    return Path(active_path) if active_path else config.strategy_dir() / "active.json"


def _ts(x) -> pd.Timestamp:
    """Naive-UTC pandas Timestamp from 'YYYY-MM-DD', ISO with Z, datetime or Timestamp."""
    t = pd.Timestamp(x)
    if t.tzinfo is not None:
        t = t.tz_convert("UTC").tz_localize(None)
    return t


def _bar_iso(t: pd.Timestamp) -> str:
    return config.iso(_ts(t))


def last_closed_bar(now=None) -> pd.Timestamp:
    """Open time of the last fully closed 1d bar (yesterday 00:00 UTC)."""
    now = _ts(now) if now is not None else _ts(config.utcnow())
    return now.floor("D") - pd.Timedelta(days=1)


def _forward_start(cfg: Optional[dict] = None) -> Optional[pd.Timestamp]:
    """protected forward.forward_start as a naive-UTC day, or None while the operator has not started paper."""
    cfg = cfg if cfg is not None else config.load_config()
    fs = (cfg.get("forward") or {}).get("forward_start")
    return _ts(fs).normalize() if fs else None


def _costs(cfg: dict) -> tuple:
    s = cfg["costs"]["spot"]
    return float(s["fee_side"]), float(s["slippage_side"])


def _read_ledger() -> list:
    return config.jsonl_read(ledger_path())


def _sleeve_state(rows: list) -> dict:
    """Latest mark per sleeve -> {sleeve_id: mark row}."""
    out = {}
    for r in rows:
        if r.get("type") == "mark":
            out[r["sleeve_id"]] = r
    return out


def _legacy_contributed(rows: list, sleeve_id: str) -> float:
    """Contributed capital of a sleeve written before flows were recorded: its init cash, rebuilt from the first
    mark (post-trade cash) plus the cost of the buy fills of that same run."""
    first = next((r for r in rows if r.get("type") == "mark" and r.get("sleeve_id") == sleeve_id), None)
    if first is None:
        return 0.0
    spent = sum(float(r["notional_usdt"]) + float(r.get("fee_usdt") or 0.0) for r in rows
                if r.get("type") == "fill" and r.get("sleeve_id") == sleeve_id and r.get("run_id") == first["run_id"]
                and r.get("side") == "buy")
    return float(first["cash_usdt"]) + spent


def _contributed(mark: dict, rows: list) -> float:
    if mark.get("contributed_usdt") is not None:
        return float(mark["contributed_usdt"])
    return _legacy_contributed(rows, mark["sleeve_id"])


def _closed_sleeves(rows: list) -> set:
    """Sleeves closed by a retire (final mark with retired flag)."""
    return {r["sleeve_id"] for r in rows if r.get("type") == "mark" and r.get("closed")}


# ── validation ───────────────────────────────────────────────────────────────
def validate_active(active: dict) -> list:
    """[] if active.json content is valid, else human-readable problems (CONTRACTS.md section 6)."""
    errs = []
    cfg = config.load_config()
    pc = cfg["paper"]
    universe = set(cfg["universe"])
    fs = _forward_start(cfg)
    cutoff = _ts(cfg["vault"]["vault_cutoff"]).normalize()
    if not isinstance(active, dict):
        return ["active.json must be an object"]
    strategies = active.get("strategies")
    if not isinstance(strategies, list):
        return ["active.json: 'strategies' must be a list"]
    seen = set()
    n_paper = 0
    total_alloc = 0.0
    try:  # validate specs with the backtest catalog when it is available
        from trader.core import catalog as _catalog
        _validate_spec = getattr(_catalog, "validate_spec", None)
    except Exception:  # pragma: no cover - catalog area not built yet
        _validate_spec = None
    for i, s in enumerate(strategies):
        tag = "strategies[%d]" % i
        if not isinstance(s, dict):
            errs.append("%s: not an object" % tag)
            continue
        sid = s.get("id")
        if not isinstance(sid, str) or not ID_RE.match(sid):
            errs.append("%s: id %r must match %s" % (tag, sid, ID_RE.pattern))
        elif sid in seen:
            errs.append("%s: duplicate id %r" % (tag, sid))
        else:
            seen.add(sid)
            tag = "strategy %s" % sid
        status = s.get("status")
        if status not in STATUSES:
            errs.append("%s: status %r not in %s" % (tag, status, STATUSES))
        spec = s.get("spec")
        if not isinstance(spec, dict):
            errs.append("%s: spec missing" % tag)
        else:
            if spec.get("mode") not in pc["allowed_modes"]:
                errs.append("%s: mode %r not allowed in paper (%s)" % (tag, spec.get("mode"), pc["allowed_modes"]))
            if spec.get("tf") not in pc["allowed_timeframes"]:
                errs.append("%s: tf %r not allowed in paper (%s)" % (tag, spec.get("tf"), pc["allowed_timeframes"]))
            if _validate_spec is not None:
                try:
                    errs.extend("%s: spec: %s" % (tag, p) for p in _validate_spec(spec))
                except Exception as e:  # noqa: BLE001
                    errs.append("%s: spec invalid (%s)" % (tag, e))
        coins = s.get("coins")
        if not isinstance(coins, list) or not coins:
            errs.append("%s: coins must be a non-empty list" % tag)
        else:
            if len(set(coins)) != len(coins):
                errs.append("%s: duplicate coins" % tag)
            bad = [c for c in coins if c not in universe]
            if bad:
                errs.append("%s: coins not in universe: %s" % (tag, bad))
            if len(coins) > pc["max_coins_per_strategy"]:
                errs.append("%s: %d coins > max %d" % (tag, len(coins), pc["max_coins_per_strategy"]))
        alloc = s.get("allocation_usdt")
        if not isinstance(alloc, (int, float)) or isinstance(alloc, bool) or alloc <= 0:
            errs.append("%s: allocation_usdt must be a positive number" % tag)
            alloc = 0.0
        elif alloc > pc["max_allocation_per_strategy_usdt"]:
            errs.append("%s: allocation %.2f > per-strategy cap %.2f"
                        % (tag, alloc, pc["max_allocation_per_strategy_usdt"]))
        elif isinstance(coins, list) and coins and alloc / len(coins) < pc["min_order_usdt"]:
            errs.append("%s: sleeve cash %.2f < min order %.2f" % (tag, alloc / len(coins), pc["min_order_usdt"]))
        act = s.get("activated_at")
        try:
            if not isinstance(act, str) or not re.match(r"^\d{4}-\d\d-\d\d$", act):
                raise ValueError
            pd.Timestamp(act)
        except Exception:  # noqa: BLE001
            errs.append("%s: activated_at %r must be YYYY-MM-DD" % (tag, act))
        else:
            # honesty seal: a sleeve can never start inside the sealed TEST window [vault_cutoff, forward_start)
            if fs is not None and pd.Timestamp(act) < fs:
                errs.append("%s: activated_at %s < forward_start %s (sealed TEST window)" % (tag, act, fs.date()))
            elif fs is None and pd.Timestamp(act) <= cutoff:
                errs.append("%s: activated_at %s <= vault_cutoff %s (sealed TEST window)" % (tag, act, cutoff.date()))
        if "screener_veto" in s and not isinstance(s["screener_veto"], bool):
            errs.append("%s: screener_veto must be a bool" % tag)
        if status != "retired":
            total_alloc += float(alloc or 0.0)
        if status == "paper":
            n_paper += 1
    if n_paper > pc["max_strategies"]:
        errs.append("%d strategies in paper > max %d" % (n_paper, pc["max_strategies"]))
    if total_alloc > pc["bankroll_usdt"] + 1e-9:
        errs.append("total allocation %.2f > bankroll %.2f" % (total_alloc, pc["bankroll_usdt"]))
    return errs


def _load_active(active_path=None) -> dict:
    p = _active_path(active_path)
    if not p.exists():
        raise PaperError("active.json not found: %s" % p)
    with open(p) as fh:
        active = json.load(fh)
    errs = validate_active(active)
    if errs:
        raise PaperError("active.json invalid: " + "; ".join(errs))
    return active


# ── market data & signals (same code path as the backtest) ───────────────────
def _load_paper_df(coin: str, asof: pd.Timestamp) -> pd.DataFrame:
    from trader.core import vault
    return vault.load(coin, "1d", view="paper", asof=asof, caller="forward")


def _target_weight(spec: dict, df: pd.DataFrame) -> float:
    from trader.core import backtest, catalog
    pos = backtest.target_positions(spec, catalog.make_d(df))
    w = float(np.nan_to_num(pos[-1]))
    return min(max(w, 0.0), 1.0)  # paper is SPOT: long-only, never levered


def _veto_allowed() -> tuple:
    try:
        from trader.core import forecast
    except Exception:  # noqa: BLE001 - screener area not built: veto never allowed
        return False, "forecast module unavailable"
    try:
        return bool(forecast.veto_allowed()), ""
    except Exception as e:  # noqa: BLE001
        return False, "veto_allowed failed: %s" % e


# ── trading core (pure, used by run_paper) ───────────────────────────────────
def _trade(state: dict, price: float, target_w: float, *, fee: float, slip: float, min_order: float,
           allow_up: bool, allow_down: bool) -> tuple:
    """Rebalance one sleeve at `price` towards weight `target_w`.

    state: {"qty","cash"} (mutated). Returns (fill dict or None, skip dict or None, prev_weight).
    """
    qty, cash = state["qty"], state["cash"]
    value = qty * price
    eq = cash + value
    prev_w = value / eq if eq > 0 else 0.0
    if eq <= 0:
        return None, {"reason": "no_equity", "notional_usdt": 0.0}, prev_w
    tgt_val = target_w * eq
    delta = tgt_val - value
    if delta > 0 and not allow_up:
        return None, ({"reason": "veto", "notional_usdt": round(delta, 6)} if abs(delta) > 1e-9 else None), prev_w
    if delta < 0 and not allow_down:
        return None, None, prev_w
    if abs(delta) <= 1e-9:
        return None, None, prev_w
    if delta > 0:  # buy: spend `notional` USDT at close*(1+slip), plus fee on the notional
        fill_price = price * (1.0 + slip)
        notional = min(delta, cash / (1.0 + fee))
        if notional < min_order or notional <= 0:
            return None, {"reason": "below_min_order", "notional_usdt": round(max(notional, delta), 6)}, prev_w
        q = notional / fill_price
        f = notional * fee
        state["qty"] = qty + q
        state["cash"] = cash - notional - f
        side = "buy"
    else:  # sell: reduce value at mark by |delta|; proceeds at close*(1-slip), minus fee
        fill_price = price * (1.0 - slip)
        q = qty if target_w <= 0 else min(qty, -delta / price)
        if 0 < q < qty and (qty - q) * price < min_order:
            q = qty  # mirror real.py: never strand a residual below the min order (it could never be sold)
        notional = q * fill_price
        if notional < min_order or q <= 0:
            return None, {"reason": "below_min_order", "notional_usdt": round(notional, 6)}, prev_w
        f = notional * fee
        state["qty"] = 0.0 if (target_w <= 0 or q >= qty) else qty - q
        state["cash"] = cash + notional - f
        side = "sell"
    fill = {"side": side, "qty": q, "price": price, "fill_price": fill_price, "notional_usdt": notional,
            "fee_usdt": f, "slippage_bps": round(slip * 1e4, 4)}
    return fill, None, prev_w


def _r(x, nd=8):
    return float(round(float(x), nd))


def _fill_row(run_id, now_iso, asof_iso, strategy_id, sleeve_id, coin, fill: dict, reason: str) -> dict:
    return {"type": "fill", "run_id": run_id, "ts": now_iso, "asof_bar": asof_iso, "strategy_id": strategy_id,
            "sleeve_id": sleeve_id, "coin": coin, "side": fill["side"], "qty": _r(fill["qty"], 12),
            "price": fill["price"], "fill_price": _r(fill["fill_price"], 10), "notional_usdt": _r(fill["notional_usdt"]),
            "fee_usdt": _r(fill["fee_usdt"]), "slippage_bps": fill["slippage_bps"],
            "prev_weight": _r(fill["prev_weight"], 6), "target_weight": _r(fill["target_weight"], 6), "reason": reason}


def _withdraw(state: dict, price: float, amount: float, *, fee: float, slip: float, min_order: float,
              can_sell: bool) -> tuple:
    """Take `amount` USDT of capital out of a sleeve (state mutated). Cash first; if cash is short and selling is
    allowed, sell coin at close*(1-slip) paying the fee (at least the min order, at most the whole position).
    Returns (fill or None, skip or None, flow) with flow = -(USDT actually withdrawn); any shortfall is retried
    on the next run because contributed capital only drops by what was really withdrawn."""
    fill = skip = None
    qty, cash = state["qty"], state["cash"]
    short = amount - cash
    if short > 1e-9 and qty > EPS_QTY:
        fill_price = price * (1.0 - slip)
        q = short / (fill_price * (1.0 - fee))
        q = max(q, min_order / fill_price)
        q = min(q, qty)
        if (qty - q) * price < min_order:
            q = qty  # no unsellable dust left behind (same rule as _trade / real.py)
        notional = q * fill_price
        if not can_sell:
            skip = {"reason": "no_data", "notional_usdt": notional}
        elif notional < min_order - 1e-9:
            skip = {"reason": "below_min_order", "notional_usdt": notional}
        else:
            eq0 = cash + qty * price
            f = notional * fee
            new_qty = 0.0 if q >= qty - EPS_QTY else qty - q
            state["qty"] = new_qty
            state["cash"] = cash = cash + notional - f
            eq1 = cash + new_qty * price
            fill = {"side": "sell", "qty": q, "price": price, "fill_price": fill_price, "notional_usdt": notional,
                    "fee_usdt": f, "slippage_bps": round(slip * 1e4, 4),
                    "prev_weight": qty * price / eq0 if eq0 > 0 else 0.0,
                    "target_weight": new_qty * price / eq1 if eq1 > 0 else 0.0}
    take = min(amount, max(state["cash"], 0.0))
    state["cash"] -= take
    return fill, skip, -take


def _plan(active: dict, asof: pd.Timestamp, rows: list, cfg: dict, run_id: str, *, settle_only: bool = False,
          now_iso: Optional[str] = None) -> dict:
    """Compute every ledger row of a run (or a settle) without writing anything."""
    _check_forward(asof, cfg)
    fs = _forward_start(cfg)
    for s in active["strategies"]:
        if pd.Timestamp(s["activated_at"]) < fs:
            raise PaperError("strategy %s: activated_at %s < forward_start %s (sealed TEST window)"
                             % (s.get("id"), s["activated_at"], fs.date()))
    fee, slip = _costs(cfg)
    min_order = float(cfg["paper"]["min_order_usdt"])
    now_iso = now_iso or config.iso()
    asof_iso = _bar_iso(asof)
    close_date = (asof + pd.Timedelta(days=1)).normalize()
    state = _sleeve_state(rows)
    closed = _closed_sleeves(rows)
    veto_ok, veto_note = (False, "")
    notes = []
    if not settle_only and any(s.get("screener_veto") for s in active["strategies"]):
        veto_ok, veto_note = _veto_allowed()
        if not veto_ok:
            notes.append("screener_veto ignored (%s)" % (veto_note or "veto not allowed yet"))
    out_rows, n_fill, n_skip = [], 0, 0
    per_strategy, sleeve_eq = {}, {}
    data_cache = {}

    def get_df(coin):
        if coin not in data_cache:
            try:
                data_cache[coin] = _load_paper_df(coin, asof)
            except Exception as e:  # noqa: BLE001
                data_cache[coin] = e
        return data_cache[coin]

    wanted = {}  # sleeve_id -> (strategy dict, coin, mode) where mode in trade|exit|hold
    for s in active["strategies"]:
        status = s["status"]
        started = close_date >= pd.Timestamp(s["activated_at"])
        for coin in s["coins"]:
            sid = "%s:%s" % (s["id"], coin)
            if sid in closed:
                continue
            if status == "retired":
                if sid in state:
                    wanted[sid] = (s, coin, "retire")
                continue
            if not started:
                continue
            wanted[sid] = (s, coin, "trade" if status == "paper" else "paused")
        if status != "retired":  # coins removed from a live strategy -> close those sleeves
            for sid in state:
                if sid.split(":")[0] == s["id"] and sid.split(":", 1)[1] not in s["coins"] and sid not in closed:
                    wanted[sid] = (s, sid.split(":", 1)[1], "remove")

    for sid, (s, coin, how) in sorted(wanted.items()):
        st = state.get(sid)
        if st is None:
            sleeve = {"qty": 0.0, "cash": 0.0}  # funded by the deposit flow below
            contributed = 0.0
            reason = "init"
        else:
            sleeve = {"qty": float(st["qty"]), "cash": float(st["cash_usdt"])}
            contributed = _contributed(st, rows)
            reason = "signal"
        df = get_df(coin)
        stale = isinstance(df, Exception) or len(df) == 0 or _ts(df.index[-1]) != asof
        if isinstance(df, Exception) or len(df) == 0:
            price = float(st["price"]) if st else None
        else:
            price = float(df["close"].iloc[-1])
        fill = skip = None
        flow = 0.0
        if price is not None and price > 0:
            # ── capital flow: sleeve capital follows allocation_usdt / len(coins) ──
            d = 0.0
            if st is None:
                d = float(s["allocation_usdt"]) / len(s["coins"])  # init deposit (settle marks it too)
            elif not settle_only and how in ("trade", "paused"):
                d = float(s["allocation_usdt"]) / len(s["coins"]) - contributed
            if d > 1e-6:
                sleeve["cash"] += d
                flow = d
            elif d < -1e-6:
                wfill, wskip, flow = _withdraw(sleeve, price, -d, fee=fee, slip=slip, min_order=min_order,
                                               can_sell=not stale)
                if wfill:
                    out_rows.append(_fill_row(run_id, now_iso, asof_iso, s["id"], sid, coin, wfill, "withdraw"))
                    n_fill += 1
                if wskip:
                    out_rows.append({"type": "skip", "run_id": run_id, "ts": now_iso, "asof_bar": asof_iso,
                                     "strategy_id": s["id"], "sleeve_id": sid, "coin": coin,
                                     "reason": wskip["reason"], "notional_usdt": _r(wskip["notional_usdt"], 6)})
                    n_skip += 1
            contributed += flow
        if price is None or price <= 0:
            skip = {"reason": "no_data", "notional_usdt": 0.0}
        elif stale and not settle_only:
            skip = {"reason": "no_data", "notional_usdt": 0.0}
        elif not settle_only:
            if how in ("retire", "remove"):
                tgt, up, down, reason = 0.0, False, True, "retire"
            else:
                tgt = _target_weight(s["spec"], df)
                up = how == "trade" and not (s.get("screener_veto") and veto_ok)
                down = True
            prev_q = sleeve["qty"]
            fill, skip, prev_w = _trade(sleeve, price, tgt, fee=fee, slip=slip, min_order=min_order,
                                        allow_up=up, allow_down=down)
            if fill:
                if reason == "signal" and prev_q > EPS_QTY and tgt > 0:
                    reason = "rebalance"  # resize of an open position (vol-target drift), not a new signal
                out_rows.append(_fill_row(run_id, now_iso, asof_iso, s["id"], sid, coin,
                                          dict(fill, prev_weight=prev_w, target_weight=tgt), reason))
                n_fill += 1
        if skip:
            out_rows.append({"type": "skip", "run_id": run_id, "ts": now_iso, "asof_bar": asof_iso,
                             "strategy_id": s["id"], "sleeve_id": sid, "coin": coin, "reason": skip["reason"],
                             "notional_usdt": _r(skip["notional_usdt"], 6)})
            n_skip += 1
        if price is None or price <= 0:
            continue
        if how == "remove" and not settle_only and sleeve["qty"] <= EPS_QTY and sleeve["cash"] > 0:
            # coin removed from a live strategy: its proceeds leave the strategy (recorded as a withdrawal)
            flow -= sleeve["cash"]
            contributed -= sleeve["cash"]
            sleeve["cash"] = 0.0
        eq = sleeve["cash"] + sleeve["qty"] * price
        mark = {"type": "mark", "run_id": run_id, "ts": now_iso, "asof_bar": asof_iso, "strategy_id": s["id"],
                "sleeve_id": sid, "coin": coin, "price": price, "qty": _r(sleeve["qty"], 12),
                "cash_usdt": _r(sleeve["cash"]), "equity_usdt": _r(eq),
                "weight": _r(sleeve["qty"] * price / eq if eq > 0 else 0.0, 6),
                "contributed_usdt": _r(contributed)}
        if abs(flow) > 1e-9:
            mark["flow_usdt"] = _r(flow)
        if stale:
            mark["stale"] = True
        if how in ("retire", "remove") and not settle_only and sleeve["qty"] <= EPS_QTY:
            mark["closed"] = True
        out_rows.append(mark)
        sleeve_eq[sid] = eq
        per_strategy[s["id"]] = per_strategy.get(s["id"], 0.0) + eq

    # total equity: every sleeve ever opened, at its latest value (closed sleeves keep their final cash)
    total = sum(float(st["equity_usdt"]) for sid, st in state.items() if sid not in sleeve_eq)
    total += sum(sleeve_eq.values())
    return {"rows": out_rows, "fills": n_fill, "skips": n_skip, "equity_usdt": total,
            "per_strategy": {k: _r(v, 6) for k, v in per_strategy.items()}, "notes": "; ".join(notes),
            "n_sleeves": len(sleeve_eq)}


# ── public API ───────────────────────────────────────────────────────────────
def _check_forward(asof: pd.Timestamp, cfg: Optional[dict] = None) -> None:
    """Honesty seal: paper only ever looks at bars of the forward period (bar open time >= forward_start).

    While forward_start is null every bar >= vault_cutoff is sealed TEST data, so paper refuses to run at all;
    otherwise asof must be >= forward_start. Applies to dry runs too (they print prices/weights/equity)."""
    fs = _forward_start(cfg)
    if fs is None:
        raise PaperError("forward_start is null: paper has not been started by the operator, and every bar "
                         ">= vault_cutoff is sealed TEST data (use a registered look instead)")
    if _ts(asof).normalize() < fs:
        raise PaperError("asof %s < forward_start %s: bars before forward_start are sealed TEST data"
                         % (_ts(asof).date(), fs.date()))


def _resolve_asof(asof, cfg: Optional[dict] = None) -> pd.Timestamp:
    last = last_closed_bar()
    t = last if asof is None else _ts(asof).normalize()
    if t > last:
        raise PaperError("asof %s is not a closed 1d bar yet (last closed %s)" % (t.date(), last.date()))
    _check_forward(t, cfg)
    return t


def _run_id(prefix: str, asof: pd.Timestamp) -> str:
    return "%s-%s-%s" % (prefix, config.utcnow().strftime("%Y%m%dT%H%M%S"), asof.strftime("%Y%m%d"))


def run_paper(active_path: Optional[str] = None, *, asof: Optional[str] = None, dry_run: bool = False) -> dict:
    """Trade all sleeves at the close of `asof` (default: last closed 1d bar). Idempotent per asof_bar."""
    cfg = config.load_config()
    asof_t = _resolve_asof(asof, cfg)
    asof_iso = _bar_iso(asof_t)
    rows = _read_ledger()
    run_id = _run_id("P", asof_t)
    done = [r for r in rows if r.get("type") == "run" and r.get("asof_bar") == asof_iso and r.get("status") == "ok"]
    if done:
        res = {"run_id": run_id, "asof_bar": asof_iso, "status": "skipped", "fills": 0, "skips": 0,
               "equity_usdt": done[-1].get("equity_usdt"), "per_strategy": {},
               "notes": "already ran for this bar (%s)" % done[-1]["run_id"]}
        if not dry_run:
            config.jsonl_append(ledger_path(), {"type": "run", "run_id": run_id, "ts": config.iso(),
                                                "asof_bar": asof_iso, "status": "skipped", "n_sleeves": 0,
                                                "equity_usdt": res["equity_usdt"], "notes": res["notes"]})
        return res
    partial = [r for r in rows if r.get("type") == "fill" and r.get("asof_bar") == asof_iso
               and not str(r.get("run_id", "")).startswith("S-")]
    if partial:
        raise PaperError("ledger has fills for %s without an ok run row (interrupted run?): operator must "
                         "inspect %s" % (asof_iso, ledger_path()))
    last_ok = [r for r in rows if r.get("type") == "run" and r.get("status") == "ok"]
    if last_ok and _ts(last_ok[-1]["asof_bar"]) > asof_t:
        raise PaperError("asof %s is before the last run bar %s (paper never trades backwards)"
                         % (asof_iso, last_ok[-1]["asof_bar"]))
    active = _load_active(active_path)
    try:
        plan = _plan(active, asof_t, rows, cfg, run_id)
    except PaperError:
        raise
    except Exception as e:  # noqa: BLE001
        if not dry_run:
            config.jsonl_append(ledger_path(), {"type": "run", "run_id": run_id, "ts": config.iso(),
                                                "asof_bar": asof_iso, "status": "error", "n_sleeves": 0,
                                                "equity_usdt": None, "notes": "%s: %s" % (type(e).__name__, e)})
        raise PaperError("paper run failed: %s" % e) from e
    res = {"run_id": run_id, "asof_bar": asof_iso, "status": "ok", "fills": plan["fills"], "skips": plan["skips"],
           "equity_usdt": _r(plan["equity_usdt"], 6), "per_strategy": plan["per_strategy"], "notes": plan["notes"]}
    if dry_run:
        res["status"] = "dry_run"
        res["rows"] = plan["rows"]
        return res
    path = ledger_path()
    for r in plan["rows"]:
        config.jsonl_append(path, r)
    config.jsonl_append(path, {"type": "run", "run_id": run_id, "ts": config.iso(), "asof_bar": asof_iso,
                               "status": "ok", "n_sleeves": plan["n_sleeves"], "equity_usdt": res["equity_usdt"],
                               "notes": plan["notes"]})
    return res


def settle(asof=None, *, active_path: Optional[str] = None) -> dict:
    """Mark-to-market only (no trading) at the close of `asof`; writes mark rows. Idempotent per bar."""
    cfg = config.load_config()
    asof_t = _resolve_asof(asof, cfg)
    asof_iso = _bar_iso(asof_t)
    rows = _read_ledger()
    already = {r["sleeve_id"] for r in rows if r.get("type") == "mark" and r.get("asof_bar") == asof_iso}
    active = _load_active(active_path)
    run_id = _run_id("S", asof_t)
    plan = _plan(active, asof_t, rows, cfg, run_id, settle_only=True)
    marks = [r for r in plan["rows"] if r["type"] == "mark" and r["sleeve_id"] not in already]
    for r in marks:
        config.jsonl_append(ledger_path(), r)
    return {"run_id": run_id, "asof_bar": asof_iso, "marks": len(marks), "already_marked": len(already),
            "equity_usdt": _r(plan["equity_usdt"], 6), "per_strategy": plan["per_strategy"]}


def equity(strategy_id: Optional[str] = None, *, since=None) -> pd.Series:
    """Daily equity (USDT) from mark rows, indexed by asof_bar (bar open time, naive UTC).

    Sleeves without a mark on a given bar carry their last equity forward (closed sleeves keep their cash).
    """
    marks = [r for r in _read_ledger() if r.get("type") == "mark"
             and (strategy_id is None or r.get("strategy_id") == strategy_id)]
    if not marks:
        return pd.Series(dtype=float, name="equity_usdt")
    df = pd.DataFrame({"t": [_ts(r["asof_bar"]) for r in marks], "sleeve": [r["sleeve_id"] for r in marks],
                       "eq": [float(r["equity_usdt"]) for r in marks]})
    piv = df.groupby(["t", "sleeve"])["eq"].last().unstack("sleeve").sort_index().ffill()
    s = piv.sum(axis=1, min_count=1)
    s.name = "equity_usdt"
    s.index.name = "asof_bar"
    if since is not None:
        s = s[s.index >= _ts(since)]
    return s


def _flow_frame(strategy_id: Optional[str] = None) -> pd.DataFrame:
    """Per-mark (t, sleeve, eq, contributed, flow) for one strategy (or all)."""
    rows = _read_ledger()
    marks = [r for r in rows if r.get("type") == "mark"
             and (strategy_id is None or r.get("strategy_id") == strategy_id)]
    legacy = {}
    recs = []
    seen = set()
    for r in marks:
        sid = r["sleeve_id"]
        if r.get("contributed_usdt") is not None:
            c = float(r["contributed_usdt"])
            fl = float(r.get("flow_usdt") or 0.0)
        else:  # ledger written before flows were recorded: one init deposit, nothing else
            if sid not in legacy:
                legacy[sid] = _legacy_contributed(rows, sid)
            c = legacy[sid]
            fl = c if sid not in seen else 0.0
        seen.add(sid)
        recs.append((_ts(r["asof_bar"]), sid, float(r["equity_usdt"]), c, fl))
    return pd.DataFrame(recs, columns=["t", "sleeve", "eq", "contributed", "flow"])


def capital(strategy_id: Optional[str] = None) -> pd.Series:
    """Contributed capital (USDT, Σ deposits - withdrawals incl. the init deposit) per bar, aligned with equity()."""
    df = _flow_frame(strategy_id)
    if df.empty:
        return pd.Series(dtype=float, name="contributed_usdt")
    piv = df.groupby(["t", "sleeve"])["contributed"].last().unstack("sleeve").sort_index().ffill()
    out = piv.sum(axis=1, min_count=1)
    out.name, out.index.name = "contributed_usdt", "asof_bar"
    return out


def flows(strategy_id: Optional[str] = None) -> pd.Series:
    """Net capital flow per bar (deposits > 0, withdrawals < 0; the init deposit is a flow)."""
    df = _flow_frame(strategy_id)
    if df.empty:
        return pd.Series(dtype=float, name="flow_usdt")
    out = df.groupby("t")["flow"].sum().sort_index()
    out.name, out.index.name = "flow_usdt", "asof_bar"
    return out


def contributed(strategy_id: Optional[str] = None, *, asof=None) -> float:
    """Capital currently contributed to a strategy (or to all paper strategies) as of `asof` (default: latest)."""
    c = capital(strategy_id)
    if asof is not None:
        c = c[c.index <= _ts(asof)]
    return float(c.iloc[-1]) if len(c) else 0.0


def returns(strategy_id: Optional[str] = None, *, since=None) -> pd.Series:
    """Daily flow-adjusted returns from mark rows: r_t = (E_t - F_t) / E_{t-1} - 1, first bar E_0 / F_0 - 1.

    Flows happen at the close before the trade, so a deposit or withdrawal is never counted as P&L. Use this (or
    net_return) instead of dividing equity by the current allocation_usdt when allocations or coins changed."""
    eq = equity(strategy_id).dropna()
    if len(eq) == 0:
        return pd.Series(dtype=float, name="return")
    fl = flows(strategy_id).reindex(eq.index).fillna(0.0)
    prev = eq.shift(1)
    first_base = fl.iloc[0] if fl.iloc[0] > 0 else eq.iloc[0]
    prev.iloc[0] = first_base
    base_eq = eq - fl
    base_eq.iloc[0] = eq.iloc[0]
    r = (base_eq / prev.where(prev > 0) - 1.0).fillna(0.0)
    r.name, r.index.name = "return", "asof_bar"
    if since is not None:
        r = r[r.index >= _ts(since)]
    return r


def net_return(strategy_id: Optional[str] = None, *, since=None, asof=None) -> float:
    """Time-weighted (flow-adjusted) total return over [since, asof] of the paper marks."""
    r = returns(strategy_id, since=since)
    if asof is not None:
        r = r[r.index <= _ts(asof)]
    return float(np.prod(1.0 + r.values) - 1.0) if len(r) else 0.0


def positions() -> dict:
    """Current {sleeve_id: {"coin","qty","cash_usdt","weight", ...}} from the latest mark rows."""
    rows = _read_ledger()
    closed = _closed_sleeves(rows)
    out = {}
    for sid, m in _sleeve_state(rows).items():
        if sid in closed:
            continue
        out[sid] = {"coin": m["coin"], "strategy_id": m["strategy_id"], "qty": m["qty"], "cash_usdt": m["cash_usdt"],
                    "weight": m["weight"], "price": m["price"], "equity_usdt": m["equity_usdt"],
                    "asof_bar": m["asof_bar"], "contributed_usdt": _contributed(m, rows)}
    return out


# ── CLI ──────────────────────────────────────────────────────────────────────
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m trader.core.paper")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_run = sub.add_parser("run")
    p_run.add_argument("--asof")
    p_run.add_argument("--dry-run", action="store_true")
    p_run.add_argument("--active")
    p_set = sub.add_parser("settle")
    p_set.add_argument("--asof")
    p_eq = sub.add_parser("equity")
    p_eq.add_argument("--id")
    p_eq.add_argument("--since")
    sub.add_parser("positions")
    p_cap = sub.add_parser("capital")
    p_cap.add_argument("--id")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "run":
            out = run_paper(a.active, asof=a.asof, dry_run=a.dry_run)
        elif a.cmd == "settle":
            out = settle(a.asof)
        elif a.cmd == "equity":
            s = equity(a.id, since=a.since)
            out = {"strategy_id": a.id, "equity": {config.iso(k): round(float(v), 6) for k, v in s.items()}}
        elif a.cmd == "capital":
            out = {"strategy_id": a.id, "contributed_usdt": round(contributed(a.id), 6),
                   "net_return_twr": round(net_return(a.id), 8),
                   "flows": {config.iso(k): round(float(v), 6) for k, v in flows(a.id).items() if abs(v) > 1e-9}}
        else:
            out = positions()
    except PaperError as e:
        print(json.dumps({"ok": False, "error": str(e)}))
        return 1
    print(json.dumps(out, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
