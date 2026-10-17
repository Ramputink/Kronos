"""Real trading on Bybit EU **spot** (Wave 1, PROTECTED core) - BUILT BUT DISABLED.

Interface: trader/CONTRACTS.md section 6; operator manual: trader/REAL.md.

Every order attempt goes through `place_order`, which refuses (raises `RealTradingDisabled`) unless ALL of:
  * protected.json `real.real_trading_enabled` is literally `true` (only an `operator:` commit may flip it).
    The gate is read from the FIXED file `<this package>/config/protected.json`; env redirects (TRADER_CONFIG,
    TRADER_JOURNAL_DIR, TRADER_ROOT) are ignored, except that tests may redirect config + ledger with env
    PHIL_CI=1 and only for dry runs (dry_run=False always uses the fixed config and the fixed ledger);
  * for a live order (dry_run=False): the git provenance check `validate.check_real_flag` passes on the fixed
    repo (the flag was set by an `operator:` commit and the working copy equals HEAD);
  * the strategy id is in `real.allowed_strategy_ids` and the coin in `real.allowed_coins`;
  * env `BYBIT_API_KEY` and `BYBIT_API_SECRET` (names from `real.api_key_env/api_secret_env`) are set.
Then hard caps raise `RealCapExceeded`: config above validate.HARD ceilings (per_order_eur <= 100, daily_eur <= 250,
max_open_positions <= 10; re-checked here, never trusted from the config), per-order EUR, daily EUR (from today's real-ledger rows), max open
positions, min order USDT, spot only, Buy/Sell only, client_order_id unique. Every attempt (dry runs and refusals
included) appends one row to `journal/real-ledger.jsonl`. Secrets are never written, printed or put in exceptions.

pybit is imported lazily, so this module imports fine on hosts without it; `doctor(offline=True)` needs no network.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import stat
import sys
from pathlib import Path
from typing import Optional

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from trader.core import config

LEDGER_NAME = "real-ledger.jsonl"
# Fixed locations derived from this file, NOT from env: TRADER_ROOT / TRADER_CONFIG / TRADER_JOURNAL_DIR can be
# set by any process running as the same user, so they must never decide whether a live order is allowed.
_FIXED_ROOT = Path(__file__).resolve().parents[1]                      # .../trader
_FIXED_CONFIG = _FIXED_ROOT / "config" / "protected.json"
_FIXED_LEDGER = _FIXED_ROOT / "journal" / LEDGER_NAME
_FIXED_PAPER_LEDGER = _FIXED_ROOT / "journal" / "paper-ledger.jsonl"
# Defence in depth: the effective ceiling is min(these, validate.HARD), whatever protected.json says.
_HARD_LOCAL = {"per_order_eur": 100.0, "daily_eur": 250.0, "max_open_positions": 10}
# env vars that could redirect the provenance check's git calls to another repository
_GIT_ENV = ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_OBJECT_DIRECTORY", "GIT_ALTERNATE_OBJECT_DIRECTORIES",
            "GIT_CEILING_DIRECTORIES", "GIT_CONFIG", "GIT_CONFIG_GLOBAL", "GIT_CONFIG_SYSTEM", "GIT_NAMESPACE",
            "GIT_CONFIG_COUNT", "GIT_CONFIG_PARAMETERS")
CLIENT_ID_RE = re.compile(r"^[A-Za-z0-9_\-]{1,36}$")  # Bybit orderLinkId: <= 36 chars
SPEND_STATUSES = ("submitted", "filled")


class RealTradingDisabled(Exception):
    pass


class RealCapExceeded(Exception):
    pass


# ── helpers ──────────────────────────────────────────────────────────────────
def _override(dry_run: bool) -> bool:
    """Env redirects are honoured only for tests (PHIL_CI=1) and only for dry runs (never reach the exchange)."""
    return bool(dry_run) and os.environ.get("PHIL_CI") == "1"


def ledger_path(dry_run: bool = True) -> Path:
    return config.journal_dir() / LEDGER_NAME if _override(dry_run) else _FIXED_LEDGER


def config_file(dry_run: bool = True) -> Path:
    return config.config_path() if _override(dry_run) else _FIXED_CONFIG


def _rcfg(dry_run: bool = True) -> dict:
    with open(config_file(dry_run)) as fh:
        return json.load(fh)["real"]


def hard_limits() -> dict:
    """Effective real-trading ceilings: min(local constants, validate.HARD)."""
    out = dict(_HARD_LOCAL)
    try:
        from trader.core import validate
        for k in out:
            v = validate.HARD.get("real." + k)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                out[k] = min(out[k], v)
    except Exception:  # noqa: BLE001 - local constants still apply
        pass
    return out


def _hard_errors(rc: dict) -> list:
    errs = []
    for k, ceil in hard_limits().items():
        v = rc.get(k)
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            errs.append("real.%s missing or not a number" % k)
        elif not math.isfinite(v) or v < 0 or v > ceil + 1e-9:
            errs.append("real.%s=%r exceeds hard ceiling %s" % (k, v, ceil))
    return errs


def _provenance_errors(rc: dict) -> list:
    """validate.check_real_flag on the FIXED repo, with git/env redirects neutralised. Fails closed."""
    saved = {k: os.environ.pop(k) for k in _GIT_ENV if k in os.environ}
    old_root = config.REPO_ROOT
    try:
        from trader.core import validate
        config.REPO_ROOT = _FIXED_ROOT.parent  # validate.repo_root() reads config.REPO_ROOT
        if Path(validate.repo_root()).resolve() / validate.CONFIG_FILE != _FIXED_CONFIG:
            return ["provenance check does not target the fixed protected.json"]
        return [str(x) for x in validate.check_real_flag({"real": rc})]
    except Exception as e:  # noqa: BLE001 - fail closed
        return ["provenance check failed: %s" % type(e).__name__]
    finally:
        config.REPO_ROOT = old_root
        os.environ.update(saved)


def _enabled(rc: dict) -> bool:
    return rc.get("real_trading_enabled") is True  # strictly the JSON boolean true


def _keys(rc: dict) -> tuple:
    return os.environ.get(rc.get("api_key_env", "BYBIT_API_KEY")) or None, \
        os.environ.get(rc.get("api_secret_env", "BYBIT_API_SECRET")) or None


def _scrub(msg, rc: Optional[dict] = None) -> str:
    """Remove any secret value from a message before it is logged or raised."""
    msg = str(msg)
    if not rc:
        try:
            rc = _rcfg(True)
        except Exception:  # noqa: BLE001 - fall back to the default env var names
            rc = {}
    for v in _keys(rc):
        if v and len(v) >= 4:
            msg = msg.replace(v, "***")
    return msg[:500]


def _tld(api_base: str) -> str:
    host = re.sub(r"^https?://", "", api_base).split("/")[0]  # api.bybit.eu
    parts = host.split(".")
    return ".".join(parts[2:]) if len(parts) >= 3 else "com"


def _http(rc: dict):
    """Authenticated pybit session against the configured Bybit host (lazy import)."""
    from pybit.unified_trading import HTTP  # lazy: optional dependency
    import logging
    key, secret = _keys(rc)
    return HTTP(testnet=False, tld=_tld(rc["api_base"]), api_key=key, api_secret=secret,
                logging_level=logging.WARNING, log_requests=False, timeout=10)


def _today() -> str:
    return config.utcnow().strftime("%Y-%m-%d")


def _rows(dry_run: bool = True) -> list:
    return config.jsonl_read(ledger_path(dry_run))


def spent_today_eur(rows: Optional[list] = None, *, dry_run: bool = True) -> float:
    rows = _rows(dry_run) if rows is None else rows
    day = _today()
    return float(sum(float(r.get("notional_eur") or 0.0) for r in rows
                     if r.get("status") in SPEND_STATUSES and str(r.get("ts", "")).startswith(day)
                     and r.get("side") == "buy"))


def _counts(r: dict, dry_run: bool) -> bool:
    """Does real-ledger row `r` move a holding in the given mode? Dry runs track their own simulated book."""
    if dry_run:
        return r.get("status") == "dry_run" and bool(r.get("dry_run"))
    return r.get("status") in SPEND_STATUSES and not r.get("dry_run")


def holdings(rows: Optional[list] = None, *, dry_run: bool = True) -> dict:
    """Real (or simulated, for dry runs) book per (strategy_id, coin) -> {"eur": net EUR cost, "qty": net base qty}.

    A sell row with `closed: true` resets that sleeve to 0 (the whole balance was sold, whatever the price did).
    `qty` is only known for live fills (dry runs have no price), so dry-run sleeves are tracked in EUR.
    """
    rows = _rows(dry_run) if rows is None else rows
    h = {}
    for r in rows:
        if not r.get("coin") or not _counts(r, dry_run):
            continue
        cur = h.setdefault((r.get("strategy_id"), r["coin"]), {"eur": 0.0, "qty": 0.0})
        eur, qty = float(r.get("notional_eur") or 0.0), float(r.get("qty") or 0.0)
        if r.get("side") == "buy":
            cur["eur"] += eur
            cur["qty"] += qty
        elif r.get("side") == "sell":
            if r.get("closed"):
                cur["eur"], cur["qty"] = 0.0, 0.0
            else:
                cur["eur"], cur["qty"] = max(0.0, cur["eur"] - eur), max(0.0, cur["qty"] - qty)
    return {k: v for k, v in h.items() if v["eur"] > 1e-9 or v["qty"] > 1e-12}


def _open_coins(rows: list) -> set:
    """Coins with an open position according to the real ledger (live rows; dry-run rows of the simulated book)."""
    out = {c for (_, c) in holdings(rows, dry_run=False)}
    out |= {c for (_, c) in holdings(rows, dry_run=True)}
    # legacy rows without a dry_run flag: net bought notional > 0
    net = {}
    for r in rows:
        if r.get("status") in SPEND_STATUSES and "dry_run" not in r and r.get("coin"):
            sgn = 1.0 if r.get("side") == "buy" else -1.0
            net[r["coin"]] = net.get(r["coin"], 0.0) + sgn * float(r.get("notional_eur") or 0.0)
    return out | {c for c, v in net.items() if v > 1e-9}


def _free_base(s, coin: str) -> float:
    """Free (not locked) balance of `coin` on the unified account. Raises if it cannot be read (fail closed)."""
    res = s.get_wallet_balance(accountType="UNIFIED", coin=coin)["result"]["list"]
    for acct in res or []:
        for c in acct.get("coin") or []:
            if c.get("coin") == coin:
                wb = float(c.get("walletBalance") or 0)
                locked = float(c.get("locked") or 0)
                return max(0.0, wb - locked)
    return 0.0


def _log(row: dict, dry_run: bool = True) -> dict:
    full = {"ts": config.iso(), "client_order_id": None, "strategy_id": None, "coin": None, "side": None,
            "notional_eur": None, "notional_usdt": None, "status": None, "reason": "", "exchange_order_id": None,
            "fill_price": None, "fee": None}
    full.update(row)
    full["reason"] = _scrub(full.get("reason") or "")
    full["runner"] = config.runner_id()
    config.jsonl_append(ledger_path(dry_run), full)
    return full


def _eur_usdt(rc: dict, live: bool) -> float:
    if live:
        try:
            import requests
            r = requests.get(rc["api_base"].rstrip("/") + "/v5/market/tickers",
                             params={"category": "spot", "symbol": "EURUSDT"}, timeout=10).json()
            px = float(r["result"]["list"][0]["lastPrice"])
            if px > 0:
                return px
        except Exception:  # noqa: BLE001 - fall back to the configured rate
            pass
    return float(rc.get("eur_usdt_fallback", 1.16))


def _floor_step(x: float, step: float) -> float:
    if step <= 0:
        return x
    return math.floor(x / step + 1e-9) * step


def _fmt(x: float, step: float) -> str:
    """Decimal string with exactly the precision of `step` (no float artefacts in the order payload)."""
    if not step or step <= 0:
        return ("%.8f" % x).rstrip("0").rstrip(".")
    dec = max(0, int(round(-math.log10(step)))) if step < 1 else 0
    return "%.*f" % (dec, x)


# ── public API ───────────────────────────────────────────────────────────────
def place_order(coin: str, side: str, notional_eur: Optional[float], *, client_order_id: str, strategy_id: str,
                dry_run: bool = True, sell_fraction: Optional[float] = None) -> dict:
    """Place (or dry-run) one spot order. Raises RealTradingDisabled / RealCapExceeded; logs every attempt.

    Buys are sized by `notional_eur` (capped by per_order_eur / daily_eur). Sells are sized from what is actually
    held, never from a EUR figure alone: with `sell_fraction` (0 < f <= 1) the order sells that fraction of this
    strategy's holding of `coin` (1 = close the position: the whole free balance when no other strategy holds the
    coin); without it, `notional_eur` is converted to coin and capped by the free balance. Live sells read the
    free base-coin balance from the exchange (Bybit spot takes the buy fee in the base coin, so a position holds
    slightly less coin than the notional bought). A sell is refused when it is below the exchange minOrderAmt,
    or when it would strand a residual worth less than minOrderAmt that this strategy could not sell later (when
    no other strategy holds the coin, the order is enlarged to the whole balance instead). The residual left
    behind is recorded on the row (`residual_qty`). per_order_eur caps buys only: a sell can only reduce a
    position and is bounded by the held balance.
    """
    dry_run = dry_run is not False  # anything but the literal False is treated as a dry run
    side_l = str(side).lower()
    base = {"client_order_id": client_order_id, "strategy_id": strategy_id, "coin": coin, "side": side_l,
            "notional_eur": notional_eur, "dry_run": dry_run}
    if sell_fraction is not None:
        base["sell_fraction"] = sell_fraction

    def refuse(exc_cls, reason):
        _log(dict(base, status="refused", reason=reason), dry_run)
        raise exc_cls(reason)

    try:
        rc = _rcfg(dry_run)
    except Exception as e:  # noqa: BLE001 - fail closed
        rc = {}
        refuse(RealTradingDisabled, "cannot read protected config: %s" % type(e).__name__)

    # 1. gates (disabled unless everything is explicitly allowed)
    if not _enabled(rc):
        refuse(RealTradingDisabled, "real_trading_enabled is false in protected config")
    if not dry_run:
        prov = _provenance_errors(rc)
        if prov:
            refuse(RealTradingDisabled, "provenance: " + "; ".join(prov))
    if strategy_id not in (rc.get("allowed_strategy_ids") or []):
        refuse(RealTradingDisabled, "strategy %r not in real.allowed_strategy_ids" % strategy_id)
    if coin not in (rc.get("allowed_coins") or []):
        refuse(RealTradingDisabled, "coin %r not in real.allowed_coins" % coin)
    key, secret = _keys(rc)
    if not key or not secret:
        refuse(RealTradingDisabled, "API key/secret env vars not set")
    # 2. sanity + hard caps (config values above the HARD ceilings invalidate the whole config)
    hard = _hard_errors(rc)
    if hard:
        refuse(RealCapExceeded, "config above hard ceilings: " + "; ".join(hard))
    if rc.get("category", "spot") != "spot":
        refuse(RealCapExceeded, "only spot is allowed (config category=%r)" % rc.get("category"))
    if side_l not in ("buy", "sell"):
        refuse(RealCapExceeded, "side must be buy or sell")
    if not CLIENT_ID_RE.match(str(client_order_id or "")):
        refuse(RealCapExceeded, "client_order_id must match %s" % CLIENT_ID_RE.pattern)
    frac = None
    if sell_fraction is not None:
        if side_l != "sell":
            refuse(RealCapExceeded, "sell_fraction is only valid for sells")
        try:
            frac = float(sell_fraction)
        except (TypeError, ValueError):
            frac = float("nan")
        if not math.isfinite(frac) or frac <= 0 or frac > 1 + 1e-9:
            refuse(RealCapExceeded, "sell_fraction must be in (0, 1]")
        frac = min(frac, 1.0)
    rows = _rows(dry_run)
    if any(r.get("client_order_id") == client_order_id and r.get("status") in ("dry_run",) + SPEND_STATUSES
           and bool(r.get("dry_run")) == bool(dry_run) for r in rows):
        refuse(RealCapExceeded, "duplicate client_order_id %s" % client_order_id)
    book = holdings(rows, dry_run=dry_run)
    mine = book.get((strategy_id, coin), {"eur": 0.0, "qty": 0.0})
    others = any(c == coin and sid != strategy_id for (sid, c) in book)
    closing = side_l == "sell" and frac is not None and frac >= 1.0
    if frac is not None:
        if mine["eur"] <= 1e-9 and mine["qty"] <= 1e-12 and (dry_run or others):
            refuse(RealCapExceeded, "nothing held for %s %s to sell" % (strategy_id, coin))
        notional_eur = mine["eur"] * frac  # estimate; live orders size from the exchange balance
        base["notional_eur"] = round(notional_eur, 6)
    try:
        notional_eur = float(notional_eur)
    except (TypeError, ValueError):
        refuse(RealCapExceeded, "notional_eur not a number")
    if not math.isfinite(notional_eur) or notional_eur < 0 or (notional_eur == 0 and (dry_run or frac is None)):
        refuse(RealCapExceeded, "notional_eur must be > 0")
    if side_l == "buy":
        if notional_eur > float(rc["per_order_eur"]) + 1e-9:
            refuse(RealCapExceeded, "order %.2f EUR > per-order cap %.2f EUR" % (notional_eur, rc["per_order_eur"]))
        spent = spent_today_eur(rows)
        if spent + notional_eur > float(rc["daily_eur"]) + 1e-9:
            refuse(RealCapExceeded, "daily cap: spent %.2f + %.2f > %.2f EUR" % (spent, notional_eur, rc["daily_eur"]))
        open_c = _open_coins(rows)
        if coin not in open_c and len(open_c) >= int(rc["max_open_positions"]):
            refuse(RealCapExceeded, "max_open_positions %d reached (%s)" % (rc["max_open_positions"], sorted(open_c)))
    elif dry_run and frac is None and notional_eur > mine["eur"] + 1e-6:
        # simulated book: a sell cannot exceed what this strategy holds
        refuse(RealCapExceeded, "sell %.2f EUR > held %.2f EUR (use sell_fraction=1 to close)"
               % (notional_eur, mine["eur"]))
    rate = _eur_usdt(rc, live=not dry_run)
    notional_usdt = notional_eur * rate
    base["notional_usdt"] = round(notional_usdt, 6)
    if dry_run:
        if notional_usdt < float(rc["min_order_usdt"]):
            refuse(RealCapExceeded, "order %.2f USDT < min %.2f USDT" % (notional_usdt, rc["min_order_usdt"]))
        extra = {"closed": True} if closing else {}
        return _log(dict(base, status="dry_run", reason="all gates and caps passed (eur_usdt=%.4f)" % rate,
                         **extra), dry_run)
    if side_l == "buy" and notional_usdt < float(rc["min_order_usdt"]):
        refuse(RealCapExceeded, "order %.2f USDT < min %.2f USDT" % (notional_usdt, rc["min_order_usdt"]))

    # 3. live order: IOC limit at last price +/- max_slippage_bps (never a naked market order)
    symbol = coin + "USDT"
    try:
        s = _http(rc)
        info = s.get_instruments_info(category="spot", symbol=symbol)["result"]["list"][0]
        lot, pf = info.get("lotSizeFilter", {}), info.get("priceFilter", {})
        base_step = float(lot.get("basePrecision") or 0)
        min_amt = max(float(lot.get("minOrderAmt") or 0), float(rc["min_order_usdt"]))
        min_qty = float(lot.get("minOrderQty") or 0)
        tick = float(pf.get("tickSize") or 0)
        last = float(s.get_tickers(category="spot", symbol=symbol)["result"]["list"][0]["lastPrice"])
        slip = float(rc.get("max_slippage_bps", 30)) / 1e4
        px = last * (1 + slip) if side_l == "buy" else last * (1 - slip)
        if tick:
            px = _floor_step(px, tick) if side_l == "sell" else math.ceil(px / tick - 1e-9) * tick
        residual = None
        if side_l == "buy":
            qty = _floor_step(notional_usdt / px, base_step)
        else:
            free = _free_base(s, coin)
            # this strategy's share of the free balance: all of it unless another strategy also holds the coin
            own = free if not others else min(free, mine["qty"])
            want = own if closing else (own * frac if frac is not None else notional_usdt / px)
            qty = _floor_step(min(want, own), base_step)
            residual = max(0.0, own - qty)
            base["residual_qty"] = residual
            if qty > 0 and residual >= max(base_step, 1e-12) and residual * px < min_amt:
                if not others:  # the remainder could never be sold on its own: close the whole balance instead
                    qty = _floor_step(own, base_step)
                    residual = max(0.0, own - qty)
                    base["residual_qty"] = residual
                    closing = True
                else:
                    refuse(RealCapExceeded, "sell would leave %.10g %s (%.2f USDT) < minOrderAmt %.2f USDT"
                           % (residual, coin, residual * px, min_amt))
            if qty <= 0 or qty * px < min_amt or qty < min_qty:
                base["residual_qty"] = own
                refuse(RealCapExceeded, "sell %.10g %s (%.2f USDT) below minOrderAmt %.2f USDT / minOrderQty %s; "
                       "free %.10g; residual %.10g %s kept as dust" % (qty, coin, qty * px, min_amt, min_qty, free,
                                                                      own, coin))
            notional_usdt = qty * px
            base.update(notional_usdt=round(notional_usdt, 6), notional_eur=round(notional_usdt / rate, 6))
        if qty <= 0:
            refuse(RealCapExceeded, "quantity rounds to zero")
        qty_s, px_s = _fmt(qty, base_step), _fmt(px, tick)
        resp = s.place_order(category="spot", symbol=symbol, side=side_l.capitalize(), orderType="Limit",
                             qty=qty_s, price=px_s, timeInForce="IOC", orderLinkId=client_order_id)
        oid = resp.get("result", {}).get("orderId")
        row = dict(base, status="submitted", exchange_order_id=oid, reason="IOC limit %s @ %s" % (qty_s, px_s))
        try:
            hist = s.get_order_history(category="spot", orderId=oid)["result"]["list"]
            if hist:
                h = hist[0]
                filled = float(h.get("cumExecQty") or 0)
                if filled > 0:
                    fee = float(h.get("cumExecFee") or 0)
                    # Bybit spot charges the buy fee in the base coin: the coin received is filled - fee
                    got = filled - fee if side_l == "buy" and 0 <= fee < filled else filled
                    row.update(status="filled", fill_price=float(h.get("avgPrice") or 0) or None, fee=fee,
                               qty=got, notional_usdt=round(float(h.get("cumExecValue") or 0), 6),
                               notional_eur=round(float(h.get("cumExecValue") or 0) / rate, 6))
                    if side_l == "sell":
                        row["residual_qty"] = max(0.0, (residual or 0.0) + qty - filled)
                        row["closed"] = bool(closing and filled >= qty - 1e-12)
                elif h.get("orderStatus") in ("Cancelled", "Rejected"):
                    row.update(status="rejected", reason="IOC not filled: %s" % h.get("orderStatus"))
        except Exception as e:  # noqa: BLE001 - order was submitted; status check is best effort
            row["reason"] += "; status check failed: %s" % type(e).__name__
        return _log(row, dry_run)
    except (RealCapExceeded, RealTradingDisabled):
        raise
    except Exception as e:  # noqa: BLE001
        return _log(dict(base, status="rejected", reason="%s: %s" % (type(e).__name__, _scrub(e, rc))), dry_run)


def real_sleeve_eur(rc: dict) -> float:
    """Size of one real sleeve (strategy, coin) in EUR: `real.sleeve_eur` if set, else `real.per_order_eur`."""
    v = rc.get("sleeve_eur", rc.get("per_order_eur"))
    try:
        v = float(v)
    except (TypeError, ValueError):
        return 0.0
    return v if math.isfinite(v) and v > 0 else 0.0


def mirror_paper(run_id: str, *, dry_run: bool = True) -> list:
    """Twin orders for the fills of paper run `run_id` whose strategy is allowed for real trading.

    Twins follow the paper sleeve's WEIGHT, not its notional: each paper fill moves its sleeve from `prev_weight`
    to `target_weight`, and the real sleeve (size `real_sleeve_eur`) makes the same move:
      * buy:  (target_weight - prev_weight) * real_sleeve_eur EUR, then capped by per_order_eur;
      * sell: the fraction (prev_weight - target_weight) / prev_weight of the real holding;
      * target_weight == 0 (signal exit or retire): close, i.e. sell the whole real holding.
    A buy, a rebalance and a full exit therefore leave the real position flat exactly when paper is flat.
    Refusals are returned (and logged), never raised.
    """
    from trader.core import paper
    dry_run = dry_run is not False
    rc = _rcfg(dry_run)
    allowed = set(rc.get("allowed_strategy_ids") or [])
    sleeve = real_sleeve_eur(rc)
    try:
        cap = min(float(rc["per_order_eur"]), hard_limits()["per_order_eur"])
    except (KeyError, TypeError, ValueError):
        cap = hard_limits()["per_order_eur"]  # place_order refuses the invalid config anyway
    # live twins come from the fixed paper ledger (TRADER_JOURNAL_DIR is honoured only for PHIL_CI dry runs)
    src = paper.ledger_path() if _override(dry_run) else _FIXED_PAPER_LEDGER
    out = []
    for f in config.jsonl_read(src):
        if f.get("type") != "fill" or f.get("run_id") != run_id or f.get("strategy_id") not in allowed:
            continue
        sid, coin, side = f["strategy_id"], f["coin"], str(f.get("side", "")).lower()
        tag = "%04x" % (sum(ord(ch) * (i + 1) for i, ch in enumerate(sid)) % 65536)  # several strategies per coin
        cid = ("m%s%s%s%s" % (run_id.replace("-", "")[-16:], tag, coin, side[:1]))[:36]
        twin = {"client_order_id": cid, "strategy_id": sid, "coin": coin, "side": side,
                "prev_weight": f.get("prev_weight"), "target_weight": f.get("target_weight")}
        try:
            pw, tw = float(f["prev_weight"]), float(f["target_weight"])
            if not (math.isfinite(pw) and math.isfinite(tw)):
                raise ValueError
        except (KeyError, TypeError, ValueError):
            out.append(dict(twin, status="refused", reason="paper fill lacks prev_weight/target_weight"))
            continue
        try:
            if side == "buy":
                eur = min(max(tw - pw, 0.0) * sleeve, cap)
                res = place_order(coin, "buy", round(eur, 2), client_order_id=cid, strategy_id=sid,
                                  dry_run=dry_run)
            elif side == "sell":
                frac = 1.0 if tw <= 1e-9 or pw <= 1e-12 else min(1.0, (pw - tw) / pw)
                if frac <= 0:
                    raise RealCapExceeded("sell fill does not lower the weight (%.6f -> %.6f)" % (pw, tw))
                res = place_order(coin, "sell", None, client_order_id=cid, strategy_id=sid, dry_run=dry_run,
                                  sell_fraction=frac)
            else:
                raise RealCapExceeded("side must be buy or sell")
            out.append(dict(res, prev_weight=pw, target_weight=tw))
        except (RealTradingDisabled, RealCapExceeded) as e:
            out.append(dict(twin, status="refused", reason=str(e)))
    return out


def doctor(*, offline: bool = False) -> dict:
    """Read-only readiness report. Works while disabled; never prints secrets. offline=True skips network."""
    rc = _rcfg(True)
    checks = []

    def add(name, ok, detail=""):
        checks.append({"name": name, "ok": bool(ok), "detail": _scrub(detail, rc)})

    enabled = _enabled(rc)
    add("config_flag", enabled, "real_trading_enabled=%s" % json.dumps(rc.get("real_trading_enabled")))
    add("allowed_strategies", bool(rc.get("allowed_strategy_ids")), "allowed_strategy_ids=%s"
        % rc.get("allowed_strategy_ids"))
    add("spot_only", rc.get("category") == "spot", "category=%s" % rc.get("category"))
    he = _hard_errors(rc)
    add("hard_ceilings", not he, "; ".join(he) or "within %s" % hard_limits())
    cf = config_file(True)
    add("fixed_config", cf == _FIXED_CONFIG, "gate read from %s%s" % (
        cf, "" if cf == _FIXED_CONFIG else " (PHIL_CI dry-run override; live orders use %s)" % _FIXED_CONFIG))
    if enabled:
        pe = _provenance_errors(rc)
        add("flag_provenance", not pe, "; ".join(pe) or "operator commit, working copy == HEAD")
    sf = Path(os.path.expanduser(rc.get("secrets_file", "~/.phil-crypto/secrets.env")))
    if sf.exists():
        mode = stat.S_IMODE(sf.stat().st_mode)
        inside = str(sf.resolve()).startswith(str(_FIXED_ROOT.parent) + os.sep)
        add("secrets_file", mode == 0o600 and not inside, "%s mode=%o inside_repo=%s" % (sf, mode, inside))
    else:
        add("secrets_file", False, "%s missing" % sf)
    key, secret = _keys(rc)
    add("api_keys_env", bool(key and secret), "%s/%s %s" % (rc["api_key_env"], rc["api_secret_env"],
                                                            "set" if key and secret else "not set"))
    rows = _rows()
    spent = spent_today_eur(rows)
    add("daily_spend", spent <= float(rc["daily_eur"]), "today %.2f / %.2f EUR" % (spent, rc["daily_eur"]))
    if offline:
        add("api_reachable", False, "skipped (offline)")
    else:
        try:
            import requests
            r = requests.get(rc["api_base"].rstrip("/") + "/v5/market/time", timeout=10).json()
            add("api_reachable", r.get("retCode") == 0, "%s retCode=%s" % (rc["api_base"], r.get("retCode")))
        except Exception as e:  # noqa: BLE001
            add("api_reachable", False, "%s: %s" % (type(e).__name__, e))
        if key and secret:
            try:
                s = _http(rc)
                info = s.get_api_key_information()["result"]
                add("key_valid", True, "readOnly=%s" % info.get("readOnly"))
                perms = info.get("permissions") or {}
                wd = any("withdraw" in str(p).lower() for v in perms.values() for p in (v or []))
                add("no_withdraw_permission", not wd, "withdraw permission %s" % ("PRESENT" if wd else "absent"))
                acc = s.get_account_info()["result"]
                add("account_type", True, "unifiedMarginStatus=%s" % acc.get("unifiedMarginStatus"))
                bal = s.get_wallet_balance(accountType="UNIFIED")["result"]["list"]
                tot = bal[0].get("totalEquity") if bal else None
                add("balances", bal is not None, "totalEquity=%s" % tot)
            except Exception as e:  # noqa: BLE001
                add("key_valid", False, "%s: %s" % (type(e).__name__, e))
        else:
            add("key_valid", False, "no API keys in env")
    ok = all(c["ok"] for c in checks)
    return {"ok": ok, "enabled": enabled, "checks": checks}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m trader.core.real")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("doctor")
    d.add_argument("--offline", action="store_true")
    m = sub.add_parser("mirror")
    m.add_argument("run_id")
    m.add_argument("--live", action="store_true", help="submit for real (still refused unless enabled)")
    a = ap.parse_args(argv)
    if a.cmd == "doctor":
        out = doctor(offline=a.offline)
        print(json.dumps(out))
        return 0 if out["ok"] else 1
    out = mirror_paper(a.run_id, dry_run=not a.live)
    print(json.dumps(out, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
