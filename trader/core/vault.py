"""The vault: the only public way to read market data (Wave 1, PROTECTED).

Views (CONTRACTS.md section 3):
  research  bars with open time < vault_cutoff. Physically truncated BEFORE anything else sees the frame,
            so no indicator, mask or funding sum computed on it can depend on sealed data.
  full      all closed bars, ONLY with a look granted by trials.request_look and accepted by
            trials.check_look(look_id, spec_hash); otherwise VaultSealed. `load_full()` is the explicit entry.
  paper     closed bars with open time <= asof (default: now), ONLY from data.paper_source (another source needs
            an accepted look) and ONLY for a declared operational caller (PAPER_CALLERS). research/sweep callers
            get VaultSealed.
            * caller 'paper' (paper settlement; accepted only from trader.core.paper/forecast/score) gets the full
              history up to asof, but only when asof is in the forward period (asof >= forward_start - 1 bar, the
              bar whose close starts the forward period) or entirely before the cutoff. While forward_start is
              null, any asof >= cutoff is VaultSealed (everything >= cutoff is sealed TEST data).
            * every other paper caller (forward, deep_retro, screener, watch, operator, test) is WINDOWED: it sees
              only [max(forward_start, cutoff) - warmup_days, asof], and VaultSealed while forward_start is null,
              so the sealed TEST bars before the warmup stay sealed.
            * an explicit caller is cross-checked against the calling module (nearest frame outside vault/data,
              matched by module name AND source file): 'screener' only from trader.strategy.screener, 'watch'
              only from trader.fleet.watch, 'paper' only from PAPER_MODULES; 'operator'/'test' only when the
              environment has TRADER_VAULT_OPERATOR=1 (operator shell / test harness, never the agent).
            * every paper-view load that returns bars in [cutoff, forward_start) is logged as an "access" row
              (see _log_access): sealed TEST bars are never handed out silently.

Python >= 3.9. Dependencies: numpy, pandas.
"""
from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path
from typing import Iterable, Optional, Union

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from trader.core import config
from trader.core import data as _data


class VaultSealed(Exception):
    """Access to sealed data (>= vault_cutoff) without a valid look."""


VIEWS = ("research", "full", "paper")
# callers allowed to read the paper (live) view. All but "paper" are windowed to the forward period (see load).
PAPER_CALLERS = ("paper", "forward", "deep_retro", "screener", "watch", "operator", "test")
FORWARD_CALLERS = ("forward", "deep_retro")
DENIED_PAPER_CALLERS = ("research", "sweep")
# Modules named in CONTRACTS §3 as paper-view users, and the caller they implicitly declare when they do not pass
# `caller=` (backward compatibility). Anything else must pass caller= explicitly or is refused (fail closed).
_IMPLICIT_PAPER_CALLERS = {"trader.core.paper": "paper", "trader.core.forecast": "paper",
                           "trader.strategy.screener": "screener", "trader.fleet.watch": "watch"}
# explicit callers that are bound to the modules that may claim them (__name__ AND source file are checked)
PAPER_MODULES = ("trader.core.paper", "trader.core.forecast", "trader.core.score")
_CALLER_MODULES = {"paper": PAPER_MODULES, "screener": ("trader.strategy.screener",),
                   "watch": ("trader.fleet.watch",)}
# 'operator'/'test' need this env flag, set by the operator's shell or the test harness outside the agent sandbox
OPERATOR_ENV = "TRADER_VAULT_OPERATOR"
_ENV_CALLERS = ("operator", "test")
_PKG_ROOT = Path(__file__).resolve().parents[2]  # directory that contains the `trader` package
PERIODS = ("train", "val", "trainval", "test", "forward", "all")
COLUMNS = ["open", "high", "low", "close", "volume", "turnover", "funding"]


def _vcfg() -> dict:
    return config.load_config()["vault"]


def cutoff() -> pd.Timestamp:
    return pd.Timestamp(_vcfg()["vault_cutoff"])


def train_end() -> pd.Timestamp:
    return pd.Timestamp(_vcfg()["train_end"])


def warmup_days() -> int:
    return int(_vcfg()["warmup_days"])


def forward_start() -> Optional[pd.Timestamp]:
    fs = (config.load_config().get("forward") or {}).get("forward_start")
    return pd.Timestamp(fs) if fs else None


def _ts(x) -> pd.Timestamp:
    t = pd.Timestamp(x)
    if t.tzinfo is not None:
        t = t.tz_convert("UTC").tz_localize(None)
    return t


def period_mask(index: pd.DatetimeIndex, period: str, *, activated_at: Optional[str] = None) -> np.ndarray:
    """Boolean mask over `index` (bar open times) for a named period (CONTRACTS.md section 2).

    start = index[0] (first bar of the loaded series), as worker.py.
      train    [start + warmup, train_end)
      val      [train_end, cutoff)            (worker.py m_val = idx >= TRAIN_END on a frame already < cutoff)
      trainval [start + warmup, cutoff)
      test     [cutoff, forward_start)  or [cutoff, end] while forward_start is null
      forward  [max(forward_start, activated_at), end]; empty while forward_start is null
      all      [start + warmup, end]
    """
    index = pd.DatetimeIndex(index)
    n = len(index)
    if period not in PERIODS:
        raise ValueError("unknown period %r (expected one of %s)" % (period, PERIODS))
    if n == 0:
        return np.zeros(0, dtype=bool)
    start = index[0] + pd.Timedelta(days=warmup_days())
    co, te, fs = cutoff(), train_end(), forward_start()
    if period == "train":
        m = (index >= start) & (index < te)
    elif period == "val":
        m = (index >= te) & (index < co)
    elif period == "trainval":
        m = (index >= start) & (index < co)
    elif period == "test":
        m = index >= co
        if fs is not None:
            m = m & (index < fs)
    elif period == "forward":
        if fs is None:
            return np.zeros(n, dtype=bool)
        lo = fs
        if activated_at:
            lo = max(lo, _ts(activated_at))
        m = index >= lo
    else:  # all
        m = index >= start
    return np.asarray(m, dtype=bool)


def _frame(df: pd.DataFrame, fund: pd.Series, *, coin, tf, view, source) -> pd.DataFrame:
    out = df.copy()
    out["funding"] = fund.reindex(out.index).fillna(0.0).values if len(out) else []
    out = out[COLUMNS]
    out.index.name = "time"
    out.attrs = {"coin": coin, "tf": tf, "view": view, "source": source,
                 "cutoff": config.iso(cutoff()),
                 "data_end": config.iso(out.index.max()) if len(out) else None}
    return out


def _check_look(look_id: Optional[str], spec_hash: Optional[str]) -> dict:
    if not look_id:
        raise VaultSealed("view='full' needs a look_id granted by trials.request_look")
    try:
        from trader.core import trials  # lazy: trials imports VaultSealed from here
    except ImportError as e:  # fail closed
        raise VaultSealed("cannot verify look %s: trials unavailable (%s)" % (look_id, e))
    try:
        res = trials.check_look(look_id, spec_hash)
    except VaultSealed:
        raise
    except Exception as e:  # noqa: BLE001 - any failure to verify is a refusal
        raise VaultSealed("look %s refused: %s: %s" % (look_id, type(e).__name__, e))
    return res if isinstance(res, dict) else {"look_id": look_id}


def _module_file(mod: str) -> Path:
    return (_PKG_ROOT / (mod.replace(".", "/") + ".py")).resolve()


def _calling_module() -> tuple:
    """(module __name__, resolved source file) of the nearest frame outside vault.py / data.py."""
    try:
        f = sys._getframe(1)
        here = (__name__, "trader.core.data")
        while f is not None and f.f_globals.get("__name__") in here:
            f = f.f_back  # skip vault.load_* wrappers and the data.load re-export
        if f is None:
            return None, None
        fn = f.f_code.co_filename
        try:
            fp = Path(fn).resolve() if fn and not fn.startswith("<") else None
        except Exception:  # noqa: BLE001
            fp = None
        return f.f_globals.get("__name__"), fp
    except Exception:  # noqa: BLE001
        return None, None


def _module_is(mod, fp, allowed) -> bool:
    """True when the calling frame really is one of `allowed`: same __name__ AND that module's own file, so a
    script that only sets __name__ (or exec's code under that name) is not accepted."""
    return mod in allowed and fp is not None and fp == _module_file(mod)


def _implicit_caller() -> Optional[str]:
    """Caller implied by the module that called load(), for the CONTRACTS §3 paper modules only."""
    mod, fp = _calling_module()
    c = _IMPLICIT_PAPER_CALLERS.get(mod)
    return c if c is not None and _module_is(mod, fp, (mod,)) else None


def _paper_gate(caller: Optional[str], source: str, paper_source: str, look_id, spec_hash) -> str:
    """Raise VaultSealed unless (caller, source) may read the paper view; returns the effective caller."""
    if caller is None:
        caller = _implicit_caller()
        if caller is None:
            raise VaultSealed("view='paper' needs caller= (one of %s); research/sweep must use view='research'"
                              % (PAPER_CALLERS,))
    if caller in DENIED_PAPER_CALLERS:
        raise VaultSealed("caller %r may not read the paper view (post-cutoff bars are sealed; use a look)" % caller)
    if caller not in PAPER_CALLERS:
        raise VaultSealed("unknown caller %r for view='paper' (expected one of %s)" % (caller, PAPER_CALLERS))
    if caller in _CALLER_MODULES:
        mod, fp = _calling_module()
        if not _module_is(mod, fp, _CALLER_MODULES[caller]):
            raise VaultSealed("caller %r is reserved to %s (called from %r)" % (caller, _CALLER_MODULES[caller], mod))
    if caller in _ENV_CALLERS and os.environ.get(OPERATOR_ENV) != "1":
        raise VaultSealed("caller %r needs %s=1 in the environment (operator shell / test harness only)"
                          % (caller, OPERATOR_ENV))
    if source != paper_source:
        # e.g. source='linear' would hand out the sealed research series in full: only with an accepted look
        _check_look(look_id, spec_hash)
    return caller


# ── access log: paper-view loads that contain sealed TEST bars ───────────────────────────────────────────────
_ACCESS_SEEN: set = set()


def _access_path() -> Path:
    """trials.jsonl once validate accepts kind 'access'; until then the sibling ledger journal/vault_access.jsonl
    (so the commit guard never meets an unknown trials kind)."""
    try:
        from trader.core import validate  # lazy
        if "access" in getattr(validate, "TRIALS_BY_KIND", {}):
            return config.journal_dir() / "trials.jsonl"
    except Exception:  # noqa: BLE001
        pass
    return config.journal_dir() / "vault_access.jsonl"


def _log_access(out: pd.DataFrame, *, caller, coin, tf, source, asof, look_id) -> Optional[dict]:
    """Log one "access" row when a paper-view frame holds bars in [cutoff, forward_start) (or >= cutoff while
    forward_start is null). Deduplicated per process on (caller, coin, tf, source, sealed range)."""
    if not len(out):
        return None
    co, fs = cutoff(), forward_start()
    m = out.index >= co
    if fs is not None:
        m = m & (out.index < fs)
    n = int(m.sum())
    if n == 0:
        return None
    sealed = out.index[m]
    key = (caller, coin, tf, source, str(sealed.min()), str(sealed.max()))
    if key in _ACCESS_SEEN:
        return None
    mod, _ = _calling_module()
    row = {"trial_id": "A-%s-%s" % (_ts(config.utcnow()).strftime("%Y%m%dT%H%M%S"), uuid.uuid4().hex[:6]),
           "ts": config.iso(), "kind": "access", "caller": caller, "runner": config.runner_id(), "view": "paper",
           "coin": coin, "tf": tf, "source": source, "asof": config.iso(_ts(asof)) if asof is not None else None,
           "module": mod, "look_id": look_id, "sealed_rows": n, "sealed_first": config.iso(sealed.min()),
           "sealed_last": config.iso(sealed.max()), "forward_start": config.iso(fs) if fs is not None else None}
    config.jsonl_append(_access_path(), row)  # a failure to log raises, so the load fails closed
    _ACCESS_SEEN.add(key)
    return row


def load(coin: str, tf: str, view: str = "research", *, source: Optional[str] = None,
         look_id: Optional[str] = None, spec_hash: Optional[str] = None,
         asof: Union[str, pd.Timestamp, None] = None, caller: Optional[str] = None) -> pd.DataFrame:
    """Bars indexed by open time, columns open,high,low,close,volume,turnover,funding; attrs
    {"coin","tf","view","source","cutoff","data_end"}. See module docstring for the views.

    `caller` is required for view='paper' (CONTRACTS §3); it is ignored by the research/full views."""
    if view not in VIEWS:
        raise ValueError("unknown view %r (expected one of %s)" % (view, VIEWS))
    dcfg = config.load_config()["data"]
    if source is None:
        source = dcfg["paper_source"] if view == "paper" else dcfg["research_source"]
    if view == "full":
        _check_look(look_id, spec_hash)  # before touching the file
    lo = None
    if view == "paper":
        caller = _paper_gate(caller, source, dcfg["paper_source"], look_id, spec_hash)  # before touching the file
        if tf not in _data.TF_DELTA:
            raise _data.DataError("unsupported tf %r" % tf)
        fs = forward_start()
        if caller == "paper":
            eff = _ts(asof) if asof is not None else _ts(config.utcnow())
            if eff >= cutoff():
                if fs is None:
                    raise VaultSealed("caller 'paper': forward_start is null, so asof %s >= cutoff is sealed TEST "
                                      "data (the operator sets forward.forward_start when paper starts)" % eff)
                if eff < fs - _data.TF_DELTA[tf]:
                    raise VaultSealed("caller 'paper': asof %s is before forward_start %s (sealed TEST period)"
                                      % (eff, fs))
        else:  # every other paper caller is windowed to the forward period (+ warmup)
            if fs is None:
                raise VaultSealed("caller %r: forward_start is null, so there is no forward period yet "
                                  "(everything >= cutoff is sealed TEST data)" % caller)
            lo = max(fs, cutoff()) - pd.Timedelta(days=warmup_days())
    raw, fund = _data.load_raw(coin, tf, source)
    if view == "research":
        raw = raw[raw.index < cutoff()]
    else:
        # closed bars only (files hold closed bars, but guard against a clock-skewed writer)
        now = _ts(config.utcnow())
        raw = raw[raw.index + _data.TF_DELTA[tf] <= now]
        if view == "paper" and asof is not None:
            raw = raw[raw.index <= _ts(asof)]
        if lo is not None:
            # windowed callers: only the warmup before the forward period plus the forward period itself;
            # the sealed TEST bars in [cutoff, forward_start - warmup) stay sealed. period_mask(...,'all') on
            # this frame starts at max(forward_start, cutoff) (start + warmup), i.e. it is the forward period.
            raw = raw[raw.index >= lo]
    out = _frame(raw, fund, coin=coin, tf=tf, view=view, source=source)
    if view == "paper":
        _log_access(out, caller=caller, coin=coin, tf=tf, source=source, asof=asof, look_id=look_id)
    if view == "research":
        if len(out) and not out.index.max() < cutoff():  # invariant; explicit so `python -O` keeps it
            raise VaultSealed("research view leaked a bar >= cutoff")
    return out


def load_full(coin: str, tf: str, *, look_id: str, spec_hash: Optional[str], source: Optional[str] = None
              ) -> pd.DataFrame:
    """Explicit entry to sealed data. Raises VaultSealed unless trials.check_look accepts (look_id, spec_hash)."""
    return load(coin, tf, "full", source=source, look_id=look_id, spec_hash=spec_hash)


def export_research(out_root, coins: Optional[Iterable[str]] = None, tfs: Optional[Iterable[str]] = None,
                    source: Optional[str] = None) -> int:
    """Write research-view CSVs (truncated at the cutoff) in the data-store layout under `out_root`, for
    shipping to sweep hosts (fleet/dispatch). Funding events are kept only up to the cutoff instant, so
    `vault.load(view="research")` with TRADER_DATA_DIR=out_root reproduces the local research view exactly
    and the remote host never holds a sealed bar. Returns the number of files written."""
    cfg = config.load_config()
    source = source or cfg["data"]["research_source"]
    coins = list(coins) if coins else list(cfg["universe"])
    tfs = list(tfs) if tfs else list(cfg["timeframes"])
    root = Path(out_root).expanduser()
    co = cutoff()
    n = 0
    for coin in coins:
        for tf in tfs:
            kp = _data.kline_path(coin, tf, source)
            if not kp.exists():
                continue
            df = _data.read_klines(kp)
            df = df[df.index < co]
            _data.write_klines(df, root / source / kp.name, tf)
            n += 1
        fp = _data.funding_path(coin)
        if source == "linear" and fp.exists():
            f = _data.read_funding(fp)
            # an event at exactly the cutoff instant settles the last research bar ((t-1ms).floor rule)
            _data.write_funding(f[f.index <= co], root / "linear" / fp.name)
            n += 1
    return n
