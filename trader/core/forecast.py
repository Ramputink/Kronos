"""Forecast ledger, settlement and Brier scoring for the screener (Wave 2, screener area).

CONTRACTS.md section 9 "Screener". Sole writer of `journal/forecasts.jsonl` (append-only, two row types):

  {"type":"forecast", "forecast_id", "ts", "asof_bar", "question_id", "coin", "kind", "options", "p",
   "baseline_p", "baseline", "model", "prompt_hash", "horizon_bars", "resolves_at_bar", "tf", "resolver",
   "baseline_n", "lag_s", ["score"]}
  {"type":"resolve", "forecast_id", "ts", "outcome", "brier", "brier_baseline", "brier_uniform", "realised"}
  (a resolve row with "void": true and outcome null is written when the resolution bar is missing from the data;
   void rows are excluded from every score)

Honesty rules implemented here
  * every forecast row carries a frozen copy of its resolver, tf and horizon, so later edits of
    strategy/screener_questions.json can never change how an old forecast is settled;
  * a forecast must be logged while the bar after `asof_bar` is still forming (lag < 1 bar after asof closed);
    anything later is refused (the model could have seen part of the outcome);
  * the baseline is computed from outcomes that were fully known at `asof_bar` (no lookahead): base_rate_365d =
    class frequencies of windows that start in the 365 days before asof and END at or before asof;
  * one forecast per (question_id, coin, asof_bar): duplicates are refused (no re-rolling the model);
  * the veto gate opens only when the screener beats BOTH its climatology baseline and the uniform (0.5) baseline
    over >= screener.veto_min_forward_days distinct forward days and >= screener.veto_min_resolved_forecasts.

Data access: vault.load(view="paper") with the implicit caller "paper" (this module is a PAPER_MODULE).

Python >= 3.9. Dependencies: numpy, pandas.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from trader.core import config

KINDS = ("binary", "choice", "score")
RESOLVERS = ("close_gt", "ret_gt", "ret_lt", "ret_bucket", "vol_gt")
BASELINES = ("base_rate_365d", "uniform")
BINARY_OPTIONS = ["no", "yes"]
TF_DELTA = {"1d": pd.Timedelta("1D"), "4h": pd.Timedelta("4h")}
MIN_BASELINE_SAMPLES = 30   # fewer complete windows than this -> uniform baseline (recorded as "uniform_fallback")
SUM_TOL = 0.02              # probabilities summing within 1 +- SUM_TOL are renormalised; beyond that refused


class ForecastError(ValueError):
    """Invalid forecast (bad probabilities, unknown question shape, late logging, ...)."""


class DuplicateForecast(ForecastError):
    """A forecast for (question_id, coin, asof_bar) already exists."""


# ── paths / small helpers ───────────────────────────────────────────────────────────────────────────────────
def forecasts_path() -> Path:
    return config.journal_dir() / "forecasts.jsonl"


def _ts(x) -> pd.Timestamp:
    t = pd.Timestamp(x)
    if t.tzinfo is not None:
        t = t.tz_convert("UTC").tz_localize(None)
    return t


def _now() -> pd.Timestamp:
    return _ts(config.utcnow())


def _bpy(tf: str) -> int:
    return int(config.load_config().get("bars_per_year", {}).get(tf, {"1d": 365, "4h": 2190}[tf]))


def n_classes(question: dict) -> int:
    if question["kind"] == "choice":
        return len(question["options"])
    return 2


def options_of(question: dict) -> List[str]:
    return list(question["options"]) if question["kind"] == "choice" else list(BINARY_OPTIONS)


def validate_question(q: dict) -> List[str]:
    """Shape check for one scored question (CONTRACTS section 9). Returns a list of errors (empty = ok)."""
    errs = []
    qid = q.get("id")
    if not isinstance(qid, str) or not qid:
        return ["question without id"]
    if q.get("kind") not in KINDS:
        errs.append("%s: kind must be one of %s" % (qid, KINDS))
    if q.get("tf") not in TF_DELTA:
        errs.append("%s: tf must be one of %s" % (qid, tuple(TF_DELTA)))
    h = q.get("horizon_bars")
    if not isinstance(h, int) or isinstance(h, bool) or h < 1:
        errs.append("%s: horizon_bars must be an int >= 1" % qid)
    r = q.get("resolver") or {}
    rt = r.get("type")
    if rt not in RESOLVERS:
        errs.append("%s: resolver.type must be one of %s" % (qid, RESOLVERS))
    if rt in ("ret_gt", "ret_lt", "vol_gt") and not isinstance(r.get("x"), (int, float)):
        errs.append("%s: resolver %s needs numeric x" % (qid, rt))
    if rt == "vol_gt" and isinstance(h, int) and h < 2:
        errs.append("%s: vol_gt needs horizon_bars >= 2" % qid)
    if q.get("kind") == "choice":
        if rt != "ret_bucket":
            errs.append("%s: choice questions resolve with ret_bucket" % qid)
        edges = r.get("edges")
        opts = q.get("options")
        if not isinstance(edges, list) or not edges or sorted(edges) != edges:
            errs.append("%s: ret_bucket edges must be a sorted non-empty list" % qid)
        elif not isinstance(opts, list) or len(opts) != len(edges) + 1:
            errs.append("%s: choice needs len(options) == len(edges) + 1" % qid)
    elif rt == "ret_bucket":
        errs.append("%s: ret_bucket is only valid for choice questions" % qid)
    if q.get("baseline", "base_rate_365d") not in BASELINES:
        errs.append("%s: baseline must be one of %s" % (qid, BASELINES))
    return errs


def normalise_p(question: dict, p) -> List[float]:
    """Scalar (binary/score: probability of 'yes') or list of class probabilities -> validated list."""
    k = n_classes(question)
    if isinstance(p, (int, float)) and not isinstance(p, bool):
        if question["kind"] == "choice":
            raise ForecastError("%s: choice forecast needs a list of %d probabilities" % (question["id"], k))
        p = [1.0 - float(p), float(p)]
    if not isinstance(p, (list, tuple)) or len(p) != k:
        raise ForecastError("%s: expected %d probabilities, got %r" % (question["id"], k, p))
    vals = []
    for x in p:
        if isinstance(x, bool) or not isinstance(x, (int, float)) or not np.isfinite(x) or x < -1e-9 or x > 1 + 1e-9:
            raise ForecastError("%s: probability out of [0,1]: %r" % (question["id"], p))
        vals.append(min(1.0, max(0.0, float(x))))
    s = sum(vals)
    if abs(s - 1.0) > SUM_TOL:
        raise ForecastError("%s: probabilities sum to %.4f, not 1" % (question["id"], s))
    return [round(v / s, 6) for v in vals]


def brier(p: Sequence[float], outcome: int) -> float:
    """Multi-class Brier sum_i (p_i - 1[i==outcome])^2 (same formula as score.brier, kept local so this module
    has no numba import on the settlement path)."""
    return float(sum((float(pi) - (1.0 if i == int(outcome) else 0.0)) ** 2 for i, pi in enumerate(p)))


# ── outcomes (shared by settlement and the baseline, so both use exactly the same rule) ───────────────────
def _grid(close: pd.Series, tf: str) -> pd.Series:
    close = close[~close.index.duplicated(keep="last")].sort_index()
    if not len(close):
        return close
    idx = pd.date_range(close.index[0], close.index[-1], freq=TF_DELTA[tf])
    return close.reindex(idx)


def outcome_series(close: pd.Series, resolver: dict, horizon: int, tf: str) -> Tuple[pd.Series, pd.Series]:
    """(outcome, realised) indexed by window START bar t: the outcome of a forecast made at the close of bar t and
    resolved at the close of bar t + horizon. NaN where a needed bar is missing. `realised` is the return
    (or the realised annual vol for vol_gt)."""
    g = _grid(close.astype(float), tf)
    fut = g.shift(-horizon)
    ret = fut / g - 1.0
    rt = resolver["type"]
    if rt == "vol_gt":
        lr = np.log(g / g.shift(1))
        vol = lr.rolling(horizon, min_periods=horizon).std(ddof=1).shift(-horizon) * np.sqrt(_bpy(tf))
        real = vol
        out = (vol > float(resolver["x"])).astype(float)
    else:
        real = ret
        if rt == "close_gt":
            out = (fut > g).astype(float)
        elif rt == "ret_gt":
            out = (ret > float(resolver["x"])).astype(float)
        elif rt == "ret_lt":
            out = (ret < float(resolver["x"])).astype(float)
        elif rt == "ret_bucket":
            edges = np.asarray(resolver["edges"], dtype=float)
            out = pd.Series(np.searchsorted(edges, ret.values, side="right").astype(float), index=ret.index)
        else:
            raise ForecastError("unknown resolver %r" % rt)
    out[real.isna()] = np.nan
    return out, real


def baseline_probs(close: pd.Series, question: dict, asof_bar) -> Tuple[List[float], str, int]:
    """(baseline_p, baseline name actually used, n windows). base_rate_365d uses only windows fully closed by
    asof_bar: start t in [asof - 365d, asof - horizon bars]."""
    k = n_classes(question)
    uni = [round(1.0 / k, 6)] * k
    if question.get("baseline", "base_rate_365d") == "uniform":
        return uni, "uniform", 0
    tf, h = question["tf"], int(question["horizon_bars"])
    a = _ts(asof_bar)
    hist = close[close.index <= a]
    out, _ = outcome_series(hist, question["resolver"], h, tf)
    lo, hi = a - pd.Timedelta(days=365), a - h * TF_DELTA[tf]
    w = out[(out.index >= lo) & (out.index <= hi)].dropna()
    if len(w) < MIN_BASELINE_SAMPLES:
        return uni, "uniform_fallback", int(len(w))
    counts = np.bincount(w.astype(int).values, minlength=k)[:k].astype(float)
    p = counts / counts.sum()
    return [round(float(x), 6) for x in p], "base_rate_365d", int(len(w))


# ── data access ─────────────────────────────────────────────────────────────────────────────────────────────
def _load_close(coin: str, tf: str, asof=None) -> pd.Series:
    from trader.core import vault  # lazy: keeps import light for the CLI summary
    df = vault.load(coin, tf, view="paper", asof=asof)  # implicit caller "paper" (PAPER_MODULES)
    return df["close"]


# ── ledger ──────────────────────────────────────────────────────────────────────────────────────────────────
def read_ledger() -> Tuple[Dict[str, dict], Dict[str, dict]]:
    """(forecasts by id, resolves by forecast id)."""
    fc, rs = {}, {}
    for r in config.jsonl_read(forecasts_path()):
        if r.get("type") == "forecast" and r.get("forecast_id"):
            fc.setdefault(r["forecast_id"], r)
        elif r.get("type") == "resolve" and r.get("forecast_id"):
            rs.setdefault(r["forecast_id"], r)
    return fc, rs


def forecast_id(question_id: str, coin: str, asof_bar, tf: str) -> str:
    d = (_ts(asof_bar) + TF_DELTA[tf])  # the forecast is made at the close of asof_bar
    stamp = d.strftime("%Y%m%d") if tf == "1d" else d.strftime("%Y%m%dT%H")
    return "F-%s-%s-%s" % (stamp, coin, question_id)


def exists(question_id: str, coin: str, asof_bar) -> bool:
    a = config.iso(_ts(asof_bar))
    for r in config.jsonl_read(forecasts_path()):
        if r.get("type") == "forecast" and r.get("question_id") == question_id and r.get("coin") == coin \
                and r.get("asof_bar") == a:
            return True
    return False


def log_forecast(question: dict, coin: str, asof_bar, p, *, model: str, prompt_hash: str,
                 close: Optional[pd.Series] = None) -> str:
    """Validate and append one forecast row; computes baseline_p. Returns the forecast_id.

    `p`: list of class probabilities (binary/score order ["no","yes"]) or, for binary/score, the probability
    of "yes". For kind "score" the model's 0-100 score must be converted by the caller (p = score/100); the raw
    score may be passed inside question["_score"] to be recorded.
    `close`: optional preloaded close series (paper view, <= asof); loaded through the vault when omitted."""
    errs = validate_question(question)
    if errs:
        raise ForecastError("; ".join(errs))
    tf, h = question["tf"], int(question["horizon_bars"])
    a = _ts(asof_bar)
    if a != a.floor(TF_DELTA[tf]):
        raise ForecastError("asof_bar %s is not a %s bar open time" % (a, tf))
    now = _now()
    asof_close = a + TF_DELTA[tf]
    if now < asof_close:
        raise ForecastError("asof_bar %s has not closed yet (now %s)" % (a, now))
    if now >= asof_close + TF_DELTA[tf]:
        raise ForecastError("late forecast: asof_bar %s closed at %s, now %s is >= 1 bar later (the outcome may "
                            "already be partly known)" % (a, asof_close, now))
    probs = normalise_p(question, p)
    if exists(question["id"], coin, a):
        raise DuplicateForecast("forecast for (%s, %s, %s) already logged" % (question["id"], coin, config.iso(a)))
    if close is None:
        close = _load_close(coin, tf, asof=a)
    close = close[close.index <= a]
    if not len(close) or close.index[-1] != a:
        raise ForecastError("%s %s: asof_bar %s is not the last closed bar of the data" % (coin, tf, a))
    bp, bname, bn = baseline_probs(close, question, a)
    fid = forecast_id(question["id"], coin, a, tf)
    row = {"type": "forecast", "forecast_id": fid, "ts": config.iso(now), "asof_bar": config.iso(a),
           "question_id": question["id"], "coin": coin, "kind": question["kind"], "options": options_of(question),
           "p": probs, "baseline_p": bp, "baseline": bname, "baseline_n": bn, "model": model,
           "prompt_hash": prompt_hash, "horizon_bars": h, "tf": tf,
           "resolves_at_bar": config.iso(a + h * TF_DELTA[tf]), "resolver": dict(question["resolver"]),
           "lag_s": int((now - asof_close).total_seconds())}
    if question.get("_score") is not None:
        row["score"] = question["_score"]
    config.jsonl_append(forecasts_path(), row)
    return fid


def resolve_due(asof=None) -> List[dict]:
    """Append resolve rows for every unresolved forecast whose resolves_at_bar is <= the last closed bar
    (as of `asof`, default now). Returns the rows written."""
    fc, rs = read_ledger()
    pending = [f for fid, f in fc.items() if fid not in rs]
    if not pending:
        return []
    cap = _ts(asof) if asof is not None else None
    groups: Dict[Tuple[str, str], List[dict]] = {}
    for f in pending:
        groups.setdefault((f["coin"], f.get("tf", "1d")), []).append(f)
    written = []
    for (coin, tf), rows in sorted(groups.items()):
        due_at = min(_ts(f["resolves_at_bar"]) for f in rows)
        if cap is not None and due_at > cap:
            continue
        try:
            close = _load_close(coin, tf, asof=cap)
        except Exception:  # noqa: BLE001 - missing data / sealed: try again next cycle
            continue
        if not len(close):
            continue
        last = close.index[-1]
        for f in sorted(rows, key=lambda r: r["forecast_id"]):
            t0, t1 = _ts(f["asof_bar"]), _ts(f["resolves_at_bar"])
            if t1 > last:
                continue
            out, real = outcome_series(close[close.index <= t1], f["resolver"], int(f["horizon_bars"]), tf)
            o = out.get(t0, np.nan)
            row = {"type": "resolve", "forecast_id": f["forecast_id"], "ts": config.iso()}
            if not np.isfinite(o):
                row.update({"outcome": None, "void": True, "reason": "missing bar at asof or resolution"})
            else:
                o = int(o)
                k = len(f["p"])
                row.update({"outcome": o, "brier": round(brier(f["p"], o), 6),
                            "brier_baseline": round(brier(f["baseline_p"], o), 6),
                            "brier_uniform": round(brier([1.0 / k] * k, o), 6),
                            "realised": round(float(real.get(t0)), 6)})
            config.jsonl_append(forecasts_path(), row)
            written.append(row)
    return written


# ── scoring ─────────────────────────────────────────────────────────────────────────────────────────────────
def _settled(since=None, question_id=None) -> List[Tuple[dict, dict]]:
    fc, rs = read_ledger()
    lo = _ts(since) if since is not None else None
    out = []
    for fid, r in rs.items():
        f = fc.get(fid)
        if f is None or r.get("void") or r.get("outcome") is None:
            continue
        if question_id is not None and f["question_id"] != question_id:
            continue
        if lo is not None and _ts(f["asof_bar"]) < lo:
            continue
        out.append((f, r))
    return out


def _summ(pairs: List[Tuple[dict, dict]]) -> dict:
    n = len(pairs)
    if n == 0:
        return {"n": 0, "brier": None, "brier_baseline": None, "brier_uniform": None, "skill": None,
                "skill_uniform": None, "days": 0}
    b = float(np.mean([r["brier"] for _, r in pairs]))
    bb = float(np.mean([r["brier_baseline"] for _, r in pairs]))
    bu = float(np.mean([r.get("brier_uniform", brier([1.0 / len(f["p"])] * len(f["p"]), r["outcome"]))
                        for f, r in pairs]))
    days = len({f["asof_bar"][:10] for f, _ in pairs})
    return {"n": n, "brier": round(b, 6), "brier_baseline": round(bb, 6), "brier_uniform": round(bu, 6),
            "skill": round(1 - b / bb, 6) if bb > 0 else None,
            "skill_uniform": round(1 - b / bu, 6) if bu > 0 else None, "days": days}


def brier_summary(*, since=None, question_id=None) -> dict:
    """{"n","brier","brier_baseline","brier_uniform","skill","skill_uniform","days"} (+ "by_question" when
    question_id is None). Skill = 1 - brier / brier_baseline (> 0 means better than the baseline)."""
    pairs = _settled(since, question_id)
    out = _summ(pairs)
    if question_id is None:
        by: Dict[str, list] = {}
        for f, r in pairs:
            by.setdefault(f["question_id"], []).append((f, r))
        out["by_question"] = {q: _summ(v) for q, v in sorted(by.items())}
    return out


def veto_status(question_id: Optional[str] = None) -> dict:
    """Why the veto is (not) allowed. Counts only forecasts whose asof_bar is in the forward period."""
    cfg = config.load_config()
    sc = cfg.get("screener", {})
    min_days = int(sc.get("veto_min_forward_days", 30))
    min_n = int(sc.get("veto_min_resolved_forecasts", 100))
    fs = cfg.get("forward", {}).get("forward_start")
    reasons = []
    if not fs:
        return {"allowed": False, "reasons": ["forward.forward_start is null"], "summary": _summ([]),
                "question_id": question_id}
    s = brier_summary(since=fs, question_id=question_id)
    if s["n"] < min_n:
        reasons.append("resolved forecasts %d < %d" % (s["n"], min_n))
    if s["days"] < min_days:
        reasons.append("settled forward days %d < %d" % (s["days"], min_days))
    if s["n"]:
        if not s["brier"] < s["brier_baseline"]:
            reasons.append("brier %.4f does not beat baseline %.4f" % (s["brier"], s["brier_baseline"]))
        if not s["brier"] < s["brier_uniform"]:
            reasons.append("brier %.4f does not beat uniform %.4f" % (s["brier"], s["brier_uniform"]))
    s.pop("by_question", None)
    return {"allowed": not reasons, "reasons": reasons, "summary": s, "question_id": question_id,
            "thresholds": {"veto_min_forward_days": min_days, "veto_min_resolved_forecasts": min_n}}


def veto_allowed(question_id: Optional[str] = None) -> bool:
    """True only when the screener (overall, or one question) beats the climatology AND uniform baselines
    over >= veto_min_forward_days settled forward days and >= veto_min_resolved_forecasts forecasts."""
    try:
        return bool(veto_status(question_id)["allowed"])
    except Exception:  # noqa: BLE001 - fail closed
        return False


# ── CLI ─────────────────────────────────────────────────────────────────────────────────────────────────────
def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="python -m trader.core.forecast")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("resolve")
    r.add_argument("--asof")
    s = sub.add_parser("summary")
    s.add_argument("--since")
    s.add_argument("--question")
    v = sub.add_parser("veto")
    v.add_argument("--question")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "resolve":
            rows = resolve_due(a.asof)
            out = {"resolved": len(rows), "rows": rows}
        elif a.cmd == "summary":
            out = brier_summary(since=a.since, question_id=a.question)
        else:
            out = veto_status(a.question)
    except Exception as e:  # noqa: BLE001
        print(json.dumps({"ok": False, "error": "%s: %s" % (type(e).__name__, e)}))
        return 1
    print(json.dumps(out, default=str))
    return 0


if __name__ == "__main__":
    # run through the package module so the vault recognises trader.core.forecast as the caller
    from trader.core import forecast as _self
    sys.exit(_self.main())
