"""Daily LLM screener: typed questions per coin, answered by Claude Haiku, logged and later settled (Wave 2).

CONTRACTS.md section 9 "Screener".

    python -m trader.strategy.screener run [--coins BTC,ETH] [--dry-run] [--print-prompt]
    python -m trader.strategy.screener questions          # validate strategy/screener_questions.json

Flow of `run`:
  1. load + validate strategy/screener_questions.json (scored questions must be 1d; "scored": false questions are
     context only: asked, returned in the run result, never logged, never used by the veto);
  2. per coin, a compact snapshot from the PAPER view (caller "screener", closed bars only): last closes, returns,
     RSI(14), ADX(14), volume ratio, realised vol, distance to EMA50/200, last funding rate (live Bybit linear
     funding, informational only, null when unavailable);
  3. coins are batched (screener.coins_per_call, default 8) into one prompt per call with a strict JSON schema;
     calls go through `claude -p --model <screener.model> --output-format json` (or the Anthropic API when
     screener.invocation == "api" or the CLI is missing and ANTHROPIC_API_KEY is set). `call_model` is injectable
     and tests never call a model;
  4. the answer is parsed strictly (unknown coins/questions ignored, out-of-range values refused per answer) and
     each valid answer is logged through forecast.log_forecast (duplicates refused, so coins already answered for
     this asof_bar are not asked again);
  5. at most screener.max_calls_per_day model calls per UTC day (attempts are counted before calling, in
     <data root>/screener_calls.json, outside git).

--dry-run never writes anything: without an injected model it only builds the prompts; with one it also parses.

Python >= 3.9. Dependencies: numpy, pandas (+ requests only for the optional API / funding paths).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from trader.core import config
from trader.core import forecast

DEFAULT_COINS = ["BTC", "ETH", "SOL", "BNB"]
DEFAULT_COINS_PER_CALL = 8
CLI_TIMEOUT_S = 300
STALE_BARS = 2          # a coin whose last closed 1d bar is older than this many bars is skipped


class ScreenerError(Exception):
    pass


# ── questions ───────────────────────────────────────────────────────────────────────────────────────────────
def questions_path() -> Path:
    return config.strategy_dir() / "screener_questions.json"


def load_questions(path=None) -> Tuple[List[dict], List[dict]]:
    """(scored questions, context-only questions). Raises ScreenerError on any invalid question."""
    p = Path(path) if path else questions_path()
    if not p.exists():  # fall back to the shipped file when a test/strategy dir has none
        p = Path(__file__).resolve().parent / "screener_questions.json"
    doc = json.loads(p.read_text(encoding="utf-8"))
    qs = doc.get("questions")
    if not isinstance(qs, list) or not qs:
        raise ScreenerError("%s: no questions" % p)
    scored, ctx, errs, seen = [], [], [], set()
    for q in qs:
        qid = q.get("id")
        if qid in seen:
            errs.append("duplicate question id %r" % qid)
        seen.add(qid)
        if not isinstance(q.get("text"), str) or "{coin}" not in q["text"]:
            errs.append("%s: text must contain {coin}" % qid)
        if q.get("scored", True) is False:
            if q.get("kind") not in ("binary",):
                errs.append("%s: context questions must be binary" % qid)
            ctx.append(q)
            continue
        errs.extend(forecast.validate_question(q))
        if q.get("tf") != "1d":
            errs.append("%s: the screener asks daily questions only (tf must be 1d)" % qid)
        scored.append(q)
    if errs:
        raise ScreenerError("; ".join(errs))
    return scored, ctx


# ── features ────────────────────────────────────────────────────────────────────────────────────────────────
def _r(x, nd=4):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return round(x, nd) if np.isfinite(x) else None


def _sig(x, n=6):
    x = float(x)
    return float("%.*g" % (n, x)) if np.isfinite(x) else None


def load_bars(coin: str, asof=None) -> pd.DataFrame:
    from trader.core import vault
    return vault.load(coin, "1d", view="paper", caller="screener", asof=asof)


def live_funding(coin: str, asof_bar: pd.Timestamp) -> Optional[float]:
    """Latest Bybit linear funding rate settled up to the close of asof_bar (informational; spot pays none)."""
    from trader.core import data
    end = asof_bar + pd.Timedelta("1D")
    start_ms = int((end - pd.Timedelta("3D")).value // 10 ** 6)
    s = data.fetch_funding("%sUSDT" % coin, start_ms=start_ms)
    s = s[s.index <= end]
    return float(s.iloc[-1]) if len(s) else None


def features(coin: str, df: pd.DataFrame, funding: Optional[float] = None) -> dict:
    """Compact snapshot of closed 1d bars (df already ends at asof_bar)."""
    from trader.core import catalog
    if len(df) < 30:
        raise ScreenerError("%s: only %d closed daily bars visible" % (coin, len(df)))
    c = df["close"].astype(float).values
    d = {"c": c, "h": df["high"].astype(float).values, "l": df["low"].astype(float).values}
    rsi = catalog.rsi(c, 14)
    adx = catalog.adx(d, 14)
    vol = df["volume"].astype(float).values
    lr = np.diff(np.log(c))
    ema = lambda n: pd.Series(c).ewm(span=n, adjust=False).mean().values[-1]  # noqa: E731
    out = {"coin": coin, "asof_bar": config.iso(df.index[-1]), "last_close": _sig(c[-1]),
           "closes_last_7": [_sig(x) for x in c[-7:]],
           "ret_1d": _r(c[-1] / c[-2] - 1), "ret_7d": _r(c[-1] / c[-8] - 1) if len(c) > 8 else None,
           "ret_30d": _r(c[-1] / c[-31] - 1) if len(c) > 31 else None,
           "rsi14": _r(rsi[-1], 1), "adx14": _r(adx[-1], 1),
           "vol_ratio_1d_30d": _r(vol[-1] / np.mean(vol[-31:-1]), 2) if np.mean(vol[-31:-1]) > 0 else None,
           "realised_vol_30d_ann": _r(np.std(lr[-30:], ddof=1) * np.sqrt(365), 3),
           "dist_from_30d_high": _r(c[-1] / np.max(c[-30:]) - 1),
           "dist_from_ema50": _r(c[-1] / ema(50) - 1) if len(c) >= 50 else None,
           "dist_from_ema200": _r(c[-1] / ema(200) - 1) if len(c) >= 200 else None,
           "funding_rate_last": _r(funding, 6) if funding is not None else None}
    return out


# ── prompt / parsing ────────────────────────────────────────────────────────────────────────────────────────
def _answer_spec(q: dict) -> str:
    if q["kind"] == "choice":
        return '{"p": [%s]}  (probabilities for options %s, in that order, summing to 1)' % (
            ", ".join(["<0..1>"] * len(q["options"])), json.dumps(q["options"]))
    if q["kind"] == "score":
        return '{"score": <integer 0..100>}'
    return '{"p": <0..1 probability of YES>}'


def build_prompt(snapshots: List[dict], scored: List[dict], ctx: List[dict]) -> str:
    lines = [
        "You are a calibrated crypto forecaster for a paper-trading research system. Your forecasts are logged "
        "and scored with the Brier score against a climatology baseline; only well-calibrated probabilities help.",
        "Rules: use ONLY the data below plus general knowledge; the snapshot ends at the close of asof_bar "
        "(UTC daily bars, Bybit spot). When you have no edge, answer close to the historical base rate. "
        "Do not use tools. Reply with ONE JSON object and nothing else (no prose, no code fences).",
        "",
        "QUESTIONS (\"{coin}\" stands for each coin):",
    ]
    for q in scored:
        lines.append("- %s [%s]: %s -> answer %s" % (q["id"], q["kind"], q["text"], _answer_spec(q)))
    for q in ctx:
        lines.append("- %s [context]: %s -> answer {\"p\": <0..1>, \"note\": \"<=120 chars\"}" % (q["id"], q["text"]))
    example = {"forecasts": [{"coin": "<COIN>", "answers": {
        q["id"]: ({"p": [0.2] * len(q["options"])} if q["kind"] == "choice" else
                  {"score": 50} if q["kind"] == "score" else {"p": 0.5}) for q in scored + ctx}}]}
    for q in ctx:
        example["forecasts"][0]["answers"][q["id"]]["note"] = ""
    lines += ["", "OUTPUT SCHEMA (one entry per coin, every question answered):", json.dumps(example),
              "", "SNAPSHOTS:", json.dumps(snapshots, separators=(",", ":"))]
    return "\n".join(lines)


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def extract_json(text: str) -> dict:
    t = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", t, re.S)
    if m:
        t = m.group(1).strip()
    if not t.startswith("{"):
        i, j = t.find("{"), t.rfind("}")
        if i < 0 or j <= i:
            raise ScreenerError("model reply holds no JSON object")
        t = t[i:j + 1]
    try:
        obj = json.loads(t)
    except ValueError as e:
        raise ScreenerError("model reply is not valid JSON: %s" % e)
    if not isinstance(obj, dict):
        raise ScreenerError("model reply is not a JSON object")
    return obj


def parse_reply(text: str, coins: List[str], scored: List[dict], ctx: List[dict]) -> Tuple[dict, dict, List[str]]:
    """-> (answers {coin: {qid: p list}}, extras {"scores": {coin: {qid: raw 0-100}}, "context": {coin: {qid:
    {"p","note"}}}}, errors). Invalid single answers are dropped with an error; the rest are kept."""
    obj = extract_json(text)
    rows = obj.get("forecasts")
    if not isinstance(rows, list):
        raise ScreenerError("model reply lacks a 'forecasts' list")
    byq = {q["id"]: q for q in scored}
    cq = {q["id"] for q in ctx}
    answers: Dict[str, dict] = {}
    scores: Dict[str, dict] = {}
    context: Dict[str, dict] = {}
    errs: List[str] = []
    for row in rows:
        coin = row.get("coin") if isinstance(row, dict) else None
        if coin not in coins:
            errs.append("unexpected coin %r in reply" % (coin,))
            continue
        if coin in answers:
            errs.append("%s answered twice; first answer kept" % coin)
            continue
        ans = row.get("answers")
        if not isinstance(ans, dict):
            errs.append("%s: answers is not an object" % coin)
            continue
        answers[coin], scores[coin], context[coin] = {}, {}, {}
        for qid, a in ans.items():
            if qid in cq:
                if isinstance(a, dict) and isinstance(a.get("p"), (int, float)) and 0 <= a["p"] <= 1:
                    context[coin][qid] = {"p": float(a["p"]), "note": str(a.get("note", ""))[:200]}
                continue
            q = byq.get(qid)
            if q is None:
                errs.append("%s: unknown question %r" % (coin, qid))
                continue
            try:
                if not isinstance(a, dict):
                    raise forecast.ForecastError("answer is not an object")
                if q["kind"] == "score":
                    s = a.get("score")
                    if isinstance(s, bool) or not isinstance(s, (int, float)) or not 0 <= s <= 100:
                        raise forecast.ForecastError("score must be a number in 0..100, got %r" % (s,))
                    answers[coin][qid] = forecast.normalise_p(q, float(s) / 100.0)
                    scores[coin][qid] = s
                else:
                    answers[coin][qid] = forecast.normalise_p(q, a.get("p"))
            except forecast.ForecastError as e:
                errs.append("%s/%s: %s" % (coin, qid, e))
        for q in scored:
            if q["id"] not in answers[coin] and q["id"] not in ans:
                errs.append("%s/%s: missing answer" % (coin, q["id"]))
    for c in coins:
        if c not in answers:
            errs.append("%s: no answer in reply" % c)
    return answers, {"scores": scores, "context": context}, errs


# ── model invocation ────────────────────────────────────────────────────────────────────────────────────────
def call_claude_cli(prompt: str, model: str) -> str:
    exe = shutil.which("claude")
    if not exe:
        raise ScreenerError("claude CLI not found on PATH")
    base = [exe, "-p", "--model", model, "--output-format", "json"]
    cwd = tempfile.gettempdir()  # keep project CLAUDE.md / memory out of the screener's context
    for args in (base + ["--tools", ""], base):
        pr = subprocess.run(args, input=prompt, capture_output=True, text=True, timeout=CLI_TIMEOUT_S, cwd=cwd)
        if pr.returncode != 0 and "--tools" in args and "unknown option" in (pr.stderr or "").lower():
            continue  # older CLI without --tools
        break
    if pr.returncode != 0:
        raise ScreenerError("claude CLI exit %d: %s" % (pr.returncode, (pr.stderr or pr.stdout)[-400:]))
    try:
        env = json.loads(pr.stdout)
    except ValueError:
        return pr.stdout
    if isinstance(env, dict):
        if env.get("is_error"):
            raise ScreenerError("claude CLI error: %s" % str(env.get("result"))[:400])
        if isinstance(env.get("structured_output"), dict):
            return json.dumps(env["structured_output"])
        if isinstance(env.get("result"), str):
            return env["result"]
    return pr.stdout


def call_anthropic_api(prompt: str, model: str) -> str:
    import requests
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise ScreenerError("ANTHROPIC_API_KEY not set")
    r = requests.post("https://api.anthropic.com/v1/messages", timeout=CLI_TIMEOUT_S,
                      headers={"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
                      json={"model": model, "max_tokens": 8192, "messages": [{"role": "user", "content": prompt}]})
    if r.status_code != 200:
        raise ScreenerError("Anthropic API %d: %s" % (r.status_code, r.text[:400]))
    return "".join(b.get("text", "") for b in r.json().get("content", []) if b.get("type") == "text")


def default_call_model(prompt: str, model: str) -> str:
    inv = config.load_config().get("screener", {}).get("invocation", "claude_cli")
    if inv == "api" or (not shutil.which("claude") and os.environ.get("ANTHROPIC_API_KEY")):
        return call_anthropic_api(prompt, model)
    return call_claude_cli(prompt, model)


# ── call budget ─────────────────────────────────────────────────────────────────────────────────────────────
def _calls_path() -> Path:
    return config.data_dir() / "screener_calls.json"


def calls_today() -> int:
    p = _calls_path()
    try:
        return int(json.loads(p.read_text()).get(config.iso()[:10], 0))
    except (OSError, ValueError, AttributeError):
        return 0


def _count_call() -> int:
    p = _calls_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        st = json.loads(p.read_text())
        if not isinstance(st, dict):
            st = {}
    except (OSError, ValueError):
        st = {}
    day = config.iso()[:10]
    st = {k: v for k, v in st.items() if k >= (pd.Timestamp(day) - pd.Timedelta("14D")).strftime("%Y-%m-%d")}
    st[day] = int(st.get(day, 0)) + 1
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(st))
    os.replace(tmp, p)
    return st[day]


# ── run ─────────────────────────────────────────────────────────────────────────────────────────────────────
def default_coins() -> List[str]:
    p = config.strategy_dir() / "schedule.json"
    try:
        coins = json.loads(p.read_text()).get("screener", {}).get("coins")
        if isinstance(coins, list) and coins:
            return [str(c) for c in coins]
    except (OSError, ValueError, AttributeError):
        pass
    return list(DEFAULT_COINS)


def run(coins: Optional[List[str]] = None, *, dry_run: bool = False,
        call_model: Optional[Callable[[str, str], str]] = None, asof=None,
        funding_fn: Optional[Callable[[str, pd.Timestamp], Optional[float]]] = live_funding,
        coins_per_call: Optional[int] = None) -> dict:
    cfg = config.load_config()
    sc = cfg.get("screener", {})
    model = sc.get("model", "claude-haiku-5-5")
    max_calls = int(sc.get("max_calls_per_day", 60))
    per_call = int(coins_per_call or sc.get("coins_per_call", DEFAULT_COINS_PER_CALL))
    universe = set(cfg.get("universe", []))
    coins = list(dict.fromkeys(coins or default_coins()))
    scored, ctx = load_questions()
    res = {"ok": True, "dry_run": dry_run, "model": model, "calls": 0, "logged": [], "skipped": {},
           "errors": [], "context": {}, "prompts": [], "asof_bars": {}}
    if universe:
        bad = [c for c in coins if c not in universe]
        for c in bad:
            res["skipped"][c] = "not in universe"
        coins = [c for c in coins if c in universe]

    snaps, frames = {}, {}
    now = pd.Timestamp(config.iso()[:19])
    for c in coins:
        try:
            df = load_bars(c, asof=asof)
            if not len(df):
                raise ScreenerError("no closed bars")
            last = df.index[-1]
            ref = pd.Timestamp(asof) if asof is not None else now
            if ref - (last + pd.Timedelta("1D")) >= pd.Timedelta(days=STALE_BARS):
                raise ScreenerError("stale data: last closed bar %s" % last.date())
            todo = [q for q in scored if not forecast.exists(q["id"], c, last)]
            if not todo:
                res["skipped"][c] = "already forecast for %s" % last.date()
                continue
            fund = None
            if funding_fn is not None:
                try:
                    fund = funding_fn(c, last)
                except Exception:  # noqa: BLE001 - informational only
                    fund = None
            snaps[c] = features(c, df, fund)
            frames[c] = df
            res["asof_bars"][c] = config.iso(last)
        except Exception as e:  # noqa: BLE001
            res["skipped"][c] = "%s: %s" % (type(e).__name__, e)

    ready = list(snaps)
    chunks = [ready[i:i + per_call] for i in range(0, len(ready), per_call)]
    for chunk in chunks:
        prompt = build_prompt([snaps[c] for c in chunk], scored, ctx)
        ph = prompt_hash(prompt)
        if dry_run and call_model is None:
            res["prompts"].append({"coins": chunk, "prompt_hash": ph, "chars": len(prompt), "prompt": prompt})
            continue
        if not dry_run:
            if calls_today() >= max_calls:
                res["errors"].append("call budget exhausted (%d/day); not asked: %s" % (max_calls, chunk))
                continue
            _count_call()
        res["calls"] += 1
        try:
            text = (call_model or default_call_model)(prompt, model)
            answers, extras, errs = parse_reply(text, chunk, scored, ctx)
        except Exception as e:  # noqa: BLE001
            res["errors"].append("call for %s failed: %s: %s" % (chunk, type(e).__name__, e))
            continue
        res["errors"].extend(errs)
        for c in chunk:
            if extras["context"].get(c):
                res["context"][c] = extras["context"][c]
        if dry_run:
            res["prompts"].append({"coins": chunk, "prompt_hash": ph, "chars": len(prompt), "answers": answers})
            continue
        for c in chunk:
            asof_bar = frames[c].index[-1]
            for q in scored:
                p = answers.get(c, {}).get(q["id"])
                if p is None:
                    continue
                qq = dict(q)
                if q["id"] in extras["scores"].get(c, {}):
                    qq["_score"] = extras["scores"][c][q["id"]]
                try:
                    # the screener's windowed close series (forward_start - warmup_days .. asof) is passed on so
                    # the baseline never reads sealed TEST bars; baseline_n in the row records the sample size
                    fid = forecast.log_forecast(qq, c, asof_bar, p, model=model, prompt_hash=ph,
                                                close=frames[c]["close"])
                    res["logged"].append(fid)
                except forecast.DuplicateForecast:
                    res["skipped"]["%s/%s" % (c, q["id"])] = "duplicate"
                except Exception as e:  # noqa: BLE001
                    res["errors"].append("%s/%s: %s: %s" % (c, q["id"], type(e).__name__, e))
    res["ok"] = not res["errors"] or bool(res["logged"])
    return res


# ── CLI ─────────────────────────────────────────────────────────────────────────────────────────────────────
def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="python -m trader.strategy.screener")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--coins")
    r.add_argument("--dry-run", action="store_true")
    r.add_argument("--print-prompt", action="store_true")
    r.add_argument("--no-funding", action="store_true")
    sub.add_parser("questions")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "questions":
            scored, ctx = load_questions()
            out = {"ok": True, "scored": [q["id"] for q in scored], "context": [q["id"] for q in ctx]}
        else:
            coins = [c.strip().upper() for c in a.coins.split(",")] if a.coins else None
            out = run(coins, dry_run=a.dry_run, funding_fn=None if a.no_funding else live_funding)
            if a.print_prompt:
                for p in out["prompts"]:
                    sys.stderr.write(p.get("prompt", "") + "\n\n")
            for p in out["prompts"]:
                p.pop("prompt", None)
    except Exception as e:  # noqa: BLE001
        print(json.dumps({"ok": False, "error": "%s: %s" % (type(e).__name__, e)}))
        return 1
    print(json.dumps(out, default=str))
    return 0 if out.get("ok") else 1


if __name__ == "__main__":
    # run through the package module so the vault recognises trader.strategy.screener as the caller
    from trader.strategy import screener as _self
    sys.exit(_self.main())
