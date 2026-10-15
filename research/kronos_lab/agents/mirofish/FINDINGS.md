# MiroFish x Kronos: feasibility (key: mirofish)

## Method
Desk research only. I ran no experiment: MiroFish needs LLM and Zep Cloud API keys, and it cannot be backtested over our cache window without lookahead. Sources: the GitHub repo github.com/666ghj/MiroFish (README, plus the backend file tree via the GitHub API) and a practitioner evaluation (WPP Research, "MiroFish: Is swarm intelligence worth the cloud bill?", cms.research.wpp.com/?p=1700).

## What MiroFish is
- Repo stats: AGPL-3.0, created 2025-11, about 78.6k stars, last push 2026-10-01. There are no tagged releases, and the README still lists its "Financial Prediction" demo as "coming soon".
- Input: seed text (news, reports, "financial signals" as text) plus a natural-language prediction question.
- Pipeline: an LLM builds an ontology and a GraphRAG graph in Zep Cloud. It then generates agent personas, and the agents act on simulated Twitter and Reddit through CAMEL-AI's OASIS engine (`run_twitter_simulation.py` / `run_reddit_simulation.py`). Last, a ReportAgent writes a prose report.
- Output: a narrative report, a log of agent posts and actions, and the option to chat with individual agents. It produces no numeric forecast, no probability, and no price series.
- It does not ingest numbers. It cannot fetch data on its own, so the seed has to be hand-written or generated.
- Cost: no trained model. Cost is entirely LLM calls, roughly agents x rounds x actions; the README warns to stay under 40 rounds. The agent count comes from ontology entities and is not user-controllable (one tester wanted 100 agents and got about 25). The Zep free tier ran out after 3 runs. Runtime is minutes to tens of minutes per run.
- Maturity: in the WPP test, the Docker image was outdated, a hardcoded token limit crashed the backend, and Zep quota killed runs mid-simulation. On a same-day S&P direction call plus top-5 gainers, the swarm gave the same (wrong) answer as one plain LLM call and its ticker picks looked random. That is n=1, so it is anecdotal, but it is the only real evaluation I found.

## Fit with Kronos
| | Kronos | MiroFish |
|---|---|---|
| Input | 256 OHLCV candles | text seed |
| Output | sampled numeric price paths | prose plus simulated social posts |
| Cost per call | ~1 GPU forward pass (s) | thousands of LLM calls ($, min) |
| Backtestable | yes (we have 900 cached windows) | effectively no |

Why it cannot be backtested honestly:
1. Historical seeds would need point-in-time news, and our cache covers ~5 months x 3 symbols x every 12h = ~900 origins.
2. The LLM agents have likely seen what happened over the test period, which leaks the future into the forecast (lookahead).
3. ~900 runs x (tens to hundreds of LLM calls) means hours of runtime and real money, and the result would still be contaminated by point 2.

## Architecture sketch (if pursued anyway)
```
news/funding/OI/liq feed --> seed builder (LLM summary, point-in-time)
Kronos paths (6x24h)     --> numeric digest: median ret, P(up), path spread, max DD  --> appended to seed
                                   |
                            MiroFish swarm (~20-40 agents, <=10 rounds)
                                   |
             ReportAgent forced to emit JSON: {bias:-1..1, conviction, vol_regime, tail_risk_flag}
                                   |
  Strategy = indicator signal (BB/RSI etc.)  x  Kronos risk sizing  x  MiroFish veto (tail_risk_flag -> flat)
```
- Direction 1, Kronos seeds agent beliefs: put the Kronos path quantiles into the seed as "the quant desk forecast". Risk: the swarm mostly echoes whatever the seed says (anchoring), so this adds cost without adding information.
- Direction 2, MiroFish as scenario generator: have it produce discrete narrative scenarios (e.g. ETF outflow, hack, CPI surprise), map each one to a drift/vol shock, and apply the shock to the Kronos sampled paths to get a stress-test fan. This is the more defensible use. It is scenario planning for risk, not alpha, which matches the one area the WPP author thought plausible.
- Use it at most as a low-frequency (daily or event-driven) discretionary overlay or veto. Never use it as an hourly signal.

## Recommendation
Do not combine MiroFish with Kronos for trading signals now. MiroFish's output is non-numeric, expensive, slow, non-reproducible (LLM sampling) and impossible to backtest without leakage. The only published market test showed no signal over a single LLM call. If the user wants a "narrative" layer, a single LLM call that returns structured JSON sentiment from point-in-time news is roughly 100x cheaper, gives about the same information, and is the right baseline any swarm must beat. That too must be forward-tested (paper trading), not backtested.

A cheap, testable alternative to the MiroFish idea is a forward paper test: log 1 LLM-JSON sentiment call plus the Kronos digest per day for 60-90 days, and compare indicator-only vs indicator+Kronos vs indicator+Kronos+LLM-veto.

## Caveats
- I did not run MiroFish (no API keys, and I wanted to avoid cost and side effects).
- The evaluation evidence is n=1 from a single practitioner, and most other web coverage is promotional.
- The project changes fast. Hardcoded limits or the Zep dependency may be fixed by now.
