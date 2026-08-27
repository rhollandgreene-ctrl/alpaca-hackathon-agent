# Daedalus

Event-driven options trading agent built for the Alpaca AI Trading Agents Hackathon (Aug 28 – Sep 4, 2026).

> Targets a dedicated, fresh Alpaca **paper trading** account. No live/production credentials or accounts are used in this repo.

**Status:** functional — paper trading only, under active tuning ahead of the hackathon.

## Overview

Daedalus watches a configurable ticker watchlist (currently 52 names) for
earnings catalysts, plus a small registry of macro/economic events
(Jackson Hole, FOMC, CPI), and turns each occurrence into a structured
`Event` — ticker, description, a sentiment score, and a confidence score
derived from live news coverage (via Tavily), deduplicated and scored with
domain-aware keyword lexicons (earnings vs. macro read the same words
differently — "weak" is bearish for a stock but dovish/bullish for
macro).

A separate execution layer consumes those events and decides whether and
how to trade: no trade, a directional call/put, or a straddle — routed by
confidence and sentiment-magnitude thresholds, sized as a fraction of
account equity, and placed as real limit orders against the Alpaca
**paper** account (`paper=True` is hardcoded, never configurable). The
core guardrail is **no pre-event bets**: an event is only eligible to
trade once the underlying occurrence has actually happened (a confirmed
earnings surprise, or a macro date that has passed) — a merely-scheduled
"upcoming" event is never eligible, at any confidence or sentiment.

Two pieces of production-mindedness worth calling out:
- **Data-source resilience**: yfinance's earnings-calendar lookup scrapes
  a live Yahoo Finance page rather than calling a stable API, and can
  silently drop or shift a ticker's row between polls. Daedalus tracks
  each ticker's last-seen earnings date, logs loudly on any drift, and
  cross-checks a more stable (if less detailed) calendar endpoint to
  recover a ticker that would otherwise vanish from detection.
- **Quota-aware polling**: news searches are cached per ticker per day,
  so a continuously-running instance searches each active ticker once a
  day rather than once per poll cycle.

## Setup

```bash
git clone https://github.com/rhollandgreene-ctrl/alpaca-hackathon-agent.git
cd alpaca-hackathon-agent
python -m venv .venv
.venv/Scripts/activate   # or: source .venv/bin/activate on macOS/Linux
pip install -r requirements.txt
cp .env.example .env     # then fill in ALPACA_API_KEY, ALPACA_SECRET_KEY,
                          # ANTHROPIC_API_KEY, TAVILY_API_KEY
```

Run a single detection pass (writes to `data/events.jsonl`):

```bash
python -m agent.event_detector --once
```

Run a single detect-and-decide pass without submitting any order
(writes to `data/trades.jsonl`):

```bash
python -m agent.options_executor --once --dry-run
```

Drop `--dry-run` to actually place orders against the paper account, and
drop `--once` on either command to run continuously on a poll loop. See
[docs/deployment.md](docs/deployment.md) for how this runs as an
always-on systemd service.

Run the test suite:

```bash
pytest
```

## MCP Tools

An MCP server (`mcp_server/server.py`) exposes 8 tools for Claude to query
and explain the system live during the demo — real event history, real
trade decisions with full reasoning and sizing math, live Alpaca account
state, a market-regime read, and a hand-maintained changelog of
calibration fixes made along the way. One tool (`get_icarus_signals`) is
explicitly mocked, since Icarus is a separate system not included in this
repo. Every tool's payload carries an `is_mocked` field so this is never
ambiguous from raw output alone.

See [docs/mcp_tools.md](docs/mcp_tools.md) for the full reference, the
real-vs-mocked table, and Claude Desktop setup instructions.

## License

MIT — see [LICENSE](LICENSE).
