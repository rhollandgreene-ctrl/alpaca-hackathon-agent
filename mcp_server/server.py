"""
Daedalus MCP server.

Exposes tools for Claude (via Claude Desktop or any MCP client) to query
and explain what the event detector / options executor are doing —
built for live interactive questioning during the hackathon demo.

See docs/mcp_tools.md for the full real-vs-mocked breakdown of every tool.
Every tool's return payload carries an explicit "is_mocked" field so this
is never ambiguous even from raw output alone.

Run directly for local testing:
    python -m mcp_server.server
Or via the MCP CLI:
    mcp dev mcp_server/server.py
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

# Load .env by explicit path — this process is spawned directly by Claude
# Desktop, so load_dotenv()'s default frame-based search can't be trusted
# to land on the repo root the way it would from an interactive shell.
REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=REPO_ROOT / ".env")

from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.trading.client import TradingClient

from mcp_server import logic

mcp = FastMCP("daedalus")

_ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
_ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

# paper=True hardcoded — same rule as agent/options_executor.py: this
# server never points at anything but the paper account, regardless of
# what ALPACA_BASE_URL says in .env.
_trading_client: Optional[TradingClient] = None
_stock_data_client: Optional[StockHistoricalDataClient] = None
if _ALPACA_API_KEY and _ALPACA_SECRET_KEY:
    _trading_client = TradingClient(api_key=_ALPACA_API_KEY, secret_key=_ALPACA_SECRET_KEY, paper=True)
    _stock_data_client = StockHistoricalDataClient(api_key=_ALPACA_API_KEY, secret_key=_ALPACA_SECRET_KEY)


@mcp.tool()
def get_market_regime() -> dict:
    """Current market regime (trending_up/trending_down/choppy/high_vol),
    computed from real recent SPY daily bars via Alpaca. This is a crude
    heuristic (realized volatility + directional consistency vs.
    configurable thresholds), not a sophisticated regime model — but it
    is real, live-computed data, not a mock."""
    if _stock_data_client is None:
        return {"is_mocked": False, "error": "ALPACA_API_KEY/ALPACA_SECRET_KEY not configured"}
    return logic.compute_market_regime(_stock_data_client)


@mcp.tool()
def get_open_positions() -> dict:
    """Current open paper-account options/equity positions, via a live
    Alpaca API call. Real data — will be an empty list if nothing is
    currently open."""
    if _trading_client is None:
        return {"is_mocked": False, "error": "ALPACA_API_KEY/ALPACA_SECRET_KEY not configured"}
    return logic.get_open_positions(_trading_client)


@mcp.tool()
def get_recent_signals(n: int = 10) -> dict:
    """The last n Event objects detected by the event detector, read from
    this project's own local log (data/events.jsonl). Real data — empty
    if the event detector hasn't been run yet in this checkout."""
    return logic.get_recent_signals(n=n)


@mcp.tool()
def run_bear_case(ticker: Optional[str] = None, event_type: Optional[str] = None) -> dict:
    """Returns real trade/event context (most recent match for the given
    ticker/event_type, or the most recent decision overall) plus a
    generic risk-factor checklist (IV crush, thesis-priced-in, reversal
    risk, macro correlation, confidence miscalibration, liquidity). This
    is a reasoning aid, not a finished argument — the calling model
    should synthesize the actual bear case from this material."""
    return logic.run_bear_case(ticker=ticker, event_type=event_type)


@mcp.tool()
def get_lessons_learned() -> dict:
    """Real calibration fixes made this week on Daedalus, read from the
    hand-maintained docs/lessons_learned.json (not hardcoded here, so it
    stays current without a code change)."""
    return logic.get_lessons_learned()


@mcp.tool()
def get_health_status() -> dict:
    """Real configuration/reachability checks: whether Alpaca and Tavily
    keys are present, and a live Alpaca get_account() call to confirm
    reachability. Tavily gets a key-presence check only (no live search
    fired) to avoid burning quota on repeated health checks."""
    return logic.get_health_status(_trading_client)


@mcp.tool()
def get_icarus_signals() -> dict:
    """MOCKED. Icarus is a separate production system not included in
    this repo. Returns an illustrative example of the payload shape a
    live integration would return — every value is fabricated, not a
    real signal. Check is_mocked (always true here) before treating this
    as live data."""
    return logic.get_icarus_signals()


@mcp.tool()
def explain_trade_decision(ticker: Optional[str] = None, order_id: Optional[str] = None) -> dict:
    """Walks through why a specific past trade decision fired (or didn't):
    which event triggered it, the confidence/sentiment/move-check values
    it cleared, which mode (directional/straddle/no-trade/skipped) was
    chosen and why, and the exact sizing math. Reads real records from
    this project's local log (data/trades.jsonl). If neither ticker nor
    order_id is given, returns the single most recent decision of any
    kind (traded, skipped, or no-trade)."""
    return logic.explain_trade_decision(ticker=ticker, order_id=order_id)


if __name__ == "__main__":
    mcp.run(transport="stdio")
