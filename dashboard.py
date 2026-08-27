"""
Daedalus dashboard.

Minimal Streamlit dashboard — functional, not polished, same lighter-weight
role as Atlas/Icarus's dashboards. Solves a real gap: the MCP tools only
work locally over stdio, so once Daedalus runs on Hetzner, this is the way
to check on it remotely via browser.

Reuses mcp_server/logic.py directly for every data read (the same
functions the MCP tools call), so this and the MCP tools stay consistent —
a fix to how events/trades are read only ever needs to happen once.

Run locally:
    streamlit run dashboard.py
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from alpaca.trading.client import TradingClient

from mcp_server import logic

# Load .env by explicit path, same reasoning as mcp_server/server.py:
# load_dotenv()'s default frame-based search can't be trusted to land on
# the repo root in every way this might get invoked (streamlit run, a
# systemd unit, etc).
REPO_ROOT = Path(__file__).resolve().parent
load_dotenv(dotenv_path=REPO_ROOT / ".env")

REFRESH_SECONDS = 30

st.set_page_config(page_title="Daedalus Dashboard", layout="wide")


@st.cache_resource
def get_trading_client():
    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")
    if not api_key or not secret_key:
        return None
    # paper=True hardcoded — same rule as agent/options_executor.py and
    # mcp_server/server.py: never configurable via ALPACA_BASE_URL.
    return TradingClient(api_key=api_key, secret_key=secret_key, paper=True)


trading_client = get_trading_client()

st.title("Daedalus")
st.caption("Event-driven options trading agent — live status")

# Fetched once, reused below by both the overview row and the Health /
# Open Positions sections — never fetched twice in the same run.
health = logic.get_health_status(trading_client)
open_positions = logic.get_open_positions(trading_client) if trading_client is not None else None

# -- Overview -------------------------------------------------------------

alpaca_detail = health["alpaca"].get("detail")

ov1, ov2, ov3, ov4 = st.columns(4)

with ov1:
    st.metric("Account Value", f"${alpaca_detail['equity']:,.2f}" if alpaca_detail else "n/a")

with ov2:
    st.metric("Cash", f"${alpaca_detail['cash']:,.2f}" if alpaca_detail and alpaca_detail.get("cash") is not None else "n/a")

with ov3:
    if alpaca_detail and alpaca_detail.get("day_pl_dollars") is not None:
        pct = alpaca_detail.get("day_pl_pct")
        st.metric(
            "Day P&L",
            f"${alpaca_detail['day_pl_dollars']:,.2f}",
            delta=f"{pct:.2f}%" if pct is not None else None,
        )
    else:
        st.metric("Day P&L", "n/a")

with ov4:
    st.metric("Open Positions", open_positions["count"] if open_positions is not None else "n/a")

# -- Health -------------------------------------------------------------

st.header("Health")
alpaca = health["alpaca"]

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Alpaca", "reachable" if alpaca["reachable"] else "unreachable")
    if alpaca.get("detail"):
        st.caption(
            f"Equity: ${alpaca['detail']['equity']:,.2f} | "
            f"Status: {alpaca['detail']['account_status']}"
        )
    elif alpaca.get("error"):
        st.caption(f"Error: {alpaca['error']}")
    elif not alpaca["api_key_configured"] or not alpaca["secret_key_configured"]:
        st.caption("ALPACA_API_KEY / ALPACA_SECRET_KEY not configured")

with col2:
    st.metric("Tavily key configured", "yes" if health["tavily"]["api_key_configured"] else "no")

with col3:
    st.metric("Anthropic key configured", "yes" if health["anthropic"]["api_key_configured"] else "no")

# Last activity = newest timestamp across either log, as a stand-in for
# "last poll time" — there's no separate poll-cycle marker in either log.
_latest_event = logic.read_jsonl(logic.EVENTS_LOG_PATH, n=1)
_latest_trade = logic.read_jsonl(logic.TRADES_LOG_PATH, n=1)
_candidates = [r.get("timestamp") for r in (_latest_event + _latest_trade) if r.get("timestamp")]
last_activity = max(_candidates) if _candidates else None
st.caption(f"Last activity: {last_activity or 'no data yet in this checkout'}")

# -- Open positions -------------------------------------------------------

st.header("Open Positions")
if open_positions is None:
    st.warning("ALPACA_API_KEY / ALPACA_SECRET_KEY not configured — can't fetch positions.")
elif open_positions["count"] == 0:
    st.write("No open positions.")
else:
    st.dataframe(pd.DataFrame(open_positions["positions"]), width="stretch")

# -- Recent signals -------------------------------------------------------

st.header("Recent Signals")
st.caption("Last 20 events detected, from data/events.jsonl")
signals = logic.get_recent_signals(n=20)
if signals["count"] == 0:
    st.write(signals["note"] or "No signals recorded yet.")
else:
    st.dataframe(pd.DataFrame(signals["signals"]), width="stretch")

# -- Recent trade decisions -------------------------------------------------

st.header("Recent Trade Decisions")
st.caption("Last 20 decisions, from data/trades.jsonl")
trade_records = logic.read_jsonl(logic.TRADES_LOG_PATH, n=20)
if not trade_records:
    st.write(
        "No trade decisions recorded yet — run "
        "`python -m agent.options_executor --once [--dry-run]`."
    )
else:
    rows = []
    for record in trade_records:
        event = record.get("event") or {}
        extra = record.get("extra") or {}
        rows.append({
            "ticker": event.get("ticker") or extra.get("ticker_resolved"),
            "decision_mode": record.get("decision_mode"),
            "decision_reason": record.get("decision_reason"),
            "traded": record.get("traded"),
            "dry_run": record.get("dry_run"),
            "order_id": record.get("order_id"),
            "timestamp": record.get("timestamp"),
        })
    st.dataframe(pd.DataFrame(rows), width="stretch")

# -- Auto-refresh -----------------------------------------------------------

st.divider()
refresh_col, caption_col = st.columns([1, 4])
with refresh_col:
    st.button("Refresh now")
with caption_col:
    st.caption(f"Auto-refreshing every {REFRESH_SECONDS}s")

time.sleep(REFRESH_SECONDS)
st.rerun()
