"""
Pure(ish) logic behind each MCP tool, kept separate from the MCP protocol
plumbing in server.py so it's directly unit-testable — call these
functions with fake clients/fixture files, no MCP client/session needed.

Every return dict carries an explicit "is_mocked" field so a caller (or a
human skimming raw tool output during the demo) can tell real data from
illustrative data without reading this file or docs/mcp_tools.md.
"""

from __future__ import annotations

import json
import os
import statistics
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from agent.event_detector import EVENTS_LOG_PATH
from agent.options_executor import MACRO_PROXY_TICKER, TRADES_LOG_PATH

REPO_ROOT = Path(__file__).resolve().parent.parent
LESSONS_LEARNED_PATH = REPO_ROOT / "docs" / "lessons_learned.json"

# -- market regime heuristic (real, computed from live Alpaca bars) ---------

REGIME_LOOKBACK_DAYS = 10
REGIME_HIGH_VOL_ANNUALIZED_PCT = 20.0
REGIME_TREND_MOVE_PCT = 3.0
REGIME_TREND_CONSISTENCY_RATIO = 0.6


# ---------------------------------------------------------------------------
# Shared JSONL helper

def read_jsonl(path: Path, n: int = 0) -> list[dict]:
    """Return up to the last n records (n=0 means all), newest first."""
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if n:
        records = records[-n:]
    return list(reversed(records))


# ---------------------------------------------------------------------------
# get_recent_signals — real, reads data/events.jsonl

def get_recent_signals(n: int = 10) -> dict:
    records = read_jsonl(EVENTS_LOG_PATH, n=n)
    return {
        "is_mocked": False,
        "source": str(EVENTS_LOG_PATH),
        "count": len(records),
        "signals": records,
        "note": None if records else (
            "No signals recorded yet in this checkout -- run "
            "'python -m agent.event_detector --once' to populate this."
        ),
    }


# ---------------------------------------------------------------------------
# get_open_positions — real, live Alpaca call

def get_open_positions(trading_client) -> dict:
    positions = trading_client.get_all_positions()
    return {
        "is_mocked": False,
        "count": len(positions),
        "positions": [
            {
                "symbol": p.symbol,
                "qty": float(p.qty),
                "side": p.side.value if hasattr(p.side, "value") else str(p.side),
                "avg_entry_price": float(p.avg_entry_price) if p.avg_entry_price is not None else None,
                "current_price": float(p.current_price) if p.current_price is not None else None,
                "market_value": float(p.market_value) if p.market_value is not None else None,
                "unrealized_pl": float(p.unrealized_pl) if p.unrealized_pl is not None else None,
                "unrealized_plpc": float(p.unrealized_plpc) if p.unrealized_plpc is not None else None,
            }
            for p in positions
        ],
    }


# ---------------------------------------------------------------------------
# get_market_regime — real, computed from live SPY daily bars

def compute_market_regime(stock_data_client, symbol: str = MACRO_PROXY_TICKER) -> dict:
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    request = StockBarsRequest(
        symbol_or_symbols=[symbol],
        timeframe=TimeFrame.Day,
        start=datetime.now(timezone.utc) - timedelta(days=30),
        limit=REGIME_LOOKBACK_DAYS + 1,
        sort="desc",
    )
    bars = stock_data_client.get_stock_bars(request)[symbol]
    closes = [float(b.close) for b in reversed(bars)]  # oldest -> newest

    method_label = "heuristic real-data regime classifier (SPY daily bars via Alpaca) — not a sophisticated model"

    if len(closes) < 3:
        return {
            "is_mocked": False,
            "method": method_label,
            "symbol": symbol,
            "regime": "unknown",
            "reason": f"insufficient bar data for {symbol} ({len(closes)} bars)",
        }

    returns = [(closes[i] - closes[i - 1]) / closes[i - 1] for i in range(1, len(closes))]
    realized_vol_annualized_pct = statistics.stdev(returns) * (252 ** 0.5) * 100
    net_move_pct = (closes[-1] - closes[0]) / closes[0] * 100
    up_days = sum(1 for r in returns if r > 0)
    down_days = sum(1 for r in returns if r < 0)
    consistency = max(up_days, down_days) / len(returns)

    if realized_vol_annualized_pct >= REGIME_HIGH_VOL_ANNUALIZED_PCT:
        regime = "high_vol"
    elif abs(net_move_pct) >= REGIME_TREND_MOVE_PCT and consistency >= REGIME_TREND_CONSISTENCY_RATIO:
        regime = "trending_up" if net_move_pct > 0 else "trending_down"
    else:
        regime = "choppy"

    return {
        "is_mocked": False,
        "method": method_label,
        "symbol": symbol,
        "regime": regime,
        "basis": {
            "lookback_trading_days": len(returns),
            "realized_vol_annualized_pct": round(realized_vol_annualized_pct, 2),
            "net_move_pct": round(net_move_pct, 2),
            "up_days": up_days,
            "down_days": down_days,
            "thresholds": {
                "high_vol_annualized_pct": REGIME_HIGH_VOL_ANNUALIZED_PCT,
                "trend_move_pct": REGIME_TREND_MOVE_PCT,
                "trend_consistency_ratio": REGIME_TREND_CONSISTENCY_RATIO,
            },
        },
    }


# ---------------------------------------------------------------------------
# get_lessons_learned — real, hand-maintained file

def get_lessons_learned() -> dict:
    if not LESSONS_LEARNED_PATH.exists():
        return {
            "is_mocked": False,
            "count": 0,
            "lessons": [],
            "note": f"{LESSONS_LEARNED_PATH} not found",
        }
    with LESSONS_LEARNED_PATH.open("r", encoding="utf-8") as f:
        lessons = json.load(f)
    return {
        "is_mocked": False,
        "source": str(LESSONS_LEARNED_PATH),
        "count": len(lessons),
        "lessons": lessons,
    }


# ---------------------------------------------------------------------------
# get_health_status — real checks

def get_health_status(trading_client=None) -> dict:
    alpaca_key = bool(os.getenv("ALPACA_API_KEY"))
    alpaca_secret = bool(os.getenv("ALPACA_SECRET_KEY"))
    tavily_key = bool(os.getenv("TAVILY_API_KEY"))
    anthropic_key = bool(os.getenv("ANTHROPIC_API_KEY"))

    alpaca_reachable = False
    alpaca_error = None
    alpaca_detail = None

    if trading_client is not None and alpaca_key and alpaca_secret:
        try:
            account = trading_client.get_account()
            alpaca_reachable = True
            equity = float(account.equity)
            cash = float(account.cash) if account.cash is not None else None
            # last_equity = equity as of the previous trading day's close
            # (Alpaca-computed, not derived here). Alpaca's account object
            # has no direct "day P&L" field, so day change is computed the
            # same way Alpaca's own web dashboard does: current equity minus
            # that prior-close equity.
            last_equity = float(account.last_equity) if account.last_equity is not None else None
            day_pl_dollars = (equity - last_equity) if last_equity is not None else None
            day_pl_pct = ((equity - last_equity) / last_equity * 100) if last_equity else None
            alpaca_detail = {
                "account_status": str(account.status),
                "options_trading_level": account.options_trading_level,
                "equity": equity,
                "cash": cash,
                "last_equity": last_equity,
                "day_pl_dollars": day_pl_dollars,
                "day_pl_pct": day_pl_pct,
            }
        except Exception as e:
            alpaca_error = str(e)

    return {
        "is_mocked": False,
        "alpaca": {
            "api_key_configured": alpaca_key,
            "secret_key_configured": alpaca_secret,
            "reachable": alpaca_reachable,
            "error": alpaca_error,
            "detail": alpaca_detail,
        },
        "tavily": {
            "api_key_configured": tavily_key,
            "note": "key-presence check only — no live search fired, to avoid burning quota on repeated health checks during a demo",
        },
        "anthropic": {
            "api_key_configured": anthropic_key,
        },
    }


# ---------------------------------------------------------------------------
# get_icarus_signals — fully mocked, Icarus is not in this repo

def get_icarus_signals() -> dict:
    return {
        "is_mocked": True,
        "note": (
            "Icarus is a separate production system, not included in this repo. "
            "This is an illustrative example of the payload shape a live "
            "integration would return — every value below is fabricated for "
            "demo purposes, not a real signal."
        ),
        "signals": [
            {
                "ticker": "AAPL",
                "signal": "STRONG_PUT",
                "fundamental_stress": 68.4,
                "sentiment_stress": 74.1,
                "reasoning": "[ILLUSTRATIVE] Contrarian stress score elevated on valuation + insider-selling cluster.",
            },
            {
                "ticker": "MU",
                "signal": "MODERATE_CALL",
                "fundamental_stress": 41.2,
                "sentiment_stress": 38.7,
                "reasoning": "[ILLUSTRATIVE] Post-earnings beat with muted immediate reaction — contrarian upside setup.",
            },
        ],
    }


# ---------------------------------------------------------------------------
# explain_trade_decision — real, reads data/trades.jsonl

def explain_trade_decision(ticker: Optional[str] = None, order_id: Optional[str] = None) -> dict:
    records = read_jsonl(TRADES_LOG_PATH)
    if not records:
        return {
            "is_mocked": False,
            "found": False,
            "note": (
                "No trade decisions recorded yet in this checkout -- run "
                "'python -m agent.options_executor --once [--dry-run]' to populate this."
            ),
        }

    match = None
    if order_id:
        match = next((r for r in records if r.get("order_id") == order_id), None)
    elif ticker:
        match = next(
            (r for r in records
             if (r.get("event") or {}).get("ticker") == ticker
             or (r.get("extra") or {}).get("ticker_resolved") == ticker),
            None,
        )
    else:
        match = records[0]

    if match is None:
        return {
            "is_mocked": False,
            "found": False,
            "note": f"No matching trade decision found for ticker={ticker!r} order_id={order_id!r}",
        }

    return {"is_mocked": False, "found": True, "record": match}


# ---------------------------------------------------------------------------
# run_bear_case — real context + structured scaffold, narrative left to the caller

def run_bear_case(ticker: Optional[str] = None, event_type: Optional[str] = None) -> dict:
    records = read_jsonl(TRADES_LOG_PATH)
    context = None
    if ticker:
        context = next(
            (r for r in records
             if (r.get("event") or {}).get("ticker") == ticker
             or (r.get("extra") or {}).get("ticker_resolved") == ticker),
            None,
        )
    if context is None and event_type:
        context = next((r for r in records if (r.get("event") or {}).get("event_type") == event_type), None)
    if context is None and records:
        context = records[0]

    return {
        "is_mocked": False,
        "reasoning_aid": True,
        "note": (
            "This tool returns real trade/event context plus a generic "
            "risk-factor checklist. It does NOT generate a finished narrative "
            "bear case — the calling model should synthesize the actual "
            "argument from this material."
        ),
        "context": context,
        "risk_factor_checklist": [
            "IV crush: was this entered right before/after the vol-driving event, and has implied vol already priced in the move?",
            "Thesis already priced in: did the underlying move before this trade could be placed, leaving little edge?",
            "Reversal risk: is the sentiment read based on early/incomplete coverage that could reverse as more sources report?",
            "Correlation/macro risk: for a single-name directional trade, could a broad macro move overwhelm the stock-specific thesis?",
            "Confidence miscalibration: does the confidence score reflect genuine multi-source agreement, or a small number of sources restating one fact?",
            "Liquidity/slippage: was the selected contract's DTE/strike liquid enough that the marketable limit fill reflects a fair price?",
        ],
    }
