"""
Daedalus options execution layer.

Consumes Event objects from event_detector.py, decides whether/how to
trade (directional call/put, straddle, or no trade) against configurable
thresholds, sizes the position, and places the order on Alpaca — paper
account only, `paper=True` is hardcoded below and is NOT read from
ALPACA_BASE_URL in .env. That env var stays for reference/other tooling;
this module refuses to point at anything but paper by design.

Design notes:
- "No pre-event bets" (the #1 requirement) is enforced two ways, because
  the Event schema carries no raw date field:
    * earnings events are only eligible when event_type ==
      "earnings_surprise" — event_detector only sets that once yfinance
      confirms a reported EPS actually exists. "earnings_upcoming" is
      never eligible, at any confidence/sentiment.
    * macro events are cross-referenced against event_detector.MACRO_EVENTS'
      `date` field and are only eligible once today >= that date, since
      check_macro_events()'s window fires both before and after it.
- decide_trade_mode() is pure (no network calls) so trade-vs-no-trade and
  directional-vs-straddle logic is unit-testable without touching Alpaca.
  The extra live "is the underlying actually moving" check required for
  straddles needs a real quote, so it lives in the orchestration layer
  (confirm_underlying_move), not in the pure decision function.
- No persistent state beyond the process lifetime: an in-memory
  "already traded today" set stops the polling loop from re-firing the
  same event every cycle while its confidence/sentiment stays qualified.
  This forgets history on restart by design — consistent with the
  no-database approach in event_detector.py. A restart mid-day can
  duplicate a trade already placed earlier that day.
- Exit logic is take-profit and max-hold ONLY — deliberately no
  stop-loss / loss-triggered exit. Every position Daedalus opens is a
  bought (long) option — is_event_eligible/decide_trade_mode/
  execute_directional/execute_straddle never construct a short or
  written position, so loss per trade is already bounded to the premium
  paid, by construction. A stop-loss would realize that already-bounded
  loss early on paths that might otherwise recover, cutting into the
  exact convexity the premium bought. So only two automated exits exist:
  take-profit (unrealized gain reaches TAKE_PROFIT_PCT of premium paid)
  and max-hold (fewer than MAX_HOLD_DAYS_BEFORE_EXPIRY days left before
  expiry, to avoid pin risk/illiquidity right at expiration).
  decide_exit() is pure (no network calls), same split as
  decide_trade_mode()/execute_directional() on the entry side.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional at import time
    load_dotenv = None

try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.enums import AssetClass, ContractType, OrderClass, OrderSide, PositionSide, TimeInForce
    from alpaca.trading.requests import (
        GetOptionContractsRequest,
        LimitOrderRequest,
        OptionLegRequest,
    )
    from alpaca.data.historical.option import OptionHistoricalDataClient
    from alpaca.data.historical.stock import StockHistoricalDataClient
    from alpaca.data.requests import OptionLatestQuoteRequest, StockSnapshotRequest
except ImportError:  # pragma: no cover - optional at import time
    TradingClient = None

from agent.event_detector import MACRO_EVENTS, Event, EventDetector

logger = logging.getLogger("daedalus.options_executor")

# Append-only JSONL history of every decision outcome (traded, skipped for
# budget, no-trade, guard-blocked) — not a database, read back by the MCP
# server's explain_trade_decision/get_recent_signals-adjacent tools.
REPO_ROOT = Path(__file__).resolve().parent.parent
TRADES_LOG_PATH = REPO_ROOT / "data" / "trades.jsonl"

# Separate from TRADES_LOG_PATH on purpose: an exit record has a different
# shape (no Event/TradeDecision, since it's not triggered by a detected
# event) and this makes opens vs. closes distinguishable at the file level
# alone, before ever reading a single field.
EXITS_LOG_PATH = REPO_ROOT / "data" / "exits.jsonl"

# DailyTradeGuard's persisted dedup snapshot for "today" — a small JSON
# object rewritten on each trade, not an append-only log like the two
# paths above. See DailyTradeGuard for why this exists: a restart mid-day
# used to lose all same-day dedup state and could re-fire an already-
# placed trade (observed live: a mid-signal restart duplicated an NVDA
# entry on 2026-08-28).
GUARD_STATE_PATH = REPO_ROOT / "data" / "trade_guard_state.json"

# OpenPositionsIndex's persisted symbol -> [entry, ...] snapshot. See
# OpenPositionsIndex for why this exists: a plain trades.jsonl scan for
# entry/exit pairing is unreliable in practice -- confirmed live, 0 of 3
# real exits linked, because /etc/logrotate.d/daedalus rotates
# data/*.jsonl daily and the scan never looks past the live file. Plain
# .json, not .jsonl, so that glob never touches it.
OPEN_POSITIONS_INDEX_PATH = REPO_ROOT / "data" / "open_positions_index.json"


def _append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# Config — tune here

MIN_CONFIDENCE_TO_TRADE = 0.5
DIRECTIONAL_SENTIMENT_THRESHOLD = 0.35
STRADDLE_MIN_CONFIDENCE = 0.55
STRADDLE_MIN_UNDERLYING_MOVE_PCT = 0.75  # percent move vs previous close, required to fire a straddle

POSITION_SIZE_PCT_OF_EQUITY = 0.015  # total premium budget per trade (both legs combined for a straddle)

TARGET_DTE_MIN = 3
TARGET_DTE_MAX = 14
STRIKE_WINDOW_PCT = 0.15  # only consider strikes within +/-15% of spot when selecting ATM

MAX_TRADES_PER_DAY = 5
POLL_INTERVAL_SECONDS = 1800

# -- Exit logic: take-profit and max-hold ONLY. Deliberately no stop-loss —
# see the module docstring's "Exit logic" note for why.
TAKE_PROFIT_PCT = 0.50  # close once unrealized gain reaches +50% of premium paid
MAX_HOLD_DAYS_BEFORE_EXPIRY = 2  # close once fewer than this many days remain before expiry

# Standard OCC option symbol: root + YYMMDD + C/P + 8-digit strike
# (e.g. NVDA260831C00222500 -> underlying NVDA, expires 2026-08-31).
_OCC_SYMBOL_RE = re.compile(r"^([A-Z]+)(\d{6})([CP])(\d{8})$")

# Macro events (Jackson Hole, FOMC, CPI, ...) have ticker=None in the Event
# schema — they're not stock-specific. This is the underlying options are
# actually traded on when a macro event fires a directional/straddle trade.
MACRO_PROXY_TICKER = "SPY"


class TradeMode(str, Enum):
    NO_TRADE = "no_trade"
    DIRECTIONAL_CALL = "directional_call"
    DIRECTIONAL_PUT = "directional_put"
    STRADDLE = "straddle"


class ExitReason(str, Enum):
    TAKE_PROFIT = "take_profit"
    MAX_HOLD = "max_hold"
    # No LOSS_STOP / STOP_LOSS member exists here — deliberately, see the
    # module docstring's "Exit logic" note.


@dataclass
class TradeDecision:
    mode: TradeMode
    reason: str


@dataclass
class ExecutionResult:
    event: Event
    decision: TradeDecision
    traded: bool
    detail: str
    order_id: Optional[str] = None
    dry_run: bool = False
    # Structured numeric context for explain_trade_decision, so the MCP
    # server doesn't have to re-parse the human-readable `detail` string.
    # Populated where relevant: ticker_resolved, is_macro_proxy, equity,
    # budget, ask/combined_ask, qty, move_pct (straddle only), contracts.
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "event": self.event.to_dict(),
            "decision_mode": self.decision.mode.value,
            "decision_reason": self.decision.reason,
            "traded": self.traded,
            "detail": self.detail,
            "order_id": self.order_id,
            "dry_run": self.dry_run,
            "extra": self.extra,
        }


# ---------------------------------------------------------------------------
# Pure decision logic — no network calls, unit-testable without Alpaca.

def is_event_eligible(
    event: Event,
    macro_events: list[dict] = MACRO_EVENTS,
    today: Optional[date] = None,
) -> tuple[bool, str]:
    """Enforces 'no pre-event bets': only eligible once the underlying
    occurrence has actually happened, not merely scheduled/anticipated."""
    today = today if today is not None else datetime.now(timezone.utc).date()

    if event.event_type == "earnings_upcoming":
        return False, "earnings_upcoming is pre-event by definition — only earnings_surprise is eligible"

    if event.event_type.startswith("macro_"):
        registry_entry = next(
            (m for m in macro_events if m["event_type"] == event.event_type), None
        )
        if registry_entry is None or registry_entry.get("date") is None:
            return False, f"no scheduled date found in macro registry for event_type={event.event_type!r}"

        event_date = registry_entry["date"]
        if today < event_date:
            return False, (
                f"{event.event_type} is scheduled for {event_date.isoformat()}, "
                f"which hasn't happened yet (today={today.isoformat()})"
            )
        return True, f"{event.event_type} scheduled {event_date.isoformat()} has occurred"

    # earnings_surprise, or any future event_type that by construction
    # only fires post-occurrence.
    return True, f"event_type={event.event_type!r} indicates the underlying occurrence already happened"


def decide_trade_mode(
    event: Event,
    macro_events: list[dict] = MACRO_EVENTS,
    today: Optional[date] = None,
) -> TradeDecision:
    """Trade-vs-no-trade and directional-vs-straddle, from thresholds alone.

    Does NOT perform the straddle's live underlying-move confirmation —
    that needs a real quote and is applied afterward in execute_event().
    """
    eligible, reason = is_event_eligible(event, macro_events=macro_events, today=today)
    if not eligible:
        return TradeDecision(TradeMode.NO_TRADE, reason)

    if event.confidence < MIN_CONFIDENCE_TO_TRADE:
        return TradeDecision(
            TradeMode.NO_TRADE,
            f"confidence={event.confidence} below MIN_CONFIDENCE_TO_TRADE={MIN_CONFIDENCE_TO_TRADE}",
        )

    if abs(event.sentiment_score) >= DIRECTIONAL_SENTIMENT_THRESHOLD:
        mode = TradeMode.DIRECTIONAL_CALL if event.sentiment_score > 0 else TradeMode.DIRECTIONAL_PUT
        return TradeDecision(
            mode,
            f"confidence={event.confidence} >= {MIN_CONFIDENCE_TO_TRADE} and "
            f"|sentiment|={abs(event.sentiment_score)} >= {DIRECTIONAL_SENTIMENT_THRESHOLD} -> clear direction",
        )

    if event.confidence >= STRADDLE_MIN_CONFIDENCE:
        return TradeDecision(
            TradeMode.STRADDLE,
            f"confidence={event.confidence} >= STRADDLE_MIN_CONFIDENCE={STRADDLE_MIN_CONFIDENCE} but "
            f"|sentiment|={abs(event.sentiment_score)} < {DIRECTIONAL_SENTIMENT_THRESHOLD} -> ambiguous direction, high signal",
        )

    return TradeDecision(
        TradeMode.NO_TRADE,
        f"confidence={event.confidence} below STRADDLE_MIN_CONFIDENCE={STRADDLE_MIN_CONFIDENCE} for an ambiguous-direction signal",
    )


def decide_exit(unrealized_plpc: float, days_to_expiry: int) -> Optional[ExitReason]:
    """Pure exit decision — no network calls, unit-testable without Alpaca.

    Take-profit and max-hold ONLY. There is deliberately no stop-loss /
    loss-triggered branch here — see the module docstring's "Exit logic"
    note. Take-profit is checked first: if a position is both up big and
    close to expiry, that's reported as a take-profit exit (the reason
    we're actually closing), not a max-hold exit.
    """
    if unrealized_plpc >= TAKE_PROFIT_PCT:
        return ExitReason.TAKE_PROFIT
    if days_to_expiry < MAX_HOLD_DAYS_BEFORE_EXPIRY:
        return ExitReason.MAX_HOLD
    return None


def parse_occ_expiration(symbol: str) -> Optional[date]:
    """Extract the expiration date from a standard OCC option symbol
    (e.g. "NVDA260831C00222500" -> date(2026, 8, 31)) without an extra
    API call — the same symbol format select_contract() already produces.
    Returns None if `symbol` doesn't match the expected shape."""
    match = _OCC_SYMBOL_RE.match(symbol)
    if not match:
        return None
    try:
        return datetime.strptime(match.group(2), "%y%m%d").date()
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Open-positions index: reliable entry/exit pairing for the post-mortem.

class OpenPositionsIndex:
    """Tracks which trade(s) opened each currently-open contract symbol.

    Replaces a plain trades.jsonl scan, which was unreliable in practice:
    confirmed live, 0 of 3 real exits successfully linked back to their
    entry, because /etc/logrotate.d/daedalus rotates data/*.jsonl daily
    and the scan only ever looked at the live file — anything from a
    prior rotation (i.e. most positions, since they're typically held
    multiple days) was invisible. This index is immune to that by
    construction: it's a plain .json file (not .jsonl), so logrotate's
    glob never touches it.

    It's also unambiguous when the same contract symbol is bought more
    than once — Alpaca merges same-symbol buys into a single position,
    so this keeps every contributing entry as a list under that symbol,
    rather than letting a later entry silently overwrite an earlier one.

    Persisted to state_path, load-on-init/save-on-write — same pattern
    as DailyTradeGuard.
    """

    def __init__(self, state_path: Path = OPEN_POSITIONS_INDEX_PATH):
        self.state_path = state_path
        self._entries: dict[str, list[dict]] = self._load()

    def _load(self) -> dict[str, list[dict]]:
        if not self.state_path.exists():
            return {}
        try:
            with self.state_path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            logger.warning("Could not read open positions index from %s — starting empty", self.state_path)
            return {}

    def _save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        with self.state_path.open("w", encoding="utf-8") as f:
            json.dump(self._entries, f)

    def record_entry(
        self, contract_symbol: str, trade_id: str, order_id: Optional[str],
        event_type: str, ticker: Optional[str],
    ) -> None:
        """Called only once a real order has actually been submitted —
        never for a dry-run or budget-skipped attempt, since no position
        exists yet to track in either case."""
        self._entries.setdefault(contract_symbol, []).append({
            "trade_id": trade_id,
            "order_id": order_id,
            "event_type": event_type,
            "ticker": ticker,
            "opened_at": datetime.now(timezone.utc).isoformat(),
        })
        self._save()

    def peek_entries(self, contract_symbol: str) -> Optional[list[dict]]:
        """Read without removing — used to populate entry_reference on
        every close attempt (including a skipped or dry-run one, which
        must not lose its index entry since the position is still open)."""
        entries = self._entries.get(contract_symbol)
        return list(entries) if entries else None

    def remove_entries(self, contract_symbol: str) -> None:
        """Called only once a real close order has actually been
        submitted — the position is genuinely gone, so its entry history
        is cleared."""
        if contract_symbol in self._entries:
            del self._entries[contract_symbol]
            self._save()


# ---------------------------------------------------------------------------
# Live execution layer

class OptionsExecutor:
    def __init__(
        self,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        dry_run: bool = False,
        position_index_path: Path = OPEN_POSITIONS_INDEX_PATH,
    ):
        api_key = api_key if api_key is not None else os.getenv("ALPACA_API_KEY")
        secret_key = secret_key if secret_key is not None else os.getenv("ALPACA_SECRET_KEY")
        self.dry_run = dry_run
        self.position_index = OpenPositionsIndex(state_path=position_index_path)

        if TradingClient is None:
            raise RuntimeError("alpaca-py is not installed — add it to requirements.txt")
        if not api_key or not secret_key:
            raise RuntimeError("ALPACA_API_KEY / ALPACA_SECRET_KEY not set in environment")

        # paper=True is hardcoded — see module docstring. Not configurable.
        self.trading_client = TradingClient(api_key=api_key, secret_key=secret_key, paper=True)
        self.stock_data_client = StockHistoricalDataClient(api_key=api_key, secret_key=secret_key)
        self.option_data_client = OptionHistoricalDataClient(api_key=api_key, secret_key=secret_key)

    # -- account / sizing -----------------------------------------------

    def get_equity(self) -> float:
        account = self.trading_client.get_account()
        return float(account.equity)

    def position_budget(self, equity: float, pct: float = POSITION_SIZE_PCT_OF_EQUITY) -> float:
        return equity * pct

    # -- straddle volatility confirmation ---------------------------------

    def confirm_underlying_move(
        self, ticker: str, min_move_pct: float = STRADDLE_MIN_UNDERLYING_MOVE_PCT
    ) -> tuple[bool, float]:
        """Live check that the underlying has actually moved today, not
        just that news coverage exists. Returns (confirmed, move_pct)."""
        snapshot = self.stock_data_client.get_stock_snapshot(
            StockSnapshotRequest(symbol_or_symbols=[ticker])
        )[ticker]

        prev_close = snapshot.previous_daily_bar.close if snapshot.previous_daily_bar else None
        latest_price = snapshot.latest_trade.price if snapshot.latest_trade else None

        if not prev_close or not latest_price:
            logger.warning("Missing snapshot data for %s — cannot confirm underlying move", ticker)
            return False, 0.0

        move_pct = abs(latest_price - prev_close) / prev_close * 100
        confirmed = move_pct >= min_move_pct
        return confirmed, round(move_pct, 3)

    # -- contract selection -------------------------------------------------

    def select_contract(self, ticker: str, option_type, spot_price: float, today: date):
        # expiration_date_gte must be today + TARGET_DTE_MIN, not just today:
        # liquid names like SPY list near-daily expirations, so a page
        # starting from today can be entirely consumed by 0-2 DTE contracts
        # before ever reaching the window we actually want.
        request = GetOptionContractsRequest(
            underlying_symbols=[ticker],
            type=option_type,
            expiration_date_gte=date.fromordinal(today.toordinal() + TARGET_DTE_MIN),
            expiration_date_lte=date.fromordinal(today.toordinal() + TARGET_DTE_MAX),
            strike_price_gte=str(round(spot_price * (1 - STRIKE_WINDOW_PCT), 2)),
            strike_price_lte=str(round(spot_price * (1 + STRIKE_WINDOW_PCT), 2)),
            limit=100,
        )
        response = self.trading_client.get_option_contracts(request)
        contracts = [c for c in response.option_contracts if c.tradable]
        if not contracts:
            return None

        # Nearest-to-spot strike (ATM); ties broken by nearest expiration.
        contracts.sort(key=lambda c: (abs(float(c.strike_price) - spot_price), (c.expiration_date - today).days))
        return contracts[0]

    def _latest_option_ask(self, symbol: str) -> Optional[float]:
        quotes = self.option_data_client.get_option_latest_quote(
            OptionLatestQuoteRequest(symbol_or_symbols=[symbol])
        )
        quote = quotes.get(symbol)
        if quote is None or not quote.ask_price:
            return None
        return float(quote.ask_price)

    def _latest_option_bid(self, symbol: str) -> Optional[float]:
        # Mirrors _latest_option_ask, but for closing: a sell-to-close
        # marketable limit should be priced at the bid, the same way an
        # opening buy is priced at the ask.
        quotes = self.option_data_client.get_option_latest_quote(
            OptionLatestQuoteRequest(symbol_or_symbols=[symbol])
        )
        quote = quotes.get(symbol)
        if quote is None or not quote.bid_price:
            return None
        return float(quote.bid_price)

    def _spot_price(self, ticker: str) -> Optional[float]:
        snapshot = self.stock_data_client.get_stock_snapshot(
            StockSnapshotRequest(symbol_or_symbols=[ticker])
        )[ticker]
        if snapshot.latest_trade:
            return float(snapshot.latest_trade.price)
        return None

    # -- order construction + submission -----------------------------------

    @staticmethod
    def _check_min_trade_budget(
        event: Event, ticker: str, mode_label: str, budget: float, cost_per_unit: float
    ) -> tuple[bool, int, str]:
        """Explicit minimum-trade-budget gate, checked before any order is
        constructed. Returns (sufficient, qty, message). `cost_per_unit` is
        ask*100 for a directional trade, or combined (call_ask+put_ask)*100
        for a straddle. Never returns a fractional/partial qty — either the
        budget covers at least one full contract (set), or the trade is
        skipped outright.

        The returned message uses a SKIPPED: prefix specifically so this
        failure mode — a real, eligible signal that couldn't be sized —
        is greppable and distinguishable in logs from "no signal fired"
        (logged by execute_event for TradeMode.NO_TRADE) and from
        "signal below threshold" (encoded in decide_trade_mode's reason).
        """
        qty = math.floor(budget / cost_per_unit)
        if qty >= 1:
            return True, qty, "ok"

        message = (
            f"SKIPPED: signal eligible for {mode_label} on {ticker}, but budget "
            f"${budget:.2f} insufficient for ask price ${cost_per_unit / 100:.2f} "
            f"(needs >= ${cost_per_unit:.2f}) | event={event.event_type} "
            f"sentiment={event.sentiment_score} confidence={event.confidence}"
        )
        return False, 0, message

    @staticmethod
    def _resolve_ticker(event: Event) -> tuple[str, str]:
        """Macro events carry ticker=None; trade the proxy instead. Returns
        (ticker, log_note) where log_note is empty for real per-ticker events."""
        if event.ticker:
            return event.ticker, ""
        return MACRO_PROXY_TICKER, f" (macro proxy for {event.event_type})"

    def execute_directional(self, event: Event, decision: TradeDecision, today: date) -> ExecutionResult:
        option_type = ContractType.CALL if decision.mode == TradeMode.DIRECTIONAL_CALL else ContractType.PUT
        ticker, proxy_note = self._resolve_ticker(event)
        base_extra = {"ticker_resolved": ticker, "is_macro_proxy": bool(proxy_note)}

        spot = self._spot_price(ticker)
        if spot is None:
            return ExecutionResult(event, decision, False, f"could not fetch spot price for {ticker}", dry_run=self.dry_run, extra=base_extra)

        contract = self.select_contract(ticker, option_type, spot, today)
        if contract is None:
            return ExecutionResult(
                event, decision, False,
                f"no tradable {option_type.value} contract found for {ticker} in DTE window",
                dry_run=self.dry_run, extra=base_extra,
            )

        ask = self._latest_option_ask(contract.symbol)
        if not ask:
            return ExecutionResult(event, decision, False, f"no live ask quote for {contract.symbol}", dry_run=self.dry_run, extra=base_extra)

        equity = self.get_equity()
        budget = self.position_budget(equity)
        sufficient, qty, budget_message = self._check_min_trade_budget(
            event, ticker, f"directional {option_type.value}", budget, ask * 100
        )
        sizing_extra = {
            **base_extra,
            "contract_symbol": contract.symbol,
            "strike": float(contract.strike_price),
            "expiration_date": str(contract.expiration_date),
            "equity": equity,
            "position_size_pct_of_equity": POSITION_SIZE_PCT_OF_EQUITY,
            "budget": budget,
            "ask": ask,
            "qty": qty,
        }
        if not sufficient:
            logger.warning(budget_message)
            return ExecutionResult(event, decision, False, budget_message, dry_run=self.dry_run, extra=sizing_extra)

        # Generated here, not earlier -- only an actually-attempted order
        # (dry-run or live) gets a trade_id; a budget-skipped signal never
        # opens a position, so there's nothing for an index entry to track.
        trade_id = uuid.uuid4().hex[:12]
        order_sizing_extra = {**sizing_extra, "trade_id": trade_id}

        order_request = LimitOrderRequest(
            symbol=contract.symbol,
            qty=qty,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
            limit_price=round(ask, 2),
        )

        detail = (
            f"BUY {qty}x {contract.symbol} ({option_type.value}, strike={contract.strike_price}, "
            f"exp={contract.expiration_date}) @ limit ${ask:.2f} | event={event.event_type} "
            f"ticker={ticker}{proxy_note} sentiment={event.sentiment_score} confidence={event.confidence} "
            f"| reason: {decision.reason} | trade_id={trade_id}"
        )

        if self.dry_run:
            logger.info("[DRY RUN] Would place order: %s", detail)
            return ExecutionResult(event, decision, False, f"[DRY RUN] {detail}", dry_run=True, extra=order_sizing_extra)

        order = self.trading_client.submit_order(order_request)
        self.position_index.record_entry(
            contract.symbol, trade_id=trade_id, order_id=str(order.id),
            event_type=event.event_type, ticker=event.ticker,
        )
        logger.info("Order submitted: %s | order_id=%s", detail, order.id)
        return ExecutionResult(event, decision, True, detail, order_id=str(order.id), dry_run=self.dry_run, extra=order_sizing_extra)

    def execute_straddle(self, event: Event, decision: TradeDecision, today: date) -> ExecutionResult:
        ticker, proxy_note = self._resolve_ticker(event)
        base_extra = {"ticker_resolved": ticker, "is_macro_proxy": bool(proxy_note)}

        confirmed, move_pct = self.confirm_underlying_move(ticker)
        move_extra = {**base_extra, "underlying_move_pct": move_pct, "move_confirmed": confirmed}
        if not confirmed:
            return ExecutionResult(
                event, decision, False,
                f"underlying move {move_pct}% below STRADDLE_MIN_UNDERLYING_MOVE_PCT="
                f"{STRADDLE_MIN_UNDERLYING_MOVE_PCT}% — volatility not confirmed, skipping straddle",
                dry_run=self.dry_run, extra=move_extra,
            )

        spot = self._spot_price(ticker)
        if spot is None:
            return ExecutionResult(event, decision, False, f"could not fetch spot price for {ticker}", dry_run=self.dry_run, extra=move_extra)

        call_contract = self.select_contract(ticker, ContractType.CALL, spot, today)
        put_contract = self.select_contract(ticker, ContractType.PUT, spot, today)
        if call_contract is None or put_contract is None:
            return ExecutionResult(
                event, decision, False,
                f"could not find both call+put contracts for {ticker} straddle",
                dry_run=self.dry_run, extra=move_extra,
            )

        call_ask = self._latest_option_ask(call_contract.symbol)
        put_ask = self._latest_option_ask(put_contract.symbol)
        if not call_ask or not put_ask:
            return ExecutionResult(event, decision, False, f"missing live ask quote for {ticker} straddle legs", dry_run=self.dry_run, extra=move_extra)

        combined_ask = call_ask + put_ask
        equity = self.get_equity()
        budget = self.position_budget(equity)
        sufficient, qty, budget_message = self._check_min_trade_budget(
            event, ticker, "straddle", budget, combined_ask * 100
        )
        sizing_extra = {
            **move_extra,
            "call_contract_symbol": call_contract.symbol,
            "put_contract_symbol": put_contract.symbol,
            "strike": float(call_contract.strike_price),
            "expiration_date": str(call_contract.expiration_date),
            "equity": equity,
            "position_size_pct_of_equity": POSITION_SIZE_PCT_OF_EQUITY,
            "budget": budget,
            "combined_ask": combined_ask,
            "qty": qty,
        }
        if not sufficient:
            logger.warning(budget_message)
            return ExecutionResult(event, decision, False, budget_message, dry_run=self.dry_run, extra=sizing_extra)

        # Both legs share one trade_id -- they're one straddle trade from
        # one entry decision, even though they're two separate Alpaca
        # positions (different symbols) that close independently.
        trade_id = uuid.uuid4().hex[:12]
        order_sizing_extra = {**sizing_extra, "trade_id": trade_id}

        legs = [
            OptionLegRequest(symbol=call_contract.symbol, ratio_qty=1, side=OrderSide.BUY),
            OptionLegRequest(symbol=put_contract.symbol, ratio_qty=1, side=OrderSide.BUY),
        ]
        order_request = LimitOrderRequest(
            qty=qty,
            time_in_force=TimeInForce.DAY,
            order_class=OrderClass.MLEG,
            legs=legs,
            limit_price=round(combined_ask, 2),
        )

        detail = (
            f"BUY {qty}x STRADDLE {ticker}{proxy_note}: call={call_contract.symbol} put={put_contract.symbol} "
            f"@ combined limit ${combined_ask:.2f} (underlying moved {move_pct}%) | event={event.event_type} "
            f"sentiment={event.sentiment_score} confidence={event.confidence} | reason: {decision.reason} "
            f"| trade_id={trade_id}"
        )

        if self.dry_run:
            logger.info("[DRY RUN] Would place order: %s", detail)
            return ExecutionResult(event, decision, False, f"[DRY RUN] {detail}", dry_run=True, extra=order_sizing_extra)

        order = self.trading_client.submit_order(order_request)
        self.position_index.record_entry(
            call_contract.symbol, trade_id=trade_id, order_id=str(order.id),
            event_type=event.event_type, ticker=event.ticker,
        )
        self.position_index.record_entry(
            put_contract.symbol, trade_id=trade_id, order_id=str(order.id),
            event_type=event.event_type, ticker=event.ticker,
        )
        logger.info("Order submitted: %s | order_id=%s", detail, order.id)
        return ExecutionResult(event, decision, True, detail, order_id=str(order.id), dry_run=self.dry_run, extra=order_sizing_extra)

    def execute_event(self, event: Event, macro_events: list[dict] = MACRO_EVENTS, today: Optional[date] = None) -> ExecutionResult:
        today = today if today is not None else datetime.now(timezone.utc).date()
        decision = decide_trade_mode(event, macro_events=macro_events, today=today)

        if decision.mode == TradeMode.NO_TRADE:
            logger.info("No trade for %s/%s: %s", event.event_type, event.ticker, decision.reason)
            result = ExecutionResult(event, decision, False, decision.reason, dry_run=self.dry_run)
        elif decision.mode in (TradeMode.DIRECTIONAL_CALL, TradeMode.DIRECTIONAL_PUT):
            result = self.execute_directional(event, decision, today)
        else:
            result = self.execute_straddle(event, decision, today)

        self._log_result(result)
        return result

    @staticmethod
    def _log_result(result: ExecutionResult) -> None:
        record = result.to_dict()
        record["timestamp"] = datetime.now(timezone.utc).isoformat()
        _append_jsonl(TRADES_LOG_PATH, record)

    # -- exit logic: take-profit / max-hold only, no stop-loss ---------------

    def check_and_close_positions(self, today: Optional[date] = None) -> list[dict]:
        """Poll-cycle exit check. Closes (sell-to-close) any open option
        position that has hit take-profit or is within
        MAX_HOLD_DAYS_BEFORE_EXPIRY of expiring. Deliberately no stop-loss
        path — see module docstring. decide_exit() is the pure decision;
        this method is the live orchestration around it, same split as
        decide_trade_mode()/execute_directional() on the entry side. This
        never touches is_event_eligible/decide_trade_mode/execute_directional/
        execute_straddle — entry-side logic is untouched by this feature.
        """
        today = today if today is not None else datetime.now(timezone.utc).date()
        positions = self.trading_client.get_all_positions()
        results = []

        for position in positions:
            if position.asset_class != AssetClass.US_OPTION:
                continue

            expiration = parse_occ_expiration(position.symbol)
            if expiration is None:
                logger.warning(
                    "Could not parse expiration from option symbol %s — skipping exit check this cycle",
                    position.symbol,
                )
                continue

            if position.unrealized_plpc is None:
                continue
            unrealized_plpc = float(position.unrealized_plpc)
            days_to_expiry = (expiration - today).days

            reason = decide_exit(unrealized_plpc, days_to_expiry)
            if reason is None:
                continue

            result = self._close_position(position, reason, unrealized_plpc, days_to_expiry, expiration)
            results.append(result)

        return results

    def _find_entry_reference(self, contract_symbol: str) -> Optional[list[dict]]:
        """Looks up OpenPositionsIndex (not a trades.jsonl scan) for every
        entry that opened this contract symbol. Immune to trades.jsonl's
        daily log rotation by construction, and returns every
        contributing entry as a list — normally length 1, but a list so
        two separate entries on the same symbol (Alpaca merges them into
        one position) don't silently collide into a single reference.
        Read-only: does not remove anything from the index — see
        OpenPositionsIndex.peek_entries / remove_entries."""
        return self.position_index.peek_entries(contract_symbol)

    def _close_position(
        self, position, reason: ExitReason, unrealized_plpc: float,
        days_to_expiry: int, expiration: date,
    ) -> dict:
        base_record = {
            "action": "close_position",
            "contract_symbol": position.symbol,
            "exit_reason": reason.value,
            "unrealized_plpc": unrealized_plpc,
            "unrealized_pl_dollars": float(position.unrealized_pl) if position.unrealized_pl is not None else None,
            "days_to_expiry": days_to_expiry,
            "expiration": expiration.isoformat(),
            "entry_reference": self._find_entry_reference(position.symbol),
            "dry_run": self.dry_run,
        }

        if position.side != PositionSide.LONG:
            # Daedalus never opens a short/written position (see module
            # docstring) — every position this code has ever opened is
            # long, so an automated close is always a sell. A short
            # position here would mean something else wrote it; refuse
            # rather than guess the correct close side.
            message = (
                f"CLOSE SKIPPED: {position.symbol} is a {position.side.value} position — "
                f"Daedalus never opens short positions, refusing to guess the correct close side"
            )
            logger.error(message)
            record = {
                **base_record, "closed": False, "qty": None, "limit_price": None,
                "order_id": None, "detail": message,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            self._log_exit(record)
            return record

        qty = abs(int(float(position.qty)))
        bid = self._latest_option_bid(position.symbol)

        if bid is None:
            message = (
                f"CLOSE SKIPPED: no live bid quote for {position.symbol}, "
                f"exit_reason={reason.value} deferred to next poll"
            )
            logger.warning(message)
            record = {
                **base_record, "closed": False, "qty": qty, "limit_price": None,
                "order_id": None, "detail": message,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            self._log_exit(record)
            return record

        order_request = LimitOrderRequest(
            symbol=position.symbol,
            qty=qty,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
            limit_price=round(bid, 2),
        )

        detail = (
            f"SELL-TO-CLOSE {qty}x {position.symbol} @ limit ${bid:.2f} | exit_reason={reason.value} "
            f"unrealized_plpc={unrealized_plpc:.4f} days_to_expiry={days_to_expiry}"
        )

        if self.dry_run:
            logger.info("[DRY RUN] Would close position (sell-to-close): %s", detail)
            record = {
                **base_record, "closed": False, "qty": qty, "limit_price": round(bid, 2),
                "order_id": None, "detail": f"[DRY RUN] {detail}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            self._log_exit(record)
            return record

        order = self.trading_client.submit_order(order_request)
        # Only removed here, on a real submitted close -- a skipped or
        # dry-run close attempt above must not clear the index, since the
        # position is still genuinely open and needs to resolve again on
        # the next poll.
        self.position_index.remove_entries(position.symbol)
        logger.info("Position CLOSE submitted (sell-to-close): %s | order_id=%s", detail, order.id)
        record = {
            **base_record, "closed": True, "qty": qty, "limit_price": round(bid, 2),
            "order_id": str(order.id), "detail": detail,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._log_exit(record)
        return record

    @staticmethod
    def _log_exit(record: dict) -> None:
        _append_jsonl(EXITS_LOG_PATH, record)


# ---------------------------------------------------------------------------
# Wiring loop: polls the detector, evaluates every event, executes qualifying ones.

class DailyTradeGuard:
    """Dedup + daily trade cap. Resets at UTC midnight. Persisted to
    state_path (a small JSON snapshot, rewritten on each trade — no
    database) so a restart mid-day doesn't lose same-day dedup and
    re-fire a trade already placed earlier today. Only today's entries
    are ever loaded back in; a state file from a different day is
    ignored outright, since this guard is daily, not permanent."""

    def __init__(
        self,
        max_trades_per_day: int = MAX_TRADES_PER_DAY,
        state_path: Path = GUARD_STATE_PATH,
        today: Optional[date] = None,
    ):
        self.max_trades_per_day = max_trades_per_day
        self.state_path = state_path
        self._day: Optional[date] = None
        self._traded_keys: set[tuple[str, Optional[str]]] = set()
        self._trade_count = 0
        self._load_state(today if today is not None else datetime.now(timezone.utc).date())

    def _load_state(self, today: date) -> None:
        if not self.state_path.exists():
            return
        try:
            with self.state_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            logger.warning("Could not read trade guard state from %s — starting with empty state", self.state_path)
            return

        if data.get("date") != today.isoformat():
            return  # stale (different day) or malformed — start fresh for today

        self._day = today
        self._traded_keys = {tuple(key) for key in data.get("keys", [])}
        self._trade_count = len(self._traded_keys)

    def _save_state(self, today: date) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        data = {"date": today.isoformat(), "keys": [list(key) for key in self._traded_keys]}
        with self.state_path.open("w", encoding="utf-8") as f:
            json.dump(data, f)

    def _roll_if_new_day(self, today: date) -> None:
        if self._day != today:
            self._day = today
            self._traded_keys = set()
            self._trade_count = 0

    def should_attempt(self, event: Event, today: Optional[date] = None) -> tuple[bool, str]:
        today = today if today is not None else datetime.now(timezone.utc).date()
        self._roll_if_new_day(today)

        key = (event.event_type, event.ticker)
        if key in self._traded_keys:
            return False, f"already traded {key} today"
        if self._trade_count >= self.max_trades_per_day:
            return False, f"MAX_TRADES_PER_DAY={self.max_trades_per_day} reached"
        return True, "ok"

    def record_trade(self, event: Event, today: Optional[date] = None) -> None:
        today = today if today is not None else datetime.now(timezone.utc).date()
        self._roll_if_new_day(today)
        self._traded_keys.add((event.event_type, event.ticker))
        self._trade_count += 1
        self._save_state(today)


def run_cycle(
    detector: EventDetector,
    executor: OptionsExecutor,
    guard: DailyTradeGuard,
    now: Optional[datetime] = None,
) -> list[ExecutionResult]:
    """One poll-evaluate-execute pass. Macro events are evaluated before
    earnings so a qualifying flagship macro signal takes priority, but
    every qualifying event this cycle gets evaluated — earnings isn't
    only consulted when macro produces nothing; it's simply checked
    every cycle, same as macro, and MAX_TRADES_PER_DAY is the only cap.

    Exit checks (take-profit / max-hold, see check_and_close_positions)
    run first, every cycle, independent of and before any new entries —
    this only ever closes what's already open; it never touches
    entry-side eligibility/sizing.
    """
    now = now or datetime.now(timezone.utc)
    today = now.date()

    exit_results = executor.check_and_close_positions(today=today)
    if exit_results:
        logger.info("Closed %d position(s) this cycle", sum(1 for r in exit_results if r["closed"]))

    macro_events = detector.check_macro_events(now=now)
    earnings_events = detector.check_earnings(now=now)
    results: list[ExecutionResult] = []

    for event in macro_events + earnings_events:
        may_attempt, guard_reason = guard.should_attempt(event, today=today)
        if not may_attempt:
            logger.info("Skipping %s/%s: %s", event.event_type, event.ticker, guard_reason)
            continue

        result = executor.execute_event(event, today=today)
        results.append(result)
        if result.traded:
            guard.record_trade(event, today=today)

    if not results:
        logger.info("No qualifying events this cycle (macro=%d, earnings=%d detected)", len(macro_events), len(earnings_events))

    return results


def run(
    detector: EventDetector,
    executor: OptionsExecutor,
    interval_seconds: int = POLL_INTERVAL_SECONDS,
) -> None:
    guard = DailyTradeGuard()
    logger.info("Starting options executor loop (interval=%ss, dry_run=%s)", interval_seconds, executor.dry_run)
    while True:
        try:
            run_cycle(detector, executor, guard)
        except Exception:
            logger.exception("Error during execution cycle")
        time.sleep(interval_seconds)


# ---------------------------------------------------------------------------
# CLI

def _parse_args():
    parser = argparse.ArgumentParser(description="Daedalus options executor")
    parser.add_argument("--once", action="store_true", help="Run a single poll-evaluate-execute cycle and exit")
    parser.add_argument("--interval", type=int, default=POLL_INTERVAL_SECONDS, help="Polling interval in seconds")
    parser.add_argument("--watchlist", type=str, default=None, help="Comma-separated tickers (default: DEFAULT_WATCHLIST)")
    parser.add_argument("--dry-run", action="store_true", help="Log decisions/orders without submitting them")
    return parser.parse_args()


def main():
    # Force UTF-8 on stdout/stderr: reason strings contain em dashes, and
    # Windows consoles default to cp1252, which mismatches a UTF-8-expecting
    # terminal (Git Bash, most CI) and renders them as mangled replacement
    # characters rather than raising — silently corrupting displayed text.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if load_dotenv is not None:
        load_dotenv()

    args = _parse_args()
    watchlist = args.watchlist.split(",") if args.watchlist else None
    detector = EventDetector(watchlist=watchlist)
    executor = OptionsExecutor(dry_run=args.dry_run)

    if args.once:
        guard = DailyTradeGuard()
        run_cycle(detector, executor, guard)
    else:
        run(detector, executor, interval_seconds=args.interval)


if __name__ == "__main__":
    main()
