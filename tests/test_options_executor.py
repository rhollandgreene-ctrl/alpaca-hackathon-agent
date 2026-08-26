"""
Tests for the Daedalus options executor decision logic.

decide_trade_mode()/is_event_eligible()/DailyTradeGuard are pure — no
Alpaca client is constructed and no network call is made, so these run
without ALPACA_API_KEY/ALPACA_SECRET_KEY being set.

The budget-sufficiency tests below do construct an OptionsExecutor, but
with fake credentials and dry_run=True, and monkeypatch every live-data
method (_spot_price, select_contract, _latest_option_ask,
confirm_underlying_move, get_equity) — constructing the alpaca-py client
objects doesn't itself make a network call, only invoking their methods
does, so this still never touches the real Alpaca API.
"""

import logging
from datetime import date, datetime, timezone
from types import SimpleNamespace
from unittest.mock import Mock

from agent.event_detector import Event
from agent.options_executor import (
    DIRECTIONAL_SENTIMENT_THRESHOLD,
    MIN_CONFIDENCE_TO_TRADE,
    POSITION_SIZE_PCT_OF_EQUITY,
    STRADDLE_MIN_CONFIDENCE,
    DailyTradeGuard,
    OptionsExecutor,
    TradeDecision,
    TradeMode,
    decide_trade_mode,
    is_event_eligible,
)

TEST_MACRO_EVENTS = [
    {
        "name": "Jackson Hole Economic Symposium",
        "event_type": "macro_jackson_hole",
        "date": date(2026, 8, 28),
        "description": "Federal Reserve's annual Jackson Hole Economic Policy Symposium",
        "search_query": "Jackson Hole Symposium Federal Reserve Powell speech",
    },
    {
        "name": "FOMC Meeting",
        "event_type": "macro_fomc",
        "date": None,
        "description": "Federal Open Market Committee interest rate decision",
        "search_query": "FOMC meeting interest rate decision outlook",
    },
]


def make_event(event_type: str, ticker=None, sentiment_score=0.0, confidence=0.0) -> Event:
    return Event(
        event_type=event_type,
        ticker=ticker,
        timestamp=datetime.now(timezone.utc).isoformat(),
        description="test event",
        sentiment_score=sentiment_score,
        confidence=confidence,
    )


# -- no pre-event bets --------------------------------------------------------

def test_earnings_upcoming_is_never_eligible_even_with_extreme_signal():
    event = make_event("earnings_upcoming", ticker="NVDA", sentiment_score=1.0, confidence=1.0)
    eligible, reason = is_event_eligible(event, macro_events=TEST_MACRO_EVENTS, today=date(2026, 8, 25))
    assert eligible is False
    assert "pre-event" in reason

    decision = decide_trade_mode(event, macro_events=TEST_MACRO_EVENTS, today=date(2026, 8, 25))
    assert decision.mode == TradeMode.NO_TRADE


def test_macro_event_before_scheduled_date_is_not_eligible():
    event = make_event("macro_jackson_hole", sentiment_score=0.9, confidence=0.9)
    decision = decide_trade_mode(event, macro_events=TEST_MACRO_EVENTS, today=date(2026, 8, 25))
    assert decision.mode == TradeMode.NO_TRADE
    assert "hasn't happened yet" in decision.reason


def test_macro_event_on_scheduled_date_is_eligible():
    event = make_event("macro_jackson_hole", sentiment_score=0.9, confidence=0.9)
    decision = decide_trade_mode(event, macro_events=TEST_MACRO_EVENTS, today=date(2026, 8, 28))
    assert decision.mode == TradeMode.DIRECTIONAL_CALL


def test_macro_event_with_no_registered_date_is_never_eligible():
    event = make_event("macro_fomc", sentiment_score=0.9, confidence=0.9)
    decision = decide_trade_mode(event, macro_events=TEST_MACRO_EVENTS, today=date(2026, 12, 1))
    assert decision.mode == TradeMode.NO_TRADE
    assert "no scheduled date" in decision.reason


def test_unknown_macro_event_type_is_not_eligible():
    event = make_event("macro_cpi", sentiment_score=0.9, confidence=0.9)
    decision = decide_trade_mode(event, macro_events=TEST_MACRO_EVENTS, today=date(2026, 8, 28))
    assert decision.mode == TradeMode.NO_TRADE


# -- trade vs. no-trade, directional vs. straddle ----------------------------

def test_earnings_surprise_strong_positive_sentiment_is_directional_call():
    event = make_event("earnings_surprise", ticker="NVDA", sentiment_score=0.6, confidence=0.8)
    decision = decide_trade_mode(event, macro_events=TEST_MACRO_EVENTS, today=date(2026, 8, 26))
    assert decision.mode == TradeMode.DIRECTIONAL_CALL


def test_earnings_surprise_strong_negative_sentiment_is_directional_put():
    event = make_event("earnings_surprise", ticker="NVDA", sentiment_score=-0.6, confidence=0.8)
    decision = decide_trade_mode(event, macro_events=TEST_MACRO_EVENTS, today=date(2026, 8, 26))
    assert decision.mode == TradeMode.DIRECTIONAL_PUT


def test_ambiguous_direction_high_confidence_is_straddle():
    event = make_event(
        "earnings_surprise", ticker="NVDA",
        sentiment_score=0.05, confidence=STRADDLE_MIN_CONFIDENCE,
    )
    decision = decide_trade_mode(event, macro_events=TEST_MACRO_EVENTS, today=date(2026, 8, 26))
    assert decision.mode == TradeMode.STRADDLE


def test_ambiguous_direction_moderate_confidence_is_no_trade():
    # Clears the base floor but not the (higher) straddle-specific bar.
    event = make_event(
        "earnings_surprise", ticker="NVDA",
        sentiment_score=0.05, confidence=MIN_CONFIDENCE_TO_TRADE + 0.01,
    )
    assert MIN_CONFIDENCE_TO_TRADE + 0.01 < STRADDLE_MIN_CONFIDENCE
    decision = decide_trade_mode(event, macro_events=TEST_MACRO_EVENTS, today=date(2026, 8, 26))
    assert decision.mode == TradeMode.NO_TRADE


def test_low_confidence_blocks_trade_even_with_extreme_sentiment():
    # This is the exact miscalibration pattern found live last session:
    # an extreme score backed by weak corroboration must not trade.
    event = make_event("earnings_surprise", ticker="NVDA", sentiment_score=0.95, confidence=0.2)
    decision = decide_trade_mode(event, macro_events=TEST_MACRO_EVENTS, today=date(2026, 8, 26))
    assert decision.mode == TradeMode.NO_TRADE
    assert "MIN_CONFIDENCE_TO_TRADE" in decision.reason


def test_thresholds_are_boundary_correct():
    just_below = make_event(
        "earnings_surprise", ticker="NVDA",
        sentiment_score=DIRECTIONAL_SENTIMENT_THRESHOLD - 0.01, confidence=0.9,
    )
    just_at = make_event(
        "earnings_surprise", ticker="NVDA",
        sentiment_score=DIRECTIONAL_SENTIMENT_THRESHOLD, confidence=0.9,
    )
    below_decision = decide_trade_mode(just_below, macro_events=TEST_MACRO_EVENTS, today=date(2026, 8, 26))
    at_decision = decide_trade_mode(just_at, macro_events=TEST_MACRO_EVENTS, today=date(2026, 8, 26))

    assert below_decision.mode == TradeMode.STRADDLE  # confidence 0.9 clears STRADDLE_MIN_CONFIDENCE
    assert at_decision.mode == TradeMode.DIRECTIONAL_CALL


# -- daily trade guard --------------------------------------------------------

def test_guard_blocks_repeat_trade_on_same_event_same_day():
    guard = DailyTradeGuard(max_trades_per_day=5)
    event = make_event("earnings_surprise", ticker="NVDA", sentiment_score=0.6, confidence=0.8)
    today = date(2026, 8, 26)

    may_attempt_first, _ = guard.should_attempt(event, today=today)
    assert may_attempt_first is True

    guard.record_trade(event, today=today)

    may_attempt_second, reason = guard.should_attempt(event, today=today)
    assert may_attempt_second is False
    assert "already traded" in reason


def test_guard_allows_different_ticker_same_day():
    guard = DailyTradeGuard(max_trades_per_day=5)
    today = date(2026, 8, 26)
    nvda_event = make_event("earnings_surprise", ticker="NVDA", confidence=0.8)
    aapl_event = make_event("earnings_surprise", ticker="AAPL", confidence=0.8)

    guard.record_trade(nvda_event, today=today)
    may_attempt, _ = guard.should_attempt(aapl_event, today=today)
    assert may_attempt is True


def test_guard_enforces_max_trades_per_day():
    guard = DailyTradeGuard(max_trades_per_day=2)
    today = date(2026, 8, 26)
    events = [make_event("earnings_surprise", ticker=t, confidence=0.8) for t in ["AAPL", "MSFT", "TSLA"]]

    for event in events[:2]:
        may_attempt, _ = guard.should_attempt(event, today=today)
        assert may_attempt is True
        guard.record_trade(event, today=today)

    may_attempt_third, reason = guard.should_attempt(events[2], today=today)
    assert may_attempt_third is False
    assert "MAX_TRADES_PER_DAY" in reason


def test_guard_resets_on_new_day():
    guard = DailyTradeGuard(max_trades_per_day=1)
    event = make_event("earnings_surprise", ticker="NVDA", confidence=0.8)

    guard.record_trade(event, today=date(2026, 8, 26))
    may_attempt_same_day, _ = guard.should_attempt(event, today=date(2026, 8, 26))
    assert may_attempt_same_day is False

    may_attempt_next_day, _ = guard.should_attempt(event, today=date(2026, 8, 27))
    assert may_attempt_next_day is True


# -- budget-sufficiency gate --------------------------------------------------

def _fake_contract(symbol, strike_price=200.0, expiration_date=date(2026, 8, 31)):
    return SimpleNamespace(
        symbol=symbol, strike_price=strike_price,
        expiration_date=expiration_date, tradable=True,
    )


def _executor_with_mocked_live_data(equity=100_000.0):
    executor = OptionsExecutor(api_key="fake", secret_key="fake", dry_run=True)
    executor.get_equity = Mock(return_value=equity)
    executor.trading_client.submit_order = Mock(
        side_effect=AssertionError("submit_order must not be called for a skipped trade")
    )
    return executor


def test_directional_skips_cleanly_when_budget_insufficient(caplog):
    executor = _executor_with_mocked_live_data()
    executor._spot_price = Mock(return_value=200.0)
    executor.select_contract = Mock(return_value=_fake_contract("NVDA260831C00200000"))
    # Ask price high enough that even the full 1.5%-of-equity budget can't
    # afford a single contract (ask*100 must exceed equity*POSITION_SIZE_PCT_OF_EQUITY).
    unaffordable_ask = (100_000.0 * POSITION_SIZE_PCT_OF_EQUITY * 2) / 100
    executor._latest_option_ask = Mock(return_value=unaffordable_ask)

    event = make_event("earnings_surprise", ticker="NVDA", sentiment_score=0.6, confidence=0.8)
    decision = TradeDecision(TradeMode.DIRECTIONAL_CALL, "cleared all gates")

    with caplog.at_level(logging.WARNING):
        result = executor.execute_directional(event, decision, today=date(2026, 8, 26))

    assert result.traded is False
    assert result.detail.startswith("SKIPPED:")
    assert "budget" in result.detail and "insufficient" in result.detail
    assert "directional call" in result.detail
    assert "NVDA" in result.detail
    executor.trading_client.submit_order.assert_not_called()

    # Distinctly logged, not silent.
    assert any(record.message.startswith("SKIPPED:") for record in caplog.records)


def test_straddle_skips_cleanly_when_budget_insufficient(caplog):
    executor = _executor_with_mocked_live_data()
    executor.confirm_underlying_move = Mock(return_value=(True, 1.2))  # volatility confirmed
    executor._spot_price = Mock(return_value=200.0)
    executor.select_contract = Mock(side_effect=[
        _fake_contract("NVDA260831C00200000"),
        _fake_contract("NVDA260831P00200000"),
    ])
    unaffordable_combined_ask = (100_000.0 * POSITION_SIZE_PCT_OF_EQUITY * 2) / 100
    executor._latest_option_ask = Mock(return_value=unaffordable_combined_ask / 2)

    event = make_event("earnings_surprise", ticker="NVDA", sentiment_score=0.05, confidence=0.9)
    decision = TradeDecision(TradeMode.STRADDLE, "cleared all gates")

    with caplog.at_level(logging.WARNING):
        result = executor.execute_straddle(event, decision, today=date(2026, 8, 26))

    assert result.traded is False
    assert result.detail.startswith("SKIPPED:")
    assert "budget" in result.detail and "insufficient" in result.detail
    assert "straddle" in result.detail
    executor.trading_client.submit_order.assert_not_called()
    assert any(record.message.startswith("SKIPPED:") for record in caplog.records)


def test_budget_skip_message_is_distinct_from_no_trade_reason():
    # The SKIPPED: budget message must not be confusable with a
    # decide_trade_mode NO_TRADE reason (below-threshold / pre-event /
    # unscheduled), so a human scanning logs can tell "nothing qualified"
    # apart from "something qualified but couldn't be sized".
    no_trade_event = make_event("earnings_upcoming", ticker="NVDA", confidence=0.9, sentiment_score=0.9)
    no_trade_decision = decide_trade_mode(no_trade_event, macro_events=TEST_MACRO_EVENTS, today=date(2026, 8, 26))
    assert not no_trade_decision.reason.startswith("SKIPPED:")

    executor = _executor_with_mocked_live_data()
    executor._spot_price = Mock(return_value=200.0)
    executor.select_contract = Mock(return_value=_fake_contract("NVDA260831C00200000"))
    executor._latest_option_ask = Mock(return_value=(100_000.0 * POSITION_SIZE_PCT_OF_EQUITY * 2) / 100)

    budget_event = make_event("earnings_surprise", ticker="NVDA", sentiment_score=0.6, confidence=0.8)
    budget_decision = TradeDecision(TradeMode.DIRECTIONAL_CALL, "cleared all gates")
    budget_result = executor.execute_directional(budget_event, budget_decision, today=date(2026, 8, 26))

    assert budget_result.detail.startswith("SKIPPED:")
    assert budget_result.detail != no_trade_decision.reason
