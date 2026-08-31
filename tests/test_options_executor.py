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

import pytest
from alpaca.trading.enums import AssetClass, OrderSide, PositionSide

from agent.event_detector import MACRO_EVENTS, Event
from agent.options_executor import (
    DIRECTIONAL_SENTIMENT_THRESHOLD,
    MAX_HOLD_DAYS_BEFORE_EXPIRY,
    MIN_CONFIDENCE_TO_TRADE,
    POSITION_SIZE_PCT_OF_EQUITY,
    STRADDLE_MIN_CONFIDENCE,
    TAKE_PROFIT_PCT,
    DailyTradeGuard,
    ExitReason,
    OpenPositionsIndex,
    OptionsExecutor,
    TradeDecision,
    TradeMode,
    decide_exit,
    decide_trade_mode,
    is_event_eligible,
    parse_occ_expiration,
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


# -- newly-added macro registry entries (jobs report, CPI, FOMC, PCE) -------
# Against the REAL MACRO_EVENTS registry (not a local fixture), same as the
# Jackson Hole eligibility tests above but verifying the actual configured
# dates -- these route through is_event_eligible/decide_trade_mode exactly
# like macro_jackson_hole; no new logic, only new registry entries.

_NEW_MACRO_EVENTS = [
    ("macro_jobs_report", date(2026, 9, 4)),
    ("macro_cpi_release", date(2026, 9, 11)),
    ("macro_fomc_meeting", date(2026, 9, 16)),
    ("macro_pce_data", date(2026, 9, 30)),
    ("macro_fomc_meeting_oct", date(2026, 10, 28)),
    ("macro_fomc_meeting_dec", date(2026, 12, 9)),
]


@pytest.mark.parametrize("event_type,scheduled_date", _NEW_MACRO_EVENTS)
def test_new_macro_event_registered_in_real_registry_with_correct_date(event_type, scheduled_date):
    entry = next(m for m in MACRO_EVENTS if m["event_type"] == event_type)
    assert entry["date"] == scheduled_date


@pytest.mark.parametrize("event_type,scheduled_date", _NEW_MACRO_EVENTS)
def test_new_macro_event_blocked_before_scheduled_date(event_type, scheduled_date):
    event = make_event(event_type, sentiment_score=0.9, confidence=0.9)
    day_before = date.fromordinal(scheduled_date.toordinal() - 1)
    decision = decide_trade_mode(event, macro_events=MACRO_EVENTS, today=day_before)
    assert decision.mode == TradeMode.NO_TRADE
    assert "hasn't happened yet" in decision.reason


@pytest.mark.parametrize("event_type,scheduled_date", _NEW_MACRO_EVENTS)
def test_new_macro_event_eligible_on_scheduled_date(event_type, scheduled_date):
    event = make_event(event_type, sentiment_score=0.9, confidence=0.9)
    decision = decide_trade_mode(event, macro_events=MACRO_EVENTS, today=scheduled_date)
    assert decision.mode == TradeMode.DIRECTIONAL_CALL


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
# state_path is always pointed at tmp_path here -- DailyTradeGuard now
# persists to disk by default (GUARD_STATE_PATH), and these tests must
# never touch the real data/trade_guard_state.json.

def test_guard_blocks_repeat_trade_on_same_event_same_day(tmp_path):
    guard = DailyTradeGuard(max_trades_per_day=5, state_path=tmp_path / "state.json")
    event = make_event("earnings_surprise", ticker="NVDA", sentiment_score=0.6, confidence=0.8)
    today = date(2026, 8, 26)

    may_attempt_first, _ = guard.should_attempt(event, today=today)
    assert may_attempt_first is True

    guard.record_trade(event, today=today)

    may_attempt_second, reason = guard.should_attempt(event, today=today)
    assert may_attempt_second is False
    assert "already traded" in reason


def test_guard_allows_different_ticker_same_day(tmp_path):
    guard = DailyTradeGuard(max_trades_per_day=5, state_path=tmp_path / "state.json")
    today = date(2026, 8, 26)
    nvda_event = make_event("earnings_surprise", ticker="NVDA", confidence=0.8)
    aapl_event = make_event("earnings_surprise", ticker="AAPL", confidence=0.8)

    guard.record_trade(nvda_event, today=today)
    may_attempt, _ = guard.should_attempt(aapl_event, today=today)
    assert may_attempt is True


def test_guard_enforces_max_trades_per_day(tmp_path):
    guard = DailyTradeGuard(max_trades_per_day=2, state_path=tmp_path / "state.json")
    today = date(2026, 8, 26)
    events = [make_event("earnings_surprise", ticker=t, confidence=0.8) for t in ["AAPL", "MSFT", "TSLA"]]

    for event in events[:2]:
        may_attempt, _ = guard.should_attempt(event, today=today)
        assert may_attempt is True
        guard.record_trade(event, today=today)

    may_attempt_third, reason = guard.should_attempt(events[2], today=today)
    assert may_attempt_third is False
    assert "MAX_TRADES_PER_DAY" in reason


def test_guard_resets_on_new_day(tmp_path):
    guard = DailyTradeGuard(max_trades_per_day=1, state_path=tmp_path / "state.json")
    event = make_event("earnings_surprise", ticker="NVDA", confidence=0.8)

    guard.record_trade(event, today=date(2026, 8, 26))
    may_attempt_same_day, _ = guard.should_attempt(event, today=date(2026, 8, 26))
    assert may_attempt_same_day is False

    may_attempt_next_day, _ = guard.should_attempt(event, today=date(2026, 8, 27))
    assert may_attempt_next_day is True


# -- daily trade guard: persistence across restart ---------------------------
# Reproduces exactly what happened live on 2026-08-28: a mid-signal
# restart lost the in-memory guard state and let NVDA's earnings_surprise
# event fire a second, duplicate entry. state_path now persists across
# separate DailyTradeGuard instances, simulating a process restart.

def test_guard_state_persists_across_restart_and_blocks_duplicate(tmp_path):
    state_path = tmp_path / "trade_guard_state.json"
    today = date(2026, 8, 28)
    event = make_event("earnings_surprise", ticker="NVDA", sentiment_score=0.733, confidence=0.667)

    guard_before_restart = DailyTradeGuard(max_trades_per_day=5, state_path=state_path, today=today)
    may_attempt, _ = guard_before_restart.should_attempt(event, today=today)
    assert may_attempt is True
    guard_before_restart.record_trade(event, today=today)

    # Simulate a process restart: a brand new instance, no shared memory
    # with the one above -- only the state file connects them.
    guard_after_restart = DailyTradeGuard(max_trades_per_day=5, state_path=state_path, today=today)

    may_attempt_after_restart, reason = guard_after_restart.should_attempt(event, today=today)
    assert may_attempt_after_restart is False
    assert "already traded" in reason


def test_guard_does_not_load_state_from_a_previous_day(tmp_path):
    # The guard is daily, not permanent -- a state file left over from
    # yesterday must not carry a "traded" key forward into today.
    state_path = tmp_path / "trade_guard_state.json"
    event = make_event("earnings_surprise", ticker="NVDA", confidence=0.8)

    yesterday_guard = DailyTradeGuard(max_trades_per_day=5, state_path=state_path, today=date(2026, 8, 27))
    yesterday_guard.record_trade(event, today=date(2026, 8, 27))

    today_guard = DailyTradeGuard(max_trades_per_day=5, state_path=state_path, today=date(2026, 8, 28))
    may_attempt, _ = today_guard.should_attempt(event, today=date(2026, 8, 28))
    assert may_attempt is True  # not blocked -- yesterday's entry must not carry over


def test_guard_state_file_survives_a_second_trade_same_day(tmp_path):
    # Not just the first trade -- the persisted set must accumulate, so a
    # restart after the *second* same-day trade still blocks both.
    state_path = tmp_path / "trade_guard_state.json"
    today = date(2026, 8, 28)
    nvda_event = make_event("earnings_surprise", ticker="NVDA", confidence=0.8)
    aapl_event = make_event("earnings_surprise", ticker="AAPL", confidence=0.8)

    guard1 = DailyTradeGuard(max_trades_per_day=5, state_path=state_path, today=today)
    guard1.record_trade(nvda_event, today=today)
    guard1.record_trade(aapl_event, today=today)

    guard2 = DailyTradeGuard(max_trades_per_day=5, state_path=state_path, today=today)
    nvda_attempt, _ = guard2.should_attempt(nvda_event, today=today)
    aapl_attempt, _ = guard2.should_attempt(aapl_event, today=today)
    assert nvda_attempt is False
    assert aapl_attempt is False


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


# -- exit logic: decide_exit() is pure — no network calls -------------------
# Take-profit and max-hold ONLY. No stop-loss / loss-triggered branch
# exists anywhere in this module; see the module docstring's "Exit logic"
# note for why (loss is already bounded to premium paid by construction,
# since every position is bought/long -- see the no-short-positions tests
# in this file already).

def test_exit_reason_has_no_stop_loss_member():
    # Structural guarantee, not just behavioral: the enum itself has no
    # loss-triggered member to accidentally wire up later.
    assert {m.value for m in ExitReason} == {"take_profit", "max_hold"}


def test_take_profit_fires_at_threshold():
    assert decide_exit(unrealized_plpc=TAKE_PROFIT_PCT, days_to_expiry=10) == ExitReason.TAKE_PROFIT


def test_take_profit_does_not_fire_just_below_threshold():
    assert decide_exit(unrealized_plpc=TAKE_PROFIT_PCT - 0.01, days_to_expiry=10) is None


def test_max_hold_fires_with_fewer_days_than_threshold():
    assert decide_exit(unrealized_plpc=0.0, days_to_expiry=MAX_HOLD_DAYS_BEFORE_EXPIRY - 1) == ExitReason.MAX_HOLD


def test_max_hold_does_not_fire_at_exactly_the_threshold():
    # Boundary: exactly MAX_HOLD_DAYS_BEFORE_EXPIRY days left is still
    # enough runway -- the condition is strictly "fewer than".
    assert decide_exit(unrealized_plpc=0.0, days_to_expiry=MAX_HOLD_DAYS_BEFORE_EXPIRY) is None


def test_neither_exit_fires_when_flat_and_far_from_expiry():
    assert decide_exit(unrealized_plpc=0.1, days_to_expiry=10) is None


def test_take_profit_takes_priority_over_max_hold_when_both_true():
    result = decide_exit(unrealized_plpc=TAKE_PROFIT_PCT + 0.1, days_to_expiry=0)
    assert result == ExitReason.TAKE_PROFIT


def test_no_stop_loss_regardless_of_how_negative_unrealized_pl_is():
    # The core guarantee this feature must not violate: a position stays
    # open on a loss, no matter how large, as long as it isn't also
    # within MAX_HOLD_DAYS_BEFORE_EXPIRY of expiring.
    assert decide_exit(unrealized_plpc=-0.99, days_to_expiry=10) is None
    assert decide_exit(unrealized_plpc=-0.01, days_to_expiry=10) is None


# -- parse_occ_expiration() ---------------------------------------------------

def test_parse_occ_expiration_valid_symbol():
    assert parse_occ_expiration("NVDA260831C00222500") == date(2026, 8, 31)


def test_parse_occ_expiration_invalid_symbol_returns_none():
    assert parse_occ_expiration("not-an-option-symbol") is None


# -- OpenPositionsIndex: reliable entry/exit pairing for the post-mortem ----
# Replaces the old best-effort trades.jsonl scan, confirmed live to fail
# 100% of the time in practice (0 of 3 real exits linked), because
# logrotate rotates data/*.jsonl daily and the scan never looked past the
# live file. This index is a plain .json file, immune to that rotation
# by construction.

def test_open_positions_index_records_and_peeks_entry(tmp_path):
    index = OpenPositionsIndex(state_path=tmp_path / "index.json")
    index.record_entry(
        "NVDA260831C00222500", trade_id="trade-a", order_id="order-a",
        event_type="earnings_surprise", ticker="NVDA",
    )

    entries = index.peek_entries("NVDA260831C00222500")
    assert entries is not None
    assert len(entries) == 1
    assert entries[0]["trade_id"] == "trade-a"
    assert entries[0]["order_id"] == "order-a"
    assert entries[0]["event_type"] == "earnings_surprise"
    assert entries[0]["ticker"] == "NVDA"


def test_open_positions_index_peek_does_not_remove(tmp_path):
    index = OpenPositionsIndex(state_path=tmp_path / "index.json")
    index.record_entry(
        "NVDA260831C00222500", trade_id="trade-a", order_id="order-a",
        event_type="earnings_surprise", ticker="NVDA",
    )

    index.peek_entries("NVDA260831C00222500")
    assert index.peek_entries("NVDA260831C00222500") is not None  # still there — peek must not consume


def test_open_positions_index_remove_entries_clears_the_symbol(tmp_path):
    index = OpenPositionsIndex(state_path=tmp_path / "index.json")
    index.record_entry(
        "NVDA260831C00222500", trade_id="trade-a", order_id="order-a",
        event_type="earnings_surprise", ticker="NVDA",
    )

    index.remove_entries("NVDA260831C00222500")

    assert index.peek_entries("NVDA260831C00222500") is None


def test_open_positions_index_survives_a_simulated_restart(tmp_path):
    state_path = tmp_path / "index.json"
    index_before_restart = OpenPositionsIndex(state_path=state_path)
    index_before_restart.record_entry(
        "NVDA260831C00222500", trade_id="trade-a", order_id="order-a",
        event_type="earnings_surprise", ticker="NVDA",
    )

    # Simulate a process restart: a brand new instance, no shared memory
    # with the one above -- only the state file connects them.
    index_after_restart = OpenPositionsIndex(state_path=state_path)
    entries = index_after_restart.peek_entries("NVDA260831C00222500")

    assert entries is not None
    assert entries[0]["trade_id"] == "trade-a"


def test_open_positions_index_resolves_two_entries_on_same_symbol_without_collision(tmp_path):
    # The exact ambiguity case: the same contract symbol bought twice on
    # different days (Alpaca merges same-symbol buys into one combined
    # position). Both entries must survive with their own trade_id --
    # neither should silently overwrite the other.
    state_path = tmp_path / "index.json"
    index_day1 = OpenPositionsIndex(state_path=state_path)
    index_day1.record_entry(
        "NVDA260831C00222500", trade_id="trade-day1", order_id="order-day1",
        event_type="earnings_surprise", ticker="NVDA",
    )

    # Simulate a restart the next day, then a second entry on the same symbol.
    index_day2 = OpenPositionsIndex(state_path=state_path)
    index_day2.record_entry(
        "NVDA260831C00222500", trade_id="trade-day2", order_id="order-day2",
        event_type="earnings_surprise", ticker="NVDA",
    )

    entries = index_day2.peek_entries("NVDA260831C00222500")
    assert entries is not None
    assert len(entries) == 2
    assert {e["trade_id"] for e in entries} == {"trade-day1", "trade-day2"}

    # Closing removes both at once -- Alpaca closes them as one combined position.
    index_day2.remove_entries("NVDA260831C00222500")
    assert index_day2.peek_entries("NVDA260831C00222500") is None


# -- OpenPositionsIndex wired into execute_directional/_close_position ------

def test_execute_directional_records_entry_in_position_index(tmp_path):
    executor = OptionsExecutor(
        api_key="fake", secret_key="fake", dry_run=False,
        position_index_path=tmp_path / "index.json",
    )
    executor.get_equity = Mock(return_value=100_000.0)
    executor._spot_price = Mock(return_value=200.0)
    executor.select_contract = Mock(return_value=_fake_contract("NVDA260831C00200000"))
    executor._latest_option_ask = Mock(return_value=2.0)
    executor.trading_client.submit_order = Mock(return_value=SimpleNamespace(id="entry-order-id"))

    event = make_event("earnings_surprise", ticker="NVDA", sentiment_score=0.6, confidence=0.8)
    decision = TradeDecision(TradeMode.DIRECTIONAL_CALL, "cleared all gates")

    result = executor.execute_directional(event, decision, today=date(2026, 8, 26))

    assert result.traded is True
    trade_id = result.extra["trade_id"]
    assert trade_id

    entries = executor.position_index.peek_entries("NVDA260831C00200000")
    assert entries is not None
    assert entries[0]["trade_id"] == trade_id
    assert entries[0]["order_id"] == "entry-order-id"
    assert entries[0]["event_type"] == "earnings_surprise"
    assert entries[0]["ticker"] == "NVDA"


def test_dry_run_entry_does_not_populate_position_index(tmp_path):
    # A dry-run entry never opens a real position, so it must not create
    # an index entry that a real close could later (incorrectly) consume.
    executor = OptionsExecutor(
        api_key="fake", secret_key="fake", dry_run=True,
        position_index_path=tmp_path / "index.json",
    )
    executor.get_equity = Mock(return_value=100_000.0)
    executor._spot_price = Mock(return_value=200.0)
    executor.select_contract = Mock(return_value=_fake_contract("NVDA260831C00200000"))
    executor._latest_option_ask = Mock(return_value=2.0)

    event = make_event("earnings_surprise", ticker="NVDA", sentiment_score=0.6, confidence=0.8)
    decision = TradeDecision(TradeMode.DIRECTIONAL_CALL, "cleared all gates")

    result = executor.execute_directional(event, decision, today=date(2026, 8, 26))

    assert result.traded is False
    assert result.extra["trade_id"]  # still stamped, for the audit trail
    assert executor.position_index.peek_entries("NVDA260831C00200000") is None


def test_close_position_consumes_and_clears_index_entry(monkeypatch, tmp_path):
    import agent.options_executor as oe_module
    monkeypatch.setattr(oe_module, "EXITS_LOG_PATH", tmp_path / "exits.jsonl")

    executor = OptionsExecutor(
        api_key="fake", secret_key="fake", dry_run=False,
        position_index_path=tmp_path / "index.json",
    )
    executor.position_index.record_entry(
        "NVDA260831C00222500", trade_id="trade-abc", order_id="order-abc",
        event_type="earnings_surprise", ticker="NVDA",
    )
    executor._latest_option_bid = Mock(return_value=3.50)
    executor.trading_client.submit_order = Mock(return_value=SimpleNamespace(id="close-order-id"))

    position = _fake_position("NVDA260831C00222500", unrealized_plpc=TAKE_PROFIT_PCT + 0.1)
    record = executor._close_position(
        position, ExitReason.TAKE_PROFIT, TAKE_PROFIT_PCT + 0.1, days_to_expiry=5, expiration=date(2026, 8, 31),
    )

    assert record["closed"] is True
    assert record["entry_reference"] is not None
    assert record["entry_reference"][0]["trade_id"] == "trade-abc"

    # Cleared after a real close -- the position is genuinely gone.
    assert executor.position_index.peek_entries("NVDA260831C00222500") is None


def test_skipped_close_does_not_clear_index_entry(monkeypatch, tmp_path):
    # A close attempt that doesn't actually execute (no live bid quote)
    # must not lose the index entry -- the position is still open and
    # needs to resolve correctly on the next poll.
    import agent.options_executor as oe_module
    monkeypatch.setattr(oe_module, "EXITS_LOG_PATH", tmp_path / "exits.jsonl")

    executor = OptionsExecutor(
        api_key="fake", secret_key="fake", dry_run=False,
        position_index_path=tmp_path / "index.json",
    )
    executor.position_index.record_entry(
        "NVDA260831C00222500", trade_id="trade-abc", order_id="order-abc",
        event_type="earnings_surprise", ticker="NVDA",
    )
    executor._latest_option_bid = Mock(return_value=None)  # no live quote

    position = _fake_position("NVDA260831C00222500", unrealized_plpc=TAKE_PROFIT_PCT + 0.1)
    record = executor._close_position(
        position, ExitReason.TAKE_PROFIT, TAKE_PROFIT_PCT + 0.1, days_to_expiry=5, expiration=date(2026, 8, 31),
    )

    assert record["closed"] is False
    assert executor.position_index.peek_entries("NVDA260831C00222500") is not None


# -- check_and_close_positions() / _close_position() live orchestration -----

def _fake_position(
    symbol, unrealized_plpc, side=PositionSide.LONG, qty="6",
    asset_class=AssetClass.US_OPTION, unrealized_pl="120.0",
):
    return SimpleNamespace(
        symbol=symbol, asset_class=asset_class, side=side, qty=qty,
        unrealized_plpc=str(unrealized_plpc), unrealized_pl=unrealized_pl,
    )


def _executor_for_exit_tests(dry_run=True, exits_log_path=None, monkeypatch=None):
    executor = OptionsExecutor(api_key="fake", secret_key="fake", dry_run=dry_run)
    executor.trading_client.submit_order = Mock(return_value=SimpleNamespace(id="close-order-id"))
    if exits_log_path is not None and monkeypatch is not None:
        import agent.options_executor as oe_module
        monkeypatch.setattr(oe_module, "EXITS_LOG_PATH", exits_log_path)
    return executor


def test_check_and_close_positions_closes_take_profit_position(monkeypatch, tmp_path):
    executor = _executor_for_exit_tests(dry_run=False, exits_log_path=tmp_path / "exits.jsonl", monkeypatch=monkeypatch)
    today = date(2026, 8, 26)
    position = _fake_position("NVDA260831C00222500", unrealized_plpc=TAKE_PROFIT_PCT + 0.1)
    executor.trading_client.get_all_positions = Mock(return_value=[position])
    executor._latest_option_bid = Mock(return_value=3.50)

    results = executor.check_and_close_positions(today=today)

    assert len(results) == 1
    assert results[0]["closed"] is True
    assert results[0]["exit_reason"] == "take_profit"
    executor.trading_client.submit_order.assert_called_once()
    submitted = executor.trading_client.submit_order.call_args[0][0]
    assert submitted.side == OrderSide.SELL
    assert submitted.symbol == "NVDA260831C00222500"
    assert submitted.qty == 6


def test_check_and_close_positions_closes_max_hold_position(monkeypatch, tmp_path):
    executor = _executor_for_exit_tests(dry_run=False, exits_log_path=tmp_path / "exits.jsonl", monkeypatch=monkeypatch)
    today = date(2026, 8, 30)  # symbol expires 2026-08-31 -> 1 day left, < MAX_HOLD_DAYS_BEFORE_EXPIRY (2)
    position = _fake_position("NVDA260831C00222500", unrealized_plpc=0.0)
    executor.trading_client.get_all_positions = Mock(return_value=[position])
    executor._latest_option_bid = Mock(return_value=1.10)

    results = executor.check_and_close_positions(today=today)

    assert len(results) == 1
    assert results[0]["closed"] is True
    assert results[0]["exit_reason"] == "max_hold"
    assert results[0]["days_to_expiry"] == 1
    executor.trading_client.submit_order.assert_called_once()


def test_check_and_close_positions_leaves_position_open_when_neither_condition_met(monkeypatch, tmp_path):
    executor = _executor_for_exit_tests(dry_run=False, exits_log_path=tmp_path / "exits.jsonl", monkeypatch=monkeypatch)
    today = date(2026, 8, 26)  # expires 2026-08-31 -> 5 days left, plenty of runway
    position = _fake_position("NVDA260831C00222500", unrealized_plpc=0.1)  # up 10%, below TAKE_PROFIT_PCT
    executor.trading_client.get_all_positions = Mock(return_value=[position])
    executor._latest_option_bid = Mock(return_value=2.50)

    results = executor.check_and_close_positions(today=today)

    assert results == []
    executor.trading_client.submit_order.assert_not_called()


def test_check_and_close_positions_never_fires_on_a_loss_alone(monkeypatch, tmp_path):
    executor = _executor_for_exit_tests(dry_run=False, exits_log_path=tmp_path / "exits.jsonl", monkeypatch=monkeypatch)
    today = date(2026, 8, 26)  # 5 days to expiry, plenty of runway
    position = _fake_position("NVDA260831C00222500", unrealized_plpc=-0.90)  # down 90%
    executor.trading_client.get_all_positions = Mock(return_value=[position])
    executor._latest_option_bid = Mock(return_value=0.25)

    results = executor.check_and_close_positions(today=today)

    assert results == []
    executor.trading_client.submit_order.assert_not_called()


def test_check_and_close_positions_skips_non_option_positions(monkeypatch, tmp_path):
    executor = _executor_for_exit_tests(dry_run=False, exits_log_path=tmp_path / "exits.jsonl", monkeypatch=monkeypatch)
    equity_position = _fake_position("AAPL", unrealized_plpc=0.99, asset_class=AssetClass.US_EQUITY)
    executor.trading_client.get_all_positions = Mock(return_value=[equity_position])

    results = executor.check_and_close_positions(today=date(2026, 8, 26))

    assert results == []
    executor.trading_client.submit_order.assert_not_called()


def test_close_position_refuses_short_positions(monkeypatch, tmp_path):
    executor = _executor_for_exit_tests(dry_run=False, exits_log_path=tmp_path / "exits.jsonl", monkeypatch=monkeypatch)
    short_position = _fake_position("NVDA260831C00222500", unrealized_plpc=TAKE_PROFIT_PCT + 0.1, side=PositionSide.SHORT)
    executor.trading_client.get_all_positions = Mock(return_value=[short_position])

    results = executor.check_and_close_positions(today=date(2026, 8, 26))

    assert len(results) == 1
    assert results[0]["closed"] is False
    assert "never opens short positions" in results[0]["detail"]
    executor.trading_client.submit_order.assert_not_called()


def test_dry_run_close_does_not_submit_order(monkeypatch, tmp_path):
    executor = _executor_for_exit_tests(dry_run=True, exits_log_path=tmp_path / "exits.jsonl", monkeypatch=monkeypatch)
    position = _fake_position("NVDA260831C00222500", unrealized_plpc=TAKE_PROFIT_PCT + 0.1)
    executor.trading_client.get_all_positions = Mock(return_value=[position])
    executor._latest_option_bid = Mock(return_value=3.50)

    results = executor.check_and_close_positions(today=date(2026, 8, 26))

    assert len(results) == 1
    assert results[0]["closed"] is False
    assert results[0]["dry_run"] is True
    assert results[0]["detail"].startswith("[DRY RUN]")
    executor.trading_client.submit_order.assert_not_called()


def test_close_detail_is_clearly_distinguishable_from_an_open(monkeypatch, tmp_path):
    # Regression guard for the exact concern raised: a close must never
    # read like a new buy-to-open in the logs.
    import agent.options_executor as oe_module
    monkeypatch.setattr(oe_module, "EXITS_LOG_PATH", tmp_path / "exits.jsonl")

    executor = OptionsExecutor(api_key="fake", secret_key="fake", dry_run=True, position_index_path=tmp_path / "index.json")
    position = _fake_position("NVDA260831C00222500", unrealized_plpc=TAKE_PROFIT_PCT + 0.1)
    executor._latest_option_bid = Mock(return_value=3.50)
    executor._find_entry_reference = Mock(return_value=None)

    record = executor._close_position(
        position, ExitReason.TAKE_PROFIT, TAKE_PROFIT_PCT + 0.1, days_to_expiry=5, expiration=date(2026, 8, 31),
    )

    assert record["action"] == "close_position"
    assert "SELL-TO-CLOSE" in record["detail"]
    assert "BUY" not in record["detail"]
