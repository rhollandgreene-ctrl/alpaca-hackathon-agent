"""
Tests for mcp_server/logic.py — the testable logic behind each MCP tool,
kept separate from MCP protocol plumbing.

All file-backed tools are exercised against tmp_path fixtures via
monkeypatching logic.EVENTS_LOG_PATH / logic.TRADES_LOG_PATH /
logic.LESSONS_LEARNED_PATH — never the real data/*.jsonl files this
project's live runs have seeded, so running this suite can't corrupt
real demo history. Alpaca-backed tools use fake client objects
(SimpleNamespace/Mock), so nothing here makes a live network call.
"""

import json
from types import SimpleNamespace
from unittest.mock import Mock

from mcp_server import logic


def _write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


# -- read_jsonl ---------------------------------------------------------------

def test_read_jsonl_returns_newest_first(tmp_path):
    path = tmp_path / "log.jsonl"
    _write_jsonl(path, [{"id": 1}, {"id": 2}, {"id": 3}])

    records = logic.read_jsonl(path)

    assert [r["id"] for r in records] == [3, 2, 1]


def test_read_jsonl_respects_n_limit(tmp_path):
    path = tmp_path / "log.jsonl"
    _write_jsonl(path, [{"id": 1}, {"id": 2}, {"id": 3}])

    records = logic.read_jsonl(path, n=2)

    assert [r["id"] for r in records] == [3, 2]


def test_read_jsonl_missing_file_returns_empty(tmp_path):
    assert logic.read_jsonl(tmp_path / "does_not_exist.jsonl") == []


def test_read_jsonl_skips_malformed_lines(tmp_path):
    path = tmp_path / "log.jsonl"
    path.write_text('{"id": 1}\nnot json\n{"id": 2}\n', encoding="utf-8")

    records = logic.read_jsonl(path)

    assert [r["id"] for r in records] == [2, 1]


# -- get_recent_signals ---------------------------------------------------------

def test_get_recent_signals_reads_real_file(monkeypatch, tmp_path):
    path = tmp_path / "events.jsonl"
    _write_jsonl(path, [{"event_type": "earnings_surprise", "ticker": "NVDA"}])
    monkeypatch.setattr(logic, "EVENTS_LOG_PATH", path)

    result = logic.get_recent_signals(n=5)

    assert result["is_mocked"] is False
    assert result["count"] == 1
    assert result["signals"][0]["ticker"] == "NVDA"
    assert result["note"] is None


def test_get_recent_signals_empty_has_helpful_note(monkeypatch, tmp_path):
    monkeypatch.setattr(logic, "EVENTS_LOG_PATH", tmp_path / "missing.jsonl")

    result = logic.get_recent_signals(n=5)

    assert result["count"] == 0
    assert "event_detector" in result["note"]


# -- get_open_positions ---------------------------------------------------------

def test_get_open_positions_maps_fields():
    fake_position = SimpleNamespace(
        symbol="NVDA260828C00212500", qty="1", side=SimpleNamespace(value="long"),
        avg_entry_price="6.78", current_price="7.10", market_value="710.0",
        unrealized_pl="32.0", unrealized_plpc="0.047",
    )
    fake_client = Mock()
    fake_client.get_all_positions.return_value = [fake_position]

    result = logic.get_open_positions(fake_client)

    assert result["is_mocked"] is False
    assert result["count"] == 1
    position = result["positions"][0]
    assert position["symbol"] == "NVDA260828C00212500"
    assert position["side"] == "long"
    assert position["avg_entry_price"] == 6.78


def test_get_open_positions_empty():
    fake_client = Mock()
    fake_client.get_all_positions.return_value = []

    result = logic.get_open_positions(fake_client)

    assert result["count"] == 0
    assert result["positions"] == []


# -- compute_market_regime -------------------------------------------------------

def _fake_bars_client(closes_newest_first):
    """closes_newest_first mirrors Alpaca's sort='desc' response ordering."""
    bars = [SimpleNamespace(close=c) for c in closes_newest_first]
    client = Mock()
    client.get_stock_bars.return_value = {"SPY": bars}
    return client


def test_compute_market_regime_high_vol():
    # Alternating +/-5% swings -> high realized volatility.
    closes = [100, 105, 100, 105, 100, 105, 100, 105, 100, 105, 100]
    client = _fake_bars_client(list(reversed(closes)))  # newest-first

    result = logic.compute_market_regime(client, symbol="SPY")

    assert result["is_mocked"] is False
    assert result["regime"] == "high_vol"


def test_compute_market_regime_trending_up():
    # Steadily rising, consistent direction, large net move.
    closes = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 112]
    client = _fake_bars_client(list(reversed(closes)))

    result = logic.compute_market_regime(client, symbol="SPY")

    assert result["regime"] == "trending_up"
    assert result["basis"]["net_move_pct"] > 0


def test_compute_market_regime_choppy():
    # Small back-and-forth moves, no real net move, no high vol.
    closes = [100, 100.5, 100.2, 100.6, 100.3, 100.5, 100.2, 100.4, 100.3, 100.5, 100.4]
    client = _fake_bars_client(list(reversed(closes)))

    result = logic.compute_market_regime(client, symbol="SPY")

    assert result["regime"] == "choppy"


def test_compute_market_regime_insufficient_data():
    client = _fake_bars_client([100, 101])  # only 2 bars

    result = logic.compute_market_regime(client, symbol="SPY")

    assert result["regime"] == "unknown"
    assert "insufficient" in result["reason"]


# -- get_lessons_learned ----------------------------------------------------------

def test_get_lessons_learned_reads_file(monkeypatch, tmp_path):
    path = tmp_path / "lessons.json"
    path.write_text(json.dumps([{"id": "test-lesson", "title": "Test"}]), encoding="utf-8")
    monkeypatch.setattr(logic, "LESSONS_LEARNED_PATH", path)

    result = logic.get_lessons_learned()

    assert result["is_mocked"] is False
    assert result["count"] == 1
    assert result["lessons"][0]["id"] == "test-lesson"


def test_get_lessons_learned_missing_file(monkeypatch, tmp_path):
    monkeypatch.setattr(logic, "LESSONS_LEARNED_PATH", tmp_path / "missing.json")

    result = logic.get_lessons_learned()

    assert result["count"] == 0
    assert result["lessons"] == []


# -- get_icarus_signals -----------------------------------------------------------

def test_get_icarus_signals_is_clearly_mocked():
    result = logic.get_icarus_signals()

    assert result["is_mocked"] is True
    assert len(result["signals"]) > 0
    assert all("ILLUSTRATIVE" in s["reasoning"] for s in result["signals"])


# -- get_health_status --------------------------------------------------------------

def test_health_status_reports_missing_keys(monkeypatch):
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    result = logic.get_health_status(trading_client=None)

    assert result["alpaca"]["api_key_configured"] is False
    assert result["alpaca"]["reachable"] is False
    assert result["tavily"]["api_key_configured"] is False
    assert result["anthropic"]["api_key_configured"] is False


def test_health_status_reachable_client(monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "fake")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "fake")

    fake_account = SimpleNamespace(status="ACTIVE", options_trading_level=3, equity="100000.0")
    fake_client = Mock()
    fake_client.get_account.return_value = fake_account

    result = logic.get_health_status(trading_client=fake_client)

    assert result["alpaca"]["reachable"] is True
    assert result["alpaca"]["error"] is None
    assert result["alpaca"]["detail"]["equity"] == 100000.0


def test_health_status_unreachable_client_reports_error(monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "fake")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "fake")

    fake_client = Mock()
    fake_client.get_account.side_effect = ConnectionError("timed out")

    result = logic.get_health_status(trading_client=fake_client)

    assert result["alpaca"]["reachable"] is False
    assert "timed out" in result["alpaca"]["error"]


# -- explain_trade_decision -----------------------------------------------------

def test_explain_trade_decision_no_data(monkeypatch, tmp_path):
    monkeypatch.setattr(logic, "TRADES_LOG_PATH", tmp_path / "missing.jsonl")

    result = logic.explain_trade_decision()

    assert result["found"] is False
    assert "options_executor" in result["note"]


def test_explain_trade_decision_defaults_to_most_recent(monkeypatch, tmp_path):
    path = tmp_path / "trades.jsonl"
    _write_jsonl(path, [
        {"event": {"ticker": "AAPL"}, "order_id": "1"},
        {"event": {"ticker": "NVDA"}, "order_id": "2"},
    ])
    monkeypatch.setattr(logic, "TRADES_LOG_PATH", path)

    result = logic.explain_trade_decision()

    assert result["found"] is True
    assert result["record"]["order_id"] == "2"  # last written = most recent


def test_explain_trade_decision_by_ticker_checks_resolved_ticker_too(monkeypatch, tmp_path):
    path = tmp_path / "trades.jsonl"
    _write_jsonl(path, [
        {"event": {"ticker": None, "event_type": "macro_jackson_hole"}, "extra": {"ticker_resolved": "SPY"}, "order_id": "1"},
        {"event": {"ticker": "NVDA"}, "extra": {}, "order_id": "2"},
    ])
    monkeypatch.setattr(logic, "TRADES_LOG_PATH", path)

    result = logic.explain_trade_decision(ticker="SPY")

    assert result["found"] is True
    assert result["record"]["order_id"] == "1"


def test_explain_trade_decision_by_order_id(monkeypatch, tmp_path):
    path = tmp_path / "trades.jsonl"
    _write_jsonl(path, [
        {"event": {"ticker": "AAPL"}, "order_id": "1"},
        {"event": {"ticker": "NVDA"}, "order_id": "2"},
    ])
    monkeypatch.setattr(logic, "TRADES_LOG_PATH", path)

    result = logic.explain_trade_decision(order_id="1")

    assert result["found"] is True
    assert result["record"]["event"]["ticker"] == "AAPL"


def test_explain_trade_decision_no_match(monkeypatch, tmp_path):
    path = tmp_path / "trades.jsonl"
    _write_jsonl(path, [{"event": {"ticker": "AAPL"}, "order_id": "1"}])
    monkeypatch.setattr(logic, "TRADES_LOG_PATH", path)

    result = logic.explain_trade_decision(ticker="TSLA")

    assert result["found"] is False


# -- run_bear_case ---------------------------------------------------------------

def test_run_bear_case_returns_checklist_and_context(monkeypatch, tmp_path):
    path = tmp_path / "trades.jsonl"
    _write_jsonl(path, [{"event": {"ticker": "NVDA"}, "order_id": "1"}])
    monkeypatch.setattr(logic, "TRADES_LOG_PATH", path)

    result = logic.run_bear_case(ticker="NVDA")

    assert result["is_mocked"] is False
    assert result["reasoning_aid"] is True
    assert result["context"]["event"]["ticker"] == "NVDA"
    assert len(result["risk_factor_checklist"]) >= 4
    assert all(isinstance(item, str) for item in result["risk_factor_checklist"])


def test_run_bear_case_without_history_still_returns_checklist(monkeypatch, tmp_path):
    monkeypatch.setattr(logic, "TRADES_LOG_PATH", tmp_path / "missing.jsonl")

    result = logic.run_bear_case(ticker="NVDA")

    assert result["context"] is None
    assert len(result["risk_factor_checklist"]) >= 4
