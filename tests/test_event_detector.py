"""
Tests for the Daedalus event detector.

These mock EventDetector._search_news directly, so no network access or
TAVILY_API_KEY is required to run them.
"""

from datetime import date, datetime, timezone

import pytest

from agent.event_detector import Event, EventDetector, score_sentiment

JACKSON_HOLE_EVENT = {
    "name": "Jackson Hole Economic Symposium",
    "event_type": "macro_jackson_hole",
    "date": date(2026, 8, 28),
    "description": "Federal Reserve's annual Jackson Hole Economic Policy Symposium",
    "search_query": "Jackson Hole Symposium Federal Reserve Powell speech",
}

BULLISH_SNIPPETS = [
    "Fed Chair signals dovish pivot at Jackson Hole, markets rally on rate cut hopes",
    "Powell's Jackson Hole speech seen as bullish for risk assets, stocks surge",
]


def _detector_with_stubbed_news(monkeypatch, snippets, **kwargs):
    detector = EventDetector(watchlist=[], macro_events=[JACKSON_HOLE_EVENT], **kwargs)
    monkeypatch.setattr(detector, "_search_news", lambda query, max_results=None: snippets)
    return detector


def test_jackson_hole_event_detected_within_lookahead_window(monkeypatch):
    detector = _detector_with_stubbed_news(monkeypatch, BULLISH_SNIPPETS)

    now = datetime(2026, 8, 25, tzinfo=timezone.utc)  # 3 days before the event
    events = detector.check_macro_events(now=now)

    assert len(events) == 1
    event = events[0]

    assert isinstance(event, Event)
    assert event.event_type == "macro_jackson_hole"
    assert event.ticker is None
    assert "Jackson Hole" in event.description
    assert -1.0 <= event.sentiment_score <= 1.0
    assert 0.0 <= event.confidence <= 1.0
    assert event.sentiment_score > 0  # bullish/dovish snippets should score positive
    assert event.confidence > 0

    # timestamp must be a valid ISO8601 string
    datetime.fromisoformat(event.timestamp)


def test_jackson_hole_event_object_is_json_serializable(monkeypatch):
    detector = _detector_with_stubbed_news(monkeypatch, BULLISH_SNIPPETS)
    now = datetime(2026, 8, 25, tzinfo=timezone.utc)

    event = detector.check_macro_events(now=now)[0]
    payload = event.to_dict()

    assert set(payload.keys()) == {
        "event_type", "ticker", "timestamp", "description",
        "sentiment_score", "confidence",
    }


def test_macro_event_outside_lookahead_window_is_not_detected(monkeypatch):
    detector = _detector_with_stubbed_news(monkeypatch, BULLISH_SNIPPETS)

    now = datetime(2026, 7, 1, tzinfo=timezone.utc)  # ~58 days out, outside 7-day window
    events = detector.check_macro_events(now=now)

    assert events == []


def test_unscheduled_macro_event_is_skipped(monkeypatch):
    unscheduled = {**JACKSON_HOLE_EVENT, "date": None, "event_type": "macro_fomc"}
    detector = EventDetector(watchlist=[], macro_events=[unscheduled])
    monkeypatch.setattr(detector, "_search_news", lambda query, max_results=None: BULLISH_SNIPPETS)

    events = detector.check_macro_events(now=datetime(2026, 8, 25, tzinfo=timezone.utc))

    assert events == []


def test_search_news_without_tavily_key_returns_empty_list():
    detector = EventDetector(watchlist=[], macro_events=[], tavily_api_key="")
    assert detector._search_news("anything") == []


@pytest.mark.parametrize(
    "snippets,expect_sign",
    [
        (["stocks surge on bullish rally, Fed beats expectations"], 1),
        (["stocks plunge as bearish selloff deepens, Fed misses expectations"], -1),
        ([], 0),
    ],
)
def test_score_sentiment_direction(snippets, expect_sign):
    score, confidence = score_sentiment(snippets)
    assert -1.0 <= score <= 1.0
    assert 0.0 <= confidence <= 1.0
    if expect_sign > 0:
        assert score > 0
    elif expect_sign < 0:
        assert score < 0
    else:
        assert score == 0.0
        assert confidence == 0.0
