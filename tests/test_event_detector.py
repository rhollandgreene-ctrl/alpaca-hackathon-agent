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
    monkeypatch.setattr(
        detector, "_search_news",
        lambda query, max_results=None, topic=None, time_range=None: snippets,
    )
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
    monkeypatch.setattr(
        detector, "_search_news",
        lambda query, max_results=None, topic=None, time_range=None: BULLISH_SNIPPETS,
    )

    events = detector.check_macro_events(now=datetime(2026, 8, 25, tzinfo=timezone.utc))

    assert events == []


def test_search_news_without_tavily_key_returns_empty_list():
    detector = EventDetector(watchlist=[], macro_events=[], tavily_api_key="")
    assert detector._search_news("anything") == []


def test_search_news_cached_reuses_result_for_same_ticker_and_day(monkeypatch):
    call_count = {"n": 0}

    def counting_search(query, max_results=None, topic=None, time_range=None):
        call_count["n"] += 1
        return [f"result for {query} (#{call_count['n']})"]

    detector = EventDetector(watchlist=[], macro_events=[], tavily_api_key="dummy")
    monkeypatch.setattr(detector, "_search_news", counting_search)

    today = date(2026, 8, 26)
    first = detector._search_news_cached("NVDA earnings preview", "NVDA", today)
    second = detector._search_news_cached("NVDA earnings preview", "NVDA", today)

    assert call_count["n"] == 1  # second call within the same day reused the cache
    assert first == second

    # A later poll cycle the following day is a cache miss and searches again.
    tomorrow = date(2026, 8, 27)
    third = detector._search_news_cached("NVDA earnings preview", "NVDA", tomorrow)

    assert call_count["n"] == 2
    assert third != first


@pytest.mark.parametrize(
    "snippets,expect_sign",
    [
        (["stocks surge on bullish rally, Fed beats expectations"], 1),
        (["stocks plunge as bearish selloff deepens, Fed misses expectations"], -1),
        ([], 0),
    ],
)
def test_score_sentiment_direction(snippets, expect_sign):
    score, confidence = score_sentiment(snippets, domain="earnings")
    assert -1.0 <= score <= 1.0
    assert 0.0 <= confidence <= 1.0
    if expect_sign > 0:
        assert score > 0
    elif expect_sign < 0:
        assert score < 0
    else:
        assert score == 0.0
        assert confidence == 0.0


# -- word-boundary matching -------------------------------------------------

def test_word_boundary_prevents_transmission_false_positive():
    # "miss" must not match inside "transmission" — this was the live bug
    # found scoring real Jackson Hole coverage (Fed's "transmission of
    # monetary policy" language triggered false negative hits).
    snippets = ["The committee discussed the transmission of monetary policy this week."]
    score, confidence = score_sentiment(snippets, domain="macro")
    assert score == 0.0
    assert confidence == 0.0


def test_word_boundary_still_matches_standalone_miss():
    # The fix must not overcorrect — a real standalone "miss" should still count.
    snippets = ["The company reported a miss versus analyst expectations."]
    score, _ = score_sentiment(snippets, domain="earnings")
    assert score < 0


# -- domain-lexicon routing --------------------------------------------------

def test_domain_lexicon_flips_polarity_for_shared_keyword():
    # Same text, opposite sign depending on domain: weak earnings are
    # bearish for a stock; weak macro/jobs data is dovish (rate-cut odds
    # rise) and so bullish for risk assets.
    weak_text = ["Guidance came in weak for the quarter."]

    earnings_score, _ = score_sentiment(weak_text, domain="earnings")
    macro_score, _ = score_sentiment(weak_text, domain="macro")

    assert earnings_score < 0
    assert macro_score > 0


def test_domain_lexicon_flips_polarity_for_strong_keyword():
    strong_text = ["The report came in strong across the board."]

    earnings_score, _ = score_sentiment(strong_text, domain="earnings")
    macro_score, _ = score_sentiment(strong_text, domain="macro")

    assert earnings_score > 0
    assert macro_score < 0


# -- duplicate-snippet deduplication -----------------------------------------

def test_duplicate_snippets_yield_lower_confidence_than_distinct_ones():
    # Same underlying fact restated three ways should count as ~1
    # corroborating source, not three — so confidence should land lower
    # than three genuinely distinct positive signals.
    duplicate_snippets = [
        "Nvidia beat earnings estimates by 6 percent this quarter",
        "Nvidia beat earnings estimates by 6 percent this quarter, results show",
        "Nvidia beat earnings estimates by 6 percent this quarter, per the filing",
    ]
    distinct_snippets = [
        "Nvidia beat earnings estimates this quarter",
        "Analysts upgraded Nvidia stock following strong guidance",
        "Nvidia shares rally as the growth outlook improves",
    ]

    _, duplicate_confidence = score_sentiment(duplicate_snippets, domain="earnings")
    _, distinct_confidence = score_sentiment(distinct_snippets, domain="earnings")

    assert duplicate_confidence < distinct_confidence


def test_dedupe_snippets_collapses_near_identical_text():
    from agent.event_detector import _dedupe_snippets

    snippets = [
        "Nvidia beat earnings estimates by 6 percent this quarter",
        "Nvidia beat earnings estimates by 6 percent this quarter, results show",
        "Completely unrelated sentence about a different topic entirely",
    ]

    deduped = _dedupe_snippets(snippets)

    assert len(deduped) == 2
