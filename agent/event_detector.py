"""
Daedalus event detection core.

Watches a ticker watchlist for earnings catalysts and a configurable
registry for macro/economic catalysts (Jackson Hole, FOMC, CPI, ...),
and emits structured Event objects for downstream consumption (options
execution, etc).

Design notes:
- Earnings dates/surprises come from yfinance (no API key required).
- News/sentiment signal comes from Tavily (TAVILY_API_KEY in .env).
- Sentiment scoring is a simple keyword heuristic, not an LLM call —
  crude by design for this session; swap `score_sentiment` for
  something smarter later without touching the rest of the pipeline.
- No persistent connections: `run_once()` does a single detection pass
  (used by tests and callers that want to drive the schedule
  themselves); `run()` wraps it in a sleep loop.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from difflib import SequenceMatcher
from typing import Optional

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional at import time
    load_dotenv = None

try:
    from tavily import TavilyClient
except ImportError:  # pragma: no cover - optional at import time
    TavilyClient = None

try:
    import yfinance as yf
except ImportError:  # pragma: no cover - optional at import time
    yf = None

logger = logging.getLogger("daedalus.event_detector")


# ---------------------------------------------------------------------------
# Config

DEFAULT_WATCHLIST = ["AAPL", "MSFT", "NVDA", "TSLA", "AMZN"]

# Macro/economic event registry. `date` is a `datetime.date` or None
# (unscheduled placeholder — fill in once the calendar is published).
MACRO_EVENTS = [
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
    {
        "name": "CPI Release",
        "event_type": "macro_cpi",
        "date": None,
        "description": "Consumer Price Index inflation data release",
        "search_query": "CPI inflation report release",
    },
]


# ---------------------------------------------------------------------------
# Output schema

@dataclass
class Event:
    event_type: str
    ticker: Optional[str]
    timestamp: str
    description: str
    sentiment_score: float
    confidence: float

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Sentiment heuristic
#
# Two domain-specific lexicons, not one shared list: the same word can mean
# opposite things for a company's earnings vs. macro/Fed commentary. "weak"
# earnings are bearish for the stock; "weak" jobs data is typically dovish
# (more likely to bring rate cuts) and so bullish for risk assets. Route each
# event through the lexicon matching its event_type via score_sentiment(...,
# domain=...).

EARNINGS_LEXICON = {
    "positive": [
        "beat", "beats", "beating", "surge", "surges", "rally", "rallies",
        "bullish", "strong", "upgrade", "upgraded", "optimistic", "growth",
        "soar", "soars", "outperform", "record high", "better-than-expected",
        "raises guidance", "raised guidance",
    ],
    "negative": [
        "miss", "misses", "missed", "plunge", "plunges", "selloff", "sell-off",
        "bearish", "weak", "weaker", "downgrade", "downgraded", "recession",
        "layoff", "layoffs", "warns", "warning", "worse-than-expected",
        "cuts guidance", "lowered guidance",
    ],
}

# Positive = dovish / bullish for risk assets. Negative = hawkish / bearish.
# Note "weak"/"strong" are intentionally inverted vs EARNINGS_LEXICON: weak
# jobs/economic data raises rate-cut odds (bullish for risk), while a strong
# labor market/economy raises hike-or-hold odds (bearish for risk).
MACRO_LEXICON = {
    "positive": [
        "rate cut", "rate cuts", "cuts rates", "cutting rates", "dovish",
        "easing", "ease policy", "pause hikes", "paused rate hikes",
        "weak jobs", "weaker jobs", "soft labor market", "cooling labor market",
        "cooling inflation", "disinflation", "inflation cooling",
        "weak", "weaker",
    ],
    "negative": [
        "rate hike", "rate hikes", "hikes rates", "hiking rates", "hawkish",
        "tightening", "sticky inflation", "inflation surge", "strong jobs",
        "strong labor market", "resilient inflation", "higher for longer",
        "tight labor market", "strong", "stronger",
    ],
}

_LEXICONS = {"earnings": EARNINGS_LEXICON, "macro": MACRO_LEXICON}


def _compile_terms(terms: list[str]) -> re.Pattern:
    # Longest-first so multi-word phrases match before their shorter substrings.
    ordered = sorted(terms, key=len, reverse=True)
    return re.compile(r"\b(?:" + "|".join(re.escape(t) for t in ordered) + r")\b")


_COMPILED = {
    domain: {
        "positive": _compile_terms(lex["positive"]),
        "negative": _compile_terms(lex["negative"]),
    }
    for domain, lex in _LEXICONS.items()
}

# Citation/footnote noise: academic references, footnote markers, and
# "sponsored by"/"presented at" framing read as live signal to a keyword
# scanner but are historical citations, not current reporting.
_CITATION_PATTERNS = [
    re.compile(r"presented at"),
    re.compile(r"sponsored by"),
    re.compile(r"see also"),
    re.compile(r"op\.\s*cit\."),
    re.compile(r"return to text"),
    re.compile(r"\(\d{4}\)[\.,]?\s*[\"“‘]"),  # (2019). "Title..."
    re.compile(r"^\s*\d+\.\s+see\b"),  # footnote marker, e.g. "6. See ..."
]
_CITATION_DOWNWEIGHT = 0.25

# Snippets are treated as near-duplicate (same underlying fact restated) above
# this similarity ratio, and only the first occurrence counts toward confidence.
_DEDUP_SIMILARITY_THRESHOLD = 0.6


def _is_citation_noise(snippet: str) -> bool:
    lowered = snippet.lower()
    return any(p.search(lowered) for p in _CITATION_PATTERNS)


def _normalize_for_dedup(snippet: str) -> str:
    return re.sub(r"\s+", " ", snippet.lower()).strip()


def _dedupe_snippets(snippets: list[str]) -> list[str]:
    """Drop snippets that are near-identical restatements of an earlier one."""
    kept: list[str] = []
    kept_normalized: list[str] = []
    for snippet in snippets:
        normalized = _normalize_for_dedup(snippet)
        if any(
            SequenceMatcher(None, normalized, existing).ratio() >= _DEDUP_SIMILARITY_THRESHOLD
            for existing in kept_normalized
        ):
            continue
        kept.append(snippet)
        kept_normalized.append(normalized)
    return kept


def score_sentiment(snippets: list[str], domain: str = "earnings") -> tuple[float, float]:
    """Domain-aware keyword-count heuristic over search-result snippets.

    Deduplicates near-identical snippets first (one fact restated across
    sources shouldn't count as multiple corroborating signals), then scores
    with word-boundary matching against the lexicon for `domain`
    ("earnings" or "macro") so the same word can carry different polarity
    in each context. Citation/footnote-like snippets are downweighted
    rather than treated as live reporting.

    Returns (sentiment_score in [-1, 1], confidence in [0, 1]).
    confidence reflects agreement across independent, non-duplicate,
    non-citation sources — not raw keyword hit count.
    """
    if not snippets:
        return 0.0, 0.0

    lexicon = _COMPILED.get(domain, _COMPILED["earnings"])
    deduped = _dedupe_snippets(snippets)

    total_pos = 0.0
    total_neg = 0.0
    leaning_weights = []  # (weight, sign) for snippets with any hit

    for snippet in deduped:
        text = snippet.lower()
        weight = _CITATION_DOWNWEIGHT if _is_citation_noise(snippet) else 1.0

        pos_hits = len(lexicon["positive"].findall(text))
        neg_hits = len(lexicon["negative"].findall(text))

        total_pos += pos_hits * weight
        total_neg += neg_hits * weight

        net = pos_hits - neg_hits
        if net != 0:
            leaning_weights.append((weight, 1 if net > 0 else -1))

    total_hits = total_pos + total_neg
    if total_hits == 0:
        return 0.0, 0.0

    score = (total_pos - total_neg) / total_hits
    majority_sign = 0 if total_pos == total_neg else (1 if total_pos > total_neg else -1)

    if majority_sign == 0 or not leaning_weights:
        # No clear directional consensus — cap confidence low regardless of volume.
        confidence = min(1.0, 0.1 * sum(w for w, _ in leaning_weights))
    else:
        total_weight = sum(w for w, _ in leaning_weights)
        agreeing_weight = sum(w for w, sign in leaning_weights if sign == majority_sign)
        agreement_ratio = agreeing_weight / total_weight
        volume_factor = min(1.0, total_weight / 3.0)
        confidence = agreement_ratio * volume_factor

    return round(score, 3), round(confidence, 3)


# ---------------------------------------------------------------------------
# Detector

class EventDetector:
    def __init__(
        self,
        watchlist: Optional[list[str]] = None,
        macro_events: Optional[list[dict]] = None,
        tavily_api_key: Optional[str] = None,
        earnings_lookahead_days: int = 14,
        earnings_lookback_days: int = 3,
        macro_lookahead_days: int = 7,
        macro_lookback_days: int = 1,
        news_max_results: int = 5,
    ):
        self.watchlist = watchlist if watchlist is not None else DEFAULT_WATCHLIST
        self.macro_events = macro_events if macro_events is not None else MACRO_EVENTS
        self.earnings_lookahead_days = earnings_lookahead_days
        self.earnings_lookback_days = earnings_lookback_days
        self.macro_lookahead_days = macro_lookahead_days
        self.macro_lookback_days = macro_lookback_days
        self.news_max_results = news_max_results

        api_key = tavily_api_key if tavily_api_key is not None else os.getenv("TAVILY_API_KEY")
        self._tavily_client = None
        if api_key and TavilyClient is not None:
            self._tavily_client = TavilyClient(api_key=api_key)
        elif not api_key:
            logger.warning("TAVILY_API_KEY not set — news/sentiment search will return no results")
        elif TavilyClient is None:
            logger.warning("tavily-python not installed — news/sentiment search will return no results")

    # -- news/sentiment ----------------------------------------------------

    def _search_news(
        self,
        query: str,
        max_results: Optional[int] = None,
        topic: Optional[str] = None,
        time_range: Optional[str] = None,
    ) -> list[str]:
        """Single seam for hitting Tavily. Tests monkeypatch this directly."""
        if self._tavily_client is None:
            logger.warning("Skipping news search (no Tavily client configured) for query=%r", query)
            return []
        try:
            response = self._tavily_client.search(
                query=query,
                max_results=max_results or self.news_max_results,
                topic=topic,
                time_range=time_range,
            )
            results = response.get("results", []) if isinstance(response, dict) else []
            return [r.get("content", "") for r in results if r.get("content")]
        except Exception:
            logger.exception("Tavily search failed for query=%r", query)
            return []

    # -- earnings ------------------------------------------------------------

    def check_earnings(self, now: Optional[datetime] = None) -> list[Event]:
        if yf is None:
            logger.warning("yfinance not installed — skipping earnings detection")
            return []

        now = now or datetime.now(timezone.utc)
        today = now.date()
        events: list[Event] = []

        for ticker in self.watchlist:
            try:
                edf = yf.Ticker(ticker).get_earnings_dates(limit=8)
            except Exception:
                logger.warning("yfinance earnings lookup failed for %s", ticker, exc_info=True)
                continue

            if edf is None or edf.empty:
                continue

            for idx, row in edf.iterrows():
                edate = idx.date() if hasattr(idx, "date") else idx
                delta_days = (edate - today).days
                reported_eps = row.get("Reported EPS")
                surprise_pct = row.get("Surprise(%)")

                if reported_eps is not None and _not_nan(reported_eps) and 0 <= (today - edate).days <= self.earnings_lookback_days:
                    events.append(self._build_earnings_event(ticker, edate, "earnings_surprise", surprise_pct, now))
                elif 0 <= delta_days <= self.earnings_lookahead_days:
                    events.append(self._build_earnings_event(ticker, edate, "earnings_upcoming", None, now))

        return events

    def _build_earnings_event(
        self, ticker: str, edate: date, event_type: str, surprise_pct, now: datetime
    ) -> Event:
        if event_type == "earnings_surprise":
            description = f"{ticker} reported earnings on {edate.isoformat()} (surprise: {surprise_pct}%)"
            # Just reported — bias toward reaction coverage, recent only.
            snippets = self._search_news(
                f"{ticker} earnings results reaction", topic="news", time_range="week"
            )
        else:
            description = f"{ticker} has upcoming earnings on {edate.isoformat()}"
            # Not yet reported — bias toward forward-looking preview/estimate
            # coverage rather than reaction to the *previous* print.
            snippets = self._search_news(
                f"{ticker} earnings preview analyst estimates ahead of earnings",
                topic="news",
            )

        score, confidence = score_sentiment(snippets, domain="earnings")

        return Event(
            event_type=event_type,
            ticker=ticker,
            timestamp=now.isoformat(),
            description=description,
            sentiment_score=score,
            confidence=confidence,
        )

    # -- macro ---------------------------------------------------------------

    def check_macro_events(self, now: Optional[datetime] = None) -> list[Event]:
        now = now or datetime.now(timezone.utc)
        today = now.date()
        events: list[Event] = []

        for macro_event in self.macro_events:
            edate = macro_event.get("date")
            if edate is None:
                continue

            delta_days = (edate - today).days
            if -self.macro_lookback_days <= delta_days <= self.macro_lookahead_days:
                snippets = self._search_news(
                    macro_event["search_query"], topic="news", time_range="week"
                )
                score, confidence = score_sentiment(snippets, domain="macro")

                events.append(Event(
                    event_type=macro_event["event_type"],
                    ticker=None,
                    timestamp=now.isoformat(),
                    description=f"{macro_event['name']}: {macro_event['description']} (scheduled {edate.isoformat()})",
                    sentiment_score=score,
                    confidence=confidence,
                ))

        return events

    # -- driver ----------------------------------------------------------

    def run_once(self, now: Optional[datetime] = None) -> list[Event]:
        now = now or datetime.now(timezone.utc)
        events = self.check_earnings(now=now) + self.check_macro_events(now=now)

        for event in events:
            logger.info("Detected event: %s", json.dumps(event.to_dict()))

        if not events:
            logger.info("No events detected this cycle")

        return events

    def run(self, interval_seconds: int = 3600) -> None:
        logger.info(
            "Starting event detector loop (interval=%ss, watchlist=%s)",
            interval_seconds, self.watchlist,
        )
        while True:
            try:
                self.run_once()
            except Exception:
                logger.exception("Error during detection cycle")
            time.sleep(interval_seconds)


def _not_nan(value) -> bool:
    try:
        return value == value  # NaN != NaN
    except Exception:
        return True


# ---------------------------------------------------------------------------
# CLI

def _parse_args():
    parser = argparse.ArgumentParser(description="Daedalus event detector")
    parser.add_argument("--once", action="store_true", help="Run a single detection pass and exit")
    parser.add_argument("--interval", type=int, default=3600, help="Polling interval in seconds (default: 3600)")
    parser.add_argument(
        "--watchlist", type=str, default=None,
        help="Comma-separated tickers (default: built-in DEFAULT_WATCHLIST)",
    )
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if load_dotenv is not None:
        load_dotenv()

    args = _parse_args()
    watchlist = args.watchlist.split(",") if args.watchlist else None
    detector = EventDetector(watchlist=watchlist)

    if args.once:
        detector.run_once()
    else:
        detector.run(interval_seconds=args.interval)


if __name__ == "__main__":
    main()
