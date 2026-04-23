"""Tests for FNSPID loader — focus on the dedup / idempotency guarantees."""

from __future__ import annotations

import pandas as pd

from bot8.data.news.fnspid_loader import _clean_frame, _headline_hash


def _make_row(symbol: str, date: str, headline: str) -> dict:
    return {
        "symbol": symbol,
        "published_at": date,
        "headline": headline,
        "body": None,
        "url": None,
        "author": None,
        "source": "fnspid",
    }


class TestCleanFrame:
    def test_dedupes_within_chunk(self) -> None:
        """Regression: republished / duplicated headlines in a single chunk must
        be collapsed before INSERT, or DuckDB PK will reject the whole batch."""
        df = pd.DataFrame(
            [
                _make_row("A", "2023-05-24", "Agilent reports Q2 earnings"),
                _make_row("A", "2023-05-24", "Agilent reports Q2 earnings"),  # dup
                _make_row("A", "2023-05-24", "Agilent reports Q2 earnings"),  # dup
                _make_row("B", "2023-05-24", "Barnes & Noble names new CEO"),
            ]
        )
        out = _clean_frame(df, since=None)
        assert len(out) == 2, f"expected 2 rows post-dedup, got {len(out)}"
        # Both unique headlines preserved
        assert set(out["symbol"]) == {"A", "B"}

    def test_keeps_distinct_headlines_same_symbol_same_day(self) -> None:
        """Two different headlines for the same symbol on the same day MUST be
        kept — they're legitimately different news items."""
        df = pd.DataFrame(
            [
                _make_row("AAPL", "2023-05-24", "Apple beats Q2 earnings"),
                _make_row("AAPL", "2023-05-24", "Apple raises dividend 5%"),
            ]
        )
        out = _clean_frame(df, since=None)
        assert len(out) == 2

    def test_drops_missing_symbol_or_headline(self) -> None:
        df = pd.DataFrame(
            [
                _make_row("AAPL", "2023-05-24", "Apple reports earnings"),
                _make_row("", "2023-05-24", "Headline without symbol"),
                _make_row("MSFT", "2023-05-24", ""),
            ]
        )
        out = _clean_frame(df, since=None)
        assert len(out) == 1
        assert out.iloc[0]["symbol"] == "AAPL"

    def test_drops_invalid_dates(self) -> None:
        df = pd.DataFrame(
            [
                _make_row("AAPL", "2023-05-24", "Valid row"),
                _make_row("MSFT", "not a date", "Invalid date row"),
            ]
        )
        out = _clean_frame(df, since=None)
        assert len(out) == 1
        assert out.iloc[0]["symbol"] == "AAPL"

    def test_since_filter(self) -> None:
        from datetime import datetime
        df = pd.DataFrame(
            [
                _make_row("AAPL", "2022-01-15", "Old news"),
                _make_row("AAPL", "2023-01-15", "Recent news"),
            ]
        )
        out = _clean_frame(df, since=datetime(2023, 1, 1))
        assert len(out) == 1
        assert out.iloc[0]["headline"] == "Recent news"

    def test_symbol_uppercased(self) -> None:
        df = pd.DataFrame([_make_row("aapl", "2023-05-24", "lowercase symbol")])
        out = _clean_frame(df, since=None)
        assert out.iloc[0]["symbol"] == "AAPL"


class TestHeadlineHash:
    def test_stable(self) -> None:
        """Same input → same hash. Critical for cross-run dedup."""
        assert _headline_hash("AAPL", "Apple beats Q2") == _headline_hash(
            "AAPL", "Apple beats Q2"
        )

    def test_case_insensitive_on_headline(self) -> None:
        """Tiny whitespace / case variations shouldn't create dup records."""
        assert _headline_hash("AAPL", "Apple beats Q2") == _headline_hash(
            "AAPL", "  apple beats q2  "
        )

    def test_different_symbols_different_hashes(self) -> None:
        assert _headline_hash("AAPL", "Market update") != _headline_hash(
            "MSFT", "Market update"
        )
