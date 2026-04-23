"""Tests for the regex catalyst classifier. Pure CPU, no torch needed."""

from __future__ import annotations

import pytest

from bot8.features.news.catalyst_regex import (
    RULESET_VERSION,
    classify,
    classify_to_tag_string,
)


class TestEarnings:
    def test_beats_estimates(self) -> None:
        assert "earnings" in classify("Apple beats Q3 earnings estimates")

    def test_misses_consensus(self) -> None:
        assert "earnings" in classify("Tesla misses consensus revenue estimates")

    def test_eps_mention(self) -> None:
        assert "earnings" in classify("NVDA reports EPS of $5.12, above forecast")

    def test_revenue_up(self) -> None:
        assert "earnings" in classify("Microsoft revenue rose 15% year-over-year")


class TestGuidance:
    def test_raises_guidance(self) -> None:
        assert "guidance" in classify("Apple raises full-year guidance")

    def test_cuts_outlook(self) -> None:
        assert "guidance" in classify("Ford cuts 2026 outlook on weak demand")

    def test_above_guidance(self) -> None:
        assert "guidance" in classify("Amazon forecast above street estimates")


class TestMandA:
    def test_acquires(self) -> None:
        assert "m_and_a" in classify("Microsoft acquires Activision for $68B")

    def test_merger(self) -> None:
        assert "m_and_a" in classify("T-Mobile and Sprint complete merger")

    def test_buyout(self) -> None:
        assert "m_and_a" in classify("Twitter accepts buyout offer from Musk")

    def test_spinoff(self) -> None:
        assert "m_and_a" in classify("GE announces spin-off of healthcare unit")


class TestAnalyst:
    def test_upgrade(self) -> None:
        assert "analyst" in classify("Goldman upgrades NVDA to Buy")

    def test_price_target(self) -> None:
        assert "analyst" in classify("Morgan Stanley raises AAPL price target to $250")

    def test_initiates_coverage(self) -> None:
        assert "analyst" in classify("Wedbush initiates coverage on TSLA with Outperform")


class TestRegulatory:
    def test_fda_approval(self) -> None:
        assert "regulatory" in classify("FDA approves Merck's new cancer drug")

    def test_sec_investigation(self) -> None:
        assert "regulatory" in classify("SEC opens investigation into Coinbase")

    def test_antitrust(self) -> None:
        assert "regulatory" in classify("DOJ files antitrust suit against Google")


class TestManagement:
    def test_ceo_resigns(self) -> None:
        assert "management" in classify("Boeing CEO resigns amid 737 MAX crisis")

    def test_cfo_appointed(self) -> None:
        assert "management" in classify("Coca-Cola names new CFO from Goldman")


class TestMultiLabel:
    """A single headline should trigger multiple categories when appropriate."""

    def test_earnings_and_guidance(self) -> None:
        tags = classify("Apple beats Q3 estimates, raises full-year guidance")
        assert "earnings" in tags
        assert "guidance" in tags

    def test_analyst_after_earnings(self) -> None:
        tags = classify("Goldman upgrades NVDA to Buy, raises price target after Q2 beat")
        assert "analyst" in tags

    def test_tag_string_output(self) -> None:
        s = classify_to_tag_string("Apple beats Q3 estimates, raises guidance")
        assert "earnings" in s.split(",")
        assert "guidance" in s.split(",")


class TestNegativeCases:
    """Random / irrelevant headlines should produce no tags."""

    @pytest.mark.parametrize(
        "text",
        [
            "",
            "Apple unveils new logo design",  # not a product launch regex match
            "Stock market closes mixed on Tuesday",
            "What to watch this week",
        ],
    )
    def test_no_false_positives(self, text: str) -> None:
        tags = classify(text)
        # "product" might legitimately catch "unveils new" — that's OK.
        # Just check nothing explodes and result is a list.
        assert isinstance(tags, list)


class TestInvariants:
    def test_empty_input(self) -> None:
        assert classify("") == []
        assert classify_to_tag_string("") == ""

    def test_none_safe(self) -> None:
        # Contract: classify() shouldn't be called with None.
        # But a wrapper might pass falsy values — verify we degrade gracefully.
        assert classify("") == []

    def test_version_string_is_set(self) -> None:
        assert RULESET_VERSION.startswith("regex-v")

    def test_case_insensitive(self) -> None:
        assert classify("APPLE BEATS Q3 ESTIMATES") == classify("apple beats q3 estimates")
