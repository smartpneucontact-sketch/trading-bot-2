"""Unit tests for the universe module. No network — logic only."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from bot8.data.market.universe import (
    _clean_col,
    _clean_columns,
    _pick_column,
    _INDUSTRY_ALIASES,
    _SECTOR_ALIASES,
    _SYMBOL_ALIASES,
)


class TestColumnCleaning:
    """Regression: Wikipedia columns often include [footnote] refs that broke
    the old matcher."""

    def test_strips_footnote_brackets(self) -> None:
        assert _clean_col("GICS Sector[5]") == "GICS Sector"
        assert _clean_col("ICB Industry[14]") == "ICB Industry"
        assert _clean_col("Ticker[1][2]") == "Ticker"

    def test_preserves_plain_names(self) -> None:
        assert _clean_col("Symbol") == "Symbol"
        assert _clean_col("  Sector  ") == "Sector"

    def test_clean_columns_df(self) -> None:
        df = pd.DataFrame(columns=["Ticker[1]", "ICB Industry[14]", "ICB Subsector[14]"])
        cleaned = _clean_columns(df)
        assert list(cleaned.columns) == ["Ticker", "ICB Industry", "ICB Subsector"]


class TestColumnAliases:
    def test_picks_first_matching_alias(self) -> None:
        cols = {"Ticker", "Company", "ICB Industry"}
        assert _pick_column(cols, _SYMBOL_ALIASES) == "Ticker"
        assert _pick_column(cols, _SECTOR_ALIASES) == "ICB Industry"

    def test_returns_none_when_no_match(self) -> None:
        assert _pick_column({"Foo", "Bar"}, _SECTOR_ALIASES) is None

    def test_gics_and_icb_both_recognized(self) -> None:
        assert _pick_column({"GICS Sector"}, _SECTOR_ALIASES) == "GICS Sector"
        assert _pick_column({"ICB Industry"}, _SECTOR_ALIASES) == "ICB Industry"
        assert _pick_column({"GICS Sub-Industry"}, _INDUSTRY_ALIASES) == "GICS Sub-Industry"
        assert _pick_column({"ICB Subsector"}, _INDUSTRY_ALIASES) == "ICB Subsector"


class TestBuildUniverseMerge:
    def test_merge_preserves_wiki_with_no_alpaca_match(self, monkeypatch) -> None:
        """Symbols that exist on Wikipedia but not in Alpaca's active assets
        must still land in the universe with is_tradable=False. Otherwise we
        silently drop tickers."""
        from bot8.data.market import universe as uni

        fake_wiki_sp500 = pd.DataFrame([
            {"symbol": "AAPL", "sector": "IT", "industry": "Hardware"},
            {"symbol": "DELISTED", "sector": "Industrials", "industry": "Defense"},
        ])
        fake_wiki_ndx = pd.DataFrame([
            {"symbol": "NVDA", "sector": "IT", "industry": "Semis"},
        ])
        fake_alpaca = pd.DataFrame([
            {"symbol": "AAPL", "is_shortable": True, "is_fractionable": True, "is_tradable": True},
            {"symbol": "NVDA", "is_shortable": True, "is_fractionable": True, "is_tradable": True},
        ])

        monkeypatch.setattr(uni, "fetch_sp500", lambda: fake_wiki_sp500)
        monkeypatch.setattr(uni, "fetch_ndx100", lambda: fake_wiki_ndx)
        monkeypatch.setattr(uni, "fetch_alpaca_assets", lambda: fake_alpaca)

        out = uni.build_universe(as_of=date(2026, 4, 20))

        assert set(out["symbol"]) == {"AAPL", "DELISTED", "NVDA"}
        delisted = out[out["symbol"] == "DELISTED"].iloc[0]
        assert delisted["is_shortable"] is False or delisted["is_shortable"] == 0
        assert delisted["is_tradable"] is False or delisted["is_tradable"] == 0

    def test_dual_membership_rows(self, monkeypatch) -> None:
        """A symbol in BOTH indexes should produce two rows (different
        index_membership values) — matters for sector-neutral portfolio
        weighting later."""
        from bot8.data.market import universe as uni

        monkeypatch.setattr(uni, "fetch_sp500", lambda: pd.DataFrame([
            {"symbol": "AAPL", "sector": "IT", "industry": "Hardware"}
        ]))
        monkeypatch.setattr(uni, "fetch_ndx100", lambda: pd.DataFrame([
            {"symbol": "AAPL", "sector": "IT", "industry": "Hardware"}
        ]))
        monkeypatch.setattr(uni, "fetch_alpaca_assets", lambda: pd.DataFrame([
            {"symbol": "AAPL", "is_shortable": True, "is_fractionable": True, "is_tradable": True}
        ]))

        out = uni.build_universe(as_of=date(2026, 4, 20))
        assert len(out) == 2
        assert set(out["index_membership"]) == {"SP500", "NDX100"}


class TestCurrentUniverse:
    def test_empty_when_no_snapshot(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv("DATA_DIR", str(tmp_path))
        from bot8.config import settings as s_mod
        s_mod.get_settings.cache_clear()

        from bot8.data.db import init_schema
        from bot8.data.market.universe import current_universe

        init_schema()
        assert current_universe() == []

    def test_filters_shortable(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv("DATA_DIR", str(tmp_path))
        from bot8.config import settings as s_mod
        s_mod.get_settings.cache_clear()

        from bot8.data.db import init_schema, session
        from bot8.data.market.universe import current_universe

        init_schema()
        with session() as con:
            con.execute(
                """
                INSERT INTO universe (symbol, as_of_date, index_membership, sector, industry, is_shortable, is_fractionable, is_tradable)
                VALUES
                  ('AAPL', DATE '2026-04-20', 'SP500', 'IT', 'HW', TRUE, TRUE, TRUE),
                  ('ZZZZ', DATE '2026-04-20', 'SP500', 'Misc', 'X', FALSE, TRUE, TRUE),
                  ('DEAD', DATE '2026-04-20', 'SP500', 'Misc', 'X', TRUE, TRUE, FALSE)
                """
            )

        # Default is require_tradable=True — DEAD is excluded
        assert "AAPL" in current_universe()
        assert "ZZZZ" in current_universe()
        assert "DEAD" not in current_universe()

        # Shortable filter picks only AAPL
        assert current_universe(require_shortable=True) == ["AAPL"]

        # Opt-out of tradable filter to see everything
        assert "DEAD" in current_universe(require_tradable=False)
