"""Unit tests for the macro loader — pure logic, no network."""

from __future__ import annotations

import pandas as pd

from bot8.data.market.macro import DEFAULT_SERIES


class TestDefaultSeries:
    def test_has_regime_essentials(self) -> None:
        """The regime filter depends on these specifically — don't drop them silently."""
        for required in ["VIX", "SPY", "HYG", "TNX", "IRX"]:
            assert required in DEFAULT_SERIES

    def test_codes_are_uppercase(self) -> None:
        for code in DEFAULT_SERIES:
            assert code == code.upper()

    def test_yfinance_tickers_non_empty(self) -> None:
        for code, yf in DEFAULT_SERIES.items():
            assert yf, f"series {code} has empty yfinance ticker"


class TestInsertSeriesIdempotency:
    def test_duplicate_insert_is_noop(self, tmp_path, monkeypatch) -> None:
        """Re-running ingestion must not create duplicate rows."""
        monkeypatch.setenv("DATA_DIR", str(tmp_path))
        from bot8.config import settings as s_mod
        s_mod.get_settings.cache_clear()

        from bot8.data.db import init_schema, session
        from bot8.data.market.macro import _insert_series

        init_schema()
        df = pd.DataFrame(
            [
                {"series_code": "VIX", "session_date": pd.Timestamp("2024-01-02").date(),
                 "open": 13.5, "high": 14.0, "low": 13.2, "close": 13.8, "volume": 0},
                {"series_code": "VIX", "session_date": pd.Timestamp("2024-01-03").date(),
                 "open": 13.8, "high": 14.2, "low": 13.6, "close": 14.0, "volume": 0},
            ]
        )
        first = _insert_series(df)
        second = _insert_series(df)  # same data
        assert first == 2
        assert second == 0

        with session(read_only=True) as con:
            n = con.execute("SELECT COUNT(*) FROM macro_daily").fetchone()[0]
            assert n == 2
