"""Tests for the news feature aggregator."""

from __future__ import annotations

from datetime import date

import pandas as pd

from bot8.features.news.aggregator import CATALYST_CATEGORIES, _explode_catalysts


class TestExplodeCatalysts:
    def test_creates_tag_column_per_category(self) -> None:
        df = pd.DataFrame({"catalyst_tags": ["earnings,guidance", "m_and_a", ""]})
        out = _explode_catalysts(df)
        for c in CATALYST_CATEGORIES:
            assert f"tag_{c}" in out.columns

    def test_flags_set_correctly(self) -> None:
        df = pd.DataFrame({"catalyst_tags": ["earnings,guidance", "m_and_a", ""]})
        out = _explode_catalysts(df)
        assert out.loc[0, "tag_earnings"] == 1
        assert out.loc[0, "tag_guidance"] == 1
        assert out.loc[0, "tag_m_and_a"] == 0
        assert out.loc[1, "tag_m_and_a"] == 1
        assert out.loc[1, "tag_earnings"] == 0
        assert (out.iloc[2].filter(like="tag_") == 0).all()

    def test_no_substring_false_positives(self) -> None:
        """'management_change' string shouldn't match 'management' regex unless
        it's the full token — verify our regex is token-aware."""
        df = pd.DataFrame({"catalyst_tags": ["managementchange"]})
        out = _explode_catalysts(df)
        assert out.loc[0, "tag_management"] == 0


class TestAggregatorIntegration:
    def test_aggregates_per_symbol_date(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv("DATA_DIR", str(tmp_path))
        from bot8.config import settings as s_mod
        s_mod.get_settings.cache_clear()

        from bot8.data.db import init_schema, session
        from bot8.features.news.aggregator import compute_news_features

        init_schema()

        # Seed news_scored with 4 headlines across 2 (symbol, date) pairs.
        rows = [
            ("AAPL", "2023-06-01 10:00:00", "hash1", "positive", 0.9, 0.85, "earnings,guidance"),
            ("AAPL", "2023-06-01 11:00:00", "hash2", "negative", -0.7, 0.80, "analyst"),
            ("AAPL", "2023-06-02 09:00:00", "hash3", "positive", 0.5, 0.70, "product"),
            ("MSFT", "2023-06-01 12:00:00", "hash4", "neutral", 0.05, 0.60, ""),
        ]
        with session() as con:
            for r in rows:
                con.execute(
                    """
                    INSERT INTO news_scored
                    (symbol, published_at, headline_hash, sentiment_label,
                     sentiment_score, sentiment_conf, catalyst_tags, model_version)
                    VALUES (?, ?, ?, ?, ?, ?, ?, 'test-v1')
                    """,
                    r,
                )

        df = compute_news_features()
        assert len(df) == 3  # 3 distinct (symbol, date) pairs

        # AAPL 2023-06-01: 2 headlines, 1 positive, 1 negative → net_sentiment = 0
        aapl_d1 = df[(df["symbol"] == "AAPL") & (df["session_date"] == date(2023, 6, 1))].iloc[0]
        assert aapl_d1["news_count"] == 2
        assert aapl_d1["pos_count"] == 1
        assert aapl_d1["neg_count"] == 1
        assert aapl_d1["net_sentiment"] == 0
        assert aapl_d1["has_earnings"] == 1
        assert aapl_d1["has_guidance"] == 1
        assert aapl_d1["has_analyst"] == 1
        assert aapl_d1["has_m_and_a"] == 0
        # sent_mean = avg(0.9, -0.7) = 0.1
        assert abs(aapl_d1["sent_mean"] - 0.1) < 1e-6
        # sent_abs_max = 0.9
        assert abs(aapl_d1["sent_abs_max"] - 0.9) < 1e-6

        # MSFT 2023-06-01: neutral only → net_sentiment = 0, no catalysts
        msft = df[df["symbol"] == "MSFT"].iloc[0]
        assert msft["news_count"] == 1
        assert msft["net_sentiment"] == 0
        assert msft["has_earnings"] == 0

    def test_ticker_aliases_collapse(self, tmp_path, monkeypatch) -> None:
        """FB news should aggregate onto META symbol."""
        monkeypatch.setenv("DATA_DIR", str(tmp_path))
        from bot8.config import settings as s_mod
        s_mod.get_settings.cache_clear()

        from bot8.data.db import init_schema, session
        from bot8.features.news.aggregator import compute_news_features

        init_schema()
        with session() as con:
            con.execute(
                """
                INSERT INTO news_scored
                (symbol, published_at, headline_hash, sentiment_label,
                 sentiment_score, sentiment_conf, catalyst_tags, model_version)
                VALUES
                  ('FB',   '2021-06-01 10:00:00', 'h1', 'positive',  0.5, 0.9, '', 'v1'),
                  ('META', '2023-06-01 10:00:00', 'h2', 'negative', -0.6, 0.8, '', 'v1')
                """
            )

        df = compute_news_features().sort_values("session_date")
        # Both rows should be under 'META' after alias collapse; 'FB' should not exist.
        assert "FB" not in df["symbol"].values
        assert (df["symbol"] == "META").all()
        assert len(df) == 2

    def test_trailing_windows_computed(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv("DATA_DIR", str(tmp_path))
        from bot8.config import settings as s_mod
        s_mod.get_settings.cache_clear()

        from bot8.data.db import init_schema, session
        from bot8.features.news.aggregator import compute_news_features

        init_schema()

        with session() as con:
            for i in range(5):
                day = f"2023-06-0{i+1} 10:00:00"
                con.execute(
                    """
                    INSERT INTO news_scored
                    (symbol, published_at, headline_hash, sentiment_label,
                     sentiment_score, sentiment_conf, catalyst_tags, model_version)
                    VALUES ('X', ?, ?, 'positive', 0.5, 0.8, '', 'test-v1')
                    """,
                    (day, f"hash{i}"),
                )

        df = compute_news_features().sort_values("session_date").reset_index(drop=True)
        # Row 3 (0-indexed) is day 4 → trailing 3d count = rows 1,2,3 = 3
        assert df.loc[3, "news_count_3d"] == 3
        # Row 4 (day 5) → trailing 7d count = all 5
        assert df.loc[4, "news_count_7d"] == 5
