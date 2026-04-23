"""Tests for portfolio construction + backtest simulation."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from bot8.portfolio.longshort import PortfolioConfig, build_daily_weights


def _make_score_frame(n_symbols: int = 100, n_days: int = 5, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start = date(2024, 1, 2)
    rows = []
    for d in range(n_days):
        dt = start + timedelta(days=d)
        for i in range(n_symbols):
            rows.append({
                "session_date": dt,
                "symbol": f"S{i:03d}",
                "score": float(rng.standard_normal()),
                "is_shortable": True,
                "sector": ["Tech", "Financial", "Health", "Energy"][i % 4],
            })
    return pd.DataFrame(rows)


class TestPortfolioConstruction:
    def test_dollar_neutral_per_day(self) -> None:
        """Sum of weights per day should be approximately 0."""
        df = _make_score_frame(n_symbols=200, n_days=5)
        w = build_daily_weights(df, score_col="score", cfg=PortfolioConfig())
        for dt, day in w.groupby("session_date"):
            net = day["weight"].sum()
            assert abs(net) < 0.05, f"day {dt} net weight = {net}"

    def test_gross_exposure_bounded_by_cap(self) -> None:
        """With 20 names per side × 2% cap, max gross per day is 2 × 20 × 0.02 = 0.8.
        Allow 0.3–2.5 range to accommodate sector-neutralization scaling."""
        df = _make_score_frame(n_symbols=200, n_days=5)
        cfg = PortfolioConfig(target_gross_exposure=2.0, max_position_weight=0.02)
        w = build_daily_weights(df, score_col="score", cfg=cfg)
        gross = w.groupby("session_date")["weight"].apply(lambda s: s.abs().sum())
        assert 0.3 < gross.mean() <= 0.8 + 1e-6

    def test_per_name_cap_respected(self) -> None:
        df = _make_score_frame(n_symbols=200, n_days=3)
        cfg = PortfolioConfig(max_position_weight=0.02)
        w = build_daily_weights(df, score_col="score", cfg=cfg)
        assert (w["weight"].abs() <= cfg.max_position_weight + 1e-6).all()

    def test_longs_and_shorts_present(self) -> None:
        df = _make_score_frame(n_symbols=200, n_days=3)
        w = build_daily_weights(df, score_col="score", cfg=PortfolioConfig())
        assert (w["weight"] > 0).any()
        assert (w["weight"] < 0).any()

    def test_shortable_filter_applied(self) -> None:
        """If is_shortable=False for bottom names, they shouldn't be shorted."""
        df = _make_score_frame(n_symbols=100, n_days=1)
        # Make the lowest-score names NOT shortable.
        df_sorted = df.sort_values("score")
        bottom_syms = set(df_sorted.head(10)["symbol"])
        df["is_shortable"] = ~df["symbol"].isin(bottom_syms)

        w = build_daily_weights(df, score_col="score",
                                cfg=PortfolioConfig(require_shortable_for_shorts=True))
        shorted = set(w[w["weight"] < 0]["symbol"])
        assert not (shorted & bottom_syms), \
            "non-shortable names were included in short book"

    def test_flat_when_universe_too_thin(self) -> None:
        df = _make_score_frame(n_symbols=20, n_days=3)
        cfg = PortfolioConfig(min_universe_per_day=50)
        w = build_daily_weights(df, score_col="score", cfg=cfg)
        # Build returns weights but they'd all be flat - check by checking sum
        assert len(w) == 0 or (w["weight"].abs().sum() == 0)
