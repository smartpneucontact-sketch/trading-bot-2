"""Training data loader.

Joins `features_quant` + `news_features_daily` + `labels_quant` into one
wide DataFrame ready for ML:

    X:     feature matrix (numeric only)
    y:     target — fwd_return_1d_zscore (cross-sectionally standardized)
    meta:  symbol, session_date, fwd_return_1d (raw, for backtest eval)

Design:
- LEFT JOIN news onto quant — most (symbol, date) rows have no news, and we
  fill missing news features with 0 so the tree models can handle them
  natively without masking. The catalyst_flags and counts being 0 is a
  legitimate "no news today" feature, not a missing value.
- Filter rows where the target is NaN (last session — can't look forward).
- Filter rows where critical long-lookback features (ret_252d) are NaN
  (warmup period). Optional: controlled via `drop_warmup` arg.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd
from loguru import logger

from bot8.data.db import session

# Columns that identify rows / are labels — must be excluded from X.
META_COLS = {
    "symbol",
    "session_date",
    "sector",
    "feature_version",
    # Label columns (present after join but belong in y / meta)
    "fwd_return_1d",
    "fwd_return_1d_demean",
    "fwd_return_1d_zscore",
    "fwd_rank_1d",
}


@dataclass(frozen=True, slots=True)
class TrainingData:
    X: pd.DataFrame                  # full feature matrix (quant + news + claude)
    y: pd.Series                     # target
    meta: pd.DataFrame               # symbol, session_date, fwd_return_1d
    quant_feature_cols: list[str]    # columns from features_quant
    news_feature_cols: list[str]     # columns from news_features_daily (FinBERT + regex)
    claude_feature_cols: list[str]   # columns from news_features_claude_daily

    @property
    def dates(self) -> pd.Series:
        return self.meta["session_date"]

    @property
    def feature_cols(self) -> list[str]:
        return self.quant_feature_cols + self.news_feature_cols + self.claude_feature_cols

    @property
    def X_quant(self) -> pd.DataFrame:
        """Quant-only feature matrix — used to train base ensemble."""
        return self.X[self.quant_feature_cols]

    @property
    def X_news(self) -> pd.DataFrame:
        """FinBERT/regex news-only feature matrix — extra input to the meta-learner."""
        return self.X[self.news_feature_cols]

    @property
    def X_claude(self) -> pd.DataFrame:
        """Claude-derived news features — extra input to the meta-learner."""
        return self.X[self.claude_feature_cols]

    @property
    def X_news_all(self) -> pd.DataFrame:
        """Both FinBERT and Claude news features concatenated."""
        return self.X[self.news_feature_cols + self.claude_feature_cols]


def load_training_data(
    since: date | None = None,
    until: date | None = None,
    drop_warmup: bool = True,
    target: str = "fwd_return_1d_zscore",
) -> TrainingData:
    """Load + join features + labels, clean, return matrices ready for sklearn."""
    logger.info("Loading features_quant + news_features_daily + labels_quant…")

    where: list[str] = []
    params: list = []
    if since:
        where.append("f.session_date >= ?")
        params.append(since)
    if until:
        where.append("f.session_date <= ?")
        params.append(until)
    where_sql = (" WHERE " + " AND ".join(where)) if where else ""

    # Claude features are optional — join only if the table exists.
    with session(read_only=True) as con:
        has_claude = bool(con.execute(
            "SELECT COUNT(*) FROM information_schema.tables "
            "WHERE table_name = 'news_features_claude_daily'"
        ).fetchone()[0])

    if has_claude:
        claude_join = """
            LEFT JOIN (
                SELECT symbol,
                       session_date,
                       sentiment_score        AS claude_sent,
                       novelty                AS claude_novelty,
                       magnitude              AS claude_magnitude,
                       confidence             AS claude_confidence,
                       sentiment_score * magnitude * confidence AS claude_weighted_signal,
                       CASE WHEN expected_direction = 'bullish' THEN 1.0
                            WHEN expected_direction = 'bearish' THEN -1.0
                            ELSE 0.0 END     AS claude_direction,
                       CAST(expected_horizon_days AS DOUBLE) AS claude_horizon,
                       CAST(primary_catalyst = 'earnings_beat' AS DOUBLE)     AS claude_is_earnings_beat,
                       CAST(primary_catalyst = 'earnings_miss' AS DOUBLE)     AS claude_is_earnings_miss,
                       CAST(primary_catalyst = 'guidance_raise' AS DOUBLE)    AS claude_is_guidance_raise,
                       CAST(primary_catalyst = 'guidance_cut' AS DOUBLE)      AS claude_is_guidance_cut,
                       CAST(primary_catalyst = 'merger_acquisition' AS DOUBLE) AS claude_is_ma,
                       CAST(primary_catalyst = 'analyst_upgrade' AS DOUBLE)   AS claude_is_upgrade,
                       CAST(primary_catalyst = 'analyst_downgrade' AS DOUBLE) AS claude_is_downgrade,
                       CAST(primary_catalyst = 'regulatory_approval' AS DOUBLE) AS claude_is_reg_approval,
                       CAST(primary_catalyst = 'regulatory_rejection' AS DOUBLE) AS claude_is_reg_rejection,
                       CAST(primary_catalyst = 'regulatory_probe' AS DOUBLE)  AS claude_is_reg_probe,
                       CAST(primary_catalyst = 'management_resignation' AS DOUBLE) AS claude_is_mgmt_resign,
                       CAST(primary_catalyst = 'bankruptcy_distress' AS DOUBLE) AS claude_is_bankruptcy
                FROM news_features_claude_daily
            ) cl USING (symbol, session_date)
        """
        claude_select = (
            ", cl.claude_sent, cl.claude_novelty, cl.claude_magnitude, "
            "cl.claude_confidence, cl.claude_weighted_signal, cl.claude_direction, "
            "cl.claude_horizon, cl.claude_is_earnings_beat, cl.claude_is_earnings_miss, "
            "cl.claude_is_guidance_raise, cl.claude_is_guidance_cut, cl.claude_is_ma, "
            "cl.claude_is_upgrade, cl.claude_is_downgrade, cl.claude_is_reg_approval, "
            "cl.claude_is_reg_rejection, cl.claude_is_reg_probe, cl.claude_is_mgmt_resign, "
            "cl.claude_is_bankruptcy"
        )
    else:
        claude_join = ""
        claude_select = ""

    sql = f"""
        SELECT f.*,
               n.* EXCLUDE (symbol, session_date, feature_version){claude_select},
               l.fwd_return_1d, l.fwd_return_1d_demean,
               l.fwd_return_1d_zscore, l.fwd_rank_1d
        FROM features_quant f
        LEFT JOIN news_features_daily n USING (symbol, session_date)
        {claude_join}
        LEFT JOIN labels_quant l USING (symbol, session_date)
        {where_sql}
    """
    with session(read_only=True) as con:
        df = con.execute(sql, params).fetchdf()

    if df.empty:
        raise RuntimeError(
            "No training rows returned. Have you run "
            "`bot8 features quant` and `bot8 features news-daily`?"
        )

    logger.info("  raw joined rows: {:,}", len(df))

    # Target must be present
    df = df.dropna(subset=[target])
    logger.info("  after dropping missing target: {:,}", len(df))

    # Drop warmup rows (ret_252d is the longest lookback; NaN = not enough history)
    if drop_warmup and "ret_252d" in df.columns:
        df = df.dropna(subset=["ret_252d"])
        logger.info("  after dropping 252d-warmup: {:,}", len(df))

    # Fill news NaNs with 0 (missing news = no news, not a missing value).
    # Covers both FinBERT aggregator output and Claude scorer output.
    news_like_cols = [c for c in df.columns if c.startswith(("news_", "sent_", "pos_",
                                                              "neg_", "net_", "has_",
                                                              "tag_", "claude_"))
                      or c.endswith("_count") or c.endswith("_count_3d")
                      or c.endswith("_count_7d")]
    for c in news_like_cols:
        if c in df.columns:
            df[c] = df[c].fillna(0)

    # Separate X / y / meta
    y = df[target].astype("float64")
    meta = df[["symbol", "session_date", "fwd_return_1d"]].reset_index(drop=True)

    excluded = META_COLS | {"fwd_return_1d_demean", "fwd_rank_1d"}
    feature_cols = [c for c in df.columns if c not in excluded]

    # Non-numeric columns (shouldn't be many, but guard)
    X = df[feature_cols].copy()
    for c in X.columns:
        if X[c].dtype == object:
            logger.warning("Dropping non-numeric column: {}", c)
            X = X.drop(columns=[c])

    # Partition columns by origin:
    #   claude_feature_cols — anything prefixed "claude_" (from the LLM scorer)
    #   news_feature_cols   — matches the aggregator's authoritative list
    #   quant_feature_cols  — everything else (from features_quant)
    from bot8.features.news.aggregator import get_news_feature_columns
    news_col_set = set(get_news_feature_columns())
    claude_feature_cols = [c for c in X.columns if c.startswith("claude_")]
    news_feature_cols = [c for c in X.columns if c in news_col_set and c not in claude_feature_cols]
    quant_feature_cols = [
        c for c in X.columns
        if c not in news_col_set and not c.startswith("claude_")
    ]

    X = X.astype("float32")
    # Let tree models handle NaN natively; do NOT impute.

    logger.info(
        "  final: X={} rows × {} cols (quant={}, news={}, claude={}), "
        "y_mean={:+.4f}, y_std={:.4f}",
        len(X), X.shape[1],
        len(quant_feature_cols), len(news_feature_cols), len(claude_feature_cols),
        y.mean(), y.std(),
    )

    return TrainingData(
        X=X.reset_index(drop=True),
        y=y.reset_index(drop=True),
        meta=meta,
        quant_feature_cols=quant_feature_cols,
        news_feature_cols=news_feature_cols,
        claude_feature_cols=claude_feature_cols,
    )
