"""News feature aggregation — per (symbol, session_date).

Reads `news_scored` (one row per headline) → collapses to one row per
symbol-day with aggregate stats suitable for joining onto `features_quant`.

Output schema (columns on `news_features_daily`):

  Primary key:
    symbol, session_date

  Headline volume:
    news_count             total headlines
    news_count_3d          trailing 3 trading days
    news_count_7d          trailing 7 trading days

  Sentiment:
    sent_mean              mean signed_score over the day
    sent_max               max signed_score
    sent_min               min signed_score
    sent_std               std of signed_score
    sent_abs_max           max(|signed_score|)
    sent_conf_weighted     Σ(score · conf) / Σ(conf)
    pos_count              # headlines with sentiment_label = 'positive'
    neg_count              # headlines with sentiment_label = 'negative'
    net_sentiment          pos_count − neg_count
    sent_mean_3d           trailing 3d sentiment mean
    sent_mean_7d           trailing 7d sentiment mean
    net_sentiment_3d       trailing 3d net sentiment
    net_sentiment_7d       trailing 7d net sentiment

  Catalyst flags (binary 0/1 for each canonical category):
    has_earnings, has_m_and_a, has_analyst, has_guidance,
    has_regulatory, has_management, has_product, has_legal,
    has_dividend_buyback, has_insider, has_macro, has_bankruptcy

  Catalyst counts (per day):
    earnings_count, m_and_a_count, analyst_count, ...

Joining:
- Symbol/day rows without news produce no row here — downstream joins should
  LEFT JOIN `news_features_daily` onto `features_quant` and fill NaN → 0.

Horizon note:
- For a 1-day horizon model, the news features for (symbol, date=t) describe
  what happened ON or BEFORE day t. The training label is `fwd_return_1d(t)`
  which runs from close(t) to close(t+1). No look-ahead bias.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
from loguru import logger

from bot8.data.db import session
from bot8.features.news.catalyst_regex import _RAW_PATTERNS

# Canonical catalyst category order. Matches catalyst_regex.py keys exactly.
CATALYST_CATEGORIES: list[str] = list(_RAW_PATTERNS.keys())

FEATURE_VERSION = "news-agg-v1"

# Canonical list of feature column names this module emits. Used by the
# training-data loader to cleanly separate quant features from news features
# when building the 2-level meta-learner stack (we must know which columns
# are "news" vs "quant" to measure news uplift).
def get_news_feature_columns() -> list[str]:
    """Returns the exact column names produced by build_news_features().

    Must stay in sync with compute_news_features() — when adding a feature,
    append it here too. Tests cross-check this list against a real run."""
    base = [
        "news_count",
        "sent_mean", "sent_max", "sent_min", "sent_std", "sent_abs_max",
        "sent_conf_weighted",
        "pos_count", "neg_count", "net_sentiment",
    ]
    for c in CATALYST_CATEGORIES:
        base.extend([f"{c}_count", f"has_{c}"])
    for n in (3, 7):
        base.extend([f"news_count_{n}d", f"sent_mean_{n}d", f"net_sentiment_{n}d"])
    return base

# Ticker alias map — FNSPID has historical tickers that don't match our bars
# universe. We collapse them onto the canonical bars symbol at load time so
# the meta-learner gets the full news signal for mega-caps that rebranded or
# have alternate share classes.
#
# Rules:
#   FB      → META      Facebook → Meta (rebrand 2022-06)
#   GOOG    → GOOGL     Alphabet Class C → Class A (same company news)
#   BRK-B   → BRK.B     Berkshire B — formatting variant
#   FISV    → FI        Fiserv rebranded 2024
#   SQ      → SQ        Block Inc. kept SQ ticker — no alias
#
# This isn't exhaustive. Add entries as we spot coverage gaps.
TICKER_ALIASES: dict[str, str] = {
    "FB": "META",
    "GOOG": "GOOGL",
    "BRK-B": "BRK.B",
    "BRKB": "BRK.B",
    "FISV": "FI",
}


def _load_scored(since: date | None) -> pd.DataFrame:
    """Pull the minimum columns we need from news_scored joined with news_raw
    (need the raw symbol + timestamp; they're the PK but scored table has them)."""
    with session(read_only=True) as con:
        sql = """
            SELECT symbol,
                   CAST(published_at AS DATE) AS session_date,
                   sentiment_label,
                   sentiment_score,
                   sentiment_conf,
                   catalyst_tags
            FROM news_scored
        """
        params: list = []
        if since:
            sql += " WHERE published_at >= ?"
            params.append(since)
        return con.execute(sql, params).fetchdf()


def _explode_catalysts(df: pd.DataFrame) -> pd.DataFrame:
    """Turn the comma-separated `catalyst_tags` column into one-hot columns."""
    # Build a DataFrame of 0/1 per category, indexed the same as df.
    flags = pd.DataFrame(
        {f"tag_{c}": 0 for c in CATALYST_CATEGORIES},
        index=df.index,
        dtype="int8",
    )
    # Iterate once — string split is expensive on millions of rows, so we
    # vectorize using the `contains` approach per category.
    tags = df["catalyst_tags"].fillna("")
    for c in CATALYST_CATEGORIES:
        # Match whole-token: split and check membership. Faster than regex
        # for this particular case (strings are short, categories are fixed).
        flags[f"tag_{c}"] = tags.str.contains(rf"(?:^|,){c}(?:,|$)", regex=True).astype("int8")
    return pd.concat([df, flags], axis=1)


def compute_news_features(since: date | None = None) -> pd.DataFrame:
    """Aggregate `news_scored` into one row per (symbol, session_date)."""
    logger.info("Loading news_scored…")
    raw = _load_scored(since)
    if raw.empty:
        logger.warning("news_scored is empty — run `bot8 features news --backfill` first")
        return pd.DataFrame()
    logger.info("  {:,} scored headlines, {} symbols", len(raw), raw["symbol"].nunique())

    # Collapse aliased tickers onto canonical bars-universe symbols. Must happen
    # before groupby so both old-ticker and new-ticker rows combine into one
    # per-day aggregate.
    alias_hits = raw["symbol"].isin(TICKER_ALIASES)
    if alias_hits.any():
        n_aliased = int(alias_hits.sum())
        raw.loc[alias_hits, "symbol"] = raw.loc[alias_hits, "symbol"].map(TICKER_ALIASES)
        logger.info("  mapped {:,} rows via ticker aliases", n_aliased)

    raw = _explode_catalysts(raw)

    # Group by (symbol, session_date) → compute aggregates
    logger.info("Aggregating per (symbol, session_date)…")
    grouped = raw.groupby(["symbol", "session_date"], sort=False)

    agg = pd.DataFrame({"news_count": grouped.size()})
    agg["sent_mean"] = grouped["sentiment_score"].mean()
    agg["sent_max"] = grouped["sentiment_score"].max()
    agg["sent_min"] = grouped["sentiment_score"].min()
    agg["sent_std"] = grouped["sentiment_score"].std().fillna(0)
    agg["sent_abs_max"] = grouped["sentiment_score"].apply(lambda s: s.abs().max())

    # Confidence-weighted sentiment: Σ(score · conf) / Σ(conf)
    score_conf = raw["sentiment_score"] * raw["sentiment_conf"]
    raw = raw.assign(_score_conf=score_conf)
    sum_score_conf = raw.groupby(["symbol", "session_date"])["_score_conf"].sum()
    sum_conf = grouped["sentiment_conf"].sum()
    agg["sent_conf_weighted"] = (sum_score_conf / sum_conf.replace(0, np.nan)).fillna(0)

    # Label counts
    agg["pos_count"] = grouped["sentiment_label"].apply(lambda s: (s == "positive").sum())
    agg["neg_count"] = grouped["sentiment_label"].apply(lambda s: (s == "negative").sum())
    agg["net_sentiment"] = agg["pos_count"] - agg["neg_count"]

    # Catalyst flags and counts
    for c in CATALYST_CATEGORIES:
        flag_col = f"tag_{c}"
        agg[f"{c}_count"] = grouped[flag_col].sum()
        agg[f"has_{c}"] = (agg[f"{c}_count"] > 0).astype("int8")

    agg = agg.reset_index()

    # ---- Trailing windows (3d, 7d) ------------------------------------
    # Need chronological order per symbol for rolling windows to be valid.
    logger.info("Computing trailing 3d/7d rollups…")
    agg = agg.sort_values(["symbol", "session_date"]).reset_index(drop=True)

    for n in (3, 7):
        # Rolling by trading-day count (not calendar days). Since we only have
        # rows where there was news, this is rolling over "days with news" —
        # slightly imprecise but a reasonable feature for the meta-learner.
        grp = agg.groupby("symbol", sort=False)
        agg[f"news_count_{n}d"] = grp["news_count"].transform(
            lambda s: s.rolling(n, min_periods=1).sum()
        )
        agg[f"sent_mean_{n}d"] = grp["sent_mean"].transform(
            lambda s: s.rolling(n, min_periods=1).mean()
        )
        agg[f"net_sentiment_{n}d"] = grp["net_sentiment"].transform(
            lambda s: s.rolling(n, min_periods=1).sum()
        )

    agg["feature_version"] = FEATURE_VERSION
    agg["session_date"] = pd.to_datetime(agg["session_date"]).dt.date
    return agg


def build_news_features(since: date | None = None) -> int:
    """End-to-end: compute + write news_features_daily. Returns rows written."""
    df = compute_news_features(since=since)
    if df.empty:
        return 0

    with session() as con:
        con.register("nf", df)
        con.execute("CREATE OR REPLACE TABLE news_features_daily AS SELECT * FROM nf")
        con.execute(
            "CREATE INDEX IF NOT EXISTS idx_nfd_symbol_date "
            "ON news_features_daily(symbol, session_date)"
        )
        n = con.execute("SELECT COUNT(*) FROM news_features_daily").fetchone()[0]
    logger.info("Wrote {:,} (symbol, date) news-feature rows ({} cols)", n, df.shape[1])
    return n
