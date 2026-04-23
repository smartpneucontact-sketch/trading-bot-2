-- bot8 DuckDB schema
-- Three broad tables: raw news, raw market bars, scored news features.
-- Everything keyed by (symbol, date) for clean joins into feature pipelines.

-- ---------------------------------------------------------------------------
-- NEWS_RAW: one row per headline, union of all sources (FNSPID / Alpaca / Yahoo / EDGAR).
-- `source` tracks provenance so we can filter or de-dup per source if needed.
-- `headline_hash` is a stable key for dedup across sources.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS news_raw (
    symbol          VARCHAR NOT NULL,
    published_at    TIMESTAMP NOT NULL,     -- UTC
    source          VARCHAR NOT NULL,       -- fnspid | alpaca | yahoo | edgar
    headline        VARCHAR NOT NULL,
    body            VARCHAR,                -- full text if available
    url             VARCHAR,
    author          VARCHAR,
    headline_hash   VARCHAR NOT NULL,       -- sha1(symbol|headline) for dedup
    ingested_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (symbol, published_at, source, headline_hash)
);

CREATE INDEX IF NOT EXISTS idx_news_raw_symbol_date ON news_raw(symbol, published_at);
CREATE INDEX IF NOT EXISTS idx_news_raw_date ON news_raw(published_at);

-- ---------------------------------------------------------------------------
-- NEWS_SCORED: one row per headline, FinBERT + regex output.
-- Kept separate from news_raw so we can re-score without re-ingesting.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS news_scored (
    symbol              VARCHAR NOT NULL,
    published_at        TIMESTAMP NOT NULL,
    headline_hash       VARCHAR NOT NULL,

    -- FinBERT
    sentiment_label     VARCHAR,            -- positive | neutral | negative
    sentiment_score     DOUBLE,             -- signed score in [-1, +1]
    sentiment_conf      DOUBLE,             -- confidence [0, 1]

    -- Regex catalyst (multi-label; comma-separated tags)
    catalyst_tags       VARCHAR,            -- 'earnings,guidance' etc.

    -- Metadata
    model_version       VARCHAR,            -- 'finbert-prosus-v1' + regex version
    scored_at           TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY (symbol, published_at, headline_hash)
);

CREATE INDEX IF NOT EXISTS idx_news_scored_symbol_date ON news_scored(symbol, published_at);

-- ---------------------------------------------------------------------------
-- BARS_DAILY: one row per (symbol, session_date). Daily OHLCV from Alpaca.
-- session_date is the trading session (America/New_York local), not UTC.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS bars_daily (
    symbol          VARCHAR NOT NULL,
    session_date    DATE NOT NULL,
    open            DOUBLE,
    high            DOUBLE,
    low             DOUBLE,
    close           DOUBLE,
    adj_close       DOUBLE,                 -- split/div adjusted
    volume          BIGINT,
    vwap            DOUBLE,
    trade_count     BIGINT,
    ingested_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (symbol, session_date)
);

CREATE INDEX IF NOT EXISTS idx_bars_symbol_date ON bars_daily(symbol, session_date);

-- ---------------------------------------------------------------------------
-- UNIVERSE: which tickers are tradable on which date.
-- Populated from S&P 500 + Nasdaq 100 + Russell 1000 mid-cap constituents.
-- Handles additions/removals over time (so we don't have survivorship bias).
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS universe (
    symbol          VARCHAR NOT NULL,
    as_of_date      DATE NOT NULL,
    index_membership VARCHAR NOT NULL,       -- 'SP500' | 'NDX100' | 'R1000MID'
    sector          VARCHAR,                 -- GICS sector
    industry        VARCHAR,                 -- GICS industry
    is_shortable    BOOLEAN,
    is_fractionable BOOLEAN,
    is_tradable     BOOLEAN,                 -- active + can place orders
    PRIMARY KEY (symbol, as_of_date, index_membership)
);

-- Migration: add is_tradable to older universe tables created before this column existed.
-- DuckDB supports ADD COLUMN IF NOT EXISTS so this is safe to run on every init.
ALTER TABLE universe ADD COLUMN IF NOT EXISTS is_tradable BOOLEAN;

-- ---------------------------------------------------------------------------
-- MACRO_DAILY: one row per (series, session_date). Sparse macro/market-wide
-- series used for regime features. series_code is a compact symbol:
--   VIX, SPY, HYG, TNX (10y yield), IRX (3m yield), GLD, DXY, USO, SHY (1-3y), LQD
-- Stored long-form (series + date + value) rather than wide to simplify adding
-- new series without schema migrations.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS macro_daily (
    series_code     VARCHAR NOT NULL,
    session_date    DATE NOT NULL,
    open            DOUBLE,
    high            DOUBLE,
    low             DOUBLE,
    close           DOUBLE,
    volume          BIGINT,
    ingested_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (series_code, session_date)
);

CREATE INDEX IF NOT EXISTS idx_macro_code_date ON macro_daily(series_code, session_date);

-- ---------------------------------------------------------------------------
-- INGEST_LOG: what was ingested when, so we can resume partial downloads safely.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS ingest_log (
    source          VARCHAR NOT NULL,        -- 'fnspid' | 'alpaca_news' | etc.
    key             VARCHAR NOT NULL,        -- e.g. file path, date range
    rows_ingested   BIGINT,
    started_at      TIMESTAMP,
    finished_at     TIMESTAMP,
    status          VARCHAR,                 -- 'ok' | 'partial' | 'error'
    error_message   VARCHAR,
    PRIMARY KEY (source, key, started_at)
);
