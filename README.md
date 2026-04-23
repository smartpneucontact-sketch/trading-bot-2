# Bot 8 — Hybrid Quant + News Trading System

Clean rewrite layered on lessons from bots 4–6. Long/short, market-neutral, daily-rebalance, news-aware.

## Philosophy

| Layer | Role |
|---|---|
| **Raw news DB** (DuckDB) | Single source of truth for headlines. Ingested from FNSPID (historical), Alpaca News + Yahoo + SEC EDGAR (live). |
| **News features** | FinBERT sentiment + regex-based catalyst classification. Free, consistent across train and inference. |
| **Quant features** | V6-style ~150 features: returns, vol, technicals, macro, regime inputs. |
| **Quant model** | Stacked ensemble (LightGBM × 3 + XGBoost + CatBoost + Ridge meta). 1-day horizon. |
| **Combiner (meta-learner)** | Ridge / LightGBM on `[quant_OOF_pred, news_features, regime]` → final rank. |
| **Portfolio** | Long top decile, short bottom decile. Dollar-neutral, sector-neutral. Conviction-weighted, capped. |
| **Risk** | Regime filter (VIX/SPY/HYG), gross/net exposure caps, per-name caps, borrow-cost penalty. |
| **Execution** | Alpaca paper (shortable check, fractional handling, fill-price polling). |
| **Backtest** | Purged + embargoed walk-forward CV. Fees, slippage, borrow costs modelled. |
| **Live** | Daily 07:00 ET news fetch → 07:30 scoring → 09:35 rebalance. |
| **Telemetry** | Trade journal (JSONL + CSV), live-vs-backtest IC tracker, factor/sector exposure logs. |

## Key design choices (vs V6)

- **1-day horizon**, not 5 — news decays fast, label overlap killed V6's CV.
- **Long/short**, not long-only — separate alpha from market beta.
- **Sector-neutral** — eliminate hidden factor tilts V6 ignored.
- **Purged + embargoed CV** — fixes the leakage that inflated V6's reported IC to 0.394.
- **Zero-cost news stack** — FinBERT + regex. Upgrade to Claude Haiku batch later if pilot shows news adds IC.

## Status

🚧 Under construction. See `TODO.md` or the TodoWrite list in the active session.

## Heads-up: macOS + iCloud Drive

If your project lives under `~/Documents/` and iCloud Drive is on, every file gets a `com.apple.provenance` xattr + `UF_HIDDEN` flag. Python 3.13's site.py skips "hidden" `.pth` files as a security measure, which breaks the `bot8` console script entry point installed by `pip install -e .`.

**Workaround**: use `./bot8.sh <command>` instead of `./bot8.sh <command>`. The wrapper invokes `python -m bot8.cli` and is immune to the flag issue.

**Permanent fix** (recommended): move the project to a non-iCloud folder, e.g. `~/Developer/bot8/`.

## Quickstart

Phase-by-phase install — only pull in the heavy stacks as you need them.

```bash
# create venv (uses system python3; swap for pyenv/conda if preferred)
python3 -m venv .venv
source .venv/bin/activate

# Phase 1 — scaffolding + DuckDB + FNSPID loader
pip install -e ".[hf,dev]"
pytest                             # smoke tests
./bot8.sh info                          # resolved config
./bot8.sh data init                     # create DuckDB + schema
./bot8.sh data fnspid --since 2020-01-01 --file 'Stock_news/nasdaq_*.csv'

# Phase 2 — FinBERT + regex news scoring
pip install -e ".[hf,nlp,dev]"
./bot8.sh features news --backfill      # (not built yet)

# Phase 3 — quant model, backtest, trading
pip install -e ".[all]"
./bot8.sh train --horizon 1d
./bot8.sh backtest --start 2021-01-01 --end 2024-12-31
./bot8.sh live premarket
```
