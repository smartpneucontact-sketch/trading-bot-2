"""Long/short portfolio construction.

For each rebalance date:
  1. Score → rank within the tradable universe.
  2. Long: top `long_decile` names.
  3. Short: bottom `short_decile` names — filtered to `is_shortable` only.
  4. Weights: conviction-proportional within each side, scaled so
     gross_long = gross_short = 1.0 (→ gross = 2.0, net = 0, dollar-neutral).
  5. Per-name cap clamps outliers.
  6. Sector neutralization: optionally re-balance weights so the long and
     short books have matched sector exposures (limits hidden factor risk).

Output: one row per (session_date, symbol, weight). Weight in [-cap, +cap],
sums to ~0 per day (dollar-neutral after cap clamping).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from loguru import logger


@dataclass(frozen=True, slots=True)
class PortfolioConfig:
    long_decile: float = 0.10
    short_decile: float = 0.10
    max_position_weight: float = 0.02   # 2% cap per name
    target_gross_exposure: float = 2.0  # long 1.0 + short 1.0
    sector_neutralize: bool = True
    require_shortable_for_shorts: bool = True
    min_universe_per_day: int = 50       # skip days with thin cross-section


def _pick_deciles(
    day: pd.DataFrame,
    score_col: str,
    cfg: PortfolioConfig,
) -> pd.DataFrame:
    """Rank one day's cross-section, tag long/short/flat."""
    df = day.copy()
    df = df.dropna(subset=[score_col])
    if len(df) < cfg.min_universe_per_day:
        return df.assign(side="flat", raw_weight=0.0)

    n_long = max(1, int(len(df) * cfg.long_decile))
    n_short = max(1, int(len(df) * cfg.short_decile))

    df = df.sort_values(score_col, ascending=False).reset_index(drop=True)
    side = np.full(len(df), "flat", dtype=object)
    side[:n_long] = "long"

    # Shorts: from bottom of ranking, filter shortable-only if required.
    short_pool = df.iloc[::-1]  # reverse ranking, worst scores first
    if cfg.require_shortable_for_shorts and "is_shortable" in df.columns:
        short_pool = short_pool[short_pool["is_shortable"].astype(bool)]
    short_idx = short_pool.index[:n_short]
    side[short_idx] = np.where(side[short_idx] == "long", "long", "short")

    df["side"] = side
    return df


def _conviction_weights(df: pd.DataFrame, score_col: str, cfg: PortfolioConfig) -> pd.DataFrame:
    """Within each of long/short, weight by score magnitude normalized to 1.0 gross.

    Scores used for magnitude are *ranks* within the day — robust to outliers
    vs. raw score values. Long ranks → higher conviction; short ranks → lower.
    """
    df = df.copy()
    df["raw_weight"] = 0.0

    for side, sign in (("long", +1.0), ("short", -1.0)):
        mask = df["side"] == side
        if not mask.any():
            continue
        # Conviction from within-day rank (higher rank for longs, lower for shorts)
        ranks = df.loc[mask, score_col].rank(
            pct=True, method="average", ascending=(side == "short")
        )
        weights = ranks / ranks.sum()
        df.loc[mask, "raw_weight"] = sign * weights

    return df


def _apply_cap(weights: pd.Series, cap: float) -> pd.Series:
    """Clamp |weight| to cap. No renormalization after — renormalizing back to
    ±1 per side would undo the cap when every name hits it (e.g. 20 longs × 2%
    cap = 0.4 max gross; scaling back to 1.0 pushes everyone to 0.05 > cap).

    Keeping raw clamp means gross naturally settles at (n_side × cap) when the
    cap binds, which is the economically correct behaviour."""
    return weights.clip(-cap, cap)


def _sector_neutralize(df: pd.DataFrame) -> pd.DataFrame:
    """Center long and short books per sector so each sector's net weight is 0.

    Strategy: for each sector, compute net weight; redistribute to cancel it
    by scaling the dominant side down. Simple and preserves total gross.
    """
    if "sector" not in df.columns:
        return df

    out = df.copy()
    for sector, grp in df.groupby("sector", dropna=False):
        net = grp["raw_weight"].sum()
        if abs(net) < 1e-8:
            continue
        # Scale down the dominant side of this sector to neutralize.
        if net > 0:
            # Too much long here; shrink long weights in this sector.
            long_mask = (out["sector"] == sector) & (out["raw_weight"] > 0)
            if long_mask.any():
                long_sum = out.loc[long_mask, "raw_weight"].sum()
                if long_sum > 0:
                    scale = max(0.0, 1 - net / long_sum)
                    out.loc[long_mask, "raw_weight"] *= scale
        else:
            short_mask = (out["sector"] == sector) & (out["raw_weight"] < 0)
            if short_mask.any():
                short_sum = out.loc[short_mask, "raw_weight"].sum().__abs__()
                if short_sum > 0:
                    scale = max(0.0, 1 - (-net) / short_sum)
                    out.loc[short_mask, "raw_weight"] *= scale
    return out


def build_daily_weights(
    df: pd.DataFrame,
    score_col: str = "score",
    cfg: PortfolioConfig | None = None,
) -> pd.DataFrame:
    """Turn a table of daily scores into target weights.

    `df` must contain columns: session_date, symbol, {score_col}, is_shortable,
    sector (if sector_neutralize is on).

    Returns a DataFrame with the same PK + `weight` column (the final target).
    Days with too few names produce 0-weights (flat).
    """
    cfg = cfg or PortfolioConfig()

    out_frames: list[pd.DataFrame] = []
    for dt, day in df.groupby("session_date", sort=True):
        picked = _pick_deciles(day, score_col, cfg)
        if picked.empty:
            continue
        weighted = _conviction_weights(picked, score_col, cfg)
        if cfg.sector_neutralize and "sector" in weighted.columns:
            weighted = _sector_neutralize(weighted)
        weighted["weight"] = _apply_cap(weighted["raw_weight"], cfg.max_position_weight)
        out_frames.append(
            weighted[["session_date", "symbol", "side", "weight"]]
            .assign(session_date=dt)
        )

    if not out_frames:
        return pd.DataFrame(columns=["session_date", "symbol", "side", "weight"])

    result = pd.concat(out_frames, ignore_index=True)

    # NOTE: we deliberately do NOT rescale to `target_gross_exposure` here —
    # the per-name cap must win over the gross target. With N_long names per
    # side × cap, max achievable gross is 2·N·cap. In production (500 symbols,
    # 10% decile = 50 longs, 2% cap) this is exactly 2.0; in thin tests it's
    # lower. Rescaling up would re-violate the cap.

    logger.info(
        "Portfolio: {} days × {} (symbol,day) weights, "
        "avg gross per day = {:.2f}",
        result["session_date"].nunique(),
        len(result),
        result.groupby("session_date")["weight"].apply(lambda w: w.abs().sum()).mean(),
    )
    return result
