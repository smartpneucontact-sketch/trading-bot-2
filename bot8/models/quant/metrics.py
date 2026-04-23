"""Core metrics for signal quality evaluation.

IC (Information Coefficient):
  Spearman rank correlation between predicted and realized returns, computed
  per day across the cross-section, then averaged. This is the canonical
  quant signal metric — robust to outliers, ignores magnitude, rewards
  consistent ordering.

ICIR (IC Information Ratio):
  mean(daily IC) / std(daily IC). Rewards consistency; a bot that scores
  IC=0.04 every day beats IC=0.10 on some days and −0.05 on others.

hit_rate:
  Fraction where sign(y_pred) == sign(y_true). Simple, direct interpretation.

See V6's IC = 0.394 — likely leaked from overlapping 5d labels in plain
k-fold. We re-compute with purged walk-forward; expect 0.03–0.08 for a
genuine quant + news signal.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


@dataclass(frozen=True, slots=True)
class SignalStats:
    ic_mean: float
    ic_std: float
    icir: float
    hit_rate: float
    n_days: int
    n_rows: int

    def pretty(self) -> str:
        return (
            f"IC={self.ic_mean:+.4f}  ICIR={self.icir:+.3f}  "
            f"hit={self.hit_rate*100:.1f}%  "
            f"days={self.n_days}  rows={self.n_rows:,}"
        )


def per_day_ic(y_true: pd.Series, y_pred: pd.Series, dates: pd.Series) -> pd.Series:
    """Return a Series of per-date Spearman IC values (index = unique date)."""
    df = pd.DataFrame({"y": y_true.values, "p": y_pred.values, "d": dates.values})

    def _rho(group: pd.DataFrame) -> float:
        if len(group) < 5:
            return np.nan
        # Spearman is robust to NaN via dropna
        clean = group.dropna(subset=["y", "p"])
        if len(clean) < 5:
            return np.nan
        rho, _ = spearmanr(clean["y"], clean["p"])
        return float(rho) if np.isfinite(rho) else np.nan

    return df.groupby("d", sort=True).apply(_rho)


def signal_stats(
    y_true: pd.Series, y_pred: pd.Series, dates: pd.Series
) -> SignalStats:
    """Compute IC mean/std, ICIR, hit rate."""
    ic = per_day_ic(y_true, y_pred, dates).dropna()
    ic_mean = float(ic.mean()) if len(ic) > 0 else np.nan
    ic_std = float(ic.std()) if len(ic) > 1 else np.nan
    icir = ic_mean / ic_std if ic_std and np.isfinite(ic_std) and ic_std > 0 else np.nan

    # Hit rate — same sign
    df = pd.DataFrame({"y": y_true.values, "p": y_pred.values}).dropna()
    hit = float((np.sign(df["y"]) == np.sign(df["p"])).mean()) if len(df) else np.nan

    return SignalStats(
        ic_mean=ic_mean,
        ic_std=ic_std,
        icir=icir,
        hit_rate=hit,
        n_days=int(ic.size),
        n_rows=int(len(y_true)),
    )
