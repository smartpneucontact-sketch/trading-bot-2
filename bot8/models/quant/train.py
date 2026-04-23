"""Quant model training orchestrator — clean 2-level stack.

Architecture:

    Level 1 (base ensemble)
        Input:  quant features only
        Output: per-model OOF predictions

    Level 2 (meta-learner) — fit TWICE for uplift measurement:
        Variant A ("quant_only"):   Ridge( base_OOF )
        Variant B ("with_news"):    Ridge( base_OOF ⊕ news_features )

        IC(B) − IC(A) is the news contribution. If positive, news is
        adding alpha over quant alone; if zero/negative, news isn't
        earning its keep.

This is the proper López de Prado / Grinold stacking pattern. Training
base-once-then-two-metas uses a single CV pass — efficient and fair.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import asdict, dataclass, field
from datetime import date, datetime

import numpy as np
import pandas as pd
from loguru import logger

from bot8.config import get_settings
from bot8.data.db import session
from bot8.models.quant.cv import purged_walk_forward
from bot8.models.quant.ensemble import (
    DEFAULT_MODELS,
    FAST_MODELS,
    BaseModelSpec,
    MetaFit,
    StackedEnsembleResult,
    fit_meta,
    fit_stacked_oof,
    predict_meta,
)
from bot8.models.quant.metrics import signal_stats
from bot8.models.quant.training_data import TrainingData, load_training_data

MODEL_VERSION = "quant-ensemble-v2"


@dataclass
class TrainingReport:
    model_version: str
    trained_at: str
    n_train_rows: int
    n_quant_features: int
    n_news_features: int
    n_claude_features: int
    n_folds: int
    base_models: list[str]
    per_model_ic: dict[str, float]
    per_model_icir: dict[str, float]

    meta_quant_only_ic: float
    meta_quant_only_icir: float
    meta_quant_only_hit: float

    meta_with_news_ic: float
    meta_with_news_icir: float
    meta_with_news_hit: float

    meta_with_claude_ic: float
    meta_with_claude_icir: float
    meta_with_claude_hit: float

    meta_with_all_ic: float          # finbert + claude together
    meta_with_all_icir: float
    meta_with_all_hit: float

    # Uplifts
    ic_uplift_finbert: float = field(init=False)
    ic_uplift_claude: float = field(init=False)
    ic_uplift_all: float = field(init=False)

    def __post_init__(self) -> None:
        self.ic_uplift_finbert = self.meta_with_news_ic - self.meta_quant_only_ic
        self.ic_uplift_claude = self.meta_with_claude_ic - self.meta_quant_only_ic
        self.ic_uplift_all = self.meta_with_all_ic - self.meta_quant_only_ic

    def pretty(self) -> str:
        lines = [
            f"model_version:     {self.model_version}",
            f"trained_at:        {self.trained_at}",
            f"rows:              {self.n_train_rows:,}",
            f"quant features:    {self.n_quant_features}",
            f"finbert features:  {self.n_news_features}",
            f"claude features:   {self.n_claude_features}",
            f"folds:             {self.n_folds}",
            "",
            "Per-base-model OOF (quant features only):",
        ]
        for m in self.base_models:
            lines.append(
                f"  {m:<12s}  IC={self.per_model_ic[m]:+.4f}  ICIR={self.per_model_icir[m]:+.3f}"
            )
        lines += [
            "",
            "Meta-learner comparisons (on identical OOF rows):",
            f"  quant_only:            IC={self.meta_quant_only_ic:+.4f}  "
            f"ICIR={self.meta_quant_only_icir:+.3f}  hit={self.meta_quant_only_hit*100:.2f}%",
            f"  with_finbert:          IC={self.meta_with_news_ic:+.4f}  "
            f"ICIR={self.meta_with_news_icir:+.3f}  hit={self.meta_with_news_hit*100:.2f}%",
            f"  with_claude (only):    IC={self.meta_with_claude_ic:+.4f}  "
            f"ICIR={self.meta_with_claude_icir:+.3f}  hit={self.meta_with_claude_hit*100:.2f}%",
            f"  with_finbert+claude:   IC={self.meta_with_all_ic:+.4f}  "
            f"ICIR={self.meta_with_all_icir:+.3f}  hit={self.meta_with_all_hit*100:.2f}%",
            "",
            f"  ΔIC (finbert alone):           {self.ic_uplift_finbert:+.4f}",
            f"  ΔIC (claude alone):            {self.ic_uplift_claude:+.4f}",
            f"  ΔIC (finbert + claude):        {self.ic_uplift_all:+.4f}",
        ]
        return "\n".join(lines)


def _save_oof_preds(
    td: TrainingData,
    result: StackedEnsembleResult,
    meta_pred_quant_only: np.ndarray,
    meta_pred_with_news: np.ndarray,
    meta_pred_with_claude: np.ndarray,
    meta_pred_with_all: np.ndarray,
) -> int:
    df = td.meta.copy()
    for name in result.base_names:
        df[f"oof_{name}"] = result.oof_preds[name].values
    df["oof_meta_quant_only"] = meta_pred_quant_only
    df["oof_meta_with_news"] = meta_pred_with_news
    df["oof_meta_with_claude"] = meta_pred_with_claude
    df["oof_meta_with_all"] = meta_pred_with_all
    df["model_version"] = MODEL_VERSION

    with session() as con:
        con.register("q", df)
        con.execute("CREATE OR REPLACE TABLE quant_oof_preds AS SELECT * FROM q")
        con.execute(
            "CREATE INDEX IF NOT EXISTS idx_qoof_symbol_date "
            "ON quant_oof_preds(symbol, session_date)"
        )
        return con.execute("SELECT COUNT(*) FROM quant_oof_preds").fetchone()[0]


def _save_artifacts(
    result: StackedEnsembleResult,
    td: TrainingData,
    fit_quant_only: MetaFit,
    fit_with_news: MetaFit,
    report: TrainingReport,
) -> None:
    s = get_settings()
    out_dir = s.models_dir / MODEL_VERSION
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "quant_feature_cols.pkl", "wb") as f:
        pickle.dump(td.quant_feature_cols, f)
    with open(out_dir / "news_feature_cols.pkl", "wb") as f:
        pickle.dump(td.news_feature_cols, f)
    with open(out_dir / "meta_quant_only.pkl", "wb") as f:
        pickle.dump(fit_quant_only, f)
    with open(out_dir / "meta_with_news.pkl", "wb") as f:
        pickle.dump(fit_with_news, f)
    with open(out_dir / "base_names.pkl", "wb") as f:
        pickle.dump(result.base_names, f)
    with open(out_dir / "report.pkl", "wb") as f:
        pickle.dump(report, f)
    (out_dir / "report.json").write_text(json.dumps(asdict(report), indent=2, default=str))
    logger.info("Saved model artifacts to {}", out_dir)


def train_quant(
    since: date | None = None,
    n_splits: int = 6,
    embargo_days: int = 5,
    fast: bool = False,
    target: str = "fwd_return_1d_zscore",
) -> TrainingReport:
    """Train base quant ensemble and both meta variants. Return unified report."""
    td = load_training_data(since=since, target=target)
    models: list[BaseModelSpec] = FAST_MODELS if fast else DEFAULT_MODELS

    folds = list(purged_walk_forward(
        td.dates, n_splits=n_splits, embargo_days=embargo_days,
    ))
    if not folds:
        raise RuntimeError("purged_walk_forward produced 0 folds — data too short?")

    # --- Level 1: base ensemble on quant-only features ----------------------
    logger.info("=== Level 1: base ensemble (quant features only) ===")
    logger.info("  X_quant shape: {}", td.X_quant.shape)
    result = fit_stacked_oof(td.X_quant, td.y, td.dates, folds, models=models)

    mask = result.oof_preds.notna().all(axis=1)
    y_oof = td.y[mask]
    dates_oof = td.dates[mask]

    per_model_ic: dict[str, float] = {}
    per_model_icir: dict[str, float] = {}
    for name in result.base_names:
        stats = signal_stats(y_oof, result.oof_preds.loc[mask, name], dates_oof)
        per_model_ic[name] = stats.ic_mean
        per_model_icir[name] = stats.icir
        logger.info("  {} OOF: {}", name, stats.pretty())

    # --- Level 2: FOUR meta-learners on identical OOF rows ------------------
    def _eval_meta(extras, name: str):
        fit = fit_meta(result, td.y, extra_features=extras, alpha=1.0, name=name)
        pred = np.full(len(td.y), np.nan)
        pred[mask.values] = predict_meta(
            fit,
            result.oof_preds.loc[mask],
            extra_features=(extras.loc[mask] if extras is not None else None),
        )
        stats = signal_stats(y_oof, pd.Series(pred[mask.values]), dates_oof)
        logger.info("META ({}) OOF: {}", name, stats.pretty())
        return fit, pred, stats

    logger.info("=== Level 2a: meta (quant_only) ===")
    fit_a, pred_a_full, stats_a = _eval_meta(None, "quant_only")

    logger.info("=== Level 2b: meta (with_finbert) ===")
    fit_b, pred_b_full, stats_b = _eval_meta(td.X_news, "with_finbert")

    if td.claude_feature_cols:
        logger.info("=== Level 2c: meta (with_claude_only) ===")
        fit_c, pred_c_full, stats_c = _eval_meta(td.X_claude, "with_claude_only")

        logger.info("=== Level 2d: meta (with_finbert+claude) ===")
        fit_d, pred_d_full, stats_d = _eval_meta(td.X_news_all, "with_finbert_claude")
    else:
        logger.warning("No Claude features present — skipping claude metas")
        fit_c = fit_d = fit_b
        pred_c_full = pred_d_full = pred_b_full
        stats_c = stats_d = stats_b

    # --- Persist + report ---------------------------------------------------
    _save_oof_preds(td, result, pred_a_full, pred_b_full, pred_c_full, pred_d_full)

    report = TrainingReport(
        model_version=MODEL_VERSION,
        trained_at=datetime.utcnow().isoformat(timespec="seconds"),
        n_train_rows=len(td.X),
        n_quant_features=len(td.quant_feature_cols),
        n_news_features=len(td.news_feature_cols),
        n_claude_features=len(td.claude_feature_cols),
        n_folds=len(folds),
        base_models=result.base_names,
        per_model_ic=per_model_ic,
        per_model_icir=per_model_icir,
        meta_quant_only_ic=stats_a.ic_mean,
        meta_quant_only_icir=stats_a.icir,
        meta_quant_only_hit=stats_a.hit_rate,
        meta_with_news_ic=stats_b.ic_mean,
        meta_with_news_icir=stats_b.icir,
        meta_with_news_hit=stats_b.hit_rate,
        meta_with_claude_ic=stats_c.ic_mean,
        meta_with_claude_icir=stats_c.icir,
        meta_with_claude_hit=stats_c.hit_rate,
        meta_with_all_ic=stats_d.ic_mean,
        meta_with_all_icir=stats_d.icir,
        meta_with_all_hit=stats_d.hit_rate,
    )
    _save_artifacts(result, td, fit_a, fit_b, report)
    return report
