"""Stacked ensemble for the quant signal.

Base models (configurable):
  - LightGBM main        — baseline gradient boosting
  - LightGBM heavy-reg   — deep regularization, guards against overfit
  - LightGBM DART        — dropout-style boosting
  - XGBoost              — different tree growth + split logic
  - CatBoost             — symmetric trees, handles categorical natively

Meta-learner:
  - Ridge regression on stacked OOF predictions (L2 = α)
  - Works well because the 5 base models have correlated but non-identical
    errors. Ridge learns their relative weights without overfitting.

OOF design:
  - For each fold in the purged walk-forward split:
      1. Train each base model on the fold's training rows.
      2. Predict on the fold's test rows → stored in oof_preds[test_idx].
  - After all folds, `oof_preds` contains one prediction per base model per
    training row — each prediction is never computed on a row that model
    has seen. This is the proper training target for the meta-learner.

All models use the default tree-model NaN handling; no imputation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import pandas as pd
from loguru import logger


class BaseEstimator(Protocol):
    def fit(self, X, y, **kwargs): ...
    def predict(self, X) -> np.ndarray: ...


@dataclass(slots=True)
class BaseModelSpec:
    name: str
    builder: callable  # returns a fresh (unfit) estimator each call
    fit_kwargs: dict = field(default_factory=dict)


def _lightgbm_main():
    import lightgbm as lgb
    return lgb.LGBMRegressor(
        objective="regression",
        n_estimators=500,
        num_leaves=63,
        max_depth=7,
        learning_rate=0.03,
        min_data_in_leaf=200,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=5,
        lambda_l1=0.1,
        lambda_l2=0.1,
        verbosity=-1,
        n_jobs=-1,
    )


def _lightgbm_heavy_reg():
    import lightgbm as lgb
    return lgb.LGBMRegressor(
        objective="regression",
        n_estimators=500,
        num_leaves=15,
        max_depth=5,
        learning_rate=0.03,
        min_data_in_leaf=500,
        feature_fraction=0.7,
        bagging_fraction=0.7,
        bagging_freq=5,
        lambda_l1=0.5,
        lambda_l2=0.5,
        verbosity=-1,
        n_jobs=-1,
    )


def _lightgbm_dart():
    import lightgbm as lgb
    return lgb.LGBMRegressor(
        objective="regression",
        boosting_type="dart",
        n_estimators=400,
        num_leaves=63,
        learning_rate=0.05,
        drop_rate=0.1,
        skip_drop=0.5,
        min_data_in_leaf=200,
        verbosity=-1,
        n_jobs=-1,
    )


def _xgboost():
    import xgboost as xgb
    return xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=400,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.5,
        verbosity=0,
        n_jobs=-1,
        tree_method="hist",
    )


def _catboost():
    from catboost import CatBoostRegressor
    return CatBoostRegressor(
        iterations=500,
        depth=6,
        learning_rate=0.05,
        l2_leaf_reg=3,
        random_seed=42,
        verbose=False,
        allow_writing_files=False,
        thread_count=-1,
    )


DEFAULT_MODELS: list[BaseModelSpec] = [
    BaseModelSpec("lgbm_main", _lightgbm_main),
    BaseModelSpec("lgbm_reg", _lightgbm_heavy_reg),
    BaseModelSpec("lgbm_dart", _lightgbm_dart),
    BaseModelSpec("xgboost", _xgboost),
    BaseModelSpec("catboost", _catboost),
]

FAST_MODELS: list[BaseModelSpec] = [
    BaseModelSpec("lgbm_main", _lightgbm_main),
    BaseModelSpec("lgbm_reg", _lightgbm_heavy_reg),
    BaseModelSpec("xgboost", _xgboost),
]


@dataclass(slots=True)
class StackedEnsembleResult:
    """Holds OOF predictions per model + fitted meta-learner + per-fold models."""
    base_names: list[str]
    oof_preds: pd.DataFrame  # columns = base model names, index aligned with X
    meta_model: object | None = None  # fitted Ridge
    full_models: dict[str, object] = field(default_factory=dict)  # base model fit on all data


def fit_stacked_oof(
    X: pd.DataFrame,
    y: pd.Series,
    dates: pd.Series,
    folds,  # Iterable[Fold]
    models: list[BaseModelSpec] | None = None,
) -> StackedEnsembleResult:
    """Fit base models in walk-forward CV, return StackedEnsembleResult.

    Does NOT fit the meta-learner yet — caller can do that after inspecting
    per-fold OOF quality.
    """
    models = models or DEFAULT_MODELS

    # Pre-allocate OOF arrays (NaN where a row was never in any test fold).
    oof = pd.DataFrame(
        {m.name: np.full(len(X), np.nan, dtype="float64") for m in models},
        index=X.index,
    )

    fold_list = list(folds)
    logger.info("Training {} base models × {} folds", len(models), len(fold_list))

    for fold in fold_list:
        logger.info(
            "Fold {}: train={:,} rows ({} → {}), test={:,} rows ({} → {})",
            fold.fold_idx,
            len(fold.train_idx),
            fold.train_dates[0].date(),
            fold.train_dates[1].date(),
            len(fold.test_idx),
            fold.test_dates[0].date(),
            fold.test_dates[1].date(),
        )
        X_tr = X.iloc[fold.train_idx]
        y_tr = y.iloc[fold.train_idx]
        X_te = X.iloc[fold.test_idx]

        for spec in models:
            model = spec.builder()
            logger.info("  fitting {} …", spec.name)
            model.fit(X_tr, y_tr, **spec.fit_kwargs)
            preds = model.predict(X_te)
            oof.loc[X.index[fold.test_idx], spec.name] = preds

    return StackedEnsembleResult(
        base_names=[m.name for m in models],
        oof_preds=oof,
    )


@dataclass(slots=True)
class MetaFit:
    """A single fitted meta-learner with its feature-name list."""
    name: str                           # 'quant_only' | 'with_news'
    model: object                       # fitted Ridge
    feature_names: list[str]            # in order of Ridge coef_


def fit_meta(
    result: StackedEnsembleResult,
    y: pd.Series,
    extra_features: pd.DataFrame | None = None,
    alpha: float = 1.0,
    name: str = "meta",
) -> MetaFit:
    """Train Ridge meta-learner on base OOF predictions, optionally concatenated
    with extra raw features (typically the news feature matrix).

    Only uses rows where all base models have a non-NaN prediction
    (i.e. rows that were in some test fold).

    News features: filled with 0 for missing — which is legitimate ("no news").
    """
    from sklearn.linear_model import Ridge

    mask = result.oof_preds.notna().all(axis=1)
    base = result.oof_preds.loc[mask].copy()
    feature_names: list[str] = list(result.base_names)

    if extra_features is not None and extra_features.shape[1] > 0:
        extras = extra_features.loc[mask].fillna(0).copy()
        feature_names += list(extras.columns)
        X_meta = pd.concat([base, extras], axis=1).values
    else:
        X_meta = base.values

    y_meta = y.loc[mask].values

    ridge = Ridge(alpha=alpha, fit_intercept=True)
    ridge.fit(X_meta, y_meta)

    # Log weights — only the non-zero ones, to stay readable when there are
    # 40+ news features stacked on top of 3-5 base models.
    nonzero = [(n, w) for n, w in zip(feature_names, ridge.coef_) if abs(w) > 1e-4]
    nonzero.sort(key=lambda t: -abs(t[1]))
    logger.info("  {} meta: {} non-trivial weights (intercept {:+.6f})",
                name, len(nonzero), ridge.intercept_)
    for fn, w in nonzero[:10]:
        logger.info("    {:<25s} {:+.4f}", fn, w)
    if len(nonzero) > 10:
        logger.info("    ... and {} more", len(nonzero) - 10)

    fit = MetaFit(name=name, model=ridge, feature_names=feature_names)
    result.meta_model = ridge  # back-compat; points at most-recent
    return fit


def predict_meta(
    fit: MetaFit,
    base_preds: pd.DataFrame,
    extra_features: pd.DataFrame | None = None,
) -> np.ndarray:
    """Apply a fitted MetaFit to new inputs. Order must match fit.feature_names."""
    cols_from_base = [c for c in fit.feature_names if c in base_preds.columns]
    rows = base_preds[cols_from_base].copy()
    if extra_features is not None:
        extra_cols = [c for c in fit.feature_names if c in extra_features.columns]
        if extra_cols:
            rows = pd.concat([rows, extra_features[extra_cols].fillna(0)], axis=1)
    # Re-order strictly to fit.feature_names
    rows = rows[fit.feature_names]
    return fit.model.predict(rows.values)
