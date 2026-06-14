from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from features import CATEGORY_FEATURE_COLUMNS, FEATURE_COLUMNS, NUMERIC_FEATURE_COLUMNS


@dataclass
class ModelResult:
    estimator: Pipeline
    metrics: dict[str, float | int | str]
    training_rows: int
    positive_rate: float


def fit_cut_model(panel: pd.DataFrame) -> ModelResult:
    train = _labeled_rows(panel)
    if train.empty:
        raise ValueError("No labeled rows are available for model training.")

    estimator = _make_estimator(train["label"].nunique() >= 2)
    estimator.fit(train[FEATURE_COLUMNS], train["label"].astype(int))
    metrics = evaluate_temporal(panel)

    return ModelResult(
        estimator=estimator,
        metrics=metrics,
        training_rows=len(train),
        positive_rate=float(train["label"].mean()),
    )


def predict_latest(model: ModelResult, panel: pd.DataFrame) -> pd.DataFrame:
    if panel.empty:
        return pd.DataFrame()

    latest = (
        panel.sort_values(["ticker", "month"])
        .groupby("ticker", as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )
    probabilities = _predict_positive_proba(model.estimator, latest[FEATURE_COLUMNS])
    latest = latest.copy()
    latest["p_cut"] = probabilities
    latest["bucket"] = latest["p_cut"].map(_bucket)
    output_columns = _unique_columns(
        [
            "ticker",
            "category",
            "month",
            "price",
            "dist_yield",
            "p_cut",
            "bucket",
            *NUMERIC_FEATURE_COLUMNS,
        ]
    )
    return latest[output_columns].sort_values(["bucket", "p_cut", "ticker"])


def evaluate_temporal(panel: pd.DataFrame) -> dict[str, float | int | str]:
    rows = _labeled_rows(panel)
    if len(rows) < 40:
        return {"status": "skipped: fewer than 40 labeled rows", "folds": 0}
    if rows["label"].nunique() < 2:
        return {"status": "skipped: only one label class present", "folds": 0}

    predictions: list[pd.DataFrame] = []
    for year in sorted(rows["month"].dt.year.unique()):
        test_start = pd.Timestamp(year=year, month=1, day=1)
        test_end = pd.Timestamp(year=year, month=12, day=31)
        train_cutoff = test_start - pd.DateOffset(months=13)
        train = rows[rows["month"] < train_cutoff]
        test = rows[(rows["month"] >= test_start) & (rows["month"] <= test_end)]

        if len(train) < 40 or len(test) < 8 or train["label"].nunique() < 2:
            continue

        estimator = _make_estimator(use_logistic=True)
        estimator.fit(train[FEATURE_COLUMNS], train["label"].astype(int))
        fold = test[["ticker", "month", "label", "baseline_decline_12m"]].copy()
        fold["score"] = _predict_positive_proba(estimator, test[FEATURE_COLUMNS])
        predictions.append(fold)

    if not predictions:
        return _fallback_temporal_eval(rows)

    scored = pd.concat(predictions, ignore_index=True)
    return _metrics_from_scored(scored, status="ok", folds=len(predictions))


def _fallback_temporal_eval(rows: pd.DataFrame) -> dict[str, float | int | str]:
    unique_months = rows["month"].drop_duplicates().sort_values()
    cutoff = unique_months.iloc[int(len(unique_months) * 0.75)]
    train = rows[rows["month"] < cutoff]
    test = rows[rows["month"] >= cutoff]
    if len(train) < 20 or len(test) < 8 or train["label"].nunique() < 2:
        return {"status": "skipped: not enough temporal train/test data", "folds": 0}

    estimator = _make_estimator(use_logistic=True)
    estimator.fit(train[FEATURE_COLUMNS], train["label"].astype(int))
    scored = test[["ticker", "month", "label", "baseline_decline_12m"]].copy()
    scored["score"] = _predict_positive_proba(estimator, test[FEATURE_COLUMNS])
    return _metrics_from_scored(scored, status="fallback temporal split", folds=1)


def _metrics_from_scored(scored: pd.DataFrame, status: str, folds: int) -> dict[str, float | int | str]:
    y_true = scored["label"].astype(int).to_numpy()
    y_score = scored["score"].to_numpy()
    risky_flag = y_score > 0.50
    baseline_flag = scored["baseline_decline_12m"].fillna(0).astype(int).to_numpy() == 1
    rng = np.random.default_rng(436)
    random_score = rng.random(len(y_true))

    if len(np.unique(y_true)) < 2:
        pr_auc = float("nan")
        baseline_pr_auc = float("nan")
        random_pr_auc = float("nan")
    else:
        pr_auc = float(average_precision_score(y_true, y_score))
        baseline_pr_auc = float(average_precision_score(y_true, baseline_flag.astype(float)))
        random_pr_auc = float(average_precision_score(y_true, random_score))

    return {
        "status": status,
        "folds": folds,
        "test_rows": int(len(scored)),
        "test_positive_rate": float(np.mean(y_true)),
        "risky_precision": float(precision_score(y_true, risky_flag, zero_division=0)),
        "pr_auc": pr_auc,
        "baseline_precision": float(precision_score(y_true, baseline_flag, zero_division=0)),
        "baseline_pr_auc": baseline_pr_auc,
        "random_pr_auc": random_pr_auc,
    }


def _make_estimator(use_logistic: bool) -> Pipeline:
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                NUMERIC_FEATURE_COLUMNS,
            ),
            ("category", encoder, CATEGORY_FEATURE_COLUMNS),
        ],
        remainder="drop",
    )
    classifier: Any
    if use_logistic:
        classifier = LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            solver="lbfgs",
        )
    else:
        classifier = DummyClassifier(strategy="prior")

    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])


def _predict_positive_proba(estimator: Pipeline, features: pd.DataFrame) -> np.ndarray:
    classifier = estimator.named_steps["classifier"]
    classes = getattr(classifier, "classes_", np.array([0, 1]))
    probabilities = estimator.predict_proba(features)
    if 1 in classes:
        positive_idx = list(classes).index(1)
        return probabilities[:, positive_idx]
    if len(classes) == 1 and int(classes[0]) == 1:
        return np.ones(len(features))
    return np.zeros(len(features))


def _labeled_rows(panel: pd.DataFrame) -> pd.DataFrame:
    if panel.empty:
        return panel.copy()
    rows = panel.dropna(subset=["label"]).copy()
    rows["month"] = pd.to_datetime(rows["month"])
    return rows.sort_values(["month", "ticker"]).reset_index(drop=True)


def _unique_columns(columns: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for column in columns:
        if column not in seen:
            seen.add(column)
            output.append(column)
    return output


def _bucket(probability: float) -> str:
    if probability < 0.20:
        return "Safe"
    if probability <= 0.50:
        return "Watch"
    return "Risky"
