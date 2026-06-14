from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "category",
    "payout_trend_24m",
    "payout_stability",
    "ever_cut_before",
    "price_ret_12m",
    "price_ret_24m",
    "dist_yield",
    "aum",
    "expense_ratio",
    "age_years",
]

NUMERIC_FEATURE_COLUMNS = [col for col in FEATURE_COLUMNS if col != "category"]
CATEGORY_FEATURE_COLUMNS = ["category"]


def build_feature_panel(
    universe: pd.DataFrame,
    data_by_ticker: dict[str, Any],
    cut_threshold: float = 0.80,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for item in universe.to_dict("records"):
        ticker = str(item["ticker"])
        category = str(item["category"])
        data = data_by_ticker.get(ticker)
        if data is None or getattr(data, "error", None):
            continue

        close = getattr(data, "monthly_close", pd.Series(dtype=float)).dropna().astype(float)
        distributions = getattr(data, "monthly_distributions", pd.Series(dtype=float)).astype(float)
        if len(close) < 36:
            continue

        distributions = distributions.reindex(close.index, fill_value=0.0)
        aum, expense_ratio, inception = _extract_info(getattr(data, "info", {}) or {})

        for idx in range(24, len(close)):
            month = close.index[idx]
            price = float(close.iloc[idx])
            if not np.isfinite(price) or price <= 0:
                continue

            trailing_12 = distributions.iloc[idx - 11 : idx + 1]
            trailing_24 = distributions.iloc[idx - 23 : idx + 1]
            prior_12 = distributions.iloc[idx - 23 : idx - 11]
            positive_12 = trailing_12[trailing_12 > 0]

            ttm_distribution = float(trailing_12.sum())
            prior_ttm_distribution = float(prior_12.sum())
            row = {
                "ticker": ticker,
                "month": month,
                "category": category,
                "payout_trend_24m": _normalized_slope(trailing_24),
                "payout_stability": _coefficient_of_variation(positive_12),
                "ever_cut_before": _ever_cut_before(distributions.iloc[: idx + 1], threshold=cut_threshold),
                "price_ret_12m": _safe_return(close.iloc[idx], close.iloc[idx - 12]),
                "price_ret_24m": _safe_return(close.iloc[idx], close.iloc[idx - 24]),
                "dist_yield": ttm_distribution / price if price > 0 else np.nan,
                "aum": aum,
                "expense_ratio": expense_ratio,
                "age_years": _age_years(inception, month),
                "price": price,
                "label": _forward_cut_label(distributions, idx, threshold=cut_threshold),
                "baseline_decline_12m": (
                    int(ttm_distribution < 0.95 * prior_ttm_distribution)
                    if prior_ttm_distribution > 0
                    else 0
                ),
            }
            rows.append(row)

    panel = pd.DataFrame(rows)
    if not panel.empty:
        panel = panel.sort_values(["month", "ticker"]).reset_index(drop=True)
    return panel


def _forward_cut_label(distributions: pd.Series, idx: int, threshold: float) -> float:
    if idx < 11 or idx + 12 >= len(distributions):
        return np.nan

    baseline = float(distributions.iloc[idx - 11 : idx + 1].mean())
    if not np.isfinite(baseline) or baseline <= 0:
        return np.nan

    for future_idx in range(idx + 1, idx + 13):
        recent = float(distributions.iloc[future_idx - 2 : future_idx + 1].mean())
        if recent < threshold * baseline:
            return 1.0
    return 0.0


def _ever_cut_before(distributions: pd.Series, threshold: float) -> int:
    if len(distributions) < 15:
        return 0
    for idx in range(12, len(distributions)):
        baseline = float(distributions.iloc[idx - 12 : idx].mean())
        if baseline <= 0:
            continue
        recent = float(distributions.iloc[max(0, idx - 2) : idx + 1].mean())
        if recent < threshold * baseline:
            return 1
    return 0


def _normalized_slope(values: pd.Series) -> float:
    y = values.astype(float).to_numpy()
    if len(y) < 2 or not np.isfinite(y).all():
        return np.nan
    scale = float(np.mean(np.abs(y)))
    if scale <= 0:
        return 0.0
    x = np.arange(len(y), dtype=float)
    slope = np.polyfit(x, y, 1)[0]
    return float(slope / scale)


def _coefficient_of_variation(values: pd.Series) -> float:
    values = values.astype(float).dropna()
    if len(values) < 2:
        return np.nan
    mean = float(values.mean())
    if mean <= 0:
        return np.nan
    return float(values.std(ddof=0) / mean)


def _safe_return(current: float, prior: float) -> float:
    if not np.isfinite(current) or not np.isfinite(prior) or prior <= 0:
        return np.nan
    return float(current / prior - 1.0)


def _extract_info(info: dict[str, Any]) -> tuple[float, float, pd.Timestamp | None]:
    aum = _first_numeric(info, ["totalAssets", "netAssets", "totalNetAssets"])
    expense_ratio = _first_numeric(
        info,
        ["annualReportExpenseRatio", "expenseRatio", "netExpenseRatio"],
    )
    if np.isfinite(expense_ratio) and expense_ratio > 1:
        expense_ratio = expense_ratio / 100.0

    inception_raw = None
    for key in ["fundInceptionDate", "startDate", "inceptionDate"]:
        if key in info and info[key] not in (None, ""):
            inception_raw = info[key]
            break
    return aum, expense_ratio, _parse_date(inception_raw)


def _first_numeric(info: dict[str, Any], keys: list[str]) -> float:
    for key in keys:
        value = info.get(key)
        try:
            if value is not None and value != "":
                parsed = float(value)
                if math.isfinite(parsed):
                    return parsed
        except (TypeError, ValueError):
            continue
    return np.nan


def _parse_date(value: Any) -> pd.Timestamp | None:
    if value is None or value == "":
        return None
    try:
        if isinstance(value, (int, float)) and value > 10_000:
            return pd.to_datetime(value, unit="s")
        return pd.to_datetime(value)
    except Exception:
        return None


def _age_years(inception: pd.Timestamp | None, as_of: pd.Timestamp) -> float:
    if inception is None or pd.isna(inception):
        return np.nan
    return max(0.0, float((as_of - inception).days / 365.25))
