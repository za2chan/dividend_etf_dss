from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf


LOGGER = logging.getLogger(__name__)


@dataclass
class Holding:
    ticker: str
    category: str
    bucket: str
    weight: float
    price: float
    shares: int
    p_cut: float
    dist_yield: float


@dataclass
class PortfolioPlan:
    name: str
    risk_level: str
    expected_yield: float
    annual_volatility: float
    projected_monthly_income: float
    holdings: list[Holding]
    notes: list[str] = field(default_factory=list)


def optimize_portfolios(
    scores: pd.DataFrame,
    data_by_ticker: dict[str, Any],
    budget: float,
    max_weight: float = 0.35,
    watch_cap: float = 0.30,
    min_post_weight: float = 0.03,
) -> list[PortfolioPlan]:
    candidates = _prepare_candidates(scores)
    notes: list[str] = []

    survivors = candidates[candidates["bucket"] != "Risky"].copy()
    if survivors.empty:
        survivors = candidates.nsmallest(min(5, len(candidates)), "p_cut").copy()
        notes.append("All ETFs were classified Risky; using the lowest P(cut) names for a fallback allocation.")

    if len(survivors) < 3:
        notes.append("Fewer than 3 ETFs survived screening; allocation constraints were relaxed.")

    tickers = list(survivors["ticker"])
    covariance = estimate_covariance(data_by_ticker, tickers)
    survivors = survivors[survivors["ticker"].isin(covariance.index)].reset_index(drop=True)
    covariance = covariance.loc[survivors["ticker"], survivors["ticker"]]

    if survivors.empty:
        raise ValueError("No candidates are available after covariance estimation.")

    y = survivors["dist_yield"].astype(float).to_numpy()
    sigma = covariance.to_numpy(dtype=float)
    n = len(survivors)
    effective_max_weight = max(max_weight, (1.0 / n) + 1e-6)
    if effective_max_weight > max_weight:
        notes.append(f"Max position weight relaxed to {effective_max_weight:.0%} because the survivor set is small.")

    watch_idx = np.flatnonzero(survivors["bucket"].to_numpy() == "Watch")
    effective_watch_cap = _effective_watch_cap(survivors, effective_max_weight, watch_cap)
    if effective_watch_cap > watch_cap + 1e-6:
        notes.append(f"Watch cap relaxed to {effective_watch_cap:.0%} to keep constraints feasible.")

    safe_weights = _solve_min_risk(y, sigma, watch_idx, effective_max_weight, effective_watch_cap)
    safe_vol = _portfolio_volatility(safe_weights, sigma)

    asset_vols = np.sqrt(np.clip(np.diag(sigma), 1e-12, None))
    high_ceiling = max(safe_vol * 2.25, float(np.nanpercentile(asset_vols, 65)))
    high_ceiling = max(high_ceiling, safe_vol * 1.05)
    high_weights = _solve_max_income(
        y,
        sigma,
        watch_idx,
        effective_max_weight,
        effective_watch_cap,
        risk_ceiling=high_ceiling,
        initial=safe_weights,
    )
    if high_weights is None:
        high_weights = _solve_max_income(
            y,
            sigma,
            watch_idx,
            effective_max_weight,
            effective_watch_cap,
            risk_ceiling=None,
            initial=safe_weights,
        )

    high_vol = _portfolio_volatility(high_weights, sigma)
    balanced_weights = _solve_balanced(
        y,
        sigma,
        watch_idx,
        effective_max_weight,
        effective_watch_cap,
        safe_vol=safe_vol,
        high_vol=max(high_vol, high_ceiling),
        initial=safe_weights,
    )

    plan_specs = [
        ("Safe", "minimum variance", safe_weights),
        ("Balanced", "best income per unit risk", balanced_weights),
        ("High-Income", "income with risk ceiling", high_weights),
    ]
    return [
        _build_plan(
            name=name,
            risk_level=risk_level,
            raw_weights=weights,
            candidates=survivors,
            covariance=covariance,
            budget=budget,
            min_post_weight=min_post_weight,
            base_notes=notes,
        )
        for name, risk_level, weights in plan_specs
    ]


def estimate_covariance(data_by_ticker: dict[str, Any], tickers: list[str], lookback_days: int = 504) -> pd.DataFrame:
    closes: dict[str, pd.Series] = {}
    for ticker in tickers:
        data = data_by_ticker.get(ticker)
        daily = getattr(data, "daily", pd.DataFrame()) if data is not None else pd.DataFrame()
        if not daily.empty and "Close" in daily:
            closes[ticker] = daily["Close"].astype(float).dropna()

    if not closes:
        return pd.DataFrame(np.eye(len(tickers)) * 0.20**2, index=tickers, columns=tickers)

    close_frame = pd.DataFrame(closes).sort_index().tail(lookback_days + 1)
    returns = close_frame.pct_change(fill_method=None).dropna(how="all")
    valid = returns.count()[returns.count() >= 60].index.tolist()
    returns = returns[valid].dropna(how="any")

    if len(valid) < 2 or len(returns) < 60:
        vols = close_frame[valid or tickers].pct_change(fill_method=None).std(skipna=True).fillna(0.012) * np.sqrt(252)
        diag = np.diag(np.clip(vols.to_numpy(dtype=float), 0.05, 0.35) ** 2)
        return pd.DataFrame(diag, index=vols.index, columns=vols.index)

    covariance = LedoitWolf().fit(returns.to_numpy(dtype=float)).covariance_ * 252
    covariance = (covariance + covariance.T) / 2
    covariance += np.eye(len(valid)) * 1e-8
    return pd.DataFrame(covariance, index=valid, columns=valid)


def _prepare_candidates(scores: pd.DataFrame) -> pd.DataFrame:
    required = ["ticker", "category", "price", "dist_yield", "p_cut", "bucket"]
    missing = [col for col in required if col not in scores]
    if missing:
        raise ValueError(f"Missing optimizer columns: {', '.join(missing)}")

    candidates = scores[required].copy()
    candidates["price"] = pd.to_numeric(candidates["price"], errors="coerce")
    candidates["dist_yield"] = pd.to_numeric(candidates["dist_yield"], errors="coerce")
    candidates["p_cut"] = pd.to_numeric(candidates["p_cut"], errors="coerce")
    candidates = candidates.replace([np.inf, -np.inf], np.nan).dropna(subset=["price", "dist_yield", "p_cut"])
    candidates = candidates[(candidates["price"] > 0) & (candidates["dist_yield"] > 0)]
    if candidates.empty:
        raise ValueError("No scored ETFs have usable price and yield inputs.")
    return candidates.sort_values(["bucket", "p_cut"]).reset_index(drop=True)


def _effective_watch_cap(candidates: pd.DataFrame, max_weight: float, watch_cap: float) -> float:
    safe_count = int((candidates["bucket"] == "Safe").sum())
    if safe_count == 0:
        return 1.0
    required_watch = max(0.0, 1.0 - safe_count * max_weight)
    return min(1.0, max(watch_cap, required_watch + 1e-6))


def _solve_min_risk(
    y: np.ndarray,
    sigma: np.ndarray,
    watch_idx: np.ndarray,
    max_weight: float,
    watch_cap: float,
) -> np.ndarray:
    cvx_solution = _solve_with_cvxpy("min_risk", y, sigma, watch_idx, max_weight, watch_cap)
    if cvx_solution is not None:
        return cvx_solution

    result = minimize(
        lambda w: float(w @ sigma @ w),
        _initial_weights(len(y)),
        method="SLSQP",
        bounds=[(0.0, max_weight)] * len(y),
        constraints=_constraints(watch_idx, watch_cap),
        options={"maxiter": 1000, "ftol": 1e-10},
    )
    return _normalize_solution(result.x if result.success else _initial_weights(len(y)), max_weight)


def _solve_max_income(
    y: np.ndarray,
    sigma: np.ndarray,
    watch_idx: np.ndarray,
    max_weight: float,
    watch_cap: float,
    risk_ceiling: float | None,
    initial: np.ndarray,
) -> np.ndarray | None:
    cvx_solution = _solve_with_cvxpy(
        "max_income",
        y,
        sigma,
        watch_idx,
        max_weight,
        watch_cap,
        risk_ceiling=risk_ceiling,
    )
    if cvx_solution is not None:
        return cvx_solution

    constraints = _constraints(watch_idx, watch_cap)
    if risk_ceiling is not None:
        constraints.append({"type": "ineq", "fun": lambda w: float(risk_ceiling**2 - w @ sigma @ w)})

    result = minimize(
        lambda w: -float(y @ w),
        initial,
        method="SLSQP",
        bounds=[(0.0, max_weight)] * len(y),
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-10},
    )
    if not result.success:
        LOGGER.info("Income optimization failed: %s", result.message)
        return None
    return _normalize_solution(result.x, max_weight)


def _solve_balanced(
    y: np.ndarray,
    sigma: np.ndarray,
    watch_idx: np.ndarray,
    max_weight: float,
    watch_cap: float,
    safe_vol: float,
    high_vol: float,
    initial: np.ndarray,
) -> np.ndarray:
    ceilings = np.linspace(max(safe_vol * 1.05, 1e-4), max(high_vol, safe_vol * 1.25), 5)
    best_weights = initial
    best_score = -np.inf
    for ceiling in ceilings:
        weights = _solve_max_income(y, sigma, watch_idx, max_weight, watch_cap, ceiling, initial)
        if weights is None:
            continue
        volatility = _portfolio_volatility(weights, sigma)
        score = float(y @ weights) / max(volatility, 1e-8)
        if score > best_score:
            best_score = score
            best_weights = weights
    return best_weights


def _solve_with_cvxpy(
    mode: str,
    y: np.ndarray,
    sigma: np.ndarray,
    watch_idx: np.ndarray,
    max_weight: float,
    watch_cap: float,
    risk_ceiling: float | None = None,
) -> np.ndarray | None:
    try:
        import cvxpy as cp
    except ModuleNotFoundError:
        return None

    try:
        w = cp.Variable(len(y))
        sigma_psd = cp.psd_wrap((sigma + sigma.T) / 2)
        constraints = [cp.sum(w) == 1, w >= 0, w <= max_weight]
        if len(watch_idx) > 0:
            constraints.append(cp.sum(w[watch_idx]) <= watch_cap)
        if risk_ceiling is not None:
            constraints.append(cp.quad_form(w, sigma_psd) <= risk_ceiling**2)

        if mode == "min_risk":
            objective = cp.Minimize(cp.quad_form(w, sigma_psd))
        else:
            objective = cp.Maximize(y @ w)

        problem = cp.Problem(objective, constraints)
        for solver in ["CLARABEL", "ECOS", "SCS", None]:
            try:
                problem.solve(solver=solver, verbose=False)
                if w.value is not None and problem.status in {"optimal", "optimal_inaccurate"}:
                    return _normalize_solution(np.asarray(w.value).ravel(), max_weight)
            except Exception:
                continue
    except Exception as exc:
        LOGGER.info("cvxpy optimization failed; falling back to scipy: %s", exc)
    return None


def _constraints(watch_idx: np.ndarray, watch_cap: float) -> list[dict[str, object]]:
    constraints: list[dict[str, object]] = [{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}]
    if len(watch_idx) > 0:
        constraints.append({"type": "ineq", "fun": lambda w: float(watch_cap - np.sum(w[watch_idx]))})
    return constraints


def _initial_weights(n: int) -> np.ndarray:
    return np.repeat(1.0 / n, n)


def _normalize_solution(weights: np.ndarray, max_weight: float) -> np.ndarray:
    weights = np.clip(np.asarray(weights, dtype=float), 0.0, max_weight)
    total = float(weights.sum())
    if total <= 0:
        return _initial_weights(len(weights))
    return weights / total


def _portfolio_volatility(weights: np.ndarray, sigma: np.ndarray) -> float:
    return float(np.sqrt(max(weights @ sigma @ weights, 0.0)))


def _post_process(weights: pd.Series, min_weight: float) -> pd.Series:
    weights = weights[weights > 1e-6].sort_values(ascending=False)
    kept = weights[weights >= min_weight]
    if kept.empty:
        kept = weights.head(1)
    return kept / kept.sum()


def _build_plan(
    name: str,
    risk_level: str,
    raw_weights: np.ndarray,
    candidates: pd.DataFrame,
    covariance: pd.DataFrame,
    budget: float,
    min_post_weight: float,
    base_notes: list[str],
) -> PortfolioPlan:
    weights = pd.Series(raw_weights, index=candidates["ticker"]).astype(float)
    weights = _post_process(weights, min_post_weight)
    selected = candidates.set_index("ticker").loc[weights.index]
    sigma = covariance.loc[weights.index, weights.index].to_numpy(dtype=float)
    weight_array = weights.to_numpy(dtype=float)
    expected_yield = float(selected["dist_yield"].to_numpy(dtype=float) @ weight_array)
    annual_volatility = _portfolio_volatility(weight_array, sigma)

    holdings: list[Holding] = []
    total_floor_invested = 0.0
    for ticker, weight in weights.items():
        row = selected.loc[ticker]
        price = float(row["price"])
        shares = int(np.floor((budget * weight) / price))
        total_floor_invested += shares * price
        holdings.append(
            Holding(
                ticker=ticker,
                category=str(row["category"]),
                bucket=str(row["bucket"]),
                weight=float(weight),
                price=price,
                shares=shares,
                p_cut=float(row["p_cut"]),
                dist_yield=float(row["dist_yield"]),
            )
        )

    notes = list(base_notes)
    if any(holding.shares == 0 for holding in holdings):
        notes.append("Budget is too small for at least one target allocation after integer share rounding.")
    if total_floor_invested < budget * 0.85:
        notes.append("Integer share rounding leaves more than 15% of the budget uninvested.")

    return PortfolioPlan(
        name=name,
        risk_level=risk_level,
        expected_yield=expected_yield,
        annual_volatility=annual_volatility,
        projected_monthly_income=budget * expected_yield / 12,
        holdings=holdings,
        notes=notes,
    )
