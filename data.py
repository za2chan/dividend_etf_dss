from __future__ import annotations

import hashlib
import logging
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


LOGGER = logging.getLogger(__name__)
MONTH_END = "ME"


@dataclass
class TickerData:
    ticker: str
    daily: pd.DataFrame
    monthly_close: pd.Series
    monthly_distributions: pd.Series
    info: dict[str, Any]
    error: str | None = None
    from_cache: bool = False


def fetch_universe_data(
    universe: pd.DataFrame,
    cache_dir: str | Path = "cache/raw",
    refresh: bool = False,
    sleep_seconds: float = 1.5,
) -> dict[str, TickerData]:
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    results: dict[str, TickerData] = {}
    tickers = list(universe["ticker"].astype(str))
    for idx, ticker in enumerate(tickers, start=1):
        LOGGER.info("Loading %s (%d/%d)", ticker, idx, len(tickers))
        data = fetch_ticker_data(ticker, cache_path, refresh=refresh)
        results[ticker] = data
        if not data.from_cache and idx < len(tickers):
            time.sleep(sleep_seconds)
    return results


def fetch_ticker_data(ticker: str, cache_dir: Path, refresh: bool = False) -> TickerData:
    cache_file = cache_dir / f"{ticker}.pkl"
    if cache_file.exists() and not refresh:
        with cache_file.open("rb") as fh:
            cached = pickle.load(fh)
        cached.from_cache = True
        return cached

    try:
        import yfinance as yf
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "yfinance is not installed and no cached data is available. "
            "Install requirements or run with --demo-data for an offline smoke test."
        ) from exc

    try:
        ticker_obj = yf.Ticker(ticker)
        daily = ticker_obj.history(period="max", auto_adjust=False)
        dividends = ticker_obj.dividends
        try:
            info = dict(ticker_obj.info or {})
        except Exception as exc:  # yfinance info often fails independently.
            LOGGER.warning("Info fetch failed for %s: %s", ticker, exc)
            info = {}

        normalized = normalize_raw_data(ticker, daily, dividends, info)
        with cache_file.open("wb") as fh:
            pickle.dump(normalized, fh)
        return normalized
    except Exception as exc:
        LOGGER.warning("Fetch failed for %s: %s", ticker, exc)
        return TickerData(
            ticker=ticker,
            daily=pd.DataFrame(),
            monthly_close=pd.Series(dtype=float, name=ticker),
            monthly_distributions=pd.Series(dtype=float, name=ticker),
            info={},
            error=str(exc),
            from_cache=False,
        )


def normalize_raw_data(
    ticker: str,
    daily: pd.DataFrame,
    dividends: pd.Series,
    info: dict[str, Any] | None = None,
) -> TickerData:
    daily = _strip_tz(daily.copy())
    daily = daily[~daily.index.duplicated(keep="last")].sort_index()
    if "Close" in daily:
        daily = daily.dropna(subset=["Close"])

    if daily.empty or "Close" not in daily:
        return TickerData(
            ticker=ticker,
            daily=pd.DataFrame(),
            monthly_close=pd.Series(dtype=float, name=ticker),
            monthly_distributions=pd.Series(dtype=float, name=ticker),
            info=info or {},
            error="empty or invalid daily history",
        )

    monthly_close = _resample(daily["Close"], "last").dropna()
    monthly_close.name = ticker

    if dividends is None or len(dividends) == 0:
        monthly_distributions = pd.Series(0.0, index=monthly_close.index, name=ticker)
    else:
        dividends = _strip_tz(dividends.copy()).astype(float)
        monthly_distributions = _resample(dividends, "sum").reindex(monthly_close.index, fill_value=0.0)
        monthly_distributions.name = ticker

    return TickerData(
        ticker=ticker,
        daily=daily,
        monthly_close=monthly_close.astype(float),
        monthly_distributions=monthly_distributions.astype(float),
        info=info or {},
    )


def generate_demo_data(universe: pd.DataFrame, years: int = 8, seed: int = 436) -> dict[str, TickerData]:
    rng = np.random.default_rng(seed)
    end = pd.Timestamp.today().normalize()
    daily_index = pd.bdate_range(end=end, periods=252 * years)
    monthly_index = pd.date_range(daily_index.min(), daily_index.max(), freq=MONTH_END)

    category_params = {
        "dividend_equity": (0.045, 0.16, 0.035),
        "covered_call": (0.025, 0.19, 0.090),
        "bond": (0.020, 0.07, 0.040),
        "reit": (0.035, 0.18, 0.050),
        "preferred_high_income": (0.025, 0.13, 0.060),
    }

    results: dict[str, TickerData] = {}
    for idx, row in enumerate(universe.to_dict("records")):
        ticker = str(row["ticker"])
        category = str(row["category"])
        drift, vol, annual_yield = category_params.get(category, (0.03, 0.15, 0.04))
        local_rng = np.random.default_rng(_stable_seed(ticker, seed))

        start_price = local_rng.uniform(25, 95)
        daily_rets = local_rng.normal(drift / 252, vol / np.sqrt(252), len(daily_index))
        prices = start_price * np.cumprod(1 + daily_rets)
        daily = pd.DataFrame(
            {
                "Open": prices * (1 + local_rng.normal(0, 0.002, len(prices))),
                "High": prices * (1 + local_rng.uniform(0, 0.006, len(prices))),
                "Low": prices * (1 - local_rng.uniform(0, 0.006, len(prices))),
                "Close": prices,
                "Volume": local_rng.integers(300_000, 6_000_000, len(prices)),
            },
            index=daily_index,
        )
        monthly_close = _resample(daily["Close"], "last").reindex(monthly_index).ffill()

        trend = np.linspace(0.95, 1.08 + rng.normal(0, 0.03), len(monthly_index))
        dist = monthly_close.to_numpy() * annual_yield / 12 * trend
        dist *= local_rng.normal(1.0, 0.08, len(dist)).clip(0.65, 1.35)

        if idx % 7 == 0 and len(dist) > 55:
            cut_at = local_rng.integers(len(dist) // 2, len(dist) - 18)
            dist[cut_at:] *= local_rng.uniform(0.55, 0.75)
        elif idx % 5 == 0 and len(dist) > 55:
            cut_at = local_rng.integers(len(dist) // 2, len(dist) - 18)
            dist[cut_at : cut_at + 8] *= local_rng.uniform(0.60, 0.78)

        monthly_distributions = pd.Series(dist, index=monthly_index, name=ticker).clip(lower=0)
        info = {
            "totalAssets": float(local_rng.uniform(400_000_000, 60_000_000_000)),
            "expenseRatio": float(local_rng.uniform(0.0004, 0.0085)),
            "fundInceptionDate": int((end - pd.DateOffset(years=years + local_rng.integers(4, 15))).timestamp()),
        }
        results[ticker] = TickerData(
            ticker=ticker,
            daily=daily,
            monthly_close=monthly_close.rename(ticker),
            monthly_distributions=monthly_distributions,
            info=info,
        )
    return results


def _strip_tz(obj: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    if obj is None or len(obj) == 0:
        return obj
    idx = pd.DatetimeIndex(pd.to_datetime(obj.index))
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    obj.index = idx
    return obj


def _resample(obj: pd.DataFrame | pd.Series, method: str) -> pd.DataFrame | pd.Series:
    try:
        grouped = obj.resample(MONTH_END)
    except ValueError:
        grouped = obj.resample("M")
    return getattr(grouped, method)()


def _stable_seed(text: str, base_seed: int) -> int:
    digest = hashlib.sha256(f"{base_seed}:{text}".encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % (2**32 - 1)
