from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from data import fetch_universe_data, generate_demo_data
from features import build_feature_panel
from model import fit_cut_model, predict_latest
from optimizer import PortfolioPlan, optimize_portfolios
from universe import candidate_universe, screen_universe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export latest ETF planner results for the static UI")
    parser.add_argument("--budget", type=float, default=50_000)
    parser.add_argument("--output", default="web/public/data/latest.json")
    parser.add_argument("--cache-dir", default="cache/raw")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--min-years", type=float, default=5.0)
    parser.add_argument("--cut-threshold", type=float, default=0.80)
    parser.add_argument("--demo-data", action="store_true")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s: %(message)s")

    candidates = candidate_universe()
    if args.demo_data:
        data_by_ticker = generate_demo_data(candidates)
        source_mode = "demo"
    else:
        data_by_ticker = fetch_universe_data(
            candidates,
            cache_dir=args.cache_dir,
            refresh=args.refresh,
            sleep_seconds=args.sleep,
        )
        source_mode = "yfinance"

    screen = screen_universe(candidates, data_by_ticker, min_years=args.min_years)
    panel = build_feature_panel(screen.universe, data_by_ticker, cut_threshold=args.cut_threshold)
    model = fit_cut_model(panel)
    scores = predict_latest(model, panel).reset_index(drop=True)

    scenarios = {
        "max_weight_35": optimize_portfolios(scores, data_by_ticker, budget=args.budget, max_weight=0.35),
        "no_max_weight": optimize_portfolios(scores, data_by_ticker, budget=args.budget, max_weight=1.0),
    }

    payload = {
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source_mode": source_mode,
            "source_data_as_of": _source_data_as_of(data_by_ticker),
            "budget": args.budget,
            "candidate_count": int(len(candidates)),
            "screened_count": int(len(screen.universe)),
            "dropped_count": int(len(screen.dropped)),
            "feature_rows": int(len(panel)),
            "labeled_rows": int(panel["label"].notna().sum()) if not panel.empty else 0,
            "cut_label_rate": float(panel["label"].dropna().mean()) if not panel.empty else None,
            "cut_threshold": args.cut_threshold,
            "bucket_thresholds": {"safe_lt": 0.20, "risky_gt": 0.50},
            "watch_cap": 0.30,
        },
        "screening": {
            "dropped": _records(screen.dropped),
            "category_counts": _category_counts(screen.universe),
        },
        "classifier": {
            "training_rows": int(model.training_rows),
            "positive_rate": float(model.positive_rate),
            "metrics": _clean(model.metrics),
        },
        "scores": _records(scores),
        "scenarios": {name: [_plan_to_dict(plan) for plan in plans] for name, plans in scenarios.items()},
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_clean(payload), indent=2), encoding="utf-8")
    print(f"Wrote {output}")
    return 0


def _plan_to_dict(plan: PortfolioPlan) -> dict[str, Any]:
    return {
        "name": plan.name,
        "risk_level": plan.risk_level,
        "expected_yield": plan.expected_yield,
        "annual_volatility": plan.annual_volatility,
        "projected_monthly_income": plan.projected_monthly_income,
        "notes": plan.notes,
        "holdings": [
            {
                "ticker": holding.ticker,
                "category": holding.category,
                "bucket": holding.bucket,
                "weight": holding.weight,
                "price": holding.price,
                "shares": holding.shares,
                "p_cut": holding.p_cut,
                "dist_yield": holding.dist_yield,
            }
            for holding in plan.holdings
        ],
    }


def _source_data_as_of(data_by_ticker: dict[str, Any]) -> str | None:
    dates: list[pd.Timestamp] = []
    for data in data_by_ticker.values():
        daily = getattr(data, "daily", pd.DataFrame())
        if not daily.empty:
            dates.append(pd.Timestamp(daily.index.max()))
    if not dates:
        return None
    return max(dates).date().isoformat()


def _category_counts(frame: pd.DataFrame) -> dict[str, int]:
    if frame.empty or "category" not in frame:
        return {}
    return {str(key): int(value) for key, value in frame["category"].value_counts().sort_index().items()}


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [_clean(record) for record in frame.to_dict("records")]


def _clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clean(item) for item in value]
    if isinstance(value, tuple):
        return [_clean(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        if np.isnan(value) or np.isinf(value):
            return None
        return value
    return value


if __name__ == "__main__":
    raise SystemExit(main())
