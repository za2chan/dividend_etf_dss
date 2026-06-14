from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd

from data import fetch_universe_data, generate_demo_data
from features import build_feature_panel
from model import fit_cut_model, predict_latest
from optimizer import PortfolioPlan, optimize_portfolios
from universe import candidate_universe, screen_universe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="US Dividend-ETF Income Planner prototype")
    parser.add_argument("--budget", type=float, required=True, help="Investment budget in USD")
    parser.add_argument("--cache-dir", default="cache/raw", help="Raw yfinance cache directory")
    parser.add_argument("--refresh", action="store_true", help="Ignore cached yfinance data")
    parser.add_argument("--sleep", type=float, default=1.5, help="Seconds to sleep between live yfinance tickers")
    parser.add_argument("--min-years", type=float, default=5.0, help="Minimum price history years for screening")
    parser.add_argument("--cut-threshold", type=float, default=0.80, help="Forward payout cut threshold")
    parser.add_argument(
        "--no-max-weight",
        action="store_true",
        help="Disable the per-ETF maximum weight constraint. Watch cap and Risky filtering still apply.",
    )
    parser.add_argument(
        "--demo-data",
        action="store_true",
        help="Run end-to-end with deterministic synthetic ETF-like data instead of yfinance",
    )
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.budget <= 0:
        print("Budget must be positive.", file=sys.stderr)
        return 2

    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s: %(message)s")

    candidates = candidate_universe()
    if args.demo_data:
        print("Using deterministic demo data. Omit --demo-data for live yfinance data.\n")
        data_by_ticker = generate_demo_data(candidates)
    else:
        try:
            data_by_ticker = fetch_universe_data(
                candidates,
                cache_dir=args.cache_dir,
                refresh=args.refresh,
                sleep_seconds=args.sleep,
            )
        except RuntimeError as exc:
            print(f"Data fetch setup failed: {exc}", file=sys.stderr)
            print("Install dependencies with: python3 -m pip install -r requirements.txt", file=sys.stderr)
            return 2

    screen = screen_universe(candidates, data_by_ticker, min_years=args.min_years)
    print(f"Screened universe: {len(screen.universe)} kept, {len(screen.dropped)} dropped")
    if not screen.dropped.empty:
        print("Dropped tickers:")
        print(screen.dropped.to_string(index=False))
        print()

    if len(screen.universe) < 3:
        print("Not enough ETFs survived screening to build portfolios.", file=sys.stderr)
        return 1

    panel = build_feature_panel(screen.universe, data_by_ticker, cut_threshold=args.cut_threshold)
    labeled_rows = int(panel["label"].notna().sum()) if not panel.empty else 0
    print(f"Feature panel: {len(panel)} rows, {labeled_rows} labeled rows")
    if panel.empty or labeled_rows == 0:
        print("No usable labeled panel rows were produced.", file=sys.stderr)
        return 1

    model = fit_cut_model(panel)
    print_model_metrics(model.metrics, model.training_rows, model.positive_rate)

    scores = predict_latest(model, panel)
    print_score_summary(scores)

    plans = optimize_portfolios(
        scores,
        data_by_ticker,
        budget=args.budget,
        max_weight=1.0 if args.no_max_weight else 0.35,
    )
    for plan in plans:
        print_plan(plan)

    return 0


def print_model_metrics(metrics: dict[str, object], training_rows: int, positive_rate: float) -> None:
    print("\nClassifier")
    print(f"Training rows: {training_rows:,} | cut label rate: {positive_rate:.1%}")
    print(f"Validation status: {metrics.get('status', 'unknown')}")
    if metrics.get("folds", 0):
        print(
            "Validation: "
            f"folds={metrics.get('folds')} "
            f"test_rows={metrics.get('test_rows')} "
            f"PR-AUC={_fmt(metrics.get('pr_auc'))} "
            f"Risky precision={_fmt(metrics.get('risky_precision'))}"
        )
        print(
            "Baselines: "
            f"decline-rule PR-AUC={_fmt(metrics.get('baseline_pr_auc'))} "
            f"decline-rule precision={_fmt(metrics.get('baseline_precision'))} "
            f"random PR-AUC={_fmt(metrics.get('random_pr_auc'))}"
        )


def print_score_summary(scores: pd.DataFrame) -> None:
    print("\nLatest ETF Scores")
    summary = scores["bucket"].value_counts().reindex(["Safe", "Watch", "Risky"], fill_value=0)
    print("Buckets: " + ", ".join(f"{name}={count}" for name, count in summary.items()))
    preview = scores[["ticker", "category", "p_cut", "bucket", "dist_yield", "price"]].copy()
    preview["p_cut"] = preview["p_cut"].map(lambda value: f"{value:.1%}")
    preview["dist_yield"] = preview["dist_yield"].map(lambda value: f"{value:.2%}")
    preview["price"] = preview["price"].map(lambda value: f"${value:,.2f}")
    print(preview.to_string(index=False))


def print_plan(plan: PortfolioPlan) -> None:
    print(f"\n{plan.name} Plan")
    print(
        f"Risk: {plan.risk_level} | "
        f"expected yield: {plan.expected_yield:.2%} | "
        f"annual vol: {plan.annual_volatility:.2%} | "
        f"projected monthly income: ${plan.projected_monthly_income:,.2f}"
    )
    rows = [
        {
            "ticker": holding.ticker,
            "category": holding.category,
            "bucket": holding.bucket,
            "weight": f"{holding.weight:.1%}",
            "shares": holding.shares,
            "price": f"${holding.price:,.2f}",
            "yield": f"{holding.dist_yield:.2%}",
            "P(cut)": f"{holding.p_cut:.1%}",
        }
        for holding in plan.holdings
    ]
    print(pd.DataFrame(rows).to_string(index=False))
    for note in dict.fromkeys(plan.notes):
        print(f"Note: {note}")


def _fmt(value: object) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if pd.isna(number):
        return "n/a"
    return f"{number:.3f}"


if __name__ == "__main__":
    raise SystemExit(main())
