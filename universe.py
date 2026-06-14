from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd


LOGGER = logging.getLogger(__name__)


CANDIDATES: dict[str, list[str]] = {
    "dividend_equity": [
        "SCHD",
        "VYM",
        "DVY",
        "VIG",
        "SDY",
        "HDV",
        "DGRO",
        "NOBL",
        "DGRW",
        "FDVV",
        "FDL",
        "PEY",
        "SPYD",
        "DHS",
        "DLN",
        "FVD",
        "RDVY",
        "DES",
        "DON",
        "DTD",
        "LVHD",
        "SPHD",
    ],
    "covered_call": [
        "JEPI",
        "JEPQ",
        "QYLD",
        "RYLD",
        "XYLD",
        "DIVO",
        "NUSI",
        "QYLG",
        "XYLG",
        "QRMI",
        "XRMI",
        "SPYI",
    ],
    "bond": [
        "AGG",
        "BND",
        "LQD",
        "HYG",
        "TLT",
        "VCIT",
        "IEF",
        "SHY",
        "IEI",
        "MUB",
        "EMB",
        "JNK",
        "BSV",
        "BIV",
        "VCSH",
        "IGSB",
        "SPAB",
        "SPSB",
        "SPIB",
        "USIG",
        "HYLB",
        "ANGL",
        "SJNK",
        "SHYG",
        "SRLN",
        "FLOT",
        "TFLO",
        "MINT",
        "NEAR",
        "JPST",
        "SGOV",
    ],
    "reit": ["VNQ", "SCHH", "RWR", "IYR", "XLRE", "REM", "REET", "USRT", "KBWY", "ICF"],
    "preferred_high_income": ["PFF", "PGX", "PFFD", "VRP", "PSK", "FPE", "PGF", "PFXF", "IPFF"],
    "multi_asset_income": ["YYY", "MDIV", "INKM", "PCEF", "CEFS"],
    "international_dividend": ["VYMI", "IDV", "DWX", "DEM", "FGD", "DGS", "PID"],
    "mlp_energy_income": ["AMLP", "MLPA", "MLPX"],
}


@dataclass(frozen=True)
class ScreenResult:
    universe: pd.DataFrame
    dropped: pd.DataFrame


def candidate_universe() -> pd.DataFrame:
    rows = [
        {"ticker": ticker, "category": category}
        for category, tickers in CANDIDATES.items()
        for ticker in tickers
    ]
    return pd.DataFrame(rows).sort_values(["category", "ticker"]).reset_index(drop=True)


def screen_universe(
    candidates: pd.DataFrame,
    data_by_ticker: dict[str, object],
    min_years: float = 5.0,
    require_distributions: bool = True,
) -> ScreenResult:
    kept: list[dict[str, object]] = []
    dropped: list[dict[str, object]] = []

    for row in candidates.to_dict("records"):
        ticker = str(row["ticker"])
        data = data_by_ticker.get(ticker)
        reason = None

        if data is None:
            reason = "no data object"
        elif getattr(data, "error", None):
            reason = getattr(data, "error")
        else:
            daily = getattr(data, "daily", pd.DataFrame())
            monthly_distributions = getattr(data, "monthly_distributions", pd.Series(dtype=float))

            if daily.empty or "Close" not in daily:
                reason = "missing daily close history"
            else:
                history_years = (daily.index.max() - daily.index.min()).days / 365.25
                if history_years < min_years:
                    reason = f"only {history_years:.1f} years of history"

            if reason is None and require_distributions:
                if monthly_distributions.empty or float(monthly_distributions.sum()) <= 0:
                    reason = "no distributions found"

        if reason:
            dropped.append({**row, "reason": reason})
            LOGGER.info("Dropping %s: %s", ticker, reason)
        else:
            kept.append(row)

    kept_df = pd.DataFrame(kept, columns=["ticker", "category"])
    dropped_df = pd.DataFrame(dropped, columns=["ticker", "category", "reason"])
    return ScreenResult(kept_df.reset_index(drop=True), dropped_df.reset_index(drop=True)
