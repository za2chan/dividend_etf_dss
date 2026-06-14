# US Dividend-ETF Income Planner — Prototype Build Spec (for Codex)

**Goal:** build a working end-to-end prototype. Screen real US dividend ETFs →
engineer features → predict each ETF's dividend-cut probability with a simple
model → run a mean-variance optimizer on the survivors → output 3 portfolio
plans. A runnable, roughly-working pipeline is the target; polish is not
required.

---

## 1. What the system does

A decision-support tool for beginner US investors who want stable monthly
dividend income (not growth, not return maximization). The user enters a budget;
the system screens a universe of US dividend ETFs, estimates each ETF's
probability of cutting its distribution within the next 12 months, filters out
the risky ones, and runs a mean-variance optimizer to produce **3 portfolio
plans** (Safe / Balanced / High-Income), each listing the ETFs, share counts,
and projected monthly income.

**Design principle:** the system does not forecast ETF returns. The income term
in the optimizer is the **observed current distribution yield**. The only
predicted quantity is the **probability of a dividend cut**, used as a risk
filter.

---

## 2. Scope

- Universe: US-listed dividend / income ETFs.
- Optimizer: plain mean-variance; no tax or account-placement layer.
- Prototype-grade: hardcoded candidate list, loose screening thresholds, and
  training on whatever history yfinance returns are all acceptable.

---

## 3. Pipeline architecture (build in this order)

```
[1] UNIVERSE      screen US dividend ETFs -> ~20-40 tickers + category
        |
[2] DATA FETCH    yfinance: daily prices + dividend history + basic info
        |
[3] FEATURES      per (ETF, month) -> backward-looking feature vector
        |
[4] LABELS        per (ETF, month) -> cut / no-cut (forward 12 mo)
        |
[5] CLASSIFIER    logistic regression -> P(cut) -> Safe / Watch / Risky
        |
[6] OPTIMIZER     mean-variance, 3 risk levels -> 3 portfolio plans
        |
[7] OUTPUT        given a budget -> print 3 plans (ETFs, shares, monthly income)
```

Each stage is its own module; `main.py` wires them end to end.

---

## 4. Stage specs

### [1] Universe
- Hardcoded candidate list of US dividend/income ETFs across categories.
  Seeds (drop any that fail to return data):
  - **dividend_equity:** `SCHD`, `VYM`, `DVY`, `VIG`, `SDY`, `HDV`, `DGRO`
  - **covered_call:** `JEPI`, `JEPQ`, `QYLD`, `RYLD`, `XYLD`, `DIVO`
  - **bond:** `AGG`, `BND`, `LQD`, `HYG`, `TLT`, `VCIT`
  - **reit:** `VNQ`, `SCHH`, `RWR`
  - **preferred / high-income:** `PFF`, `PGX`, `SPHD`
- Attach a `category` label to each (from the grouping above).
- Screening: keep tickers that return >= ~5 years of history and pay
  distributions. If AUM/expense `info` fields are missing, log and keep the
  ticker (do not filter on them).
- Output: `{ticker, category}`.

### [2] Data fetch (yfinance)
- `yf.Ticker(t).history(period="max", auto_adjust=False)` -> daily OHLCV.
  - Use **`Close`** (not `Adj Close`; Adj Close folds dividends back into price
    and removes the NAV-erosion signal).
  - Keep **daily** closes for the covariance matrix; also produce a
    **month-end** resample (`.resample("ME").last()`) for the classifier.
- `yf.Ticker(t).dividends` -> tz-strip the index, `.resample("ME").sum()` ->
  monthly distribution totals.
- `yf.Ticker(t).info` -> AUM / expense ratio / inception if present, else `NaN`.
- Rate limiting: yfinance returns `429 Too Many Requests` under load. Add
  `time.sleep(~1.5s)` between tickers, wrap fetches in try/except, and cache raw
  data to disk (parquet/pickle) so re-runs don't re-fetch.

### [3] Features (computed at month `t`, backward-looking only)
Per `(ticker, month t)` row:

| feature | definition |
|---|---|
| `category` | ETF category (one-hot at model time) |
| `payout_trend_24m` | normalized slope of monthly distribution over past 24 mo |
| `payout_stability` | coefficient of variation of distribution over past 12 mo (ignore zero-months) |
| `ever_cut_before` | 1 if a >20% sustained payout drop occurred any time up to `t` |
| `price_ret_12m`, `price_ret_24m` | `Close` return over past 12 / 24 mo |
| `dist_yield` | trailing-12-mo distribution / current price (reused by optimizer) |
| `aum`, `expense_ratio`, `age_years` | from `info` if available, else `NaN` |

Median-impute (or drop) NaN features before training.

### [4] Labels (computed at month `t`, forward-looking 12 mo)
- `baseline = mean(monthly distribution over [t-11, t])`.
- For each future month `tau` in `(t, t+12]`:
  `recent = mean(distribution over [tau-2, tau])` (3-mo trailing).
  If `recent < 0.80 * baseline` -> **label = 1 (cut)**, break.
- Else label = 0. Label = `NaN` if the forward 12-mo window is not fully
  observed (drop those rows).
- 3-month trailing window and 0.80 threshold (= 20% drop) are the defaults;
  expose the threshold as a parameter.

### [5] Classifier
- Panel: stack all `(ticker, month, features..., label)` rows; drop NaN labels.
- Model: **regularized logistic regression** (`sklearn`,
  `class_weight="balanced"`, L2). One-hot `category`, standardize numerics. A
  shallow gradient-boosted model (`LightGBM`/`GradientBoosting`) may be fit as a
  comparison. Use a tabular model — the engineered features make this a tabular
  classification problem, so a sequence model is unnecessary.
- Validation: **walk-forward (rolling window)**. Train on earlier years, test on
  the next year, roll forward (train <=2020 -> test 2021; train <=2021 -> test
  2022; ...), with a small embargo so forward labels do not leak. A single
  temporal split (train < 2022, test >= 2022) is acceptable to get it running
  first.
- Metrics: precision on the Risky flag and PR-AUC (the panel is imbalanced;
  do not use accuracy). Compare against a baseline rule ("flag any ETF whose
  payout has declined for 12+ months") and a random-label sanity check.
- Output: calibrated `P(cut)` per ETF -> bucket
  `Safe (<0.2) / Watch (0.2-0.5) / Risky (>0.5)`.

### [6] Optimizer (mean-variance)
Score every universe ETF with the trained classifier on the latest month, then:
- Filter: drop `Risky` ETFs; cap aggregate weight of `Watch` ETFs at <= 30%.
- Inputs:
  - `y_i` = current distribution yield (income term, observed).
  - `Sigma` = covariance matrix from **daily `Close` returns** over the last
    ~1-2 years with **Ledoit-Wolf shrinkage** (`sklearn.covariance.LedoitWolf`).
- Three solves (`cvxpy`), common constraints `sum(w_i) = 1`,
  `0 <= w_i <= 0.35`, Risky excluded, Watch capped:
  - **Safe:** `min  w^T Sigma w`
  - **High-Income:** `max  sum(w_i y_i)  s.t.  w^T Sigma w <= sigma_high^2`
  - **Balanced:** maximize income-per-unit-risk `sum(w_i y_i) / sqrt(w^T Sigma w)`.
    Since a ratio objective is not a clean QP, sweep 3-5 risk ceilings between
    the Safe and High-Income risk levels, solve `max income s.t. risk <= sigma_k`
    for each, and pick the point with the highest income/risk ratio.
- Post-process: drop any weight < 3% and renormalize.

### [7] Output (given a budget B)
For each of the 3 plans, print:
- plan name + risk level + projected **monthly income** = `B * sum(w_i y_i) / 12`.
- per-ETF: ticker, weight, **integer share count** = `floor(B*w_i / price_i)`,
  and its safety bucket (Safe / Watch / Risky).
- a one-line note if the budget is too small for a sensible allocation.

---

## 5. File layout
```
universe.py        # [1] candidate list + screening -> {ticker, category}
data.py            # [2] yfinance fetch + caching (daily + monthly + info)
features.py        # [3]+[4] feature engineering + label construction
model.py           # [5] panel build, logistic regression, walk-forward eval
optimizer.py       # [6] filter + mean-variance (cvxpy), 3 plans
main.py            # [7] wire end-to-end; takes a budget; prints 3 plans
requirements.txt   # yfinance pandas numpy scikit-learn cvxpy pyarrow  (lightgbm optional)
```

---

## 6. Implementation notes

- yfinance is unofficial and rate-limited: sleep between calls, cache raw data,
  wrap in try/except, and expect occasional `429`s. Some `info` fields are
  missing for some tickers — tolerate `NaN`.
- Use `Close`, not `Adj Close`.
- Dividend cuts are rare -> the panel is imbalanced; use
  `class_weight="balanced"` and PR-AUC.
- Features look backward, labels look forward, validation is time-ordered — no
  leakage.
- Only the 3 plans are shown, so ~3-5 optimizer solves are enough (no full
  frontier curve).

## 7. Definition of done
1. `python main.py --budget 50000` runs end-to-end without manual steps.
2. It fetches (or loads cached) data for the screened universe.
3. It trains the classifier and prints a validation metric plus the baseline
   comparison.
4. It prints 3 portfolio plans with tickers, share counts, and monthly income.
5. Code is modular per Section 5; raw data is cached so re-runs are fast.
