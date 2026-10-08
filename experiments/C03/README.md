# C03: Market-proxy exposure timing

## What this experiment tests

This experiment changes the size of an otherwise unchanged Markowitz ML portfolio according to a market trend signal. It compounds the previous 12 consecutive completed monthly excess returns of a capitalization-weighted proxy built from the permitted stock data. A negative signal halves the original portfolio's exposure, reducing its estimated risk budget from 10% to 5%; otherwise exposure remains unchanged. Incomplete signal history also uses full exposure. It does not refit the forecast or covariance model, and the proxy is not an outside market-index input.

## Saved results

| Portfolio | Scaling/view | Control | Sharpe | Delta Sharpe | Max drawdown |
| --- | --- | --- | ---: | ---: | ---: |
| markowitz_ml_timed | ORIGINAL covariance, variable10%/5% budget | P01/markowitz_ml | 2.2654 | -0.0762 | 33.11% |

Metrics cover January 1990 through December 2023. Returns and volatility are annualized from monthly excess returns. Drawdown uses compounded monthly excess returns. Sharpe improvements across components are not additive.

Recorded producer time: 8.985 seconds (0.1 minutes).
Recorded producer-stage elapsed time. This may reuse earlier forecasts or risk estimates and excludes upstream fits, checks and review; it is not a cold end-to-end runtime estimate.

## Code used by this experiment

These are the principal implementation files currently staged for this experiment. Shared preprocessing, benchmark risk and performance calculations also use [baseline.py](../../src/baseline.py), [comparison_models.py](../../src/comparison_models.py) and [evaluate_baseline.py](../../src/evaluate_baseline.py), as applicable. The full standalone execution package is still being prepared; remaining private-runner imports are listed in [the source index](../../src/source-index.json).

- [extended_allocations.py](../../src/extended_allocations.py): Industry-portfolio construction and combination, plus the fixed market-proxy timing multiplier.

Historical model-parameter validation remains part of the scientific implementation. Separate correctness checks, numerical verifiers and operational schedulers are excluded.

## Method details

- Signal is12consecutive completed monthly compounded capitalization-weighted EXCESS-proxy returns.
- Negative signal halves exposure to5% original-model risk; otherwise10%; incomplete history uses full exposure.

## Interpretation limits

- Full 408-month backtest, with gross returns before trading costs. Historical selections use only labels complete by actual formation; test-period performance does not select parameters.
- The proxy is built from permitted stocks with lagged capitalization, not external market data or a total-return index. Fixed rule is not tuned on test outcomes.

Configuration choices were informed by previously observed evaluation-period results. These results do not constitute a genuinely unseen holdout or establish future performance. Public release requires source-version reconciliation, attribution review and final packaging.
