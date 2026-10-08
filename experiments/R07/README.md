# R07: Expanding factor-covariance history

## What this experiment tests

This experiment estimates factor covariance from all available completed factor history instead of restricting it to the most recent 2,520 observations. It retains the benchmark's exponential half-lives of 504 trading observations for correlation and 84 for variance, preserving historical ages and gaps. Stock exposures, daily factor coefficients, specific risk, forecasts and allocation rules stay original. The change is the amount of available covariance history, not a new forecast model or a new decay function.

## Saved results

| Portfolio | Scaling/view | Control | Sharpe | Delta Sharpe | Max drawdown |
| --- | --- | --- | ---: | ---: | ---: |
| minimum_variance_native | native net-one MVP | P01/minimum_variance | 0.9478 | -0.0006 | 25.09% |
| markowitz_xgboost_native | variant covariance10% | P01/markowitz_ml | 2.3450 | +0.0034 | 30.14% |
| markowitz_xgboost_reference | ORIGINAL covariance10% | P01/markowitz_ml | 2.3433 | +0.0017 | 30.52% |

Metrics cover January 1990 through December 2023. Returns and volatility are annualized from monthly excess returns. Drawdown uses compounded monthly excess returns. Sharpe improvements across components are not additive.

Recorded producer time: 24.697 seconds (0.4 minutes).
Recorded producer-stage elapsed time. This may reuse earlier forecasts or risk estimates and excludes upstream fits, checks and review; it is not a cold end-to-end runtime estimate.

## Code used by this experiment

These are the principal implementation files currently staged for this experiment. Shared preprocessing, benchmark risk and performance calculations also use [baseline.py](../../src/baseline.py), [comparison_models.py](../../src/comparison_models.py) and [evaluate_baseline.py](../../src/evaluate_baseline.py), as applicable. The full standalone execution package is still being prepared; remaining private-runner imports are listed in [the source index](../../src/source-index.json).

- [extended_risk.py](../../src/extended_risk.py): Expanding-history factor covariance with the original exponential decay rules.

Historical model-parameter validation remains part of the scientific implementation. Separate correctness checks, numerical verifiers and operational schedulers are excluded.

## Method details

- Keep original exponential correlation/variance half-lives504/84days; actual historical ages and gaps.

## Interpretation limits

- Full 408-month backtest, with gross returns before trading costs. Historical selections use only labels complete by actual formation; test-period performance does not select parameters.
- Expansion changes F history only. Exponential weighting still gives old history very small positive weights.

Configuration choices were informed by previously observed evaluation-period results. These results do not constitute a genuinely unseen holdout or establish future performance. Public release requires source-version reconciliation, attribution review and final packaging.
