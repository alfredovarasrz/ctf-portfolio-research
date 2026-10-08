# T07_RISK: Industry median risk preprocessing

## What this experiment tests

This experiment uses the same country/month/industry median hierarchy as T07, but only for missing risk-characteristic exposures. It preserves original ranking, observed zeros, minimum observations and constant-column handling, then consistently rebuilds daily factors, residuals and risk covariance. Stock forecasts remain original. It separates industry-aware risk imputation from the predictor-imputation experiment T07.

## Saved results

| Portfolio | Scaling/view | Control | Sharpe | Delta Sharpe | Max drawdown |
| --- | --- | --- | ---: | ---: | ---: |
| minimum_variance_native | native net-one MVP | P01/minimum_variance | 0.9157 | -0.0328 | 28.19% |
| markowitz_xgboost_native | variant covariance10% | P01/markowitz_ml | 2.3474 | +0.0058 | 29.78% |
| markowitz_xgboost_reference | ORIGINAL covariance10% | P01/markowitz_ml | 2.3676 | +0.0260 | 26.99% |

Metrics cover January 1990 through December 2023. Returns and volatility are annualized from monthly excess returns. Drawdown uses compounded monthly excess returns. Sharpe improvements across components are not additive.

Recorded producer time: 304.147 seconds (5.1 minutes).
Recorded producer-stage elapsed time. This may reuse earlier forecasts or risk estimates and excludes upstream fits, checks and review; it is not a cold end-to-end runtime estimate.

## Code used by this experiment

These are the principal implementation files currently staged for this experiment. Shared preprocessing, benchmark risk and performance calculations also use [baseline.py](../../src/baseline.py), [comparison_models.py](../../src/comparison_models.py) and [evaluate_baseline.py](../../src/evaluate_baseline.py), as applicable. All shared Python dependencies are packaged in `src`; [the source index](../../src/source-index.json) records the import graph and source hashes.

- [risk_preprocessing.py](../../src/risk_preprocessing.py): Risk-exposure normalization, missing-value filling and constant-column Gaussian treatment.

Historical model-parameter validation remains part of the scientific implementation. Separate correctness checks, numerical verifiers and operational schedulers are excluded.

## Method details

- Country/month/FF12industry median then country/month/zero fallback.
- Preserve rawzero-0.5, country/month rank groups, minimum10observations and constant-zero policy.

## Interpretation limits

- Full 408-month backtest, with gross returns before trading costs. Historical selections use only labels complete by actual formation; test-period performance does not select parameters.
- Independent from T04/T07 predictor median cells; no accumulated imputation change.

Configuration choices were informed by previously observed evaluation-period results. These results do not constitute a genuinely unseen holdout or establish future performance. Entry-point packaging and per-run source-version reconciliation remain pending where not identified below.
