# T07: Industry-group median imputation

## What this experiment tests

This experiment fills missing forecast predictors using stocks in the same country, month and Fama–French 12 industry. It first uses the median of observed transformed values in that industry group, then falls back to the country/month median and finally zero. Both original Ridge and XGBoost forecasts are refitted with unchanged historical search settings. Ranking, observed-zero treatment and Barra risk remain original. P01 is the main control, while T04 shows what changes when industry information is added to the imputation groups.

## Saved results

| Portfolio | Scaling/view | Control | Sharpe | Delta Sharpe | Max drawdown |
| --- | --- | --- | ---: | ---: | ---: |
| industry_median_ridge_deciles | equal-weight deciles; no fixed volatility budget | P01/ridge | 0.7059 | -0.0338 | 57.01% |
| industry_median_ridge_markowitz | ORIGINAL covariance10% Markowitz | P01/markowitz_ridge | 2.1180 | -0.0208 | 38.04% |
| industry_median_xgboost_deciles | equal-weight deciles; no fixed volatility budget | P01/factor_ml | 0.4754 | -0.2286 | 52.63% |
| industry_median_xgboost_markowitz | ORIGINAL covariance10% Markowitz | P01/markowitz_ml | 1.6734 | -0.6682 | 33.74% |

Metrics cover January 1990 through December 2023. Returns and volatility are annualized from monthly excess returns. Drawdown uses compounded monthly excess returns. Sharpe improvements across components are not additive.

Recorded producer time: 3252.644 seconds (54.2 minutes).
Recorded producer-stage elapsed time. This may reuse earlier forecasts or risk estimates and excludes upstream fits, checks and review; it is not a cold end-to-end runtime estimate.

## Code used by this experiment

These are the principal implementation files currently staged for this experiment. Shared preprocessing, benchmark risk and performance calculations also use [baseline.py](../../src/baseline.py), [comparison_models.py](../../src/comparison_models.py) and [evaluate_baseline.py](../../src/evaluate_baseline.py), as applicable. The full standalone execution package is still being prepared; remaining private-runner imports are listed in [the source index](../../src/source-index.json).

- [median_predictors.py](../../src/median_predictors.py): Country/month and industry-aware missing-predictor median filling.
- [baseline.py](../../src/baseline.py): Monthly predictor preprocessing, historical stock Ridge fitting and decile allocation.
- [comparison_models.py](../../src/comparison_models.py): Original stock XGBoost forecasts, Barra-style risk estimation and benchmark portfolio calculations.

Historical model-parameter validation remains part of the scientific implementation. Separate correctness checks, numerical verifiers and operational schedulers are excluded.

## Method details

- Country/month/FF12industry first, with country/month then zero fallback.
- Keep country/month ranking and raw-zero override-0.5; exclude previously filled missing cells from median estimation.
- Apply independently to original Ridge and XGBoost predictors.

## Interpretation limits

- Full 408-month backtest, with gross returns before trading costs. Historical selections use only labels complete by actual formation; test-period performance does not select parameters.
- This transforms predictors only, not Barra exposures. Observed centered rank0 is not missing.

Configuration choices were informed by previously observed evaluation-period results. These results do not constitute a genuinely unseen holdout or establish future performance. Public release requires source-version reconciliation, attribution review and final packaging.
