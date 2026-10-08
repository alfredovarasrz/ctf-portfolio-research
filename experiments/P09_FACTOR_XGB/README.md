# P09_FACTOR_XGB: Factor-level XGBoost forecasts mapped to stocks

## What this experiment tests

This experiment changes P08's factor-level forecasting model from Ridge to XGBoost. It keeps the same completed monthly sums of original daily Barra coefficients, observed lag-one factor inputs, 414 coordinates and mapping through current stock exposures. A single multioutput vector-leaf XGBoost model uses the original P01 annual depth, leaf penalty and sampling settings, but selects its own shared tree count using actual mapped stock-return validation error. Learning rate is 0.01, patience is 25 rounds and the predeclared ceiling is 1,000 rounds. Only Markowitz ML is evaluated, with original risk and allocation and no C02 or R08 combination in this standalone run.

## Saved results

| Portfolio | Scaling/view | Control | Sharpe | Delta Sharpe | Max drawdown |
| --- | --- | --- | ---: | ---: | ---: |
| markowitz_ml | ORIGINAL covariance, estimated 10% annual budget | P01/markowitz_ml | 3.2899 | +0.9484 | 14.33% |

Metrics cover January 1990 through December 2023. Returns and volatility are annualized from monthly excess returns. Drawdown uses compounded monthly excess returns. Sharpe improvements across components are not additive.

Recorded producer time: 3591.866 seconds (59.9 minutes).
Recorded producer-stage elapsed time. This may reuse earlier forecasts or risk estimates and excludes upstream fits, checks and review; it is not a cold end-to-end runtime estimate.

## Code used by this experiment

These are the principal implementation files currently staged for this experiment. Shared preprocessing, benchmark risk and performance calculations also use [baseline.py](../../src/baseline.py), [comparison_models.py](../../src/comparison_models.py) and [evaluate_baseline.py](../../src/evaluate_baseline.py), as applicable. The full standalone execution package is still being prepared; remaining private-runner imports are listed in [the source index](../../src/source-index.json).

- [factor_xgb_paths.py](../../src/factor_xgb_paths.py): P09 multioutput factor XGBoost fitting and tree-count selection by mapped stock-return error.
- [factor_forecast_paths.py](../../src/factor_forecast_paths.py): P07 factor-mean forecasts and P08/P08_SIX lagged-factor Ridge forecasts with mapped stock-error selection.

Historical model-parameter validation remains part of the scientific implementation. Separate correctness checks, numerical verifiers and operational schedulers are excluded.

Configuration choices were informed by previously observed evaluation-period results. These results do not constitute a genuinely unseen holdout or establish future performance. Per-experiment source-version reconciliation and execution packaging remain pending in this initial research publication.
