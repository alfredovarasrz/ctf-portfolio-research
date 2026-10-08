# P06: Three-month bagged forecasts

## What this experiment tests

This experiment tests bagging, a form of averaging models fitted to resampled historical data. It creates five XGBoost training samples using circular three-month blocks, keeping each month's complete stock cross-section together, fits a model to each sample and averages their predictions. It uses seed 1 and inherits the original annual XGBoost configuration and tree count rather than repeating that search. The original rolling history, preprocessing, covariance and portfolio allocation remain fixed.

## Saved results

| Portfolio | Scaling/view | Control | Sharpe | Delta Sharpe | Max drawdown |
| --- | --- | --- | ---: | ---: | ---: |
| factor_ml | equal-weight deciles; no fixed volatility budget | P01/factor_ml | 0.7581 | +0.0541 | 54.67% |
| markowitz_ml | ORIGINAL covariance10% Markowitz | P01/markowitz_ml | 2.2823 | -0.0593 | 31.55% |

Metrics cover January 1990 through December 2023. Returns and volatility are annualized from monthly excess returns. Drawdown uses compounded monthly excess returns. Sharpe improvements across components are not additive.

Recorded producer time: 1075.542 seconds (17.9 minutes).
Recorded producer-stage elapsed time. This may reuse earlier forecasts or risk estimates and excludes upstream fits, checks and review; it is not a cold end-to-end runtime estimate.

## Code used by this experiment

These are the principal implementation files currently staged for this experiment. Shared preprocessing, benchmark risk and performance calculations also use [baseline.py](../../src/baseline.py), [comparison_models.py](../../src/comparison_models.py) and [evaluate_baseline.py](../../src/evaluate_baseline.py), as applicable. The full standalone execution package is still being prepared; remaining private-runner imports are listed in [the source index](../../src/source-index.json).

- [extended_forecasts.py](../../src/extended_forecasts.py): Predictor PCA, rolling age-weighted forecasts and three-month-block bagging.

Historical model-parameter validation remains part of the scientific implementation. Separate correctness checks, numerical verifiers and operational schedulers are excluded.

## Method details

- Five bags; circular three-month blocks; whole stock cross-sections remain together; seed1.
- Reuse original P01 selected tree configuration and tree count; no extra unchanged hyperparameter search.

## Interpretation limits

- Full 408-month backtest, with gross returns before trading costs. Historical selections use only labels complete by actual formation; test-period performance does not select parameters.
- Five fitted bags are a small ensemble, distinct from the1000-draw inference exercises.

Configuration choices were informed by previously observed evaluation-period results. These results do not constitute a genuinely unseen holdout or establish future performance. Public release requires source-version reconciliation, attribution review and final packaging.
