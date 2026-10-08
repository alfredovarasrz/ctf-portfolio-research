# P03: Predictor PCA

## What this experiment tests

This experiment compresses the forecast predictors with principal component analysis before fitting XGBoost. It learns PCA from each historical training subset, then chooses how much predictor variance to retain, jointly with the original XGBoost configuration, using historical stock-return prediction error. The candidate retained fractions are 50%, 70%, 90% and 95%. After selection, it refits PCA and XGBoost on the complete eligible historical window. The original Barra risk model and portfolio rules remain fixed; this is predictor PCA, not PCA of the risk covariance.

## Saved results

| Portfolio | Scaling/view | Control | Sharpe | Delta Sharpe | Max drawdown |
| --- | --- | --- | ---: | ---: | ---: |
| factor_ml | equal-weight deciles; no fixed volatility budget | P01/factor_ml | 0.6744 | -0.0295 | 57.91% |
| markowitz_ml | ORIGINAL covariance10% Markowitz | P01/markowitz_ml | 1.8900 | -0.4516 | 34.83% |

Metrics cover January 1990 through December 2023. Returns and volatility are annualized from monthly excess returns. Drawdown uses compounded monthly excess returns. Sharpe improvements across components are not additive.

Recorded producer time: 2499.619 seconds (41.7 minutes).
Recorded producer-stage elapsed time. This may reuse earlier forecasts or risk estimates and excludes upstream fits, checks and review; it is not a cold end-to-end runtime estimate.

## Code used by this experiment

These are the principal implementation files currently staged for this experiment. Shared preprocessing, benchmark risk and performance calculations also use [baseline.py](../../src/baseline.py), [comparison_models.py](../../src/comparison_models.py) and [evaluate_baseline.py](../../src/evaluate_baseline.py), as applicable. All shared Python dependencies are packaged in `src`; [the source index](../../src/source-index.json) records the import graph and source hashes.

- [extended_forecasts.py](../../src/extended_forecasts.py): Predictor PCA, rolling age-weighted forecasts and three-month-block bagging.

Historical model-parameter validation remains part of the scientific implementation. Separate correctness checks, numerical verifiers and operational schedulers are excluded.

## Method details

- Retained predictor variance candidates50%,70%,90%,95%; jointly selected with the original20tree configurations by historical stock-return MSE.
- Final PCA refits on complete past training history; normal two-stage tree-count selection.

## Interpretation limits

- Full 408-month backtest, with gross returns before trading costs. Historical selections use only labels complete by actual formation; test-period performance does not select parameters.
- Predictor variance retained is not stock-return variance explained. The risk-factor basis is unchanged.

Configuration choices were informed by previously observed evaluation-period results. These results do not constitute a genuinely unseen holdout or establish future performance. Entry-point packaging and per-run source-version reconciliation remain pending where not identified below.
