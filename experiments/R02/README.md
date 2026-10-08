# R02: Specific-risk penalty

## What this experiment tests

This experiment tunes the separate Ridge regression used to predict log stock-specific volatility when the risk model requires a fallback estimate. It selects the penalty using completed historical calibration cross-sections and held-out security groups. It does not tune the daily factor regression: original exposures, daily factor coefficients, factor covariance and eligible directly observed specific volatility remain unchanged. The resulting change is confined to the fallback specific-risk prediction, with original stock forecasts and portfolio rules.

## Saved results

| Portfolio | Scaling/view | Control | Sharpe | Delta Sharpe | Max drawdown |
| --- | --- | --- | ---: | ---: | ---: |
| minimum_variance_native | native net-one MVP | P01/minimum_variance | 0.9483 | -0.0001 | 25.26% |
| markowitz_xgboost_native | variant covariance10% | P01/markowitz_ml | 2.3415 | -0.0001 | 30.40% |
| markowitz_xgboost_reference | ORIGINAL covariance10% | P01/markowitz_ml | 2.3415 | -0.0001 | 30.40% |

Metrics cover January 1990 through December 2023. Returns and volatility are annualized from monthly excess returns. Drawdown uses compounded monthly excess returns. Sharpe improvements across components are not additive.

Recorded producer time: 1089.400 seconds (18.2 minutes).
Recorded producer-stage elapsed time. This may reuse earlier forecasts or risk estimates and excludes upstream fits, checks and review; it is not a cold end-to-end runtime estimate.

## Code used by this experiment

These are the principal implementation files currently staged for this experiment. Shared preprocessing, benchmark risk and performance calculations also use [baseline.py](../../src/baseline.py), [comparison_models.py](../../src/comparison_models.py) and [evaluate_baseline.py](../../src/evaluate_baseline.py), as applicable. The full standalone execution package is still being prepared; remaining private-runner imports are listed in [the source index](../../src/source-index.json).

- [risk_penalty_paths.py](../../src/risk_penalty_paths.py): Daily factor and specific-risk Ridge penalty selection on historical cross-sections.
- [risk_penalty_replay.py](../../src/risk_penalty_replay.py): Consistent risk-state reconstruction after changing a risk-regression penalty.

Historical model-parameter validation remains part of the scientific implementation. Separate correctness checks, numerical verifiers and operational schedulers are excluded.

## Method details

- Five deterministic security folds, not five return-forecast time blocks.
- Latest120completed calibration cross-sections; R01 screens last original factor-fit day per month, R02 each monthly log-volatility cross-section.
- Integers1..10000 plus subunit values/current0.0001,10000-point refinement and predeclared boundary extensions.
- Risk glmnet-compatible no-intercept scaling n*lambda/RMS(target); constant training columns excluded.

## Interpretation limits

- Full 408-month backtest, with gross returns before trading costs. Historical selections use only labels complete by actual formation; test-period performance does not select parameters.
- Do not compare these lambda values directly with ordinary Ridge or XGBoost leaf penalties. Annual choices are causal and do not rewrite previous years.

Configuration choices were informed by previously observed evaluation-period results. These results do not constitute a genuinely unseen holdout or establish future performance. Public release requires source-version reconciliation, attribution review and final packaging.
