# R08_VARIANCE: Risk variance decay

## What this experiment tests

This experiment changes the factor-variance decay rule while keeping the original 2,520-observation history window. It replaces exponential variance weights with hyperbolic weights, 1/(1+age/h), and chooses h annually from 42, 84 and 168 trading observations using historical covariance error. Correlation weighting, exposures, factor coefficients, specific risk and stock forecasts stay original. Unlike R08_EXPANDING_VARIANCE, it does not include the full expanding factor history.

## Saved results

| Portfolio | Scaling/view | Control | Sharpe | Delta Sharpe | Max drawdown |
| --- | --- | --- | ---: | ---: | ---: |
| minimum_variance_native | native net-one MVP | P01/minimum_variance | 1.0304 | +0.0819 | 26.21% |
| markowitz_xgboost_native | variant covariance10% | P01/markowitz_ml | 2.3809 | +0.0393 | 24.41% |
| markowitz_xgboost_reference | ORIGINAL covariance10% | P01/markowitz_ml | 2.4048 | +0.0632 | 24.33% |

Metrics cover January 1990 through December 2023. Returns and volatility are annualized from monthly excess returns. Drawdown uses compounded monthly excess returns. Sharpe improvements across components are not additive.

Recorded producer time: 28.318 seconds (0.5 minutes).
Recorded producer-stage elapsed time. This may reuse earlier forecasts or risk estimates and excludes upstream fits, checks and review; it is not a cold end-to-end runtime estimate.

## Code used by this experiment

These are the principal implementation files currently staged for this experiment. Shared preprocessing, benchmark risk and performance calculations also use [baseline.py](../../src/baseline.py), [comparison_models.py](../../src/comparison_models.py) and [evaluate_baseline.py](../../src/evaluate_baseline.py), as applicable. All shared Python dependencies are packaged in `src`; [the source index](../../src/source-index.json) records the import graph and source hashes.

- [tuned_risk_decay.py](../../src/tuned_risk_decay.py): Historical selection of hyperbolic correlation or variance decay within a rolling window.
- [extended_risk.py](../../src/extended_risk.py): Historical covariance weighting and shared risk transformations.

Historical model-parameter validation remains part of the scientific implementation. Separate correctness checks, numerical verifiers and operational schedulers are excluded.

## Method details

- Weight1/(1+age/h), power1; h=42,84,168trading days.
- History: original2520daily observations, five calendar blocks.
- Choose annual h by equal-fold covariance-entry MSE against uniform unbiased held-out covariance; preserve full ages across held-out gaps.

## Interpretation limits

- Full 408-month backtest, with gross returns before trading costs. Historical selections use only labels complete by actual formation; test-period performance does not select parameters.
- h is the age at half current observation weight for a hyperbola, not an exponential half-life. Correlation and variance cells never change both decay components together.

Configuration choices were informed by previously observed evaluation-period results. These results do not constitute a genuinely unseen holdout or establish future performance. Entry-point packaging and per-run source-version reconciliation remain pending where not identified below.
