# R08_EXPANDING_CORRELATION: Expanding history with tuned correlation decay

## What this experiment tests

This version combines all available completed factor history with the hyperbolic correlation weights used in R08_CORRELATION. It selects h annually from 252, 504 and 1,008 trading observations using covariance-entry error over completed historical calendar blocks. Factor variances retain the original exponential rule. Exposures, specific risk and stock forecasts remain unchanged. Its distinction from the rolling version is expanding history, not a different set of h candidates.

## Saved results

| Portfolio | Scaling/view | Control | Sharpe | Delta Sharpe | Max drawdown |
| --- | --- | --- | ---: | ---: | ---: |
| minimum_variance_native | native net-one MVP | P01/minimum_variance | 0.7840 | -0.1644 | 29.91% |
| markowitz_xgboost_native | variant covariance10% | P01/markowitz_ml | 2.1879 | -0.1537 | 31.56% |
| markowitz_xgboost_reference | ORIGINAL covariance10% | P01/markowitz_ml | 2.1875 | -0.1541 | 32.59% |

Metrics cover January 1990 through December 2023. Returns and volatility are annualized from monthly excess returns. Drawdown uses compounded monthly excess returns. Sharpe improvements across components are not additive.

Recorded producer time: 68.740 seconds (1.1 minutes).
Recorded producer-stage elapsed time. This may reuse earlier forecasts or risk estimates and excludes upstream fits, checks and review; it is not a cold end-to-end runtime estimate.

## Code used by this experiment

These are the principal implementation files currently staged for this experiment. Shared preprocessing, benchmark risk and performance calculations also use [baseline.py](../../src/baseline.py), [comparison_models.py](../../src/comparison_models.py) and [evaluate_baseline.py](../../src/evaluate_baseline.py), as applicable. The full standalone execution package is still being prepared; remaining private-runner imports are listed in [the source index](../../src/source-index.json).

- [risk_history_regimes.py](../../src/risk_history_regimes.py): Expanding covariance history, tuned variance/correlation decay and drawdown-conditioned covariance.
- [extended_risk.py](../../src/extended_risk.py): Historical covariance weighting and shared risk transformations.

Historical model-parameter validation remains part of the scientific implementation. Separate correctness checks, numerical verifiers and operational schedulers are excluded.

## Method details

- Weight1/(1+age/h), power1; h=252,504,1008trading days.
- History: all completed usable observations, whole-history blocks capped96calendar months; all blocks rotate.
- Choose annual h by equal-fold covariance-entry MSE against uniform unbiased held-out covariance; preserve full ages across held-out gaps.

## Interpretation limits

- Full 408-month backtest, with gross returns before trading costs. Historical selections use only labels complete by actual formation; test-period performance does not select parameters.
- h is the age at half current observation weight for a hyperbola, not an exponential half-life. Correlation and variance cells never change both decay components together.

Configuration choices were informed by previously observed evaluation-period results. These results do not constitute a genuinely unseen holdout or establish future performance. Public release requires source-version reconciliation, attribution review and final packaging.
