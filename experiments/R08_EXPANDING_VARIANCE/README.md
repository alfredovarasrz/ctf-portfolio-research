# R08_EXPANDING_VARIANCE: Expanding history with tuned variance decay

## What this experiment tests

This experiment estimates factor covariance using all available completed factor history and changes the age weighting for factor variances. Variance weights are hyperbolic, 1/(1+age/h), and h is selected annually from 42, 84 and 168 trading observations using historical covariance-entry error. Correlations retain the original exponential half-life of 504. Stock exposures, specific risk, original factor coefficients and stock forecasts remain fixed. It tests expanding history with a different variance-decay rule; the standalone result is separate from its later combination with P08 and C02.

## Saved results

| Portfolio | Scaling/view | Control | Sharpe | Delta Sharpe | Max drawdown |
| --- | --- | --- | ---: | ---: | ---: |
| minimum_variance_native | native net-one MVP | P01/minimum_variance | 1.0209 | +0.0725 | 27.82% |
| markowitz_xgboost_native | variant covariance10% | P01/markowitz_ml | 2.4275 | +0.0859 | 22.59% |
| markowitz_xgboost_reference | ORIGINAL covariance10% | P01/markowitz_ml | 2.4499 | +0.1083 | 23.51% |

Metrics cover January 1990 through December 2023. Returns and volatility are annualized from monthly excess returns. Drawdown uses compounded monthly excess returns. Sharpe improvements across components are not additive.

Recorded producer time: 68.585 seconds (1.1 minutes).
Recorded producer-stage elapsed time. This may reuse earlier forecasts or risk estimates and excludes upstream fits, checks and review; it is not a cold end-to-end runtime estimate.

## Code used by this experiment

These are the principal implementation files currently staged for this experiment. Shared preprocessing, benchmark risk and performance calculations also use [baseline.py](../../src/baseline.py), [comparison_models.py](../../src/comparison_models.py) and [evaluate_baseline.py](../../src/evaluate_baseline.py), as applicable. All shared Python dependencies are packaged in `src`; [the source index](../../src/source-index.json) records the import graph and source hashes.

- [risk_history_regimes.py](../../src/risk_history_regimes.py): Expanding covariance history, tuned variance/correlation decay and drawdown-conditioned covariance.
- [extended_risk.py](../../src/extended_risk.py): Historical covariance weighting and shared risk transformations.

Historical model-parameter validation remains part of the scientific implementation. Separate correctness checks, numerical verifiers and operational schedulers are excluded.

## Method details

- Weight1/(1+age/h), power1; h=42,84,168trading days.
- History: all completed usable observations, whole-history blocks capped96calendar months; all blocks rotate.
- Choose annual h by equal-fold covariance-entry MSE against uniform unbiased held-out covariance; preserve full ages across held-out gaps.

## Interpretation limits

- Full 408-month backtest, with gross returns before trading costs. Historical selections use only labels complete by actual formation; test-period performance does not select parameters.
- h is the age at half current observation weight for a hyperbola, not an exponential half-life. Correlation and variance cells never change both decay components together.

Configuration choices were informed by previously observed evaluation-period results. These results do not constitute a genuinely unseen holdout or establish future performance. The local entry point is packaged below. Saved full-period results remain unchanged.

## Run this experiment

The [run script](code/run.py) estimates this experiment from the three supplied
tables and then evaluates its monthly weights. [Execution instructions](code/README.md)
describe its inputs and outputs. Use a new output folder for a cold run.
The original results above are preserved; new outputs go to the requested folder.

## Complete shared dependencies

[artifact_utils.py](../../src/artifact_utils.py), [baseline.py](../../src/baseline.py), [capped_calendar_validation.py](../../src/capped_calendar_validation.py), [comparison_models.py](../../src/comparison_models.py), [evaluate_baseline.py](../../src/evaluate_baseline.py), [experiment_io.py](../../src/experiment_io.py), [extended_risk.py](../../src/extended_risk.py), [market_cap_proxy.py](../../src/market_cap_proxy.py), [market_cap_value_digest.py](../../src/market_cap_value_digest.py), [research_forecasts.py](../../src/research_forecasts.py), [research_risk.py](../../src/research_risk.py), [risk_allocations.py](../../src/risk_allocations.py), [risk_history_regimes.py](../../src/risk_history_regimes.py), [tuned_risk_decay.py](../../src/tuned_risk_decay.py).
