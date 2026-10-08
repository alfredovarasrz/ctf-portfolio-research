# T02: Risk percentile exposures

## What this experiment tests

This experiment removes the second normalization step from characteristic-risk exposures. Varying characteristics retain their country/month percentile ranks and zero/missing-value rules, but are no longer cross-sectionally centered and divided by their sample standard deviation after ranking. Constant characteristics still have zero exposure, and industry indicators are unchanged. The daily regressions, residuals and risk covariance are rebuilt consistently, while stock forecasts and downstream allocation rules remain original.

## Saved results

| Portfolio | Scaling/view | Control | Sharpe | Delta Sharpe | Max drawdown |
| --- | --- | --- | ---: | ---: | ---: |
| minimum_variance_native | native net-one MVP | P01/minimum_variance | 0.6741 | -0.2744 | 36.18% |
| markowitz_xgboost_native | variant covariance10% | P01/markowitz_ml | 2.0251 | -0.3165 | 51.70% |
| markowitz_xgboost_reference | ORIGINAL covariance10% | P01/markowitz_ml | 2.2390 | -0.1026 | 28.15% |

Metrics cover January 1990 through December 2023. Returns and volatility are annualized from monthly excess returns. Drawdown uses compounded monthly excess returns. Sharpe improvements across components are not additive.

Recorded producer time: 265.021 seconds (4.4 minutes).
Recorded producer-stage elapsed time. This may reuse earlier forecasts or risk estimates and excludes upstream fits, checks and review; it is not a cold end-to-end runtime estimate.

## Code used by this experiment

These are the principal implementation files currently staged for this experiment. Shared preprocessing, benchmark risk and performance calculations also use [baseline.py](../../src/baseline.py), [comparison_models.py](../../src/comparison_models.py) and [evaluate_baseline.py](../../src/evaluate_baseline.py), as applicable. All shared Python dependencies are packaged in `src`; [the source index](../../src/source-index.json) records the import graph and source hashes.

- [risk_preprocessing.py](../../src/risk_preprocessing.py): Risk-exposure normalization, missing-value filling and constant-column Gaussian treatment.

Historical model-parameter validation remains part of the scientific implementation. Separate correctness checks, numerical verifiers and operational schedulers are excluded.

## Method details

- Retain raw-zero/missing/minimum10-observation rules and zero exposure for rounded-constant columns.
- No subsequent monthly mean subtraction or unit sample-SD division for varying characteristics.

## Interpretation limits

- Full 408-month backtest, with gross returns before trading costs. Historical selections use only labels complete by actual formation; test-period performance does not select parameters.
- Fixed Ridge penalty means this tests scale and effective regularization together, not pure coordinate relabeling.

Configuration choices were informed by previously observed evaluation-period results. These results do not constitute a genuinely unseen holdout or establish future performance. The local entry point is packaged below. Saved full-period results remain unchanged.

## Run this experiment

The [run script](code/run.py) estimates this experiment from the three supplied
tables and then evaluates its monthly weights. [Execution instructions](code/README.md)
describe its inputs and outputs. Use a new output folder for a cold run.
The original results above are preserved; new outputs go to the requested folder.

## Complete shared dependencies

[artifact_utils.py](../../src/artifact_utils.py), [baseline.py](../../src/baseline.py), [comparison_models.py](../../src/comparison_models.py), [evaluate_baseline.py](../../src/evaluate_baseline.py), [experiment_io.py](../../src/experiment_io.py), [market_cap_proxy.py](../../src/market_cap_proxy.py), [market_cap_value_digest.py](../../src/market_cap_value_digest.py), [research_forecasts.py](../../src/research_forecasts.py), [research_risk.py](../../src/research_risk.py), [risk_allocations.py](../../src/risk_allocations.py), [risk_preprocessing.py](../../src/risk_preprocessing.py).
