# R03: Ledoit–Wolf factor covariance

## What this experiment tests

This experiment tests Ledoit–Wolf shrinkage of the covariance between estimated Barra factors. It starts from up to 2,520 completed daily factor vectors and shrinks their unweighted empirical covariance toward a scaled identity target, with shrinkage intensity estimated from the data. A matched unweighted empirical covariance is also evaluated, so the comparison can separate shrinkage from the change away from the benchmark's age weights. Stock exposures, specific risk and expected returns stay fixed. Native and original-risk reference scaling are reported separately; reference-scaled Minimum Variance no longer necessarily has net weight one.

## Saved results

| Portfolio | Scaling/view | Control | Sharpe | Delta Sharpe | Max drawdown |
| --- | --- | --- | ---: | ---: | ---: |
| native_minimum_variance_reference | ORIGINAL covariance10%; reference-scaled MVP is not net-one | reference-scaled control | 0.9497 | +0.0000 | 39.95% |
| native_markowitz_ridge_reference | ORIGINAL covariance10% | reference-scaled control | 2.1388 | +0.0000 | 38.38% |
| native_markowitz_xgboost_reference | ORIGINAL covariance10% | reference-scaled control | 2.3416 | +0.0000 | 30.41% |
| uniform_empirical_minimum_variance_native | native net-one MVP | P01/minimum_variance | 1.0358 | +0.0874 | 27.27% |
| uniform_empirical_minimum_variance_reference | ORIGINAL covariance10%; reference-scaled MVP is not net-one | native_minimum_variance_reference | 1.0534 | +0.1037 | 29.57% |
| uniform_empirical_markowitz_ridge_native | variant covariance10% | P01/markowitz_ridge | 2.3663 | +0.2275 | 37.76% |
| uniform_empirical_markowitz_ridge_reference | ORIGINAL covariance10% | native_markowitz_ridge_reference | 2.4519 | +0.3131 | 32.83% |
| uniform_empirical_markowitz_xgboost_native | variant covariance10% | P01/markowitz_ml | 2.3706 | +0.0290 | 21.58% |
| uniform_empirical_markowitz_xgboost_reference | ORIGINAL covariance10% | native_markowitz_xgboost_reference | 2.4085 | +0.0669 | 20.65% |
| ledoit_wolf_minimum_variance_native | native net-one MVP | uniform_empirical_minimum_variance_native | 0.8496 | -0.1862 | 30.17% |
| ledoit_wolf_minimum_variance_reference | ORIGINAL covariance10%; reference-scaled MVP is not net-one | uniform_empirical_minimum_variance_reference | 0.8401 | -0.2133 | 38.51% |
| ledoit_wolf_markowitz_ridge_native | variant covariance10% | uniform_empirical_markowitz_ridge_native | 1.9730 | -0.3933 | 33.72% |
| ledoit_wolf_markowitz_ridge_reference | ORIGINAL covariance10% | uniform_empirical_markowitz_ridge_reference | 2.1876 | -0.2643 | 32.89% |
| ledoit_wolf_markowitz_xgboost_native | variant covariance10% | uniform_empirical_markowitz_xgboost_native | 2.1425 | -0.2281 | 20.88% |
| ledoit_wolf_markowitz_xgboost_reference | ORIGINAL covariance10% | uniform_empirical_markowitz_xgboost_reference | 2.1742 | -0.2343 | 20.82% |

Metrics cover January 1990 through December 2023. Returns and volatility are annualized from monthly excess returns. Drawdown uses compounded monthly excess returns. Sharpe improvements across components are not additive.

Recorded producer time: 78.046 seconds (1.3 minutes).
Recorded producer-stage elapsed time. This may reuse earlier forecasts or risk estimates and excludes upstream fits, checks and review; it is not a cold end-to-end runtime estimate.

## Code used by this experiment

These are the principal implementation files currently staged for this experiment. Shared preprocessing, benchmark risk and performance calculations also use [baseline.py](../../src/baseline.py), [comparison_models.py](../../src/comparison_models.py) and [evaluate_baseline.py](../../src/evaluate_baseline.py), as applicable. All shared Python dependencies are packaged in `src`; [the source index](../../src/source-index.json) records the import graph and source hashes.

- [batch_risk.py](../../src/batch_risk.py): Ledoit–Wolf and fixed-90% factor-covariance PCA transformations, including native/reference allocation views.

Historical model-parameter validation remains part of the scientific implementation. Separate correctness checks, numerical verifiers and operational schedulers are excluded.

## Method details

- Latest2520daily original factor coefficients; data-estimated Ledoit-Wolf intensity.
- Matched uniform empirical F separates the new time weighting from shrinkage.

## Interpretation limits

- Full 408-month backtest, with gross returns before trading costs. Historical selections use only labels complete by actual formation; test-period performance does not select parameters.
- The shrinkage target is factor-space identity, not the complete stock covariance. Original EW and uniform empirical controls must stay distinct.

Configuration choices were informed by previously observed evaluation-period results. These results do not constitute a genuinely unseen holdout or establish future performance. The local entry point is packaged below. Saved full-period results remain unchanged.

## Run this experiment

The [run script](code/run.py) estimates this experiment from the three supplied
tables and then evaluates its monthly weights. [Execution instructions](code/README.md)
describe its inputs and outputs. Use a new output folder for a cold run.
The original results above are preserved; new outputs go to the requested folder.

## Complete shared dependencies

[artifact_utils.py](../../src/artifact_utils.py), [baseline.py](../../src/baseline.py), [batch_risk.py](../../src/batch_risk.py), [comparison_models.py](../../src/comparison_models.py), [evaluate_baseline.py](../../src/evaluate_baseline.py), [experiment_io.py](../../src/experiment_io.py), [market_cap_proxy.py](../../src/market_cap_proxy.py), [market_cap_value_digest.py](../../src/market_cap_value_digest.py), [research_forecasts.py](../../src/research_forecasts.py), [research_risk.py](../../src/research_risk.py).
