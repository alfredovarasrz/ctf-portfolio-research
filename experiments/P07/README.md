# P07: Historical factor-mean forecasts

## What this experiment tests

This experiment replaces stock-level machine-learning forecasts with forecasts derived from the original Barra factor coefficients. Daily cross-sectional regressions explain observed stock excess returns using the preceding month-end characteristic and industry exposures. At each annual forecast anchor, P07 takes the mean of the latest 2,520 completed daily coefficient vectors and multiplies it by 21 as an approximate monthly forecast. That factor forecast is frozen for the annual chunk; each month's known stock exposures map it into expected stock returns. Those expected returns feed the decile or Markowitz allocator. The factor coefficients are regression estimates, not automatically tradable factor-portfolio returns.

## Saved results

| Portfolio | Scaling/view | Control | Sharpe | Delta Sharpe | Max drawdown |
| --- | --- | --- | ---: | ---: | ---: |
| factor_ml | equal-weight deciles; no fixed volatility budget | P01/factor_ml | 1.1333 | +0.4293 | 54.40% |
| markowitz_ml | ORIGINAL covariance10% Markowitz | P01/markowitz_ml | 3.4330 | +1.0914 | 11.61% |

Metrics cover January 1990 through December 2023. Returns and volatility are annualized from monthly excess returns. Drawdown uses compounded monthly excess returns. Sharpe improvements across components are not additive.

Recorded producer time: 31.001 seconds (0.5 minutes).
Recorded producer-stage elapsed time. This may reuse earlier forecasts or risk estimates and excludes upstream fits, checks and review; it is not a cold end-to-end runtime estimate.

## Code used by this experiment

These are the principal implementation files currently staged for this experiment. Shared preprocessing, benchmark risk and performance calculations also use [baseline.py](../../src/baseline.py), [comparison_models.py](../../src/comparison_models.py) and [evaluate_baseline.py](../../src/evaluate_baseline.py), as applicable. All shared Python dependencies are packaged in `src`; [the source index](../../src/source-index.json) records the import graph and source hashes.

- [factor_forecast_paths.py](../../src/factor_forecast_paths.py): P07 factor-mean forecasts and P08/P08_SIX lagged-factor Ridge forecasts with mapped stock-error selection.

Historical model-parameter validation remains part of the scientific implementation. Separate correctness checks, numerical verifiers and operational schedulers are excluded.

## Method details

- Latest2520daily coefficients; daily mean multiplied by21; annual forecast-parameter freeze.

## Interpretation limits

- Full 408-month backtest, with gross returns before trading costs. Historical selections use only labels complete by actual formation; test-period performance does not select parameters.
- Mean times21 is a linear horizon approximation, not compounding tradable factor returns. No new original Barra regression is claimed.

Configuration choices were informed by previously observed evaluation-period results. These results do not constitute a genuinely unseen holdout or establish future performance. The local entry point is packaged below. Saved full-period results remain unchanged.

## Run this experiment

The [run script](code/run.py) estimates this experiment from the three supplied
tables and then evaluates its monthly weights. [Execution instructions](code/README.md)
describe its inputs and outputs. Use a new output folder for a cold run.
The original results above are preserved; new outputs go to the requested folder.

## Complete shared dependencies

[artifact_utils.py](../../src/artifact_utils.py), [baseline.py](../../src/baseline.py), [comparison_models.py](../../src/comparison_models.py), [evaluate_baseline.py](../../src/evaluate_baseline.py), [experiment_io.py](../../src/experiment_io.py), [extended_risk.py](../../src/extended_risk.py), [factor_forecast_paths.py](../../src/factor_forecast_paths.py), [market_cap_proxy.py](../../src/market_cap_proxy.py), [market_cap_value_digest.py](../../src/market_cap_value_digest.py), [research_forecasts.py](../../src/research_forecasts.py), [research_risk.py](../../src/research_risk.py), [ridge_path.py](../../src/ridge_path.py).
