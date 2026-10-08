# P08_SIX: Six-penalty factor Ridge control

## What this experiment tests

This is the smaller-search control for P08. It uses the same monthly sums of daily Barra coefficients, lag-one factor predictor, 120 completed target months, historical validation loss and mapping back to stocks. The only intended forecasting change is the Ridge penalty search, restricted here to the original six positive penalties instead of P08's wide grid and refinement. Original risk estimation and allocation remain fixed, allowing the results to show the effect of the penalty search itself.

## Saved results

| Portfolio | Scaling/view | Control | Sharpe | Delta Sharpe | Max drawdown |
| --- | --- | --- | ---: | ---: | ---: |
| factor_ml | equal-weight deciles; no fixed volatility budget | P01/factor_ml | 1.3069 | +0.6030 | 50.21% |
| markowitz_ml | ORIGINAL covariance10% Markowitz | P01/markowitz_ml | 3.4516 | +1.1100 | 9.48% |

Metrics cover January 1990 through December 2023. Returns and volatility are annualized from monthly excess returns. Drawdown uses compounded monthly excess returns. Sharpe improvements across components are not additive.

Recorded producer time: 42.345 seconds (0.7 minutes).
Recorded producer-stage elapsed time. This may reuse earlier forecasts or risk estimates and excludes upstream fits, checks and review; it is not a cold end-to-end runtime estimate.

## Code used by this experiment

These are the principal implementation files currently staged for this experiment. Shared preprocessing, benchmark risk and performance calculations also use [baseline.py](../../src/baseline.py), [comparison_models.py](../../src/comparison_models.py) and [evaluate_baseline.py](../../src/evaluate_baseline.py), as applicable. All shared Python dependencies are packaged in `src`; [the source index](../../src/source-index.json) records the import graph and source hashes.

- [factor_forecast_paths.py](../../src/factor_forecast_paths.py): P07 factor-mean forecasts and P08/P08_SIX lagged-factor Ridge forecasts with mapped stock-error selection.
- [ridge_path.py](../../src/ridge_path.py): Efficient evaluation of ordinary Ridge coefficient paths.

Historical model-parameter validation remains part of the scientific implementation. Separate correctness checks, numerical verifiers and operational schedulers are excluded.

## Method details

- Monthly targets are sums of original daily factor coefficients; previous-month factor vector predicts all next-month factor coordinates.
- Original120completed target-month eligibility and five calendar blocks; omit missing contiguous lags without extending the window.
- Penalty minimizes actual mapped stock-return MSE, not factor MSE.
- Ordinary Ridge unpenalized intercept and n*lambda; original six positive Ridge lambdas only.

## Interpretation limits

- Full 408-month backtest, with gross returns before trading costs. Historical selections use only labels complete by actual formation; test-period performance does not select parameters.
- Linear factor aggregates are not compounded tradable factor portfolios. Positive penalty is needed for the short, high-dimensional design.
- P08_SIX isolates the search-grid change from P08; no fine-grid precision claim.

Configuration choices were informed by previously observed evaluation-period results. These results do not constitute a genuinely unseen holdout or establish future performance. The local entry point is packaged below. Saved full-period results remain unchanged.

## Run this experiment

The [run script](code/run.py) estimates this experiment from the three supplied
tables and then evaluates its monthly weights. [Execution instructions](code/README.md)
describe its inputs and outputs. Use a new output folder for a cold run.
The original results above are preserved; new outputs go to the requested folder.

## Complete shared dependencies

[artifact_utils.py](../../src/artifact_utils.py), [baseline.py](../../src/baseline.py), [comparison_models.py](../../src/comparison_models.py), [evaluate_baseline.py](../../src/evaluate_baseline.py), [experiment_io.py](../../src/experiment_io.py), [extended_risk.py](../../src/extended_risk.py), [factor_forecast_paths.py](../../src/factor_forecast_paths.py), [market_cap_proxy.py](../../src/market_cap_proxy.py), [market_cap_value_digest.py](../../src/market_cap_value_digest.py), [research_forecasts.py](../../src/research_forecasts.py), [research_risk.py](../../src/research_risk.py), [ridge_path.py](../../src/ridge_path.py).
