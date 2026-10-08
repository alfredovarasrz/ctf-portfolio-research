# P04_INHERITED: All-history forecasts with inherited annual tree parameters

## What this experiment tests

This experiment replaces the stock forecast's rolling 120-month fitting history with all available completed history and treats observations equally. For each annual anchor it inherits the original P01 choices for XGBoost depth, leaf penalty, row sampling and column sampling, then reselects the tree count on its own expanding history. It refits the forecast model and evaluates both Factor ML deciles and Markowitz ML using unchanged original risk and allocation rules. The accepted result preserves 18 original four-thread annual fits and 16 ten-thread continuation fits. It is distinct from the unfinished original P04 search that would retune the full tree grid.

## Saved results

| Portfolio | Scaling/view | Control | Sharpe | Delta Sharpe | Max drawdown |
| --- | --- | --- | ---: | ---: | ---: |
| factor_ml | equal-weight deciles; no fixed risk budget | P01/factor_ml | 1.2281 | +0.5242 | 39.66% |
| markowitz_ml | ORIGINAL covariance, estimated 10% annual budget | P01/markowitz_ml | 2.9534 | +0.6118 | 30.33% |

Metrics cover January 1990 through December 2023. Returns and volatility are annualized from monthly excess returns. Drawdown uses compounded monthly excess returns. Sharpe improvements across components are not additive.

Recorded producer time: 9227.110 seconds (153.8 minutes).
Recorded producer-stage elapsed time. This may reuse earlier forecasts or risk estimates and excludes upstream fits, checks and review; it is not a cold end-to-end runtime estimate. P04 inherited preserves mixed original FOUR-thread and TEN-thread continuation work.

## Code used by this experiment

These are the principal implementation files currently staged for this experiment. Shared preprocessing, benchmark risk and performance calculations also use [baseline.py](../../src/baseline.py), [comparison_models.py](../../src/comparison_models.py) and [evaluate_baseline.py](../../src/evaluate_baseline.py), as applicable. All shared Python dependencies are packaged in `src`; [the source index](../../src/source-index.json) records the import graph and source hashes.

- [inherited_expanding_forecasts.py](../../src/inherited_expanding_forecasts.py): Expanding forecast variants that inherit original annual tree settings.
- [expanding_forecasts.py](../../src/expanding_forecasts.py): All-history forecast fitting and hyperbolic observation-weight selection.

Historical model-parameter validation remains part of the scientific implementation. Separate correctness checks, numerical verifiers and operational schedulers are excluded.

Configuration choices were informed by previously observed evaluation-period results. These results do not constitute a genuinely unseen holdout or establish future performance. Entry-point packaging and per-run source-version reconciliation remain pending where not identified below.
