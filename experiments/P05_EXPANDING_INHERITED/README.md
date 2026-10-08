# P05_EXPANDING_INHERITED: All-history hyperbolic fitting weights with inherited tree parameters

## What this experiment tests

This experiment uses all available completed stock-forecast history, like P04_INHERITED, but discounts older training observations with hyperbolic age weights. It inherits the original P01 depth, leaf penalty and sampling settings, then historically selects the age-weight parameter from 12, 36, 60 and 120 months and reselects the tree count on its own history. Factor ML deciles and Markowitz ML use the rebuilt forecasts with the original risk model and allocation rules. It differs from the rolling-history P05 experiment and from the unfinished original expanding full-grid search.

## Saved results

| Portfolio | Scaling/view | Control | Sharpe | Delta Sharpe | Max drawdown |
| --- | --- | --- | ---: | ---: | ---: |
| factor_ml | equal-weight deciles; no fixed risk budget | P01/factor_ml | 0.9527 | +0.2488 | 44.12% |
| markowitz_ml | ORIGINAL covariance, estimated 10% annual budget | P01/markowitz_ml | 2.7240 | +0.3824 | 34.49% |

Metrics cover January 1990 through December 2023. Returns and volatility are annualized from monthly excess returns. Drawdown uses compounded monthly excess returns. Sharpe improvements across components are not additive.

Recorded producer time: 16505.117 seconds (275.1 minutes).
Recorded producer-stage elapsed time. This may reuse earlier forecasts or risk estimates and excludes upstream fits, checks and review; it is not a cold end-to-end runtime estimate.

## Code used by this experiment

These are the principal implementation files currently staged for this experiment. Shared preprocessing, benchmark risk and performance calculations also use [baseline.py](../../src/baseline.py), [comparison_models.py](../../src/comparison_models.py) and [evaluate_baseline.py](../../src/evaluate_baseline.py), as applicable. All shared Python dependencies are packaged in `src`; [the source index](../../src/source-index.json) records the import graph and source hashes.

- [inherited_expanding_forecasts.py](../../src/inherited_expanding_forecasts.py): Expanding forecast variants that inherit original annual tree settings.
- [expanding_forecasts.py](../../src/expanding_forecasts.py): All-history forecast fitting and hyperbolic observation-weight selection.

Historical model-parameter validation remains part of the scientific implementation. Separate correctness checks, numerical verifiers and operational schedulers are excluded.

Configuration choices were informed by previously observed evaluation-period results. These results do not constitute a genuinely unseen holdout or establish future performance. Entry-point packaging and per-run source-version reconciliation remain pending where not identified below.
