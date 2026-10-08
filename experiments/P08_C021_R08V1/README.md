# P08_C021_R08V1: P08 only + C02 + R08

## What this experiment tests

This is the selected combined strategy. P08 supplies the expected stock-return vector by predicting the next monthly Barra factor coefficients and mapping them through current known exposures. R08 expanding variance supplies the modified allocation covariance, and C02 regular adds its historically selected diagonal stock-weight penalty. The original same-date covariance sets the penalty units and estimated 10% annual risk budget. This candidate was independently rebuilt from raw inputs and evaluated over 1990–2023. Its daily Barra penalty remains 0.0001, distinct from P08_R01.

## Saved results

| Portfolio | Scaling/view | Control | Sharpe | Delta Sharpe | Max drawdown |
| --- | --- | --- | ---: | ---: | ---: |
| markowitz_ml | original covariance risk scaling | P01/markowitz_ml | 3.5862 | +1.2447 | 20.23% |

Metrics cover January 1990 through December 2023. Returns and volatility are annualized from monthly excess returns. Drawdown uses compounded monthly excess returns. Sharpe improvements across components are not additive.

Recorded producer time: 706.112 seconds (11.8 minutes).
Independently cold producer fit at TEN threads; excludes numerical checks and review.

## Code used by this experiment

The runnable research implementation of this selected combined strategy will
be packaged in [code/](code/README.md). The exact self-contained contest version
will be in [submission/](../../submission/README.md). Both implement the same
P08-only, C02 regular and R08 expanding-variance strategy.

These are the principal implementation files currently staged for this experiment. Shared preprocessing, benchmark risk and performance calculations also use [baseline.py](../../src/baseline.py), [comparison_models.py](../../src/comparison_models.py) and [evaluate_baseline.py](../../src/evaluate_baseline.py), as applicable. All shared Python dependencies are packaged in `src`; [the source index](../../src/source-index.json) records the import graph and source hashes.

- [exploratory_combination_paths.py](../../src/exploratory_combination_paths.py): The two fixed combined forecasts and their C02/R08 Markowitz allocation.
- [factor_forecast_paths.py](../../src/factor_forecast_paths.py): P07 factor-mean forecasts and P08/P08_SIX lagged-factor Ridge forecasts with mapped stock-error selection.
- [risk_history_regimes.py](../../src/risk_history_regimes.py): Expanding covariance history, tuned variance/correlation decay and drawdown-conditioned covariance.
- [batch_allocations.py](../../src/batch_allocations.py): Industry-neutral optimization and the regular C02 diagonal-penalty allocation with historical payoff-bank selection.

Historical model-parameter validation remains part of the scientific implementation. Separate correctness checks, numerical verifiers and operational schedulers are excluded.

Configuration choices were informed by previously observed evaluation-period results. These results do not constitute a genuinely unseen holdout or establish future performance. Entry-point packaging and per-run source-version reconciliation remain pending where not identified below.

## Running this experiment

The local entry point is [code/run.py](code/run.py). From the repository root:

```sh
python experiments/P08_C021_R08V1/code/run.py --data /path/to/ctf-tables --output /path/to/new-results --threads 10
```

The data folder must contain `ctff_chars.parquet`, `ctff_features.parquet` and `ctff_daily_ret.parquet`. Install [the research requirements](../../requirements-research.txt) first. The script fits from the supplied tables, constructs monthly weights, and then evaluates them using subsequent returns. It does not require old private forecasts, caches or verification files. Use a separate output directory for each experiment. Generated outputs are not automatically published.
