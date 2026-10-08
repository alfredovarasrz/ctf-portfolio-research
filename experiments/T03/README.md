# T03: Percentile-ranked forecasts

## What this experiment tests

This experiment replaces each original stock expected-return forecast with its country/month percentile rank centered at zero. The forecast models themselves are unchanged. Decile portfolios retain the same ordering and serve as an invariance comparison, while Markowitz now optimizes using rank scores rather than calibrated expected returns. Original covariance stays fixed. Prediction MSE is omitted for these scores because their units are ranks, not returns.

## Saved results

| Portfolio | Scaling/view | Control | Sharpe | Delta Sharpe | Max drawdown |
| --- | --- | --- | ---: | ---: | ---: |
| ridge_rank_deciles | equal-weight deciles; no fixed volatility budget | P01/ridge | 0.7397 | +0.0000 | 54.62% |
| ridge_rank_markowitz | ORIGINAL covariance10% Markowitz | P01/markowitz_ridge | 1.9814 | -0.1575 | 26.31% |
| xgboost_rank_deciles | equal-weight deciles; no fixed volatility budget | P01/factor_ml | 0.7039 | +0.0000 | 59.15% |
| xgboost_rank_markowitz | ORIGINAL covariance10% Markowitz | P01/markowitz_ml | 2.0621 | -0.2795 | 16.19% |

Metrics cover January 1990 through December 2023. Returns and volatility are annualized from monthly excess returns. Drawdown uses compounded monthly excess returns. Sharpe improvements across components are not additive.

Recorded producer time: 26.443 seconds (0.4 minutes).
Recorded producer-stage elapsed time. This may reuse earlier forecasts or risk estimates and excludes upstream fits, checks and review; it is not a cold end-to-end runtime estimate.

## Code used by this experiment

These are the principal implementation files currently staged for this experiment. Shared preprocessing, benchmark risk and performance calculations also use [baseline.py](../../src/baseline.py), [comparison_models.py](../../src/comparison_models.py) and [evaluate_baseline.py](../../src/evaluate_baseline.py), as applicable. All shared Python dependencies are packaged in `src`; [the source index](../../src/source-index.json) records the import graph and source hashes.

- [baseline.py](../../src/baseline.py): Monthly predictor preprocessing, historical stock Ridge fitting and decile allocation.
- [research_risk.py](../../src/research_risk.py): Risk-artifact construction and allocation using saved risk estimates.

The percentile transformation of forecasts step currently resides in the original local `run_batch_experiment.py` runner. Its simple public entry point has not yet been extracted; the files linked above supply the surrounding forecast and allocation functions.

Historical model-parameter validation remains part of the scientific implementation. Separate correctness checks, numerical verifiers and operational schedulers are excluded.

## Method details

- Maximum-tie ranks centered by0.5; separate original Ridge and XGBoost streams.

## Interpretation limits

- Full 408-month backtest, with gross returns before trading costs. Historical selections use only labels complete by actual formation; test-period performance does not select parameters.
- Monotone ranking leaves decile membership unchanged under the preserved tie policy; decile outputs are invariance controls. Scores are not returns, so return-MSE is not meaningful for them.

Configuration choices were informed by previously observed evaluation-period results. These results do not constitute a genuinely unseen holdout or establish future performance. Entry-point packaging and per-run source-version reconciliation remain pending where not identified below.

## Running this experiment

The local entry point is [code/run.py](code/run.py). From the repository root:

```sh
python experiments/T03/code/run.py --data /path/to/ctf-tables --output /path/to/new-results --threads 10
```

The data folder must contain `ctff_chars.parquet`, `ctff_features.parquet` and `ctff_daily_ret.parquet`. Install [the research requirements](../../requirements-research.txt) first. The script fits from the supplied tables, constructs monthly weights, and then evaluates them using subsequent returns. It does not require old private forecasts, caches or verification files. Use a separate output directory for each experiment. Generated outputs are not automatically published.
