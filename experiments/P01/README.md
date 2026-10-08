# P01: Restored benchmark comparison

## What this experiment tests

P01 is the restored benchmark comparison that serves as the control for most experiments. It fits stock-return forecasts from all 402 ranked characteristics, using either Ridge or XGBoost, and estimates the original Barra-style covariance from characteristic and industry exposures. The saved portfolios include Ridge deciles, XGBoost deciles (Factor ML), Minimum Variance, Ridge Markowitz and XGBoost Markowitz (Markowitz ML). Forecast models are refitted annually using 120 completed target months and the benchmark's five historical calendar folds. The original six Ridge penalties and 20 XGBoost configurations are retained.

## Saved results

| Portfolio | Scaling/view | Control | Sharpe | Delta Sharpe | Max drawdown |
| --- | --- | --- | ---: | ---: | ---: |
| ridge | equal-weight deciles; no fixed volatility budget | baseline control | 0.7397 | +0.0000 | 54.62% |
| factor_ml | equal-weight deciles; no fixed volatility budget | baseline control | 0.7039 | +0.0000 | 59.15% |
| minimum_variance | native net-one MVP | baseline control | 0.9484 | +0.0000 | 25.26% |
| markowitz_ridge | ORIGINAL covariance10% Markowitz | baseline control | 2.1388 | +0.0000 | 38.38% |
| markowitz_ml | ORIGINAL covariance10% Markowitz | baseline control | 2.3416 | +0.0000 | 30.41% |

Metrics cover January 1990 through December 2023. Returns and volatility are annualized from monthly excess returns. Drawdown uses compounded monthly excess returns. Sharpe improvements across components are not additive.

Recorded producer time: 3253.613 seconds (54.2 minutes).
Recorded producer-stage elapsed time. This may reuse earlier forecasts or risk estimates and excludes upstream fits, checks and review; it is not a cold end-to-end runtime estimate.

## Code used by this experiment

These are the principal implementation files currently staged for this experiment. Shared preprocessing, benchmark risk and performance calculations also use [baseline.py](../../src/baseline.py), [comparison_models.py](../../src/comparison_models.py) and [evaluate_baseline.py](../../src/evaluate_baseline.py), as applicable. All shared Python dependencies are packaged in `src`; [the source index](../../src/source-index.json) records the import graph and source hashes.

- [baseline.py](../../src/baseline.py): Monthly predictor preprocessing, historical stock Ridge fitting and decile allocation.
- [comparison_models.py](../../src/comparison_models.py): Original stock XGBoost forecasts, Barra-style risk estimation and benchmark portfolio calculations.

- [research_forecasts.py](../../src/research_forecasts.py): Restored benchmark forecast orchestration with historical calendar-fold tuning.

Historical model-parameter validation remains part of the scientific implementation. Separate correctness checks, numerical verifiers and operational schedulers are excluded.

## Method details

- 402 country/month centered percentile-ranked characteristics; missing midpoint0; raw zeros -0.5.
- Annual refits on the previous120completed months; five historical calendar blocks, four train and one validate.
- Ridge lambda candidates0.0001,0.001,0.01,0.1,1,10. XGBoost keeps20original configurations and two-stage tree-count selection.
- Barra risk uses industry plus characteristic exposures,2520daily observations for F, exponential correlation/variance half-lives504/84days, and separate specific-risk estimation.

## Interpretation limits

- Full 408-month backtest, with gross returns before trading costs. Historical selections use only labels complete by actual formation; test-period performance does not select parameters.
- Markowitz has10% predicted annual volatility; native minimum variance has weights summing to1. Factor ML equal-weights each selected long/short side; it does not use Barra allocation.

Configuration choices were informed by previously observed evaluation-period results. These results do not constitute a genuinely unseen holdout or establish future performance. Entry-point packaging and per-run source-version reconciliation remain pending where not identified below.

## Running this experiment

The local entry point is [code/run.py](code/run.py). From the repository root:

```sh
python experiments/P01/code/run.py --data /path/to/ctf-tables --output /path/to/new-results --threads 10
```

The data folder must contain `ctff_chars.parquet`, `ctff_features.parquet` and `ctff_daily_ret.parquet`. Install [the research requirements](../../requirements-research.txt) first. The script fits from the supplied tables, constructs monthly weights, and then evaluates them using subsequent returns. It does not require old private forecasts, caches or verification files. Use a separate output directory for each experiment. Generated outputs are not automatically published.
