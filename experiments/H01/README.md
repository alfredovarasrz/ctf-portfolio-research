# H01: Dense Ridge penalty search

## What this experiment tests

This experiment expands the penalty search for the stock-level Ridge return model. Instead of the original six penalties, it searches a broad grid containing small positive values and integers 1–10,000, refines the region around the historical validation minimum, and applies a fixed boundary-extension rule. The selected penalty minimizes stock-return prediction error in the completed historical training window. Predictor preprocessing, the 120-month rolling history, the covariance model and portfolio allocation rules remain unchanged.

## Saved results

| Portfolio | Scaling/view | Control | Sharpe | Delta Sharpe | Max drawdown |
| --- | --- | --- | ---: | ---: | ---: |
| dense_ridge_deciles | equal-weight deciles; no fixed volatility budget | P01/ridge | 0.7469 | +0.0072 | 53.35% |
| dense_ridge_markowitz | ORIGINAL covariance10% Markowitz | P01/markowitz_ridge | 2.0721 | -0.0667 | 46.14% |

Metrics cover January 1990 through December 2023. Returns and volatility are annualized from monthly excess returns. Drawdown uses compounded monthly excess returns. Sharpe improvements across components are not additive.

Recorded producer time: 30.457 seconds (0.5 minutes).
Recorded producer-stage elapsed time. This may reuse earlier forecasts or risk estimates and excludes upstream fits, checks and review; it is not a cold end-to-end runtime estimate.

## Code used by this experiment

These are the principal implementation files currently staged for this experiment. Shared preprocessing, benchmark risk and performance calculations also use [baseline.py](../../src/baseline.py), [comparison_models.py](../../src/comparison_models.py) and [evaluate_baseline.py](../../src/evaluate_baseline.py), as applicable. The full standalone execution package is still being prepared; remaining private-runner imports are listed in [the source index](../../src/source-index.json).

- [batch_forecasts.py](../../src/batch_forecasts.py): Broad historical stock Ridge penalty search.
- [ridge_path.py](../../src/ridge_path.py): Efficient evaluation of ordinary Ridge coefficient paths.

Historical model-parameter validation remains part of the scientific implementation. Separate correctness checks, numerical verifiers and operational schedulers are excluded.

## Method details

- Integers1..10000 plus1000log-spaced subunit values1e-8..1 and original six;10000-point local refinement.
- Boundary extension by two decades, at most three times, floor1e-12; historical mean fold prediction MSE chooses lambda.

## Interpretation limits

- Full 408-month backtest, with gross returns before trading costs. Historical selections use only labels complete by actual formation; test-period performance does not select parameters.
- Ordinary Ridge uses n*lambda and an unpenalized intercept. Dense numerical coverage is not statistical precision.

Configuration choices were informed by previously observed evaluation-period results. These results do not constitute a genuinely unseen holdout or establish future performance. Per-experiment source-version reconciliation and execution packaging remain pending in this initial research publication.
