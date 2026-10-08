# P02: Equal forecast ensemble

## What this experiment tests

This experiment averages the original stock-level Ridge and XGBoost expected-return forecasts with a fixed 50/50 weight. It then applies either the decile allocation rule or Markowitz optimization to the averaged forecast. It averages expected returns, not the two portfolios' stock weights. No ensemble weight is learned, and the original forecast fits, covariance and allocation rules remain unchanged.

## Saved results

| Portfolio | Scaling/view | Control | Sharpe | Delta Sharpe | Max drawdown |
| --- | --- | --- | ---: | ---: | ---: |
| equal_ensemble_deciles | equal-weight deciles; no fixed volatility budget | P01/factor_ml | 0.7401 | +0.0362 | 61.02% |
| equal_ensemble_markowitz | ORIGINAL covariance10% Markowitz | P01/markowitz_ml | 2.5991 | +0.2575 | 29.70% |

Metrics cover January 1990 through December 2023. Returns and volatility are annualized from monthly excess returns. Drawdown uses compounded monthly excess returns. Sharpe improvements across components are not additive.

Recorded producer time: 12.462 seconds (0.2 minutes).
Recorded producer-stage elapsed time. This may reuse earlier forecasts or risk estimates and excludes upstream fits, checks and review; it is not a cold end-to-end runtime estimate.

## Code used by this experiment

These are the principal implementation files currently staged for this experiment. Shared preprocessing, benchmark risk and performance calculations also use [baseline.py](../../src/baseline.py), [comparison_models.py](../../src/comparison_models.py) and [evaluate_baseline.py](../../src/evaluate_baseline.py), as applicable. The full standalone execution package is still being prepared; remaining private-runner imports are listed in [the source index](../../src/source-index.json).

- [baseline.py](../../src/baseline.py): Monthly predictor preprocessing, historical stock Ridge fitting and decile allocation.
- [comparison_models.py](../../src/comparison_models.py): Original stock XGBoost forecasts, Barra-style risk estimation and benchmark portfolio calculations.
- [research_risk.py](../../src/research_risk.py): Risk-artifact construction and allocation using saved risk estimates.

The 50/50 averaging step currently resides in the original local `run_batch_experiment.py` runner. Its simple public entry point has not yet been extracted; the files linked above supply the surrounding forecast and allocation functions.

Historical model-parameter validation remains part of the scientific implementation. Separate correctness checks, numerical verifiers and operational schedulers are excluded.

## Method details

- Fixed equal forecast weights, not tuned ensemble weights.

## Interpretation limits

- Full 408-month backtest, with gross returns before trading costs. Historical selections use only labels complete by actual formation; test-period performance does not select parameters.

Configuration choices were informed by previously observed evaluation-period results. These results do not constitute a genuinely unseen holdout or establish future performance. Per-experiment source-version reconciliation and execution packaging remain pending in this initial research publication.
