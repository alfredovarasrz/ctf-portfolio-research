# C02: Penalized portfolio weights

## What this experiment tests

This experiment tests whether penalizing large stock positions improves portfolio construction. It adds gamma times the identity matrix to the covariance used to choose the allocation direction, equivalent to an L2 penalty on stock weights. Gamma is q times the current original median stock variance. Once a year, the model chooses q from 16 fixed candidates using only previously completed candidate-portfolio outcomes: historical Sharpe for Markowitz and variance for Minimum Variance. The first annual period uses q=0 because its payoff bank is empty. Forecasts and original risk estimates stay fixed, and the original covariance sets the Markowitz portfolio's estimated 10% annual risk budget.

## Saved results

| Portfolio | Scaling/view | Control | Sharpe | Delta Sharpe | Max drawdown |
| --- | --- | --- | ---: | ---: | ---: |
| ridge_penalized_markowitz | ORIGINAL covariance10% Markowitz | P01/markowitz_ridge | 2.1612 | +0.0223 | 35.03% |
| penalized_minimum_variance | native net-one MVP | P01/minimum_variance | 0.9358 | -0.0126 | 23.87% |
| xgboost_penalized_markowitz | ORIGINAL covariance10% Markowitz | P01/markowitz_ml | 2.4404 | +0.0988 | 28.61% |

Metrics cover January 1990 through December 2023. Returns and volatility are annualized from monthly excess returns. Drawdown uses compounded monthly excess returns. Sharpe improvements across components are not additive.

Recorded producer time: 39.348 seconds (0.7 minutes).
Recorded producer-stage elapsed time. This may reuse earlier forecasts or risk estimates and excludes upstream fits, checks and review; it is not a cold end-to-end runtime estimate.

## Code used by this experiment

These are the principal implementation files currently staged for this experiment. Shared preprocessing, benchmark risk and performance calculations also use [baseline.py](../../src/baseline.py), [comparison_models.py](../../src/comparison_models.py) and [evaluate_baseline.py](../../src/evaluate_baseline.py), as applicable. The full standalone execution package is still being prepared; remaining private-runner imports are listed in [the source index](../../src/source-index.json).

- [batch_allocations.py](../../src/batch_allocations.py): Industry-neutral optimization and the regular C02 diagonal-penalty allocation with historical payoff-bank selection.

Historical model-parameter validation remains part of the scientific implementation. Separate correctness checks, numerical verifiers and operational schedulers are excluded.

## Method details

- Candidate q=0 plus15log-spaced values1e-5..100.
- Select annually from at most120completed as-of candidate outcomes, split into five score blocks. MML maximizes mean fold Sharpe; MVP minimizes mean fold variance.
- No usable five-block history implies q0; first annual anchor has no candidate bank, so q0 remains fixed for the initial12formations.

## Interpretation limits

- Full 408-month backtest, with gross returns before trading costs. Historical selections use only labels complete by actual formation; test-period performance does not select parameters.
- q penalizes weight size, not transaction cost or turnover. Its objective is portfolio performance/risk, not forecasting MSE.
- The completed grid is finite; outer Sharpe cannot justify model adoption.

Configuration choices were informed by previously observed evaluation-period results. These results do not constitute a genuinely unseen holdout or establish future performance. Per-experiment source-version reconciliation and execution packaging remain pending in this initial research publication.
