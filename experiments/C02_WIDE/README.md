# C02_WIDE: Broad and refined allocation penalties

## What this experiment tests

This version tests whether C02's result depends on its small initial penalty grid. It keeps the same forecasts, covariance, allocation equations and historical payoff-bank selection, but expands the candidate q values to include subunit values and integers 1–10,000, followed by local refinement. Penalties are selected using completed historical candidate outcomes, never the current portfolio's future return. Original C02 is a secondary comparison. The search remains finite, so a boundary choice is not proof of a globally optimal penalty.

## Saved results

| Portfolio | Scaling/view | Control | Sharpe | Delta Sharpe | Max drawdown |
| --- | --- | --- | ---: | ---: | ---: |
| penalized_minimum_variance | native net-one MVP | P01/minimum_variance | 0.9363 | -0.0122 | 23.81% |
| ridge_penalized_markowitz | ORIGINAL covariance10% Markowitz | P01/markowitz_ridge | 2.1504 | +0.0116 | 34.84% |
| xgboost_penalized_markowitz | ORIGINAL covariance10% Markowitz | P01/markowitz_ml | 2.4113 | +0.0697 | 28.06% |

Metrics cover January 1990 through December 2023. Returns and volatility are annualized from monthly excess returns. Drawdown uses compounded monthly excess returns. Sharpe improvements across components are not additive.

Recorded producer time: 596.775 seconds (9.9 minutes).
Recorded producer-stage elapsed time. This may reuse earlier forecasts or risk estimates and excludes upstream fits, checks and review; it is not a cold end-to-end runtime estimate.

## Code used by this experiment

These are the principal implementation files currently staged for this experiment. Shared preprocessing, benchmark risk and performance calculations also use [baseline.py](../../src/baseline.py), [comparison_models.py](../../src/comparison_models.py) and [evaluate_baseline.py](../../src/evaluate_baseline.py), as applicable. The full standalone execution package is still being prepared; remaining private-runner imports are listed in [the source index](../../src/source-index.json).

- [spectral_allocation_path.py](../../src/spectral_allocation_path.py): Efficient computation of broad C02 allocation-penalty paths.
- [batch_allocations.py](../../src/batch_allocations.py): Industry-neutral optimization and the regular C02 diagonal-penalty allocation with historical payoff-bank selection.

The broad-search orchestration currently resides in the original local `wide_allocation.py` module. It has not yet been separated from private runner dependencies for publication. The linked spectral module supplies the efficient penalty-path calculations.

Historical model-parameter validation remains part of the scientific implementation. Separate correctness checks, numerical verifiers and operational schedulers are excluded.

## Method details

- Candidate q=0,original16,200subunit log points1e-8..1,integers1..10000;10000-point refinement between coarse neighbors.
- Select annually from at most120completed as-of candidate outcomes, split into five score blocks. MML maximizes mean fold Sharpe; MVP minimizes mean fold variance.
- No usable five-block history implies q0; first annual anchor has no candidate bank, so q0 remains fixed for the initial12formations.

## Interpretation limits

- Full 408-month backtest, with gross returns before trading costs. Historical selections use only labels complete by actual formation; test-period performance does not select parameters.
- q penalizes weight size, not transaction cost or turnover. Its objective is portfolio performance/risk, not forecasting MSE.
- The completed grid is finite; outer Sharpe cannot justify model adoption.

Configuration choices were informed by previously observed evaluation-period results. These results do not constitute a genuinely unseen holdout or establish future performance. Per-experiment source-version reconciliation and execution packaging remain pending in this initial research publication.
