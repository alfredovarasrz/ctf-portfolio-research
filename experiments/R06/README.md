# R06: Grouped risk-characteristic basis

## What this experiment tests

This experiment groups the characteristic-risk coordinates into 21 fixed hierarchical groups. The annual grouping uses historical daily factor coefficients after removing dependence on a dataset-derived market proxy, with Ward linkage; industry indicators remain separate. The risk model then consistently rebuilds factor regressions, residuals, covariance and specific-risk state under the grouped exposure mapping. Stock forecasts and allocation rules remain original. It tests a fixed grouping design, not an exhaustive search over the number of groups or a complete replication of a published empirical result.

## Saved results

| Portfolio | Scaling/view | Control | Sharpe | Delta Sharpe | Max drawdown |
| --- | --- | --- | ---: | ---: | ---: |
| minimum_variance_native | native net-one MVP | P01/minimum_variance | 0.6485 | -0.3000 | 33.12% |
| markowitz_xgboost_native | variant covariance10% | P01/markowitz_ml | 1.8156 | -0.5260 | 44.45% |
| markowitz_xgboost_reference | ORIGINAL covariance10% | P01/markowitz_ml | 1.9793 | -0.3623 | 17.53% |

Metrics cover January 1990 through December 2023. Returns and volatility are annualized from monthly excess returns. Drawdown uses compounded monthly excess returns. Sharpe improvements across components are not additive.

Recorded producer time: 945.642 seconds (15.8 minutes).
Recorded producer-stage elapsed time. This may reuse earlier forecasts or risk estimates and excludes upstream fits, checks and review; it is not a cold end-to-end runtime estimate.

## Code used by this experiment

These are the principal implementation files currently staged for this experiment. Shared preprocessing, benchmark risk and performance calculations also use [baseline.py](../../src/baseline.py), [comparison_models.py](../../src/comparison_models.py) and [evaluate_baseline.py](../../src/evaluate_baseline.py), as applicable. The full standalone execution package is still being prepared; remaining private-runner imports are listed in [the source index](../../src/source-index.json).

- [extended_risk_rebuilds.py](../../src/extended_risk_rebuilds.py): Historical exposure-PCA/grouping bases and consistent daily factor/residual/risk reconstruction.

Historical model-parameter validation remains part of the scientific implementation. Separate correctness checks, numerical verifiers and operational schedulers are excluded.

## Method details

- Ward linkage after removing dataset-derived market-proxy dependence; latest2520original daily factor rows.
- Fixed21groups, annual mapping; industries separate; consistent factor/residual/F/D rebuild.

## Interpretation limits

- Full 408-month backtest, with gross returns before trading costs. Historical selections use only labels complete by actual formation; test-period performance does not select parameters.
- Algorithmic analogue of the replication-crisis paper, not its published factor taxonomy. Saved grouping map/directions/risk/raw proof is not independent Ward or full daily-state regeneration.

Configuration choices were informed by previously observed evaluation-period results. These results do not constitute a genuinely unseen holdout or establish future performance. Public release requires source-version reconciliation, attribution review and final packaging.
