# R05: Projected risk-characteristic basis

## What this experiment tests

This experiment applies PCA to the characteristic exposures used inside the risk model, rather than to forecast predictors or to the final covariance matrix. A historical basis is learned from 120 completed monthly exposure cross-sections, with equal weight per month and a fixed 90% retained eigenvalue mass. The 12 industry indicators stay separate, and the annual mapping is frozen before subsequent monthly use. Daily factor regressions, residuals and covariance are rebuilt consistently in that basis. Forecasts and portfolio rules remain original. The saved completed result uses the corrected fixed-basis implementation; earlier failed attempts remain in the private archive.

## Saved results

| Portfolio | Scaling/view | Control | Sharpe | Delta Sharpe | Max drawdown |
| --- | --- | --- | ---: | ---: | ---: |
| minimum_variance_native | native net-one MVP | P01/minimum_variance | 0.9668 | +0.0184 | 24.74% |
| markowitz_xgboost_native | variant covariance10% | P01/markowitz_ml | 2.2521 | -0.0894 | 33.15% |
| markowitz_xgboost_reference | ORIGINAL covariance10% | P01/markowitz_ml | 2.3086 | -0.0330 | 25.71% |

Metrics cover January 1990 through December 2023. Returns and volatility are annualized from monthly excess returns. Drawdown uses compounded monthly excess returns. Sharpe improvements across components are not additive.

Recorded producer time: 940.642 seconds (15.7 minutes).
Recorded producer-stage elapsed time. This may reuse earlier forecasts or risk estimates and excludes upstream fits, checks and review; it is not a cold end-to-end runtime estimate.

## Code used by this experiment

These are the principal implementation files currently staged for this experiment. Shared preprocessing, benchmark risk and performance calculations also use [baseline.py](../../src/baseline.py), [comparison_models.py](../../src/comparison_models.py) and [evaluate_baseline.py](../../src/evaluate_baseline.py), as applicable. The full standalone execution package is still being prepared; remaining private-runner imports are listed in [the source index](../../src/source-index.json).

- [extended_risk_rebuilds.py](../../src/extended_risk_rebuilds.py): Historical exposure-PCA/grouping bases and consistent daily factor/residual/risk reconstruction.

Historical model-parameter validation remains part of the scientific implementation. Separate correctness checks, numerical verifiers and operational schedulers are excluded.

## Method details

- Past120monthly exposure cross-sections; equal-month average exposure covariance; keep90% eigenvalue mass.
- Industry12indicators stay separate; annual mapping frozen.

## Interpretation limits

- Full 408-month backtest, with gross returns before trading costs. Historical selections use only labels complete by actual formation; test-period performance does not select parameters.
- Distinct from R04. Proof checks saved basis/transform/marginal-risk/allocation/raw receipts; it does not independently regenerate every PCA basis and entire daily state.

Configuration choices were informed by previously observed evaluation-period results. These results do not constitute a genuinely unseen holdout or establish future performance. Per-experiment source-version reconciliation and execution packaging remain pending in this initial research publication.
