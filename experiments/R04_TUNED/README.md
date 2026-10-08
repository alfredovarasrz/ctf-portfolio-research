# R04_TUNED: Historically tuned factor-covariance PCA

## What this experiment tests

This version of R04 historically selects the retained common-risk fraction rather than fixing it at 90%. Once a year it compares 50%, 70%, 90% and 95% retention using factor-covariance entry error across five completed historical calendar blocks. It applies the selected fraction to each current factor covariance and restores omitted stock marginal variance to specific risk. The selection criterion is covariance error, not portfolio Sharpe or outer-period returns. Original exposures, factor history, forecasts and age-weighting rules remain fixed.

## Saved results

| Portfolio | Scaling/view | Control | Sharpe | Delta Sharpe | Max drawdown |
| --- | --- | --- | ---: | ---: | ---: |
| minimum_variance_native | native net-one MVP | P01/minimum_variance | 0.7781 | -0.1704 | 27.35% |
| markowitz_xgboost_native | variant covariance10% | P01/markowitz_ml | 1.9047 | -0.4369 | 62.92% |
| markowitz_xgboost_reference | ORIGINAL covariance10% | P01/markowitz_ml | 2.2742 | -0.0673 | 28.87% |

Metrics cover January 1990 through December 2023. Returns and volatility are annualized from monthly excess returns. Drawdown uses compounded monthly excess returns. Sharpe improvements across components are not additive.

Recorded producer time: 351.639 seconds (5.9 minutes).
Recorded producer-stage elapsed time. This may reuse earlier forecasts or risk estimates and excludes upstream fits, checks and review; it is not a cold end-to-end runtime estimate.

## Code used by this experiment

These are the principal implementation files currently staged for this experiment. Shared preprocessing, benchmark risk and performance calculations also use [baseline.py](../../src/baseline.py), [comparison_models.py](../../src/comparison_models.py) and [evaluate_baseline.py](../../src/evaluate_baseline.py), as applicable. The full standalone execution package is still being prepared; remaining private-runner imports are listed in [the source index](../../src/source-index.json).

- [tuned_covariance_pca.py](../../src/tuned_covariance_pca.py): Historical selection of the retained factor-covariance eigenvalue mass.
- [batch_risk.py](../../src/batch_risk.py): Ledoit–Wolf and fixed-90% factor-covariance PCA transformations, including native/reference allocation views.

Historical model-parameter validation remains part of the scientific implementation. Separate correctness checks, numerical verifiers and operational schedulers are excluded.

## Method details

- Retained fractions50%,70%,90%,95%;2520completed daily rows, five completed calendar blocks.
- Four blocks estimate EW covariance, held-out block supplies uniform unbiased covariance; equal mean fold covariance-entry MSE.
- Apply selected fraction to each current original F and restore omitted stock marginal variance to D.

## Interpretation limits

- Full 408-month backtest, with gross returns before trading costs. Historical selections use only labels complete by actual formation; test-period performance does not select parameters.
- Factor-space loss does not optimize stock diagonal restoration, Sharpe or portfolio utility. Finite fraction grid only.

Configuration choices were informed by previously observed evaluation-period results. These results do not constitute a genuinely unseen holdout or establish future performance. Per-experiment source-version reconciliation and execution packaging remain pending in this initial research publication.
