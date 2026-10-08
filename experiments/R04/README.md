# R04: Factor covariance PCA

## What this experiment tests

This experiment reduces the common-risk covariance to its leading principal components, retaining at least 90% of factor-covariance eigenvalue mass at each formation date. It restores each stock's omitted marginal factor variance to that stock's specific variance, preserving stock marginal variances while changing common covariance between stocks. The threshold is fixed rather than selected from outer returns. Original exposures, factor history, forecasts and allocation rules remain unchanged. Native and original-risk reference scaling have separate controls. This does not reduce the dimension of the original daily factor regression.

## Saved results

| Portfolio | Scaling/view | Control | Sharpe | Delta Sharpe | Max drawdown |
| --- | --- | --- | ---: | ---: | ---: |
| native_minimum_variance_reference | ORIGINAL covariance10%; reference-scaled MVP is not net-one | reference-scaled control | 0.9497 | +0.0000 | 39.95% |
| native_markowitz_ridge_reference | ORIGINAL covariance10% | reference-scaled control | 2.1388 | +0.0000 | 38.38% |
| native_markowitz_xgboost_reference | ORIGINAL covariance10% | reference-scaled control | 2.3416 | +0.0000 | 30.41% |
| pca90_minimum_variance_native | native net-one MVP | P01/minimum_variance | 0.8631 | -0.0854 | 27.90% |
| pca90_minimum_variance_reference | ORIGINAL covariance10%; reference-scaled MVP is not net-one | native_minimum_variance_reference | 0.9472 | -0.0026 | 28.66% |
| pca90_markowitz_ridge_native | variant covariance10% | P01/markowitz_ridge | 2.1873 | +0.0485 | 51.08% |
| pca90_markowitz_ridge_reference | ORIGINAL covariance10% | native_markowitz_ridge_reference | 2.2614 | +0.1225 | 28.96% |
| pca90_markowitz_xgboost_native | variant covariance10% | P01/markowitz_ml | 2.2949 | -0.0467 | 40.26% |
| pca90_markowitz_xgboost_reference | ORIGINAL covariance10% | native_markowitz_xgboost_reference | 2.3859 | +0.0443 | 19.89% |

Metrics cover January 1990 through December 2023. Returns and volatility are annualized from monthly excess returns. Drawdown uses compounded monthly excess returns. Sharpe improvements across components are not additive.

Recorded producer time: 47.380 seconds (0.8 minutes).
Recorded producer-stage elapsed time. This may reuse earlier forecasts or risk estimates and excludes upstream fits, checks and review; it is not a cold end-to-end runtime estimate.

## Code used by this experiment

These are the principal implementation files currently staged for this experiment. Shared preprocessing, benchmark risk and performance calculations also use [baseline.py](../../src/baseline.py), [comparison_models.py](../../src/comparison_models.py) and [evaluate_baseline.py](../../src/evaluate_baseline.py), as applicable. The full standalone execution package is still being prepared; remaining private-runner imports are listed in [the source index](../../src/source-index.json).

- [batch_risk.py](../../src/batch_risk.py): Ledoit–Wolf and fixed-90% factor-covariance PCA transformations, including native/reference allocation views.

Historical model-parameter validation remains part of the scientific implementation. Separate correctness checks, numerical verifiers and operational schedulers are excluded.

## Method details

- Keep smallest number of factor-covariance eigenmodes carrying at least90% eigenvalue mass.
- Add diagonal of B*(F-Fretained)*B' to stock-specific variance.

## Interpretation limits

- Full 408-month backtest, with gross returns before trading costs. Historical selections use only labels complete by actual formation; test-period performance does not select parameters.
- Eigenvectors mix original factors; retention is factor variance, not portfolio risk explained. Off-diagonal stock covariance changes while marginal stock variances are restored.

Configuration choices were informed by previously observed evaluation-period results. These results do not constitute a genuinely unseen holdout or establish future performance. Per-experiment source-version reconciliation and execution packaging remain pending in this initial research publication.
