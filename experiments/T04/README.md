# T04: Country/month median imputation

## What this experiment tests

This experiment changes how missing inputs are filled for stock-return forecasting. After the original country/month rank transformation, a missing cell receives the median of the observed transformed values in its country/month instead of zero; an entirely missing group falls back to zero. Observed raw zeros and tie rules stay unchanged. Both Ridge and XGBoost forecasts are refitted with their original historical parameter searches, while the Barra risk model and portfolio construction remain fixed.

## Saved results

| Portfolio | Scaling/view | Control | Sharpe | Delta Sharpe | Max drawdown |
| --- | --- | --- | ---: | ---: | ---: |
| country_median_ridge_deciles | equal-weight deciles; no fixed volatility budget | P01/ridge | 0.7420 | +0.0023 | 55.31% |
| country_median_ridge_markowitz | ORIGINAL covariance10% Markowitz | P01/markowitz_ridge | 2.1041 | -0.0347 | 38.27% |
| country_median_xgboost_deciles | equal-weight deciles; no fixed volatility budget | P01/factor_ml | 0.3906 | -0.3134 | 53.31% |
| country_median_xgboost_markowitz | ORIGINAL covariance10% Markowitz | P01/markowitz_ml | 0.7910 | -1.5506 | 35.50% |

Metrics cover January 1990 through December 2023. Returns and volatility are annualized from monthly excess returns. Drawdown uses compounded monthly excess returns. Sharpe improvements across components are not additive.

Recorded producer time: 3296.812 seconds (54.9 minutes).
Recorded producer-stage elapsed time. This may reuse earlier forecasts or risk estimates and excludes upstream fits, checks and review; it is not a cold end-to-end runtime estimate.

## Code used by this experiment

These are the principal implementation files currently staged for this experiment. Shared preprocessing, benchmark risk and performance calculations also use [baseline.py](../../src/baseline.py), [comparison_models.py](../../src/comparison_models.py) and [evaluate_baseline.py](../../src/evaluate_baseline.py), as applicable. The full standalone execution package is still being prepared; remaining private-runner imports are listed in [the source index](../../src/source-index.json).

- [median_predictors.py](../../src/median_predictors.py): Country/month and industry-aware missing-predictor median filling.
- [baseline.py](../../src/baseline.py): Monthly predictor preprocessing, historical stock Ridge fitting and decile allocation.
- [comparison_models.py](../../src/comparison_models.py): Original stock XGBoost forecasts, Barra-style risk estimation and benchmark portfolio calculations.

Historical model-parameter validation remains part of the scientific implementation. Separate correctness checks, numerical verifiers and operational schedulers are excluded.

## Method details

- Country/monthmedian, with zero fallback if none observed.
- Keep country/month ranking and raw-zero override-0.5; exclude previously filled missing cells from median estimation.
- Apply independently to original Ridge and XGBoost predictors.

## Interpretation limits

- Full 408-month backtest, with gross returns before trading costs. Historical selections use only labels complete by actual formation; test-period performance does not select parameters.
- This transforms predictors only, not Barra exposures. Observed centered rank0 is not missing.

Configuration choices were informed by previously observed evaluation-period results. These results do not constitute a genuinely unseen holdout or establish future performance. Per-experiment source-version reconciliation and execution packaging remain pending in this initial research publication.
