# T05: Benchmark constant-column noise

## What this experiment tests

This experiment restores the benchmark's seeded Gaussian-noise treatment for risk characteristics whose sample standard deviation rounds to zero at six decimals. It replaces those constant cross-sectional values with standard-normal draws before standardization, using seed 1 and a fixed chronological, feature and stock-ID order. Daily factors, residuals and risk estimates are rebuilt consistently. Forecast predictors and nonconstant characteristics remain original. The baseline's zero-exposure treatment remains a separate control.

## Saved results

| Portfolio | Scaling/view | Control | Sharpe | Delta Sharpe | Max drawdown |
| --- | --- | --- | ---: | ---: | ---: |
| minimum_variance_native | native net-one MVP | P01/minimum_variance | 0.9485 | +0.0000 | 25.26% |
| markowitz_xgboost_native | variant covariance10% | P01/markowitz_ml | 2.3416 | -0.0000 | 30.41% |
| markowitz_xgboost_reference | ORIGINAL covariance10% | P01/markowitz_ml | 2.3416 | -0.0000 | 30.41% |

Metrics cover January 1990 through December 2023. Returns and volatility are annualized from monthly excess returns. Drawdown uses compounded monthly excess returns. Sharpe improvements across components are not additive.

Recorded producer time: 268.702 seconds (4.5 minutes).
Recorded producer-stage elapsed time. This may reuse earlier forecasts or risk estimates and excludes upstream fits, checks and review; it is not a cold end-to-end runtime estimate.

## Code used by this experiment

These are the principal implementation files currently staged for this experiment. Shared preprocessing, benchmark risk and performance calculations also use [baseline.py](../../src/baseline.py), [comparison_models.py](../../src/comparison_models.py) and [evaluate_baseline.py](../../src/evaluate_baseline.py), as applicable. The full standalone execution package is still being prepared; remaining private-runner imports are listed in [the source index](../../src/source-index.json).

- [risk_preprocessing.py](../../src/risk_preprocessing.py): Risk-exposure normalization, missing-value filling and constant-column Gaussian treatment.

Historical model-parameter validation remains part of the scientific implementation. Separate correctness checks, numerical verifiers and operational schedulers are excluded.

## Method details

- Trigger: sample SD rounds to0 at six decimals; N(0,1) draws; seed1.
- Chronological months, canonical feature order and sorted stock IDs; rebuild daily factor/residual/F/D consistently.

## Interpretation limits

- Full 408-month backtest, with gross returns before trading costs. Historical selections use only labels complete by actual formation; test-period performance does not select parameters.
- It does not jitter every raw zero. Standardization can make artificial variation material. Executed trusted-R fixture establishes bounded draw parity, not equivalence under every possible R encounter order.

Configuration choices were informed by previously observed evaluation-period results. These results do not constitute a genuinely unseen holdout or establish future performance. Public release requires source-version reconciliation, attribution review and final packaging.
