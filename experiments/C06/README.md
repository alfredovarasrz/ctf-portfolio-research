# C06: Markowitz industry sleeves

## What this experiment tests

This experiment combines the same industry-contained Markowitz portfolios as C04 and C05 using their estimated returns and joint covariance. It chooses nonnegative tangency coefficients, normalizes them to sum to one, and scales the resulting stock allocation to 10% estimated annual risk. This asks whether return-sensitive industry weighting helps compared with equal or minimum-variance industry weights. Original stock forecasts and risk estimates remain unchanged.

## Saved results

| Portfolio | Scaling/view | Control | Sharpe | Delta Sharpe | Max drawdown |
| --- | --- | --- | ---: | ---: | ---: |
| markowitz_sleeves | industry-contained net-zero sleeves; final ORIGINAL covariance10% | P01/markowitz_ml; C01 joint industry constraint is secondary | 2.1605 | -0.1811 | 21.13% |

Metrics cover January 1990 through December 2023. Returns and volatility are annualized from monthly excess returns. Drawdown uses compounded monthly excess returns. Sharpe improvements across components are not additive.

Recorded producer time: 9.238 seconds (0.2 minutes).
Recorded producer-stage elapsed time. This may reuse earlier forecasts or risk estimates and excludes upstream fits, checks and review; it is not a cold end-to-end runtime estimate.

## Code used by this experiment

These are the principal implementation files currently staged for this experiment. Shared preprocessing, benchmark risk and performance calculations also use [baseline.py](../../src/baseline.py), [comparison_models.py](../../src/comparison_models.py) and [evaluate_baseline.py](../../src/evaluate_baseline.py), as applicable. The full standalone execution package is still being prepared; remaining private-runner imports are listed in [the source index](../../src/source-index.json).

- [extended_allocations.py](../../src/extended_allocations.py): Industry-portfolio construction and combination, plus the fixed market-proxy timing multiplier.

Historical model-parameter validation remains part of the scientific implementation. Separate correctness checks, numerical verifiers and operational schedulers are excluded.

## Method details

- Each active sleeve starts at10% standalone original predicted annual risk; cross-sleeve covariance is retained.
- Outer coefficients are nonnegative and sum to1; combined portfolio is then scaled to10% original predicted annual risk.
- Absent/one-stock/zero-direction industries are inactive.

## Interpretation limits

- Full 408-month backtest, with gross returns before trading costs. Historical selections use only labels complete by actual formation; test-period performance does not select parameters.
- C05 is minimum risk across neutral sleeves, not native stock-net-one MVP. Industry-sleeve optimization differs from C01 joint constraints.

Configuration choices were informed by previously observed evaluation-period results. These results do not constitute a genuinely unseen holdout or establish future performance. Public release requires source-version reconciliation, attribution review and final packaging.
