# C05: Minimum-variance industry sleeves

## What this experiment tests

This experiment uses the same industry-contained, net-zero Markowitz portfolios as C04, but chooses their combination weights to minimize estimated portfolio variance. Those industry coefficients must be nonnegative and sum to one, and the final stock portfolio is rescaled to 10% estimated annual risk. The optimization retains covariance between industries. It tests the combination of industry portfolios, not the benchmark Minimum Variance strategy that invests directly in stocks with net weight one.

## Saved results

| Portfolio | Scaling/view | Control | Sharpe | Delta Sharpe | Max drawdown |
| --- | --- | --- | ---: | ---: | ---: |
| minimum_sleeves | industry-contained net-zero sleeves; final ORIGINAL covariance10% | P01/markowitz_ml; C01 joint industry constraint is secondary | 2.1304 | -0.2112 | 28.12% |

Metrics cover January 1990 through December 2023. Returns and volatility are annualized from monthly excess returns. Drawdown uses compounded monthly excess returns. Sharpe improvements across components are not additive.

Recorded producer time: 9.312 seconds (0.2 minutes).
Recorded producer-stage elapsed time. This may reuse earlier forecasts or risk estimates and excludes upstream fits, checks and review; it is not a cold end-to-end runtime estimate.

## Code used by this experiment

These are the principal implementation files currently staged for this experiment. Shared preprocessing, benchmark risk and performance calculations also use [baseline.py](../../src/baseline.py), [comparison_models.py](../../src/comparison_models.py) and [evaluate_baseline.py](../../src/evaluate_baseline.py), as applicable. All shared Python dependencies are packaged in `src`; [the source index](../../src/source-index.json) records the import graph and source hashes.

- [extended_allocations.py](../../src/extended_allocations.py): Industry-portfolio construction and combination, plus the fixed market-proxy timing multiplier.

Historical model-parameter validation remains part of the scientific implementation. Separate correctness checks, numerical verifiers and operational schedulers are excluded.

## Method details

- Each active sleeve starts at10% standalone original predicted annual risk; cross-sleeve covariance is retained.
- Outer coefficients are nonnegative and sum to1; combined portfolio is then scaled to10% original predicted annual risk.
- Absent/one-stock/zero-direction industries are inactive.

## Interpretation limits

- Full 408-month backtest, with gross returns before trading costs. Historical selections use only labels complete by actual formation; test-period performance does not select parameters.
- C05 is minimum risk across neutral sleeves, not native stock-net-one MVP. Industry-sleeve optimization differs from C01 joint constraints.

Configuration choices were informed by previously observed evaluation-period results. These results do not constitute a genuinely unseen holdout or establish future performance. The local entry point is packaged below. Saved full-period results remain unchanged.

## Run this experiment

The [run script](code/run.py) estimates this experiment from the three supplied
tables and then evaluates its monthly weights. [Execution instructions](code/README.md)
describe its inputs and outputs. Use a new output folder for a cold run.
The original results above are preserved; new outputs go to the requested folder.

## Complete shared dependencies

[artifact_utils.py](../../src/artifact_utils.py), [baseline.py](../../src/baseline.py), [batch_allocations.py](../../src/batch_allocations.py), [comparison_models.py](../../src/comparison_models.py), [evaluate_baseline.py](../../src/evaluate_baseline.py), [experiment_io.py](../../src/experiment_io.py), [extended_allocations.py](../../src/extended_allocations.py), [market_cap_proxy.py](../../src/market_cap_proxy.py), [market_cap_value_digest.py](../../src/market_cap_value_digest.py), [research_forecasts.py](../../src/research_forecasts.py), [research_risk.py](../../src/research_risk.py).
