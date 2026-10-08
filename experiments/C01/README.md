# C01: Industry-neutral allocations

## What this experiment tests

This experiment asks whether industry bets account for part of the Markowitz portfolio's performance. It optimizes stock weights subject to the long and short weights summing to zero within each represented Fama–French 12 industry. For example, a +5% weight in one industry's stocks must be offset by −5% in other stocks from that same industry. A separate portfolio requires only zero total net weight, so the comparison isolates the added industry constraints. Both original Ridge and XGBoost forecasts are tested, with the original covariance and estimated 10% annual risk scaling unchanged. Industry neutrality does not imply market-beta neutrality.

## Saved results

| Portfolio | Scaling/view | Control | Sharpe | Delta Sharpe | Max drawdown |
| --- | --- | --- | ---: | ---: | ---: |
| ridge_net_neutral | ORIGINAL covariance10% Markowitz | P01/markowitz_ridge | 2.1552 | +0.0164 | 29.25% |
| ridge_industry_neutral | ORIGINAL covariance10% Markowitz | ridge_net_neutral | 2.1693 | +0.0141 | 29.35% |
| xgboost_net_neutral | ORIGINAL covariance10% Markowitz | P01/markowitz_ml | 1.9963 | -0.3453 | 29.10% |
| xgboost_industry_neutral | ORIGINAL covariance10% Markowitz | xgboost_net_neutral | 2.0123 | +0.0160 | 29.65% |

Metrics cover January 1990 through December 2023. Returns and volatility are annualized from monthly excess returns. Drawdown uses compounded monthly excess returns. Sharpe improvements across components are not additive.

Recorded producer time: 22.296 seconds (0.4 minutes).
Recorded producer-stage elapsed time. This may reuse earlier forecasts or risk estimates and excludes upstream fits, checks and review; it is not a cold end-to-end runtime estimate.

## Code used by this experiment

These are the principal implementation files currently staged for this experiment. Shared preprocessing, benchmark risk and performance calculations also use [baseline.py](../../src/baseline.py), [comparison_models.py](../../src/comparison_models.py) and [evaluate_baseline.py](../../src/evaluate_baseline.py), as applicable. All shared Python dependencies are packaged in `src`; [the source index](../../src/source-index.json) records the import graph and source hashes.

- [batch_allocations.py](../../src/batch_allocations.py): Industry-neutral optimization and the regular C02 diagonal-penalty allocation with historical payoff-bank selection.

Historical model-parameter validation remains part of the scientific implementation. Separate correctness checks, numerical verifiers and operational schedulers are excluded.

## Method details

- Sum of stock weights within every industry equals0; fair comparison also imposes overall net0.
- Both original Ridge and XGBoost forecasts;10% original predicted annual risk.

## Interpretation limits

- Full 408-month backtest, with gross returns before trading costs. Historical selections use only labels complete by actual formation; test-period performance does not select parameters.
- Industry neutrality differs from beta neutrality. Native fully invested MVP is not this net-zero portfolio.

Configuration choices were informed by previously observed evaluation-period results. These results do not constitute a genuinely unseen holdout or establish future performance. The local entry point is packaged below. Saved full-period results remain unchanged.

## Run this experiment

The [run script](code/run.py) estimates this experiment from the three supplied
tables and then evaluates its monthly weights. [Execution instructions](code/README.md)
describe its inputs and outputs. Use a new output folder for a cold run.
The original results above are preserved; new outputs go to the requested folder.

## Complete shared dependencies

[artifact_utils.py](../../src/artifact_utils.py), [baseline.py](../../src/baseline.py), [batch_allocations.py](../../src/batch_allocations.py), [comparison_models.py](../../src/comparison_models.py), [evaluate_baseline.py](../../src/evaluate_baseline.py), [experiment_io.py](../../src/experiment_io.py), [market_cap_proxy.py](../../src/market_cap_proxy.py), [market_cap_value_digest.py](../../src/market_cap_value_digest.py), [research_forecasts.py](../../src/research_forecasts.py), [research_risk.py](../../src/research_risk.py).
