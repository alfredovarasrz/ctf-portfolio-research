# R10_DD10: Factor covariance conditioned on 10% drawdown

## What this experiment tests

This experiment conditions factor covariance on whether a capitalization-weighted excess-return proxy built from the permitted stocks is at least 10% below its observed running peak. Within the original 2,520-day window, it uses historical factor observations whose already-known proxy state matches the current state, retaining the original correlation and variance half-lives and full historical age gaps. If fewer than 63 matching days are available, it uses the original covariance. Exposures, specific risk, factor coefficients, stock forecasts and allocation remain original. This tests state-dependent risk estimation, not future knowledge of a downturn.

## Saved results

| Portfolio | Scaling/view | Control | Sharpe | Delta Sharpe | Max drawdown |
| --- | --- | --- | ---: | ---: | ---: |
| minimum_variance_native | native net-one MVP | P01/minimum_variance | 0.9454 | -0.0030 | 26.73% |
| markowitz_xgboost_native | variant covariance10% | P01/markowitz_ml | 2.2679 | -0.0737 | 35.20% |
| markowitz_xgboost_reference | ORIGINAL covariance10% | P01/markowitz_ml | 2.4061 | +0.0645 | 24.76% |

Metrics cover January 1990 through December 2023. Returns and volatility are annualized from monthly excess returns. Drawdown uses compounded monthly excess returns. Sharpe improvements across components are not additive.

Recorded producer time: 29.170 seconds (0.5 minutes).
Recorded producer-stage elapsed time. This may reuse earlier forecasts or risk estimates and excludes upstream fits, checks and review; it is not a cold end-to-end runtime estimate.

## Code used by this experiment

These are the principal implementation files currently staged for this experiment. Shared preprocessing, benchmark risk and performance calculations also use [baseline.py](../../src/baseline.py), [comparison_models.py](../../src/comparison_models.py) and [evaluate_baseline.py](../../src/evaluate_baseline.py), as applicable. All shared Python dependencies are packaged in `src`; [the source index](../../src/source-index.json) records the import graph and source hashes.

- [risk_history_regimes.py](../../src/risk_history_regimes.py): Expanding covariance history, tuned variance/correlation decay and drawdown-conditioned covariance.

Historical model-parameter validation remains part of the scientific implementation. Separate correctness checks, numerical verifiers and operational schedulers are excluded.

## Method details

- 10% fixed drawdown from observed running peak of permitted-stock capitalization-weighted EXCESS proxy.
- Original2520-day window and504/84EW settings; full unfiltered age gaps retained; at least63matching days or exact original F fallback.

## Interpretation limits

- Full 408-month backtest, with gross returns before trading costs. Historical selections use only labels complete by actual formation; test-period performance does not select parameters.
- Each threshold is a separate original-parent experiment, not a test-period-selected winner. This is conditional covariance, not downside clipping/semivariance or a total-return bear-market index.

Configuration choices were informed by previously observed evaluation-period results. These results do not constitute a genuinely unseen holdout or establish future performance. Entry-point packaging and per-run source-version reconciliation remain pending where not identified below.
