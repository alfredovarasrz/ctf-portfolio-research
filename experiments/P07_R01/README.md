# P07_R01: P07 forecasts with historically tuned daily Barra Ridge

## What this experiment tests

This variant applies the tuned daily Barra Ridge penalty from R01 to P07's factor-mean forecasting approach. The changed daily regression produces different factor coefficients and residuals, so both the factor forecasts and risk estimates change. It then maps the historical factor-mean forecast through known stock exposures as in P07. Results include native changed-risk scaling and an original-risk reference version, compared with both P01 and original P07. The run reused accepted R01 risk histories, so its recorded producer time is incremental.

## Saved results

| Portfolio | Scaling/view | Control | Sharpe | Delta Sharpe | Max drawdown |
| --- | --- | --- | ---: | ---: | ---: |
| markowitz_ml | native changed-risk scaling | P01/markowitz_ml | 2.8645 | +0.5229 | 19.29% |
| markowitz_ml_reference | original-risk reference scaling | P01/markowitz_ml | 3.0457 | +0.7041 | 16.38% |

Metrics cover January 1990 through December 2023. Returns and volatility are annualized from monthly excess returns. Drawdown uses compounded monthly excess returns. Sharpe improvements across components are not additive.

Recorded producer time: 27.781 seconds (0.5 minutes).
Incremental producer time with accepted R01 risk histories reused; not a cold end-to-end runtime.

## Code used by this experiment

These are the principal implementation files currently staged for this experiment. Shared preprocessing, benchmark risk and performance calculations also use [baseline.py](../../src/baseline.py), [comparison_models.py](../../src/comparison_models.py) and [evaluate_baseline.py](../../src/evaluate_baseline.py), as applicable. The full standalone execution package is still being prepared; remaining private-runner imports are listed in [the source index](../../src/source-index.json).

- [tuned_factor_forecast_paths.py](../../src/tuned_factor_forecast_paths.py): P07/P08 factor forecasts based on the tuned R01 daily factor stream.
- [risk_penalty_paths.py](../../src/risk_penalty_paths.py): Daily factor and specific-risk Ridge penalty selection on historical cross-sections.
- [risk_penalty_replay.py](../../src/risk_penalty_replay.py): Consistent risk-state reconstruction after changing a risk-regression penalty.

Historical model-parameter validation remains part of the scientific implementation. Separate correctness checks, numerical verifiers and operational schedulers are excluded.

Configuration choices were informed by previously observed evaluation-period results. These results do not constitute a genuinely unseen holdout or establish future performance. Public release requires source-version reconciliation, attribution review and final packaging.
