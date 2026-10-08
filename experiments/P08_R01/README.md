# P08_R01: P08 forecasts with historically tuned daily Barra Ridge

## What this experiment tests

This variant applies the tuned daily Barra Ridge penalty from R01 before P08's lagged-factor Ridge forecast. Both the factor forecasting inputs and risk estimates therefore change; the separate forecasting Ridge penalty is still selected using historical mapped stock-return error. Native changed-risk scaling and original-risk reference scaling are reported against P01 and original P08. This standalone variant does not include C02 or R08 expanding variance, and it is not the final selected combination. Its recorded producer time is incremental because accepted R01 risk histories were reused.

## Saved results

| Portfolio | Scaling/view | Control | Sharpe | Delta Sharpe | Max drawdown |
| --- | --- | --- | ---: | ---: | ---: |
| markowitz_ml | native changed-risk scaling | P01/markowitz_ml | 2.7064 | +0.3648 | 14.64% |
| markowitz_ml_reference | original-risk reference scaling | P01/markowitz_ml | 2.9291 | +0.5875 | 11.58% |

Metrics cover January 1990 through December 2023. Returns and volatility are annualized from monthly excess returns. Drawdown uses compounded monthly excess returns. Sharpe improvements across components are not additive.

Recorded producer time: 39.840 seconds (0.7 minutes).
Incremental producer time with accepted R01 risk histories reused; not a cold end-to-end runtime.

## Code used by this experiment

These are the principal implementation files currently staged for this experiment. Shared preprocessing, benchmark risk and performance calculations also use [baseline.py](../../src/baseline.py), [comparison_models.py](../../src/comparison_models.py) and [evaluate_baseline.py](../../src/evaluate_baseline.py), as applicable. The full standalone execution package is still being prepared; remaining private-runner imports are listed in [the source index](../../src/source-index.json).

- [tuned_factor_forecast_paths.py](../../src/tuned_factor_forecast_paths.py): P07/P08 factor forecasts based on the tuned R01 daily factor stream.
- [risk_penalty_paths.py](../../src/risk_penalty_paths.py): Daily factor and specific-risk Ridge penalty selection on historical cross-sections.
- [risk_penalty_replay.py](../../src/risk_penalty_replay.py): Consistent risk-state reconstruction after changing a risk-regression penalty.
- [ridge_path.py](../../src/ridge_path.py): Efficient evaluation of ordinary Ridge coefficient paths.

Historical model-parameter validation remains part of the scientific implementation. Separate correctness checks, numerical verifiers and operational schedulers are excluded.

Configuration choices were informed by previously observed evaluation-period results. These results do not constitute a genuinely unseen holdout or establish future performance. Per-experiment source-version reconciliation and execution packaging remain pending in this initial research publication.
