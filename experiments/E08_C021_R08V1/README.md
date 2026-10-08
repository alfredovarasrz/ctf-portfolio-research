# E08_C021_R08V1: Equal thirds + C02 + R08

## What this experiment tests

This combined strategy averages three expected stock-return vectors with equal one-third weights: original stock Ridge, original stock XGBoost and P08's factor-based forecast. It feeds that average into Markowitz optimization with C02 regular and R08 expanding variance. Expected returns are averaged before optimization; the three portfolios' stock weights are not averaged. The original same-date covariance calibrates the C02 penalty units and estimated 10% annual risk scaling. This candidate was independently rebuilt from raw inputs and selected retrospectively for exploratory evaluation.

## Saved results

| Portfolio | Scaling/view | Control | Sharpe | Delta Sharpe | Max drawdown |
| --- | --- | --- | ---: | ---: | ---: |
| markowitz_ml | original covariance risk scaling | P01/markowitz_ml | 3.5615 | +1.2199 | 18.05% |

Metrics cover January 1990 through December 2023. Returns and volatility are annualized from monthly excess returns. Drawdown uses compounded monthly excess returns. Sharpe improvements across components are not additive.

Recorded producer time: 4016.130 seconds (66.9 minutes).
Independently cold producer fit at TEN threads; excludes numerical checks and review.

## Code used by this experiment

These are the principal implementation files currently staged for this experiment. Shared preprocessing, benchmark risk and performance calculations also use [baseline.py](../../src/baseline.py), [comparison_models.py](../../src/comparison_models.py) and [evaluate_baseline.py](../../src/evaluate_baseline.py), as applicable. The full standalone execution package is still being prepared; remaining private-runner imports are listed in [the source index](../../src/source-index.json).

- [exploratory_combination_paths.py](../../src/exploratory_combination_paths.py): The two fixed combined forecasts and their C02/R08 Markowitz allocation.
- [factor_forecast_paths.py](../../src/factor_forecast_paths.py): P07 factor-mean forecasts and P08/P08_SIX lagged-factor Ridge forecasts with mapped stock-error selection.
- [risk_history_regimes.py](../../src/risk_history_regimes.py): Expanding covariance history, tuned variance/correlation decay and drawdown-conditioned covariance.
- [batch_allocations.py](../../src/batch_allocations.py): Industry-neutral optimization and the regular C02 diagonal-penalty allocation with historical payoff-bank selection.
- [baseline.py](../../src/baseline.py): Monthly predictor preprocessing, historical stock Ridge fitting and decile allocation.
- [comparison_models.py](../../src/comparison_models.py): Original stock XGBoost forecasts, Barra-style risk estimation and benchmark portfolio calculations.

Historical model-parameter validation remains part of the scientific implementation. Separate correctness checks, numerical verifiers and operational schedulers are excluded.

Configuration choices were informed by previously observed evaluation-period results. These results do not constitute a genuinely unseen holdout or establish future performance. Per-experiment source-version reconciliation and execution packaging remain pending in this initial research publication.
