# V01: Archived forward-only comparison

## What this experiment tests

This archived earlier comparison uses forward-only validation inside each completed historical training window. Each held-out historical block is predicted using earlier blocks, instead of the benchmark's rotating calendar-fold method that fits four historical blocks and validates on the fifth. Ridge and XGBoost forecasts feed the same decile and Markowitz rules, with the original risk model and Minimum Variance control. It is a completed methodological alternative outside the 41-package standalone roster, not a replacement for P01 or a contest requirement.

## Saved results

| Portfolio | Control | Sharpe | Delta Sharpe | Max drawdown |
| --- | --- | ---: | ---: | ---: |
| ridge | P01/ridge | 0.7020 | -0.0377 | 54.62% |
| factor_ml | P01/factor_ml | 0.7193 | +0.0153 | 49.34% |
| minimum_variance | P01/minimum_variance | 0.9484 | +0.0000 | 25.26% |
| markowitz_ridge | P01/markowitz_ridge | 2.0165 | -0.1223 | 31.75% |
| markowitz_ml | P01/markowitz_ml | 2.2087 | -0.1329 | 21.91% |

Recorded producer time: 2556.590 seconds. Recorded historical producer-stage time; excludes upstream work and correctness checks.

## Code used by this experiment

These are the principal implementation files currently staged for this experiment. Shared preprocessing, benchmark risk and performance calculations also use [baseline.py](../../src/baseline.py), [comparison_models.py](../../src/comparison_models.py) and [evaluate_baseline.py](../../src/evaluate_baseline.py), as applicable. The full standalone execution package is still being prepared; remaining private-runner imports are listed in [the source index](../../src/source-index.json).

- [research_forecasts.py](../../src/research_forecasts.py): Forward-only historical forecast fitting for the archived V01 comparison.
- [baseline.py](../../src/baseline.py): Monthly predictor preprocessing, historical stock Ridge fitting and decile allocation.
- [comparison_models.py](../../src/comparison_models.py): Original stock XGBoost forecasts, Barra-style risk estimation and benchmark portfolio calculations.

Historical model-parameter validation remains part of the scientific implementation. Separate correctness checks, numerical verifiers and operational schedulers are excluded.

Results cover 1990–2023 and exclude costs. This archived comparison is outside the 41-package standalone roster and is presented separately. Retrospective configuration selection does not constitute an unseen holdout.
