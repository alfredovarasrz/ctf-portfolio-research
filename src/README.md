# Shared scientific source code

`src` means source code. These 29 Python files contain the data transformations, forecasting models, risk estimators, historical parameter-selection procedures and portfolio calculations used by the experiments. They are unchanged snapshots of the current scientific code from the private research archive.

Experiment folders link to the principal files for their methods. Common calculations remain here so multiple experiments can use one implementation.

| File | Role |
| --- | --- |
| [baseline.py](baseline.py) | Monthly predictor preprocessing, historical stock Ridge fitting and decile allocation. |
| [batch_allocations.py](batch_allocations.py) | Industry-neutral optimization and the regular C02 diagonal-penalty allocation with historical payoff-bank selection. |
| [batch_forecasts.py](batch_forecasts.py) | Broad historical stock Ridge penalty search. |
| [batch_risk.py](batch_risk.py) | Ledoit–Wolf and fixed-90% factor-covariance PCA transformations, including native/reference allocation views. |
| [capped_calendar_validation.py](capped_calendar_validation.py) | Historical calendar-block partitioning for expanding-model parameter selection. |
| [comparison_models.py](comparison_models.py) | Original stock XGBoost forecasts, Barra-style risk estimation and benchmark portfolio calculations. |
| [evaluate_baseline.py](evaluate_baseline.py) | Performance calculations for weights and subsequent returns. |
| [expanding_forecasts.py](expanding_forecasts.py) | All-history forecast fitting and hyperbolic observation-weight selection. |
| [exploratory_combination_paths.py](exploratory_combination_paths.py) | The two fixed combined forecasts and their C02/R08 Markowitz allocation. |
| [extended_allocations.py](extended_allocations.py) | Industry-portfolio construction and combination, plus the fixed market-proxy timing multiplier. |
| [extended_forecasts.py](extended_forecasts.py) | Predictor PCA, rolling age-weighted forecasts and three-month-block bagging. |
| [extended_risk.py](extended_risk.py) | Historical covariance weighting and shared risk transformations. |
| [extended_risk_rebuilds.py](extended_risk_rebuilds.py) | Historical exposure-PCA/grouping bases and consistent daily factor/residual/risk reconstruction. |
| [factor_forecast_paths.py](factor_forecast_paths.py) | P07 factor-mean forecasts and P08/P08_SIX lagged-factor Ridge forecasts with mapped stock-error selection. |
| [factor_xgb_paths.py](factor_xgb_paths.py) | P09 multioutput factor XGBoost fitting and tree-count selection by mapped stock-return error. |
| [inherited_expanding_forecasts.py](inherited_expanding_forecasts.py) | Expanding forecast variants that inherit original annual tree settings. |
| [market_cap_value_digest.py](market_cap_value_digest.py) | Small shared utility for stable capitalization-value identity handling in the local research code. |
| [median_predictors.py](median_predictors.py) | Country/month and industry-aware missing-predictor median filling. |
| [research_forecasts.py](research_forecasts.py) | Restored benchmark forecast orchestration and the forward-only historical fitting alternative used by V01. |
| [research_risk.py](research_risk.py) | Risk-artifact construction and allocation using saved risk estimates. |
| [ridge_path.py](ridge_path.py) | Efficient evaluation of ordinary Ridge coefficient paths. |
| [risk_history_regimes.py](risk_history_regimes.py) | Expanding covariance history, tuned variance/correlation decay and drawdown-conditioned covariance. |
| [risk_penalty_paths.py](risk_penalty_paths.py) | Daily factor and specific-risk Ridge penalty selection on historical cross-sections. |
| [risk_penalty_replay.py](risk_penalty_replay.py) | Consistent risk-state reconstruction after changing a risk-regression penalty. |
| [risk_preprocessing.py](risk_preprocessing.py) | Risk-exposure normalization, missing-value filling and constant-column Gaussian treatment. |
| [spectral_allocation_path.py](spectral_allocation_path.py) | Efficient computation of broad C02 allocation-penalty paths. |
| [tuned_covariance_pca.py](tuned_covariance_pca.py) | Historical selection of the retained factor-covariance eigenvalue mass. |
| [tuned_factor_forecast_paths.py](tuned_factor_forecast_paths.py) | P07/P08 factor forecasts based on the tuned R01 daily factor stream. |
| [tuned_risk_decay.py](tuned_risk_decay.py) | Historical selection of hyperbolic correlation or variance decay within a rolling window. |

## Preparation status

This source tree is not yet independently runnable. Some modules import scientific functions from local runners that also contain private execution machinery. Those dependencies must be separated when preparing the public entry points. The remaining imports and snapshot hashes are listed in [source-index.json](source-index.json). Staged source versions also need reconciliation with each completed run before release.

Historical parameter validation is part of the models and remains included. Correctness tests, numerical verifiers, fixtures, operational schedulers and monitoring helpers are excluded.

The selected strategy's self-contained contest entry point and pinned dependencies are available in [submission/](../submission/README.md). They do not depend on these research snapshots. The current [paper draft](../Portfolio_Model_Writeup_Draft.docx) is at the repository top level.
