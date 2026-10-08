# Experiment results


Saved results, organized October 7, 2026. No model was rerun for this table. All figures are before costs. Controls and scaling variants are preserved in each experiment folder.

| Experiment | Description | Results |
| --- | --- | --- |
| [C01](../experiments/C01/README.md) | Industry-neutral allocations | 4 portfolio rows |
| [C02](../experiments/C02/README.md) | Penalized portfolio weights | 3 portfolio rows |
| [C02_WIDE](../experiments/C02_WIDE/README.md) | Broad and refined allocation penalties | 3 portfolio rows |
| [C03](../experiments/C03/README.md) | Market-proxy exposure timing | 1 portfolio rows |
| [C04](../experiments/C04/README.md) | Equal industry sleeves | 1 portfolio rows |
| [C05](../experiments/C05/README.md) | Minimum-variance industry sleeves | 1 portfolio rows |
| [C06](../experiments/C06/README.md) | Markowitz industry sleeves | 1 portfolio rows |
| [H01](../experiments/H01/README.md) | Dense Ridge penalty search | 2 portfolio rows |
| [P01](../experiments/P01/README.md) | Restored benchmark comparison | 5 portfolio rows |
| [P02](../experiments/P02/README.md) | Equal forecast ensemble | 2 portfolio rows |
| [P03](../experiments/P03/README.md) | Predictor PCA | 2 portfolio rows |
| [P05](../experiments/P05/README.md) | Forecast-history decay | 2 portfolio rows |
| [P06](../experiments/P06/README.md) | Three-month bagged forecasts | 2 portfolio rows |
| [P07](../experiments/P07/README.md) | Historical factor-mean forecasts | 2 portfolio rows |
| [P08](../experiments/P08/README.md) | Broad and refined factor Ridge forecasts | 2 portfolio rows |
| [P08_SIX](../experiments/P08_SIX/README.md) | Six-penalty factor Ridge control | 2 portfolio rows |
| [R01](../experiments/R01/README.md) | Daily factor penalty | 3 portfolio rows |
| [R02](../experiments/R02/README.md) | Specific-risk penalty | 3 portfolio rows |
| [R03](../experiments/R03/README.md) | Ledoit–Wolf factor covariance | 15 portfolio rows |
| [R04](../experiments/R04/README.md) | Factor covariance PCA | 9 portfolio rows |
| [R04_TUNED](../experiments/R04_TUNED/README.md) | Historically tuned factor-covariance PCA | 3 portfolio rows |
| [R05](../experiments/R05/README.md) | Projected risk-characteristic basis | 3 portfolio rows |
| [R06](../experiments/R06/README.md) | Grouped risk-characteristic basis | 3 portfolio rows |
| [R07](../experiments/R07/README.md) | Expanding factor-covariance history | 3 portfolio rows |
| [R08_CORRELATION](../experiments/R08_CORRELATION/README.md) | Risk correlation decay | 3 portfolio rows |
| [R08_EXPANDING_CORRELATION](../experiments/R08_EXPANDING_CORRELATION/README.md) | Expanding history with tuned correlation decay | 3 portfolio rows |
| [R08_EXPANDING_VARIANCE](../experiments/R08_EXPANDING_VARIANCE/README.md) | Expanding history with tuned variance decay | 3 portfolio rows |
| [R08_VARIANCE](../experiments/R08_VARIANCE/README.md) | Risk variance decay | 3 portfolio rows |
| [R10_DD10](../experiments/R10_DD10/README.md) | Factor covariance conditioned on 10% drawdown | 3 portfolio rows |
| [R10_DD15](../experiments/R10_DD15/README.md) | Factor covariance conditioned on 15% drawdown | 3 portfolio rows |
| [R10_DD20](../experiments/R10_DD20/README.md) | Factor covariance conditioned on 20% drawdown | 3 portfolio rows |
| [T02](../experiments/T02/README.md) | Risk percentile exposures | 3 portfolio rows |
| [T03](../experiments/T03/README.md) | Percentile-ranked forecasts | 4 portfolio rows |
| [T04](../experiments/T04/README.md) | Country/month median imputation | 4 portfolio rows |
| [T04_RISK](../experiments/T04_RISK/README.md) | Country median risk preprocessing | 3 portfolio rows |
| [T05](../experiments/T05/README.md) | Benchmark constant-column noise | 3 portfolio rows |
| [T07](../experiments/T07/README.md) | Industry-group median imputation | 4 portfolio rows |
| [T07_RISK](../experiments/T07_RISK/README.md) | Industry median risk preprocessing | 3 portfolio rows |
| [P04_INHERITED](../experiments/P04_INHERITED/README.md) | All-history forecasts with inherited annual tree parameters | 2 portfolio rows |
| [P05_EXPANDING_INHERITED](../experiments/P05_EXPANDING_INHERITED/README.md) | All-history hyperbolic fitting weights with inherited tree parameters | 2 portfolio rows |
| [P09_FACTOR_XGB](../experiments/P09_FACTOR_XGB/README.md) | Factor-level XGBoost forecasts mapped to stocks | 1 portfolio rows |
| [P07_R01](../experiments/P07_R01/README.md) | P07 forecasts with historically tuned daily Barra Ridge | 2 portfolio rows |
| [P08_R01](../experiments/P08_R01/README.md) | P08 forecasts with historically tuned daily Barra Ridge | 2 portfolio rows |
| [E08_C021_R08V1](../experiments/E08_C021_R08V1/README.md) | Equal thirds + C02 + R08 | 1 portfolio rows |
| [P08_C021_R08V1](../experiments/P08_C021_R08V1/README.md) | P08 only + C02 + R08 | 1 portfolio rows |

## Archived completed comparison

[V01](../experiments/V01/README.md) preserves the earlier forward-only baseline alternative separately from the 41-package standalone scope. It adds five portfolio rows.
