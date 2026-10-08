# Portfolio forecasting and risk experiments

[Global Factor Data / JKP Factors](https://jkpfactors.com/) | [Common Task Framework](https://jkpfactors.com/ctf) | [Data access](https://jkpfactors.com/ctf/dataset-access) | [Leaderboard](https://jkpfactors.com/ctf/leaderboard)

## Context

This repository contains research for the Common Task Framework portfolio-model competition hosted on the Global Factor Data (JKP Factors) website. The competition runs submitted models on a common stock dataset and evaluates the portfolio weights they produce. Our objective is to study how return forecasting, risk estimation and portfolio construction affect historical performance, then provide a reproducible implementation and methodology paper for the selected strategy.

This initial publication includes the self-contained final model, the final PDF write-up, saved experiment results and shared research source snapshots. The final model can run independently. Individual experiment execution packages are still being cleaned and assembled; the shared source tree should currently be read as research material.

The research follows the competition host's supplied R implementations of Minimum Variance, Factor ML and Markowitz ML, translated into Python. Ridge Markowitz adapts the supplied Markowitz allocation to stock-level Ridge forecasts. Only the permitted stock-characteristic, feature-list and daily-return tables enter the models. Historical performance is evaluated from January 1990 through December 2023, with earlier available data used for estimation. The input `ctff_test` flags determine the evaluation observations; the final model must not assume that every future dataset has the same dates.

## Research approach

We began with four principal reference portfolios:

| Portfolio | Expected returns | Portfolio construction |
| --- | --- | --- |
| Minimum Variance | No return forecast | Minimize estimated stock variance with weights summing to one |
| Ridge Markowitz | Stock-level Ridge forecasts | Markowitz allocation using the original Barra-style covariance |
| Factor ML | Stock-level XGBoost forecasts | Equal-weight long and short predicted-return deciles |
| Markowitz ML | The same stock-level XGBoost forecasts | Markowitz allocation using the original Barra-style covariance |

The name Factor ML belongs to the supplied stock-forecast benchmark. It differs from the forecasts of Barra factor coefficients studied in P07 and P08. The saved benchmark comparison also includes Ridge deciles as an additional reference.

We then tested individual changes to forecasting, risk estimation, preprocessing and allocation. Each experiment was compared with the relevant original portfolios or matched controls, keeping other components fixed where the experiment allowed it. Not every change applies to all four portfolios. The saved comparisons report annual mean excess return, realized volatility, Sharpe ratio, maximum drawdown, gross exposure and monthly turnover. Experiment READMEs explain what changed, what remained fixed and which portfolio each result is compared against.

After examining those individual results, we chose components to combine. We independently rebuilt two strategies: one averaging original stock Ridge, original stock XGBoost and P08 forecasts in equal thirds, and one using only P08 forecasts. Both combined forecasts with C02 regular and R08 expanding variance. Expected-return vectors are combined before optimization; portfolio weights are not averaged.

The selected strategy is **P08-only + C02 regular + R08 expanding variance**, saved as `P08_C021_R08V1`.

- **P08** predicts the next month's estimated Barra factor coefficients from the previous completed month's coefficients using Ridge, then maps the forecast through known stock exposures to obtain expected stock returns.
- **R08 expanding variance** uses all completed factor history and historically selected hyperbolic weights for factor variances, while retaining the original correlation weighting and specific-risk method.
- **C02 regular** adds a historically selected diagonal penalty to the allocation covariance to penalize large stock weights. The original same-date covariance sets the penalty units and estimated 10% annual risk scaling.

| Saved historical statistic | Original Markowitz ML | Equal-thirds combination | Selected P08-only combination |
| --- | ---: | ---: | ---: |
| Annualized Sharpe | 2.3416 | 3.5615 | 3.5862 |
| Sharpe change vs original | 0.0000 | +1.2199 | +1.2447 |
| Maximum drawdown | 30.41% | 18.05% | 20.23% |

**These results include no trading costs.** Financing and stock-borrow costs are also excluded. The selected configuration was chosen after inspecting historical evaluation-period performance, so the selection-bias limitations below apply.

## Folder structure

| Path | What it contains |
| --- | --- |
| [experiments/](results/README.md) | One folder per experiment, containing its README and compact `results.json`; runnable experiment code will be packaged in each experiment's `code/` subfolder |
| [src/](src/README.md) | Shared Python source code for model calculations, historical parameter selection and performance measurement |
| [results/](results/README.md) | Consolidated saved results, including [the portfolio CSV](results/portfolio_results.csv) and experiment JSON |
| [submission/](submission/README.md) | The selected strategy's self-contained [model.py](submission/model.py), pinned dependencies and verified [final portfolio weights CSV](submission/weights.csv) |

The final [PDF write-up](submission/Alfredo%20Vara%20-%20PS3.pdf), **Factor Forecasting and Regularized Portfolio Allocation**, is available in `submission/` alongside the model, dependencies and final portfolio weights. The large internal research reports are excluded. Robustness notes are summarized below, so there is no separate `docs` folder.

Each experiment's runnable code will use the relevant common functions in `src/`.
The selected combined strategy already has its own experiment folder,
[P08_C021_R08V1](experiments/P08_C021_R08V1/README.md), with a
[code subfolder](experiments/P08_C021_R08V1/code/README.md) reserved for its complete
research execution entry point. That code is still being packaged.

The consolidated experiment JSON collects the individual experiment result
records in one file. The CSV provides their portfolio-level metrics in a table,
so users can compare results without opening every experiment folder.

### What shared source code means

`src` is short for source code. It contains 29 Python files with the calculations used across the experiments. For example, `baseline.py` prepares stock predictors and fits the original Ridge model; `comparison_models.py` supplies the original XGBoost and Barra-style risk calculations; and `factor_forecast_paths.py` implements the P07/P08 factor forecasts. These are code files, not fitted models, saved forecasts or additional experiments.

Several experiments use the same underlying calculations. Keeping those functions in one shared place avoids maintaining a different copy in every experiment folder. Each experiment README links to its principal source files and explains their roles. The [source README](src/README.md) gives a complete file guide. Some original runners still combine scientific calculations with private execution machinery; separating those dependencies into simple runnable entry points remains part of preparation.

### The submission code

`submission/` contains the selected strategy's self-contained [model.py](submission/model.py), its pinned [requirements.txt](submission/requirements.txt) and an execution guide. Every scientific function needed by the selected model is inside that one Python file. It calculates forecasts, risk and monthly weights from the three supplied tables and returns `id`, `eom`, `w`. It imports no local experiment or shared-source files and reads no old fitted artifacts. A fresh full-period run matched all 885,698 accepted weights exactly, including a byte-identical weights CSV. Historical parameter selection remains in the file; full-period performance evaluation remains outside it. Competition acceptance has not yet been obtained.

## Rules, data and reproduction

Everything prepared for the contest must follow the [official contest rules](https://jkpfactors.com/ctf/rules). Only the three supplied tables may enter the model. At each portfolio formation date, all inputs and training/validation labels used in that weight calculation must already be available. The submitted model must rebalance monthly, be deterministic, cover the supplied `ctff_test` keys and satisfy the required interface, dependency and resource limits.

Historical hyperparameter validation remains in the model code. P08's forecast-penalty selection, R08's decay selection and C02's historical payoff-bank selection define the strategy. These are separate from auxiliary tests and numerical verifiers used privately to check implementation correctness. Those correctness-checking files are excluded from this repository.

Users must obtain data through the official data-access process. Raw data, transformed panels, fitted caches, logs and credentials are not distributed here. The original research archive remains private. Unfinished original full-search P04/P05 drafts are not presented as completed experiments; the completed inherited variants have separate folders.

The current source snapshots use NumPy/SciPy, Polars, Ibis/DuckDB and XGBoost where applicable. Runnable research entry points, experiment-specific dependency pins and source-version reconciliation are still being prepared. Reproduction commands will be added when those entry points are complete. This initial publication holds 45 completed experiment packages plus one archived completed comparison, 138 scalar portfolio-result rows and 29 unchanged source snapshots. The selected contest model has its own pinned dependencies and does not use Ibis or the shared source tree. GitHub publication is separate from contest submission; no contest upload or acceptance is claimed.

## Interpretation

Returns are monthly excess returns. Annual mean is the arithmetic monthly mean multiplied by 12; annual volatility is the monthly sample standard deviation multiplied by the square root of 12; Sharpe is their ratio. Maximum drawdown uses compounded monthly excess returns. These statistics should not be read as CAGR or after-cost returns.

Every Sharpe change must be read against its stated control. Some experiments report both native scaling under the changed covariance and reference scaling under the original covariance. Those versions answer different questions and remain separately labeled. Component Sharpe improvements are not additive, and a combined result cannot by itself attribute improvement to one component. The small Sharpe difference between the two final combinations does not establish a reliable advantage.

## Limitations and robustness

The experiment shortlist and final combination were selected after observing results from the historical evaluation period. This can overfit the choice of configuration to that period even when individual portfolio weights respect formation-date information availability. The final comparison is exploratory and does not provide a genuinely unseen holdout or establish statistical significance for the selected combined model.

Future work should predeclare compatible candidate combinations, use estimation and validation data to select one fixed strategy, and evaluate that strategy on genuinely unseen subsequent data. Bootstrap resampling of an already inspected period does not create an unseen test set.

The earlier I01 exercise used 1,000 common circular 12-month block-bootstrap draws with seed 1 for paired mean-return and Sharpe comparisons. Its inference is conditional on the earlier saved fitted strategies and pointwise. It does not refit models, correct the full configuration-search process, quantify drawdown uncertainty or cover the two final combined portfolios. The write-up summarizes the relevant robustness findings and their limitations.

The 10% annual risk level is estimated at formation using the original covariance. It is not a guarantee of realized volatility. The selected strategy has substantial gross exposure and turnover, making omitted trading, financing and borrow costs material. Factor coefficients are regression estimates rather than automatically tradable factor-portfolio returns. Finite hyperparameter searches can select boundary values without demonstrating a global optimum. Local research verification does not constitute contest approval.
