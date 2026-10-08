# Portfolio forecasting and risk experiments

[Global Factor Data / JKP Factors](https://jkpfactors.com/) | [Common Task Framework](https://jkpfactors.com/ctf) | [Data access](https://jkpfactors.com/ctf/dataset-access) | [Leaderboard](https://jkpfactors.com/ctf/leaderboard)

## Context

This repository contains research for the Common Task Framework portfolio-model competition hosted on the Global Factor Data (JKP Factors) website. The competition runs submitted models on a common stock dataset and evaluates the portfolio weights they produce. Our objective is to study how return forecasting, risk estimation and portfolio construction affect historical performance, then provide a reproducible implementation and methodology paper for the selected strategy.

This repository includes the self-contained final model, final PDF write-up, saved results and shared scientific code with complete local dependencies. All 46 experiment folders have runnable local entry points and execution instructions.

The research follows the Minimum Variance, Factor ML and Markowitz ML R benchmarks supplied by the competition hosts, Global Factor Data / JKP Factors, translated into Python. Ridge Markowitz adapts the supplied Markowitz allocation to stock-level Ridge forecasts. Only the permitted stock-characteristic, feature-list and daily-return tables enter the models. Historical performance is evaluated from January 1990 through December 2023, with earlier available data used for estimation. The input `ctff_test` flags determine the evaluation observations; the final model must not assume that every future dataset has the same dates.

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
| [Factor Forecasting and Regularized Portfolio Allocation.pdf](Factor%20Forecasting%20and%20Regularized%20Portfolio%20Allocation.pdf) | Final methodology paper |
| [experiments/](experiments/) | One folder per experiment, containing its README, saved `results.json` and a runnable `code/` subfolder |
| [src/](src/README.md) | Shared Python source code for model calculations, historical parameter selection and performance measurement |
| [results/](results/README.md) | Consolidated saved results, including [the portfolio CSV](results/portfolio_results.csv) and experiment JSON |
| [submission/](submission/README.md) | The selected strategy's self-contained [model.py](submission/model.py), pinned dependencies and verified [final portfolio weights CSV](submission/weights.csv) |

The final [PDF write-up](Factor%20Forecasting%20and%20Regularized%20Portfolio%20Allocation.pdf), **Factor Forecasting and Regularized Portfolio Allocation**, sits at the repository’s top level beside this README. The model, dependencies and final portfolio weights are in `submission/`. The large internal research reports are excluded. Robustness notes are summarized below, so there is no separate `docs` folder.

Each experiment's run script uses the relevant common functions in `src/`.
The selected combined strategy's [research entry point](experiments/P08_C021_R08V1/code/run.py)
calls the self-contained submission model and evaluates the resulting weights locally.

The consolidated experiment JSON collects the individual experiment result
records in one file. The CSV provides their portfolio-level metrics in a table,
so users can compare results without opening every experiment folder.

### What shared source code means

`src` is short for source code. It contains 34 Python files for scientific calculations and local input/output. For example, `baseline.py` prepares stock predictors and fits the original Ridge model; `comparison_models.py` supplies the original XGBoost and Barra-style risk calculations; and `factor_forecast_paths.py` implements the P07/P08 factor forecasts. These are code files, not fitted models, saved forecasts or additional experiments.

Several experiments use the same underlying calculations. Keeping those functions in one shared place avoids maintaining a different copy in every experiment folder. Each experiment README links to its principal source files and explains their roles. The [source README](src/README.md) gives a complete file guide. The shared files no longer import private execution code. Each experiment has a short run script that calls these functions directly.

### The submission code

`submission/` contains the selected strategy's self-contained [model.py](submission/model.py), its pinned [requirements.txt](submission/requirements.txt) and an execution guide. Every scientific function needed by the selected model is inside that one Python file. It calculates forecasts, risk and monthly weights from the three supplied tables and returns `id`, `eom`, `w`. It imports no local experiment or shared-source files and reads no old fitted artifacts. A fresh full-period run matched all 885,698 accepted weights exactly, including a byte-identical weights CSV. Historical parameter selection remains in the file; full-period performance evaluation remains outside it. Competition acceptance has not yet been obtained.

## Reproduction

Use Python 3.13. Install the local research dependencies:

```sh
python -m pip install -r requirements-research.txt
```

Research imports require NumPy, Pandas, Polars, PyArrow, SciPy, threadpoolctl and XGBoost, plus their pinned supporting dependencies. Ibis and DuckDB are no longer needed by the public code. Obtain the three permitted CTF tables through the data-access instructions above; raw data are not included in this repository.

All 46 experiment folders contain a run script and execution instructions.

| Group | Experiment folders |
| --- | --- |
| Benchmark and historical-validation comparisons | [P01](experiments/P01/code/README.md), [H01](experiments/H01/code/README.md), [V01](experiments/V01/code/README.md) |
| Return forecasts | [P02](experiments/P02/code/README.md), [P03](experiments/P03/code/README.md), [P04_INHERITED](experiments/P04_INHERITED/code/README.md), [P05](experiments/P05/code/README.md), [P05_EXPANDING_INHERITED](experiments/P05_EXPANDING_INHERITED/code/README.md), [P06](experiments/P06/code/README.md), [P07](experiments/P07/code/README.md), [P08](experiments/P08/code/README.md), [P08_SIX](experiments/P08_SIX/code/README.md), [P09_FACTOR_XGB](experiments/P09_FACTOR_XGB/code/README.md) |
| Forecast and risk preprocessing | [T02](experiments/T02/code/README.md), [T03](experiments/T03/code/README.md), [T04](experiments/T04/code/README.md), [T04_RISK](experiments/T04_RISK/code/README.md), [T05](experiments/T05/code/README.md), [T07](experiments/T07/code/README.md), [T07_RISK](experiments/T07_RISK/code/README.md) |
| Risk estimation | [R01](experiments/R01/code/README.md), [R02](experiments/R02/code/README.md), [R03](experiments/R03/code/README.md), [R04](experiments/R04/code/README.md), [R04_TUNED](experiments/R04_TUNED/code/README.md), [R05](experiments/R05/code/README.md), [R06](experiments/R06/code/README.md), [R07](experiments/R07/code/README.md), [R08_CORRELATION](experiments/R08_CORRELATION/code/README.md), [R08_VARIANCE](experiments/R08_VARIANCE/code/README.md), [R08_EXPANDING_CORRELATION](experiments/R08_EXPANDING_CORRELATION/code/README.md), [R08_EXPANDING_VARIANCE](experiments/R08_EXPANDING_VARIANCE/code/README.md), [R10_DD10](experiments/R10_DD10/code/README.md), [R10_DD15](experiments/R10_DD15/code/README.md), [R10_DD20](experiments/R10_DD20/code/README.md) |
| Allocation | [C01](experiments/C01/code/README.md), [C02](experiments/C02/code/README.md), [C02_WIDE](experiments/C02_WIDE/code/README.md), [C03](experiments/C03/code/README.md), [C04](experiments/C04/code/README.md), [C05](experiments/C05/code/README.md), [C06](experiments/C06/code/README.md) |
| Combined models | [P07_R01](experiments/P07_R01/code/README.md), [P08_R01](experiments/P08_R01/code/README.md), [E08_C021_R08V1](experiments/E08_C021_R08V1/code/README.md), [P08_C021_R08V1](experiments/P08_C021_R08V1/code/README.md) |

For example, from the repository root:

```sh
python experiments/P01/code/run.py --data /path/to/ctf-tables --output /path/to/new-results --threads 10
```

Each run reads the supplied tables and fits its own models. Use a separate output directory for each experiment. Within a run, risk estimates are shared across applicable portfolios. Local research scripts save weights, monthly performance and fitting records; optional generated risk artifacts support later method variants. These files remain outside the public Git history. The scripts use the input `ctff_test` flags rather than hardcoding 1990–2023.

The scripts and their shared dependencies run without the private research repository. Each entry point has passed bounded execution on supplied historical data. The 38 newly packaged scripts also pass future-data removal checks. Their scientific functions and settings are reconciled with the original research code. Existing results cover 46 experiment folders and 138 scalar portfolio records. The cleanup preserves those results; bounded checks do not constitute fresh full-period reproduction.

The contest entry point is [submission/model.py](submission/model.py), with [its separate requirements](submission/requirements.txt). It receives the three Pandas tables and returns weights. Only that self-contained script is prepared for contest execution. The research scripts' local file reading, caching and performance evaluation are not part of the contest entry point. See the [official rules](https://jkpfactors.com/ctf/rules). No contest upload or acceptance is claimed.

## Benchmark attribution

The original Minimum Variance, Factor ML and Markowitz ML benchmark implementations are attributed to the competition hosts, [Global Factor Data / JKP Factors](https://jkpfactors.com/), which supplied them through the [Common Task Framework](https://jkpfactors.com/ctf). We translated their R implementations into Python, retaining their core preprocessing, historical parameter selection, covariance estimation and portfolio-construction methods as reference models. Ridge Markowitz is our adaptation of the supplied Markowitz construction using stock-level Ridge forecasts. The experiment extensions and selected combined strategy build on these host-provided benchmarks.

## Interpretation

Returns are monthly excess returns. Annual mean is the arithmetic monthly mean multiplied by 12; annual volatility is the monthly sample standard deviation multiplied by the square root of 12; Sharpe is their ratio. Maximum drawdown uses compounded monthly excess returns. These statistics should not be read as CAGR or after-cost returns.

Every Sharpe change must be read against its stated control. Some experiments report both native scaling under the changed covariance and reference scaling under the original covariance. Those versions answer different questions and remain separately labeled. Component Sharpe improvements are not additive, and a combined result cannot by itself attribute improvement to one component. The small Sharpe difference between the two final combinations does not establish a reliable advantage.

## Limitations and robustness

The experiment shortlist and final combination were selected after observing results from the historical evaluation period. This can overfit the choice of configuration to that period even when individual portfolio weights respect formation-date information availability. The final comparison is exploratory and does not provide a genuinely unseen holdout or establish statistical significance for the selected combined model.

Future work should predeclare compatible candidate combinations, use estimation and validation data to select one fixed strategy, and evaluate that strategy on genuinely unseen subsequent data. Bootstrap resampling of an already inspected period does not create an unseen test set.

The earlier I01 exercise used 1,000 common circular 12-month block-bootstrap draws with seed 1 for paired mean-return and Sharpe comparisons. Its inference is conditional on the earlier saved fitted strategies and pointwise. It does not refit models, correct the full configuration-search process, quantify drawdown uncertainty or cover the two final combined portfolios. The write-up summarizes the relevant robustness findings and their limitations.

The 10% annual risk level is estimated at formation using the original covariance. It is not a guarantee of realized volatility. The selected strategy has substantial gross exposure and turnover, making omitted trading, financing and borrow costs material. Factor coefficients are regression estimates rather than automatically tradable factor-portfolio returns. Finite hyperparameter searches can select boundary values without demonstrating a global optimum. Local research verification does not constitute contest approval.
