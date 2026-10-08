# Final contest model

[model.py](model.py) implements the selected **P08-only + C02 regular + R08 expanding variance** strategy (`P08_C021_R08V1`). It is one self-contained Python file. Every scientific function it needs is included in that file; it imports standard Python and installed scientific packages, without calling any experiment or shared-source files.

The file follows the supplied benchmarks' libraries/settings, utilities, portfolio construction and `main` structure. [requirements.txt](requirements.txt) pins the model dependencies and their transitive dependencies, with package hashes, for Python 3.13.

## Interface and execution

The competition calls:

```python
weights = main(chars, features, daily_ret)
```

The arguments are the three supplied pandas DataFrames. Polars handles the internal table operations directly. The function returns a pandas DataFrame with exactly `id`, `eom` and `w`, including every supplied `ctff_test` stock-month. Test dates and features come from those inputs. The model contains no fixed evaluation-period dates or hardcoded stock universe.

Industry classification accepts whole SIC codes supplied as integers, floats or numeric strings. Missing, invalid, nonfinite or fractional codes receive the Other industry classification.

The model fits from these inputs on each invocation. It reads and writes no files, creates no temporary directories and uses no Ibis query layer. A bounded in-memory cache retains stock-loss sufficient statistics for the latest 144 completed months. A month's labels enter that cache only when its return month has completed. No saved research forecasts, selected parameters or risk estimates are read. Portfolio weights are returned to the competition.

The bottom of `model.py` contains a deterministic synthetic example. Run `python model.py` to exercise the interface without external data or file access. Importing the file does not run this example. The model prints daily-history progress, annual forecast refits, risk and allocation choices, and the final output count.

## Calculations retained

- **Original Barra exposures and risk.** Characteristics receive country/month percentile ranks and monthly sample-standard-deviation scaling, with the accepted zero, missing-value and constant-characteristic conventions. FF12 industry indicators are included. Daily Ridge regressions use preceding-month exposures and completed daily returns, with the original fixed `1e-4` penalty and glmnet-equivalent normalization. Their residuals produce the original specific-risk estimates.
- **P08 forecasts.** Completed daily factor coefficients are summed within each month. A multivariate Ridge model predicts the next monthly coefficient vector from the previous vector. It refits every twelve supplied test return months, using up to 120 completed target months and five historical calendar validation blocks. The selected positive penalty minimizes actual stock-return prediction error after mapping the factor forecast through preceding-month stock exposures. Current known exposures map the forecast back to a stock expected-return vector.
- **R08 expanding variance.** The allocation covariance uses all completed factor history, exponential correlation weights and hyperbolic variance weights. The variance half-weight age is selected annually from 42, 84 and 168 observations using historical covariance validation. Factor covariance updates monthly. The original bounded covariance still supplies the C02 penalty scale and the estimated-volatility normalization, matching the tested combined strategy.
- **C02 regularization.** Each month the model calculates allocations for the fixed sixteen-point penalty grid. Annual penalty selection uses at most 120 completed outcomes from these previously generated allocations. Each outcome becomes available only on its return date. The selected allocation is scaled to 10% estimated annual volatility under the original covariance.

Historical forecast losses, covariance validation and the historical candidate-payoff calculation remain because they determine model parameters and weights. Full-period evaluation, realized portfolio-performance reporting, charts and research checkpoint orchestration are excluded.

## Verification status

The simplified file passed a real 123-month, 100-stock check with all 402 features and the supplied pandas interface. Every weight matched the previous export exactly. Input reordering, future truncation and future mutation left the relevant earlier weights exactly unchanged.

The SIC correction was checked against all 1,369,324 rows of the supplied monthly table, whose SIC column is stored as strings. Every existing industry classification is unchanged. A complete real 123-month, 100-stock run also produced identical weights when SIC was supplied as floats instead of strings, including input reordering and future truncation checks. Unused state fields and payoff metadata were removed without changing the model calculations.

Earlier 135-month checks covered an annual refit and a midyear cutoff. Removing future data or changing future characteristics, monthly labels and daily returns left all relevant earlier weights exactly unchanged. The numerical layout used for monthly factor summation remains unchanged to preserve these exact results.

A fresh full-period run of the SIC-corrected file from the three current raw tables produced all 885,698 weights across 408 months in 7.1 minutes at ten numerical threads. Every weight matched the accepted selected research run exactly, with a maximum absolute difference of zero. The serialized weights CSV was byte-identical and measured 34,589,479 bytes, below the 50 MB output limit. Peak memory was 6.9 GiB, including the full pandas input tables. The previously accepted source and its earlier verification evidence remain preserved separately.

The earlier real 123-month pandas check also passed with only the ten pinned public runtime packages visible and Ibis absent. This used the installed distributions in an isolated environment, rather than a clean network installation. The synthetic local example passed in that environment. The current source is 33,666 bytes, has six explicit error checks and contains no file, network, shell or dynamic-execution calls.

These local checks do not constitute competition acceptance. The competition applies its own execution and dependency/security checks. The model is published in this GitHub repository; no contest submission has occurred. The research execution entry point remains a separate packaging task in [the experiment's code folder](../experiments/P08_C021_R08V1/code/README.md).
