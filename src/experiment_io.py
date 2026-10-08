"""Input and output helpers for local research scripts."""
import argparse
import os
from pathlib import Path


def arguments(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--data', type=Path, required=True, help='Folder containing the three CTF Parquet tables')
    parser.add_argument('--output', type=Path, required=True, help='Folder for newly generated research outputs')
    parser.add_argument('--threads', type=int, choices=range(1, 11), default=10)
    args = parser.parse_args()
    for name in ('POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        os.environ[name] = str(args.threads)
    args.output.mkdir(parents=True, exist_ok=True)
    return args


def load_inputs(folder):
    import polars as pl
    from artifact_utils import digest
    filenames = ('ctff_chars.parquet', 'ctff_features.parquet', 'ctff_daily_ret.parquet')
    identity = {name: digest(folder / name) for name in filenames}
    print('Reading supplied characteristics, feature list and daily returns', flush=True)
    return (pl.scan_parquet(folder / filenames[0]), pl.read_parquet(folder / filenames[1]),
            pl.scan_parquet(folder / filenames[2]), identity)


def fit_stock_forecasts(chars, features, threads, source_factory=None):
    from dataclasses import replace
    from baseline import DEFAULT_SETTINGS, MonthlySource
    from comparison_models import FORECAST_SETTINGS
    from research_forecasts import benchmark_forecasts
    source_factory = source_factory or MonthlySource
    ridge, ridge_fits = benchmark_forecasts(chars, features,
        replace(DEFAULT_SETTINGS, blas_threads=threads), source_factory, kind='ridge')
    xgboost, xgboost_fits = benchmark_forecasts(chars, features,
        replace(FORECAST_SETTINGS, threads=threads), source_factory, kind='xgboost')
    return ridge, xgboost, {'ridge': ridge_fits, 'xgboost': xgboost_fits}


def save_portfolio(folder, name, weights, predictions, metadata):
    from evaluate_baseline import evaluate
    monthly, statistics = evaluate(weights, predictions, metadata)
    weights.sort('eom', 'id').write_csv(folder / f'{name}-weights.csv')
    monthly.write_csv(folder / f'{name}-returns.csv')
    print(f"{name}: {statistics['months']} months, Sharpe {statistics['sharpe_annual']:.4f}", flush=True)
    return statistics


def save_results(folder, experiment, statistics, fits):
    from artifact_utils import write_json
    write_json(folder / 'forecast_fits.json', fits)
    write_json(folder / 'summary.json', {'experiment': experiment, 'portfolios': statistics,
                                       'trading_costs_included': False})
