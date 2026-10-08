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


def fit_xgboost_forecast(chars, features, threads):
    from dataclasses import replace
    from comparison_models import FORECAST_SETTINGS
    from research_forecasts import benchmark_forecasts
    prediction, fits = benchmark_forecasts(chars, features,
        replace(FORECAST_SETTINGS, threads=threads), kind='xgboost')
    return prediction, {'xgboost': fits}


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


def build_original_risk(chars, features, daily, folder, threads, identity):
    from dataclasses import replace
    from comparison_models import RISK_SETTINGS
    from research_risk import build_risk_artifacts
    settings = replace(RISK_SETTINGS, threads=threads)
    print('Estimating daily factors and stock-specific risk', flush=True)
    build_risk_artifacts(chars, features, daily, folder, settings=settings, identity=identity)
    return settings


def save_forecast_portfolios(folder, predictions, metadata, risk, settings):
    from baseline import predictions_to_weights
    from research_risk import allocate_from_risk_artifacts
    _, markowitz, records = allocate_from_risk_artifacts(risk, predictions, settings=settings)
    statistics = {
        'factor_ml': save_portfolio(folder, 'factor_ml', predictions_to_weights(predictions),
                                    predictions, metadata),
        'markowitz_ml': save_portfolio(folder, 'markowitz_ml', markowitz, predictions, metadata)}
    return statistics, records


def cap_weighted_proxy(chars, risk, cutoff):
    """Use preceding raw market equity and the observed daily regression universe."""
    import json
    import polars as pl
    from market_cap_proxy import raw_caps, build_proxy
    dates = []
    for folder in sorted((risk / 'months').iterdir()):
        if folder.is_dir() and not folder.name.endswith('.tmp') and folder.name <= str(cutoff):
            preceding = json.loads((folder / 'complete.json').read_text()).get('preceding_exposure_date')
            if preceding is not None:
                dates.append(preceding)
    caps, _ = raw_caps(chars.with_columns(pl.col('eom').cast(pl.Date)), cutoff, dates)
    proxy, _, _ = build_proxy(caps, risk, cutoff)
    return proxy
