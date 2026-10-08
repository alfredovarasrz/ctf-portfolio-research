"""Percentile ranking of return forecasts. Local research, including performance evaluation."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'src'))
from experiment_io import arguments


def main(args):
    from dataclasses import replace
    from baseline import MonthlySource, canonical_features, predictions_to_weights
    from comparison_models import RISK_SETTINGS
    from experiment_io import load_inputs, fit_stock_forecasts, save_portfolio, save_results
    from research_risk import build_risk_artifacts, allocate_from_risk_artifacts

    chars, features, daily, identity = load_inputs(args.data)
    metadata = MonthlySource(chars, canonical_features(features)).metadata
    settings = replace(RISK_SETTINGS, threads=args.threads)
    ridge, xgboost, fits = fit_stock_forecasts(chars, features, args.threads)
    print('Estimating daily factors and stock-specific risk', flush=True)
    risk = args.output / 'risk'
    build_risk_artifacts(chars, features, daily, risk, settings=settings, identity=identity)
    import polars as pl
    forecasts = {name + '_rank': prediction.with_columns(
        (pl.col('pred').rank(method='max').over('excntry', 'eom') /
         pl.len().over('excntry', 'eom') - .5).alias('pred'))
        for name, prediction in {'ridge': ridge, 'xgboost': xgboost}.items()}

    statistics = {}
    for name, prediction in forecasts.items():
        deciles = predictions_to_weights(prediction)
        _, markowitz, _ = allocate_from_risk_artifacts(risk, prediction, settings=settings)
        statistics[name + '_deciles'] = save_portfolio(args.output, name + '_deciles', deciles, None, metadata)
        statistics[name + '_markowitz'] = save_portfolio(args.output, name + '_markowitz', markowitz, None, metadata)

    save_results(args.output, 'T03', statistics, fits)


if __name__ == '__main__':
    main(arguments('Percentile ranking of return forecasts'))
