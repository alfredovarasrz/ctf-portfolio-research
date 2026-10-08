"""Country median filling of missing predictors. Local research, including performance evaluation."""
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
    from median_predictors import median_source_factory
    ridge, xgboost, fits = fit_stock_forecasts(chars, features, args.threads, median_source_factory('country'))
    print('Estimating daily factors and stock-specific risk', flush=True)
    risk = args.output / 'risk'
    build_risk_artifacts(chars, features, daily, risk, settings=settings, identity=identity)
    forecasts = {'ridge': ridge, 'xgboost': xgboost}

    statistics = {}
    for name, prediction in forecasts.items():
        deciles = predictions_to_weights(prediction)
        _, markowitz, _ = allocate_from_risk_artifacts(risk, prediction, settings=settings)
        statistics[name + '_deciles'] = save_portfolio(args.output, name + '_deciles', deciles, prediction, metadata)
        statistics[name + '_markowitz'] = save_portfolio(args.output, name + '_markowitz', markowitz, prediction, metadata)
    
    save_results(args.output, 'T04', statistics, fits)


if __name__ == '__main__':
    main(arguments('Country median filling of missing predictors'))
