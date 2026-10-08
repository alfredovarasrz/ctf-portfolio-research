"""Forward-only historical forecast validation. Local research, including performance evaluation."""
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
    from baseline import DEFAULT_SETTINGS
    from comparison_models import FORECAST_SETTINGS
    from research_forecasts import forward_ridge_returns, forward_forecast_returns
    ridge, ridge_fits = forward_ridge_returns(chars, features, replace(DEFAULT_SETTINGS, blas_threads=args.threads))
    xgboost, xgboost_fits = forward_forecast_returns(chars, features, replace(FORECAST_SETTINGS, threads=args.threads))
    fits = {'ridge': ridge_fits, 'xgboost': xgboost_fits}
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

    save_results(args.output, 'V01', statistics, fits)


if __name__ == '__main__':
    main(arguments('Forward-only historical forecast validation'))
