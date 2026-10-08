"""Expanding history with inherited tree settings. Local research, including performance evaluation."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'src'))
from experiment_io import arguments


def main(args):
    from dataclasses import replace
    from baseline import MonthlySource, canonical_features
    from comparison_models import FORECAST_SETTINGS
    from artifact_utils import write_json
    from experiment_io import load_inputs, fit_xgboost_forecast, save_results, build_original_risk

    chars, features, daily, identity = load_inputs(args.data)
    names = canonical_features(features)
    metadata = MonthlySource(chars, names).metadata
    risk = args.output / 'risk'
    settings = build_original_risk(chars, features, daily, risk, args.threads, identity)
    from inherited_expanding_forecasts import parent_selection_contract
    _, parent_fits = fit_xgboost_forecast(chars, features, args.threads)
    parent = args.output / 'parent'
    parent.mkdir(exist_ok=True)
    write_json(parent / 'forecast_fits.json', parent_fits)
    forecast_settings = replace(FORECAST_SETTINGS, threads=args.threads)
    selections = parent_selection_contract(parent, forecast_settings)
    from inherited_expanding_forecasts import inherited_expanding_forecast_returns
    from experiment_io import save_forecast_portfolios
    prediction, fits, _ = inherited_expanding_forecast_returns(chars, features, 'P04_INHERITED',
        forecast_settings, parent_selections=selections,
        checkpoint_dir=args.output / 'forecasts', identity=identity)
    statistics, allocation = save_forecast_portfolios(args.output, prediction, metadata, risk, settings)
    write_json(args.output / 'allocation_records.json', allocation)
    save_results(args.output, 'P04_INHERITED', statistics, fits)


if __name__ == '__main__':
    main(arguments('Expanding history with inherited tree settings'))
