"""Joint factor-return XGBoost forecasts. Local research, including performance evaluation."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'src'))
from experiment_io import arguments


def main(args):
    from dataclasses import replace
    from baseline import MonthlySource, canonical_features
    from comparison_models import FORECAST_SETTINGS
    from artifact_utils import write_json
    from experiment_io import (load_inputs, fit_xgboost_forecast, save_portfolio,
                               save_results, build_original_risk)

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
    from factor_xgb_paths import forecast_xgb_family
    from research_risk import allocate_from_risk_artifacts
    prediction, fits = forecast_xgb_family(chars, names, risk, args.output / 'forecasts',
        identity, selections, forecast_settings, maximum_rounds=1000)
    _, weights, allocation = allocate_from_risk_artifacts(risk, prediction, settings=settings)
    statistics = {'markowitz_ml': save_portfolio(args.output, 'markowitz_ml', weights, prediction, metadata)}
    write_json(args.output / 'allocation_records.json', allocation)
    save_results(args.output, 'P09_FACTOR_XGB', statistics, fits)


if __name__ == '__main__':
    main(arguments('Joint factor-return XGBoost forecasts'))
