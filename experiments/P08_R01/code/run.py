"""Factor forecasts and risk with a tuned daily Ridge penalty. Local research, including performance evaluation."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'src'))
from experiment_io import arguments


def main(args):
    from dataclasses import replace
    from baseline import MonthlySource, canonical_features
    from comparison_models import FORECAST_SETTINGS
    from artifact_utils import write_json
    from experiment_io import load_inputs, save_portfolio, save_results, build_original_risk

    chars, features, daily, identity = load_inputs(args.data)
    names = canonical_features(features)
    metadata = MonthlySource(chars, names).metadata
    risk = args.output / 'risk'
    settings = build_original_risk(chars, features, daily, risk, args.threads, identity)
    from risk_penalty_replay import build_factor_penalty_risk, iter_penalty_risk
    from tuned_factor_forecast_paths import forecast_family
    from risk_allocations import allocate_risk_rows
    derived = args.output / 'penalty-risk'
    build_factor_penalty_risk(risk, derived, settings=settings, identity=identity)
    prediction, fits = forecast_family(chars, names, 'P08_R01', derived,
        args.output / 'forecasts', identity, replace(FORECAST_SETTINGS, threads=args.threads))
    streams, records = allocate_risk_rows(iter_penalty_risk(risk, derived), prediction, settings)
    statistics = {}
    for old, name in {'markowitz_xgboost_native':'markowitz_ml',
                      'markowitz_xgboost_reference':'markowitz_ml_reference'}.items():
        statistics[name] = save_portfolio(args.output, name, streams[old], prediction, metadata)
    write_json(args.output / 'allocation_records.json', records)
    save_results(args.output, 'P08_R01', statistics, fits)


if __name__ == '__main__':
    main(arguments('Factor forecasts and risk with a tuned daily Ridge penalty'))
