"""Industry-median risk characteristics. Local research, including performance evaluation."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'src'))
from experiment_io import arguments


def main(args):
    from baseline import MonthlySource, canonical_features
    from artifact_utils import write_json
    from experiment_io import (load_inputs, fit_xgboost_forecast, save_portfolio,
                               save_results, build_original_risk)

    chars, features, daily, identity = load_inputs(args.data)
    names = canonical_features(features)
    metadata = MonthlySource(chars, names).metadata
    risk = args.output / 'risk'
    settings = build_original_risk(chars, features, daily, risk, args.threads, identity)
    from risk_allocations import allocate_risk_rows
    prediction, fits = fit_xgboost_forecast(chars, features, args.threads)

    from risk_preprocessing import build_preprocessed_risk
    from risk_allocations import read_rebuilt_risk
    derived = args.output / 'changed-risk'
    build_preprocessed_risk(chars, features, daily, derived, 'industry_median', settings=settings, identity=identity)
    rows = read_rebuilt_risk(risk, derived)

    streams, records = allocate_risk_rows(rows, prediction, settings)
    statistics = {name: save_portfolio(args.output, name, weights,
        prediction if name.startswith('markowitz') else None, metadata)
        for name, weights in streams.items()}
    write_json(args.output / 'allocation_records.json', records)
    save_results(args.output, 'T07_RISK', statistics, fits)


if __name__ == '__main__':
    main(arguments('Industry-median risk characteristics'))
