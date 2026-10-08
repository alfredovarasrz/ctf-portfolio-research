"""Markowitz industry sleeves. Local research, including performance evaluation."""
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
    from extended_allocations import allocate_sleeves
    prediction, fits = fit_xgboost_forecast(chars, features, args.threads)
    streams, records = allocate_sleeves(risk, prediction, modes=('markowitz',), settings=settings)
    label = 'markowitz' + '_sleeves'
    statistics = {label: save_portfolio(args.output, label, streams['markowitz'], prediction, metadata)}
    write_json(args.output / 'allocation_records.json', records)
    save_results(args.output, 'C06', statistics, fits)


if __name__ == '__main__':
    main(arguments('Markowitz industry sleeves'))
