"""Wide historical allocation-penalty search. Local research, including performance evaluation."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'src'))
from experiment_io import arguments


def main(args):
    from baseline import MonthlySource, canonical_features
    from artifact_utils import write_json
    from experiment_io import (load_inputs, fit_stock_forecasts, save_portfolio,
                               save_results, build_original_risk)

    chars, features, daily, identity = load_inputs(args.data)
    names = canonical_features(features)
    metadata = MonthlySource(chars, names).metadata
    risk = args.output / 'risk'
    settings = build_original_risk(chars, features, daily, risk, args.threads, identity)
    from wide_allocation import allocate_wide
    ridge, xgboost, fits = fit_stock_forecasts(chars, features, args.threads)
    predictions = {'ridge': ridge, 'xgboost': xgboost}
    streams, allocation = allocate_wide(risk, predictions, metadata,
        cache_dir=args.output / 'allocations', identity=identity, settings=settings)
    statistics = {}
    for name, weights in streams.items():
        forecast = ridge if name.startswith('ridge') else xgboost if name.startswith('xgboost') else None
        statistics[name] = save_portfolio(args.output, name, weights, forecast, metadata)
    write_json(args.output / 'allocation_records.json', allocation)
    save_results(args.output, 'C02_WIDE', statistics, fits)


if __name__ == '__main__':
    main(arguments('Wide historical allocation-penalty search'))
