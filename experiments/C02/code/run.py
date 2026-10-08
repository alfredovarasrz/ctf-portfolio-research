"""Historically selected weight regularization. Local research, including performance evaluation."""
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
    ridge, xgboost, fits = fit_stock_forecasts(chars, features, args.threads)
    statistics = {}
    records = {}
    for name, prediction in {'ridge': ridge, 'xgboost': xgboost}.items():
        from batch_allocations import allocate_penalized
        minimum, markowitz, allocation, bank = allocate_penalized(risk, prediction, metadata, settings=settings)
        label = name + '_penalized_markowitz'
        statistics[label] = save_portfolio(args.output, label, markowitz, prediction, metadata)
        if name == 'ridge':
            statistics['penalized_minimum_variance'] = save_portfolio(
                args.output, 'penalized_minimum_variance', minimum, None, metadata)
        bank.write_parquet(args.output / (name + '-candidate-bank.parquet'))
        records[name] = allocation
    write_json(args.output / 'allocation_records.json', records)
    save_results(args.output, 'C02', statistics, fits)


if __name__ == '__main__':
    main(arguments('Historically selected weight regularization'))
