"""Factor covariance shrinkage. Local research, including performance evaluation."""
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
    from batch_risk import allocate_variant_models
    ridge, xgboost, fits = fit_stock_forecasts(chars, features, args.threads)
    predictions = {'ridge': ridge, 'xgboost': xgboost}
    statistics = {}
    records = {}
    for variant in ('native', 'uniform_empirical', 'ledoit_wolf'):
        streams, records[variant] = allocate_variant_models(risk, predictions, variant, settings=settings)
        for name, weights in streams.items():
            if variant == 'native' and name.endswith('_native'):
                continue
            forecast = next((key for key in predictions if name.startswith('markowitz_' + key + '_')), None)
            label = variant + '_' + name
            statistics[label] = save_portfolio(args.output, label, weights, predictions.get(forecast), metadata)
    write_json(args.output / 'allocation_records.json', records)
    save_results(args.output, 'R03', statistics, fits)


if __name__ == '__main__':
    main(arguments('Factor covariance shrinkage'))
