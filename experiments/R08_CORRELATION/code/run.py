"""Historically selected correlation age weights. Local research, including performance evaluation."""
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

    from tuned_risk_decay import transform_tuned_decay
    rows = transform_tuned_decay(risk, 'correlation', threads=args.threads)

    streams, records = allocate_risk_rows(rows, prediction, settings)
    statistics = {name: save_portfolio(args.output, name, weights,
        prediction if name.startswith('markowitz') else None, metadata)
        for name, weights in streams.items()}
    write_json(args.output / 'allocation_records.json', records)
    save_results(args.output, 'R08_CORRELATION', statistics, fits)


if __name__ == '__main__':
    main(arguments('Historically selected correlation age weights'))
