"""Drawdown-conditioned factor covariance. Local research, including performance evaluation."""
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
    import polars as pl
    from experiment_io import cap_weighted_proxy
    proxy = cap_weighted_proxy(chars, risk, metadata.filter(pl.col('ctff_test'))['eom'].max())

    from risk_history_regimes import iter_history_risk
    rows = iter_history_risk(risk, 'R10_DD20', proxy=proxy, threads=args.threads, identity=identity)

    streams, records = allocate_risk_rows(rows, prediction, settings)
    statistics = {name: save_portfolio(args.output, name, weights,
        prediction if name.startswith('markowitz') else None, metadata)
        for name, weights in streams.items()}
    write_json(args.output / 'allocation_records.json', records)
    save_results(args.output, 'R10_DD20', statistics, fits)


if __name__ == '__main__':
    main(arguments('Drawdown-conditioned factor covariance'))
