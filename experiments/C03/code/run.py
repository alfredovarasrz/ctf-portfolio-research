"""Market-trend exposure budget. Local research, including performance evaluation."""
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
    from extended_allocations import time_weights
    from research_risk import allocate_from_risk_artifacts
    prediction, fits = fit_xgboost_forecast(chars, features, args.threads)
    import polars as pl
    from experiment_io import cap_weighted_proxy
    proxy = cap_weighted_proxy(chars, risk, metadata.filter(pl.col('ctff_test'))['eom'].max())

    _, original, _ = allocate_from_risk_artifacts(risk, prediction, settings=settings)
    weights, allocation = time_weights(original, proxy, risk, settings=settings)
    statistics = {'markowitz_ml_timed': save_portfolio(args.output, 'markowitz_ml_timed', weights, prediction, metadata)}
    write_json(args.output / 'allocation_records.json', allocation)
    save_results(args.output, 'C03', statistics, fits)


if __name__ == '__main__':
    main(arguments('Market-trend exposure budget'))
