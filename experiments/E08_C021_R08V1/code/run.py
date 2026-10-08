"""Equal-thirds forecasts with regularized allocation and expanding variance. Local research, including performance evaluation."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'src'))
from experiment_io import arguments


def main(args):
    from dataclasses import replace
    from baseline import MonthlySource, canonical_features
    from comparison_models import FORECAST_SETTINGS
    from artifact_utils import write_json
    from experiment_io import (load_inputs, save_portfolio,
                               save_results, build_original_risk)

    chars, features, daily, identity = load_inputs(args.data)
    names = canonical_features(features)
    metadata = MonthlySource(chars, names).metadata
    risk = args.output / 'risk'
    settings = build_original_risk(chars, features, daily, risk, args.threads, identity)
    from factor_forecast_paths import forecast_family
    from risk_history_regimes import iter_history_risk
    from combined_allocations import allocate_combination, stock_forecasts
    import polars as pl
    ridge, xgboost, fits = stock_forecasts(chars, features, args.threads)
    factor, factor_fits = forecast_family(chars, names, 'P08', risk,
        args.output / 'factor-forecasts', identity, replace(FORECAST_SETTINGS, threads=args.threads))
    keys = ['id', 'eom', 'excntry']
    joined = ridge.join(xgboost, on=keys, validate='1:1', suffix='_xgb').join(
        factor, on=keys, validate='1:1', suffix='_factor')
    prediction = joined.select(*keys, ((pl.col('pred') + pl.col('pred_xgb') + pl.col('pred_factor')) / 3).alias('pred'))
    rows = iter_history_risk(risk, 'R08_EXPANDING_VARIANCE', threads=args.threads,
        selection_cache=args.output / 'selection', identity=identity)
    weights, allocation, bank = allocate_combination(rows, prediction, metadata, settings)
    statistics = {'markowitz_ml': save_portfolio(args.output, 'markowitz_ml', weights, prediction, metadata)}
    write_json(args.output / 'allocation_records.json', allocation)
    write_json(args.output / 'candidate_bank.json', bank)
    save_results(args.output, 'E08_C021_R08V1', statistics, dict(fits, factor=factor_fits))


if __name__ == '__main__':
    main(arguments('Equal-thirds forecasts with regularized allocation and expanding variance'))
