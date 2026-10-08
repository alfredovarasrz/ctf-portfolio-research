"""Age-weighted stock-return training. Local research, including performance evaluation."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'src'))
from experiment_io import arguments


def main(args):
    from dataclasses import replace
    from baseline import MonthlySource, canonical_features
    from comparison_models import FORECAST_SETTINGS
    from artifact_utils import write_json
    from experiment_io import load_inputs, save_results, build_original_risk

    chars, features, daily, identity = load_inputs(args.data)
    names = canonical_features(features)
    metadata = MonthlySource(chars, names).metadata
    risk = args.output / 'risk'
    settings = build_original_risk(chars, features, daily, risk, args.threads, identity)
    from extended_forecasts import extended_forecast_returns
    from experiment_io import save_forecast_portfolios
    prediction, fits, _ = extended_forecast_returns(chars, features, 'P05',
        replace(FORECAST_SETTINGS, threads=args.threads), baseline_fits=None,
        checkpoint_dir=args.output / 'forecasts', identity=identity)
    statistics, allocation = save_forecast_portfolios(args.output, prediction, metadata, risk, settings)
    write_json(args.output / 'allocation_records.json', allocation)
    save_results(args.output, 'P05', statistics, fits)


if __name__ == '__main__':
    main(arguments('Age-weighted stock-return training'))
