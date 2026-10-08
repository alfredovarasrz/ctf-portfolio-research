"""Factor-return forecasts, expanding variance and regularized allocation."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))
from experiment_io import arguments


def main(args):
    import polars as pl
    from baseline import MonthlySource, canonical_features
    from experiment_io import load_inputs, save_portfolio, save_results

    chars, features, daily, _ = load_inputs(args.data)
    sys.path.insert(0, str(ROOT / 'submission'))
    import model
    from dataclasses import replace
    model.RISK_SETTINGS = replace(model.RISK_SETTINGS, threads=args.threads)
    weights = pl.from_pandas(model.main(chars.collect().to_pandas(), features.to_pandas(),
                                      daily.collect().to_pandas()))
    metadata = MonthlySource(chars, canonical_features(features)).metadata
    statistics = save_portfolio(args.output, 'markowitz_ml', weights, None, metadata)
    save_results(args.output, 'P08_C021_R08V1', {'markowitz_ml': statistics}, {})


if __name__ == '__main__':
    main(arguments('Selected combined portfolio'))
