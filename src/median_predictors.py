"""T04/T07 forecast-only median fills with unchanged benchmark percentile ranks."""
from hashlib import sha256
import json
from pathlib import Path

import polars as pl

from baseline import META, MonthlySource, as_lazy, prepare_pred_data
from comparison_models import ff12_class


def prepare_median_data(raw, features, group='country'):
    """Replace only missing cells after the existing country/month ranking."""
    if group not in ('country', 'industry'):
        raise ValueError('Median group must be country or industry')
    raw = raw.with_columns(pl.col(features).cast(pl.Float64).fill_nan(None)).sort(['id', 'eom'])
    ranked = prepare_pred_data(raw, features)
    # Restore original missing cells before computing medians. An observed rank
    # of zero remains observed; raw zeros keep the benchmark's -0.5 override.
    ranked = ranked.with_columns([
        pl.when(raw[name].is_null()).then(None).otherwise(pl.col(name)).alias(name)
        for name in features])
    country = ['excntry', 'eom']
    groups = country
    if group == 'industry':
        ranked = ranked.with_columns(pl.Series('__median_industry', ff12_class(raw['sic'].to_list())))
        groups = country + ['__median_industry']
    ranked = ranked.with_columns([
        pl.col(name).fill_null(pl.col(name).median().over(groups)
                              .fill_null(pl.col(name).median().over(country)).fill_null(0.))
        .alias(name) for name in features])
    return ranked.drop('__median_industry') if group == 'industry' else ranked


class MedianSource(MonthlySource):
    """Raw monthly staged reads; inherited stats cache is unique to this source."""
    def __init__(self, chars, features, cache_size=144, *, group='country', prepared_dir=None, identity=None):
        super().__init__(chars, features, cache_size)
        if group not in ('country', 'industry'):
            raise ValueError('Median group must be country or industry')
        self.group = group
        self.sic = as_lazy(chars).select(pl.col('id').cast(pl.Int64), pl.col('eom').cast(pl.Date),
                                        pl.col('sic').cast(pl.String)) if group == 'industry' else None
        self.prepared = Path(prepared_dir) if prepared_dir is not None else None
        if self.prepared is not None:
            if identity is None:
                raise ValueError('Median prepared cache requires explicit input identity')
            self.prepared.mkdir(parents=True, exist_ok=True)
            metadata = dict(group=group, features=features, identity=identity,
                            convention='observed-centered-country-month-ranks-v1',
                            source_sha256={name: sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                           for name in ('median_predictors.py', 'baseline.py', 'comparison_models.py')})
            metadata = json.loads(json.dumps(metadata, sort_keys=True))
            marker = self.prepared / 'identity.json'
            if marker.exists() and json.loads(marker.read_text()) != metadata:
                raise ValueError('Median prepared-cache identity mismatch; use a separate path')
            temporary = marker.with_name(marker.name + '.tmp')
            temporary.write_text(json.dumps(metadata, sort_keys=True) + '\n')
            temporary.replace(marker)

    def load(self, formation_date):
        path = self.prepared / (str(formation_date) + '.parquet') if self.prepared is not None else None
        if path is not None and path.exists():
            return pl.read_parquet(path)
        raw = self.load_raw(formation_date)
        if self.sic is not None:
            sic = self.sic.filter(pl.col('eom') == formation_date).collect(engine='streaming')
            raw = raw.join(sic, on=['id', 'eom'])
        prepared = prepare_median_data(raw, self.features, self.group).select(META + self.features)
        if path is not None:
            temporary = path.with_name(path.name + '.tmp')
            prepared.write_parquet(temporary, compression='zstd')
            temporary.replace(path)
        return prepared


def median_source_factory(group='country', *, prepared_dir=None, identity=None):
    """Use with original baseline.run_pipeline or comparison.forecast_returns."""
    def source_factory(chars, features, cache_size=144):
        return MedianSource(chars, features, cache_size, group=group,
                            prepared_dir=prepared_dir, identity=identity)
    return source_factory
