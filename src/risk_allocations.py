"""Benchmark allocation and risk scaling for changed covariance estimates."""
from datetime import date

import numpy as np
import polars as pl
from threadpoolctl import threadpool_limits

from comparison_models import optimize_portfolios, portfolio_variance


def allocate_risk_rows(risks, predictions, settings):
    """Keep net-one minimum variance and both Markowitz risk-scaling views."""
    names = ('minimum_variance_native', 'markowitz_xgboost_native',
             'markowitz_xgboost_reference')
    streams = {name: [] for name in names}
    records = []
    with threadpool_limits(limits=settings.threads):
        for risk in risks:
            ids, formation, exposures = risk['ids'], risk['eom'], risk['B']
            record = risk['record']
            if date.fromisoformat(record.get('basis_information_cutoff', str(formation))) > formation:
                raise ValueError('Exposure basis uses future information')
            keys = pl.DataFrame({'id': ids, 'eom': [formation] * len(ids)})
            matched = keys.join(predictions.select('id', 'eom', 'pred'),
                                on=['id', 'eom'], how='left', validate='1:1').sort('id')
            if matched['pred'].null_count() or not matched['pred'].is_finite().all():
                raise ValueError('Expected returns missing for risk-model stocks')
            minimum, markowitz, _ = optimize_portfolios(
                ids, formation, exposures, risk['D'], risk['F'],
                matched['pred'].to_numpy(), settings)
            original_vol = float(np.sqrt(252 * portfolio_variance(
                markowitz['w'].to_numpy(), risk['reference_D'], risk['reference_F'],
                risk.get('reference_B', exposures))))
            reference = markowitz.with_columns(
                (pl.col('w') * settings.target_annual_volatility / original_vol).alias('w'))
            for name, frame in zip(names, (minimum, markowitz, reference)):
                streams[name].append(frame)
                records.append(dict(record, output_key=name,
                    scaling='reference' if name.endswith('_reference') else 'native'))
    return {name: pl.concat(rows).sort('eom', 'id') for name, rows in streams.items()}, records


def read_rebuilt_risk(original, derived):
    """Read a newly rebuilt exposure/penalty model with the original risk reference."""
    import json
    from pathlib import Path
    original, derived = Path(original), Path(derived)
    for record in json.loads((derived / 'records.json').read_text()):
        formation = date.fromisoformat(record['formation_date'])
        with np.load(derived / 'months' / str(formation) / 'risk.npz', allow_pickle=False) as data:
            values = {name: data[name].copy() for name in ('ids', 'B', 'F', 'D')}
        with np.load(original / 'months' / str(formation) / 'risk.npz', allow_pickle=False) as data:
            reference = {'reference_' + name: data[name].copy() for name in ('B', 'F', 'D')}
        yield dict(**values, **reference, eom=formation,
                   record=dict(record, basis_information_cutoff=str(formation)))
