"""Single-change covariance experiments using the saved baseline risk stream."""
from collections import deque
from datetime import date
import json
from pathlib import Path

import numpy as np
import polars as pl
from threadpoolctl import threadpool_limits

from comparison_models import RISK_SETTINGS, optimize_portfolios, portfolio_variance


# Covariance estimators --------------------------------------------------------
def ledoit_wolf_factor_covariance(values):
    """Canonical Ledoit-Wolf identity-target covariance, with MLE normalization.

    Formula follows sklearn's official ledoit_wolf_shrinkage implementation.
    The fourth-moment sum uses row squared norms, avoiding a second K by K fit.
    Return the matched empirical covariance, shrunk covariance and intensity.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or len(values) < 2 or not values.shape[1] or not np.isfinite(values).all():
        raise ValueError('Covariance needs at least two finite factor-return rows')
    centered = values - values.mean(axis=0)
    n, k = centered.shape
    empirical = centered.T @ centered / n
    trace = float(np.trace(empirical))
    mu = trace / k
    delta = float(np.sum(empirical ** 2) - trace ** 2 / k) / k
    row_square_norm = np.sum(centered ** 2, axis=1)
    beta = float(np.sum(row_square_norm ** 2) / n - np.sum(empirical ** 2)) / (k * n)
    # Only numerical roundoff can make the theoretical nonnegative terms <0.
    alpha = 0. if delta <= 0 else float(np.clip(beta / delta, 0., 1.))
    shrunk = (1. - alpha) * empirical
    shrunk[np.diag_indices(k)] += alpha * mu
    return empirical, (shrunk + shrunk.T) / 2, alpha


def pca_factor_covariance(exposures, covariance, diagonal, retained_variance=.90):
    """Truncate eigenmodes of F; move omitted stock marginal variance into D.

    B keeps its original coordinates. This is a low-rank F approximation, not
    a historical exposure projection or a refit of daily factor regressions.
    """
    if not 0 < retained_variance <= 1:
        raise ValueError('Retained factor variance must be in (0,1]')
    covariance = (covariance + covariance.T) / 2
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    scale = max(float(np.max(np.abs(eigenvalues))), 1e-20)
    if eigenvalues.min() < -1e-9 * scale:
        raise ValueError('Factor covariance is materially indefinite')
    eigenvalues = np.maximum(eigenvalues, 0.)
    order = np.argsort(-eigenvalues, kind='stable')
    eigenvalues = eigenvalues[order]; eigenvectors = eigenvectors[:, order]
    total = float(eigenvalues.sum())
    if total == 0:
        return np.zeros_like(covariance), diagonal.copy(), dict(retained_components=0,
                    retained_factor_variance_fraction=1., discarded_stock_variance_mean=0.)
    positive = int(np.count_nonzero(eigenvalues > 0))
    count = positive if retained_variance == 1 else min(positive,
                    int(np.searchsorted(np.cumsum(eigenvalues), retained_variance * total)) + 1)
    vectors = eigenvectors[:, :count]
    retained = (vectors * eigenvalues[:count]) @ vectors.T
    # Row factor algebra, never N-by-N stock covariance construction.
    discarded = eigenvectors[:, count:]
    omitted = np.sum((exposures @ discarded) ** 2 * eigenvalues[count:], axis=1)
    adjusted_diagonal = diagonal + omitted
    return (retained + retained.T) / 2, adjusted_diagonal, dict(
        retained_components=count, retained_factor_variance_fraction=float(eigenvalues[:count].sum() / total),
        discarded_stock_variance_mean=float(omitted.mean()))


# Saved-history reuse ---------------------------------------------------------
def transform_risk(artifact_dir, variant, *, retained_variance=.90):
    """Yield scored-month B/F/D without refitting or copying parent daily streams.

    Variants: native, uniform_empirical, ledoit_wolf, pca90. Every history includes only
    dates at or before the yielded formation date and at most the parent's
    covariance_days. Latest checkpoint metadata supplies committed dates only;
    its future-fitted factor/state arrays are deliberately never loaded.
    """
    if variant not in ('native', 'uniform_empirical', 'ledoit_wolf', 'pca90'):
        raise ValueError(f'Unknown covariance variant {variant}')
    root = Path(artifact_dir)
    manifest = json.loads((root / 'manifest.json').read_text())
    with np.load(root / 'checkpoint.npz', allow_pickle=False) as checkpoint:
        metadata = json.loads(str(checkpoint['metadata']))
    records = {r['formation_date']: r for r in metadata['records']}
    last_month = date.fromisoformat(metadata['last_month'])
    factor_names = manifest['factor_order']
    factor_history = deque(maxlen=manifest['settings']['covariance_days'])
    directories = sorted(p for p in (root / 'months').iterdir()
                         if p.is_dir() and not p.name.endswith('.tmp') and p.name <= str(last_month))
    for directory in directories:
        d = date.fromisoformat(directory.name)
        marker = json.loads((directory / 'complete.json').read_text())
        if marker['available_through'] != str(d) or marker['factor_order'] != factor_names:
            raise ValueError('Risk month cutoff or factor order differs from its parent')
        if variant in ('uniform_empirical', 'ledoit_wolf'):
            stream = pl.read_parquet(directory / 'factor_returns.parquet').sort('date')
            if stream.columns != ['date'] + factor_names or (stream.height and stream['date'].max() > d):
                raise ValueError('Factor-return stream contains future dates or a different factor basis')
            factor_history.extend(stream.select(factor_names).to_numpy())
        if str(d) not in records:
            continue
        with np.load(directory / 'risk.npz', allow_pickle=False) as saved:
            ids = saved['ids'].copy(); exposures = saved['B'].copy()
            covariance = saved['F'].copy(); diagonal = saved['D'].copy()
            if str(saved['eom']) != str(d):
                raise ValueError('Scored covariance formation date differs from its directory')
        reference_covariance = covariance.copy()
        reference_diagonal = diagonal.copy()
        record = dict(records[str(d)], covariance_variant=variant,
                      covariance_information_cutoff=str(d), shrinkage_intensity=None,
                      covariance_normalization='native separate-half-life unbiased')
        if variant in ('uniform_empirical', 'ledoit_wolf'):
            empirical, shrunk, alpha = ledoit_wolf_factor_covariance(np.array(factor_history))
            covariance = empirical if variant == 'uniform_empirical' else shrunk
            record.update(factor_return_days=len(factor_history),
                          shrinkage_intensity=0. if variant == 'uniform_empirical' else alpha,
                          covariance_normalization='uniform centered MLE denominator n')
        elif variant == 'pca90':
            covariance, diagonal, diagnostics = pca_factor_covariance(exposures, covariance, diagonal, retained_variance)
            record.update(diagnostics, retained_variance_rule=retained_variance,
                          discarded_risk_policy='restore omitted stock marginal variance to D')
        yield dict(ids=ids, eom=d, B=exposures, F=covariance, D=diagonal,
                   reference_F=reference_covariance, reference_D=reference_diagonal, record=record)


# Portfolio construction ------------------------------------------------------
def allocate_variant(artifact_dir, predictions=None, variant='native', *,
                     settings=RISK_SETTINGS, scaling='native', retained_variance=.90):
    """Return native weights or weights scaled to the original model's causal risk.

    Native MVP sums to one; native Markowitz targets 10% using the variant risk.
    Reference scaling targets the configured volatility for both allocations
    under the original monthly B/F/D. Its MVP no longer has net weight one.
    """
    if scaling not in ('native', 'reference'):
        raise ValueError('Scaling must be native or reference')
    minimum_weights = []; markowitz_weights = []; records = []
    with threadpool_limits(limits=settings.threads):
        for risk in transform_risk(artifact_dir, variant, retained_variance=retained_variance):
            ids, d, x = risk['ids'], risk['eom'], risk['B']
            mu = None
            if predictions is not None:
                keys = pl.DataFrame({'id': ids, 'eom': [d] * len(ids)})
                matched = keys.join(predictions.select('id', 'eom', 'pred'),
                                    on=['id', 'eom'], how='left').sort('id')
                if matched.height != len(ids) or matched['pred'].null_count() or not matched['pred'].is_finite().all():
                    raise ValueError('Expected returns missing, duplicate or invalid for risk-model securities')
                mu = matched['pred'].to_numpy()
            minimum, markowitz, _ = optimize_portfolios(ids, d, x, risk['D'], risk['F'], mu, settings)
            record = dict(risk['record'], scaling=scaling)
            for name, frame in (('minimum_variance', minimum), ('markowitz', markowitz)):
                if frame is None:
                    continue
                w = frame['w'].to_numpy()
                original_vol = np.sqrt(252 * portfolio_variance(w, risk['reference_D'], risk['reference_F'], x))
                if not np.isfinite(original_vol) or original_vol <= 0:
                    raise ValueError('Original reference risk is invalid for portfolio scaling')
                if scaling == 'reference':
                    w = w * settings.target_annual_volatility / original_vol
                    frame = frame.with_columns(pl.Series('w', w))
                record[name + '_variant_predicted_annual_volatility'] = float(np.sqrt(
                    252 * portfolio_variance(w, risk['D'], risk['F'], x)))
                record[name + '_reference_predicted_annual_volatility'] = float(np.sqrt(
                    252 * portfolio_variance(w, risk['reference_D'], risk['reference_F'], x)))
                (minimum_weights if name == 'minimum_variance' else markowitz_weights).append(frame)
            records.append(record)
    if not minimum_weights:
        raise ValueError('No committed scored risk months')
    return pl.concat(minimum_weights), pl.concat(markowitz_weights) if markowitz_weights else None, records


def allocate_variant_models(artifact_dir, predictions_dict, variant='native', *,
                            settings=RISK_SETTINGS, scalings=('native', 'reference'),
                            retained_variance=.90):
    """Allocate several forecast streams/scalings in one monthly covariance pass.

    Output keys are minimum_variance_<scaling> and
    markowitz_<prediction_dict_key>_<scaling>. Each record identifies its output
    key, forecast stream, scaling and risk under both variant/reference models.
    Parent streams remain read-only and no additional covariance cache is built.
    """
    if not scalings or len(set(scalings)) != len(scalings) or any(v not in ('native', 'reference') for v in scalings):
        raise ValueError('Distinct native/reference scaling views required')
    weights = {}; records = []
    with threadpool_limits(limits=settings.threads):
        for risk in transform_risk(artifact_dir, variant, retained_variance=retained_variance):
            ids, d, x = risk['ids'], risk['eom'], risk['B']
            keys = pl.DataFrame({'id': ids, 'eom': [d] * len(ids)})
            portfolios = {}
            # Keep original optimizer math. The first stream supplies the common
            # MVP; subsequent MVP calculations are discarded rather than cached.
            for name, predictions in sorted(predictions_dict.items()):
                matched = keys.join(predictions.select('id', 'eom', 'pred'),
                                    on=['id', 'eom'], how='left').sort('id')
                if matched.height != len(ids) or matched['pred'].null_count() or not matched['pred'].is_finite().all():
                    raise ValueError('Expected returns missing, duplicate or invalid for risk-model securities')
                minimum, markowitz, _ = optimize_portfolios(ids, d, x, risk['D'], risk['F'],
                                                           matched['pred'].to_numpy(), settings)
                if 'minimum_variance' not in portfolios:
                    portfolios['minimum_variance'] = (minimum, None)
                portfolios['markowitz_' + name] = (markowitz, name)
            if not portfolios:
                minimum, _, _ = optimize_portfolios(ids, d, x, risk['D'], risk['F'], None, settings)
                portfolios['minimum_variance'] = (minimum, None)
            for portfolio_name, (frame, prediction_name) in portfolios.items():
                original_weights = frame['w'].to_numpy()
                original_vol = float(np.sqrt(252 * portfolio_variance(
                    original_weights, risk['reference_D'], risk['reference_F'], x)))
                if not np.isfinite(original_vol) or original_vol <= 0:
                    raise ValueError('Original reference risk is invalid for portfolio scaling')
                for scaling in scalings:
                    w = original_weights if scaling == 'native' else original_weights * settings.target_annual_volatility / original_vol
                    output_key = portfolio_name + '_' + scaling
                    weights.setdefault(output_key, []).append(frame.with_columns(pl.Series('w', w)))
                    records.append(dict(risk['record'], output_key=output_key,
                        prediction_model=prediction_name, scaling=scaling,
                        variant_predicted_annual_volatility=float(np.sqrt(
                            252 * portfolio_variance(w, risk['D'], risk['F'], x))),
                        reference_predicted_annual_volatility=float(np.sqrt(
                            252 * portfolio_variance(w, risk['reference_D'], risk['reference_F'], x)))))
    if not weights:
        raise ValueError('No committed scored risk months')
    return {key: pl.concat(frames) for key, frames in weights.items()}, records
