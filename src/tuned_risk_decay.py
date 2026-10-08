"""R08: one historically selected p1 hyperbolic risk-decay component at a time."""
from collections import deque
from datetime import date
import hashlib

import numpy as np
from threadpoolctl import threadpool_limits

from comparison_models import RISK_SETTINGS, factor_covariance
from extended_risk import _parent_months, _factor_stream, covariance_with_weights


def decay_factor_covariance(values, ages, settings, component, half_weight=None):
    """Fit covariance on supplied training rows while retaining calendar-age gaps.

    half_weight=None is the original exponential control. The named component
    alone uses p1 hyperbolic 1/(1+age/h), whose unnormalized weight halves at h.
    """
    values = np.asarray(values, dtype=float); ages = np.asarray(ages, dtype=float)
    if component not in ('correlation', 'variance') or ages.shape != (len(values),) or np.any(ages <= 0):
        raise ValueError('Named component and positive matching ages required')
    if half_weight is not None and half_weight <= 0:
        raise ValueError('Positive hyperbolic half-weight age required')
    covariances = {}
    for name, original in (('correlation', settings.correlation_half_life), ('variance', settings.variance_half_life)):
        weights = (1. / (1. + ages / half_weight) if name == component and half_weight is not None
                   else .5 ** (ages / original))
        covariances[name] = covariance_with_weights(values, weights)
    sd = np.sqrt(np.maximum(np.diag(covariances['correlation']), 0.))
    outer = np.outer(sd, sd)
    correlation = np.divide(covariances['correlation'], outer,
                            out=np.zeros_like(outer), where=outer > 0)
    variance_sd = np.sqrt(np.maximum(np.diag(covariances['variance']), 0.))
    covariance = correlation * np.outer(variance_sd, variance_sd)
    return (covariance + covariance.T) / 2


def select_decay_rate(values, dates, settings=RISK_SETTINGS, *, component, candidates=None):
    """Five completed historical calendar blocks: fit four, validate one.

    Hyperparameters minimize mean squared factor-covariance error, never test
    Sharpe. All supplied rows must be known at the actual outer formation.
    Training ages are positions in the FULL original bounded window, including
    held-out gaps, anchored at that outer formation. Final refit uses every row.
    """
    values = np.asarray(values, dtype=float)
    dates = list(dates)
    if component not in ('correlation', 'variance') or len(dates) != len(values):
        raise ValueError('Named component and aligned historical dates required')
    original = settings.correlation_half_life if component == 'correlation' else settings.variance_half_life
    candidates = sorted(set(map(float, candidates if candidates is not None else (.5 * original, original, 2 * original))))
    if not candidates or candidates[0] <= 0 or not np.isfinite(candidates).all():
        raise ValueError('Finite positive decay candidates required')
    if len(dates) != len(set(dates)) or dates != sorted(dates):
        raise ValueError('Historical dates must be unique and chronological')
    months = sorted(set((d.year, d.month) for d in dates))
    record = dict(component=component, hyperbolic_power=1., candidate_half_weight_days=candidates,
        validation_policy='five contiguous historical calendar blocks; four train and one validate',
        validation_target='uniform centered unbiased held-out factor covariance, denominator n-1',
        validation_loss='mean squared factor covariance entries; equal fold weight',
        weighting_age_policy='full historical trading-observation ages; held-out gaps retained',
        tuning_months=len(months), tuning_days=len(values), selected_half_weight_days=None,
        selection_fallback=None, candidates=[], exponential_control_folds=[])
    # An insufficient tiny validation sample stays on the original estimator.
    if len(months) < 5 or len(values) < 10:
        record['selection_fallback'] = 'original exponential; insufficient five-block history'
        return None, record
    blocks = np.array_split(np.arange(len(months)), 5)
    month_index = {m: i for i, m in enumerate(months)}
    row_month = np.array([month_index[(d.year, d.month)] for d in dates])
    ages = np.arange(len(values), 0, -1, dtype=float)
    folds = []
    for block in blocks:
        validation = np.isin(row_month, block); training = ~validation
        if validation.sum() < 2 or training.sum() < 2:
            record['selection_fallback'] = 'original exponential; insufficient observations in a calendar fold'
            return None, record
        held_out = values[validation]
        centered = held_out - held_out.mean(axis=0)
        target = centered.T @ centered / (len(held_out) - 1)
        validation_dates = [d for d, selected in zip(dates, validation) if selected]
        training_dates = [d for d, selected in zip(dates, training) if selected]
        folds.append((training, target, dict(training_days=int(training.sum()), validation_days=int(validation.sum()),
            validation_start=str(validation_dates[0]), validation_end=str(validation_dates[-1]),
            training_first_day=str(training_dates[0]), training_last_day=str(training_dates[-1]))))
    for candidate in [None] + candidates:
        scores = []
        for training, target, metadata in folds:
            fitted = decay_factor_covariance(values[training], ages[training], settings, component, candidate)
            scores.append(dict(metadata, mean_squared_covariance_error=float(np.mean((fitted - target) ** 2))))
        if candidate is None:
            record['exponential_control_folds'] = scores
        else:
            record['candidates'].append(dict(half_weight_days=candidate, folds=scores,
                mean_validation_loss=float(np.mean([r['mean_squared_covariance_error'] for r in scores]))))
    # Exact score ties choose the lower predeclared half-weight age.
    winner = min(record['candidates'], key=lambda candidate: (candidate['mean_validation_loss'], candidate['half_weight_days']))
    record['selected_half_weight_days'] = winner['half_weight_days']
    return winner['half_weight_days'], record


def transform_tuned_decay(artifact_dir, component, *, cutoff=None, candidates=None, threads=None):
    """Yield original B/D/mu-compatible risk streams, with annual selected F decay.

    Correlation and variance are DISTINCT experiments. Expanding joint variants
    are not silently enabled. The same original latest covariance_days stream
    supplies tuning and each month-end refit. No persistent future-fitted state.
    """
    if component not in ('correlation', 'variance'):
        raise ValueError('Decay component must be correlation or variance')
    root, manifest, metadata, directories = _parent_months(artifact_dir, cutoff)
    parent_sha = hashlib.sha256((root / 'manifest.json').read_bytes()).hexdigest()
    settings = type(RISK_SETTINGS)(**manifest['settings'])
    execution_threads = settings.threads if threads is None else threads
    if not isinstance(execution_threads, int) or execution_threads < 1:
        raise ValueError('Execution thread override must be a positive integer')
    records = {r['formation_date']: r for r in metadata['records']}
    names = manifest['factor_order']; history = deque(maxlen=settings.covariance_days)
    dates = deque(maxlen=settings.covariance_days); selections = {}; last_date = None
    with threadpool_limits(limits=execution_threads):
        for directory in directories:
            d = date.fromisoformat(directory.name)
            stream, _ = _factor_stream(directory, names, d)
            if stream.height:
                if last_date is not None and stream['date'].min() <= last_date:
                    raise ValueError('Factor dates overlap or move backward')
                last_date = stream['date'].max()
                history.extend(stream.select(names).to_numpy()); dates.extend(stream['date'].to_list())
            if str(d) not in records:
                continue
            year = d.year + int(d.month == 12)
            values = np.array(history)
            if year not in selections:
                half_weight, selection = select_decay_rate(values, dates, settings, component=component, candidates=candidates)
                selection.update(tuning_information_cutoff=str(d), tuning_last_factor_day=str(last_date), forecast_year=year)
                selections[year] = (half_weight, selection)
            half_weight, selection = selections[year]
            with np.load(directory / 'risk.npz', allow_pickle=False) as saved:
                ids = saved['ids'].copy(); exposures = saved['B'].copy()
                reference = saved['F'].copy(); diagonal = saved['D'].copy()
                if str(saved['eom']) != str(d):
                    raise ValueError('Scored formation date differs from month directory')
            covariance = (factor_covariance(values, settings) if half_weight is None else decay_factor_covariance(
                values, np.arange(len(values), 0, -1, dtype=float), settings, component, half_weight))
            record = dict(records[str(d)], covariance_variant='tuned_hyperbolic_' + component,
                parent_manifest_sha256=parent_sha, covariance_information_cutoff=str(d),
                factor_returns_through=str(last_date), factor_return_days=len(values),
                covariance_history='original bounded history', decay_selection=selection,
                execution_threads=execution_threads)
            yield dict(ids=ids, eom=d, B=exposures, F=covariance, D=diagonal,
                       reference_F=reference, reference_D=diagonal.copy(), record=record)
