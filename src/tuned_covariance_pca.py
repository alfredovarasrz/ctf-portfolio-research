"""R04_TUNED: annual historical selection of factor-covariance PCA mass."""
from collections import deque
from dataclasses import asdict
from datetime import date
import hashlib
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from batch_risk import pca_factor_covariance
from comparison_models import RISK_SETTINGS
from extended_risk import _parent_months, _factor_stream, covariance_with_weights
from research_resources import write_json

PCA_GRID = (.5, .7, .9, .95)


def fold_covariance(values, ages, settings=RISK_SETTINGS):
    """Original separate EW half-lives with full-window ages through fold gaps."""
    values = np.asarray(values, dtype=float); ages = np.asarray(ages, dtype=float)
    if (values.ndim != 2 or len(values) < 2 or ages.shape != (len(values),)
            or not np.isfinite(values).all() or not np.isfinite(ages).all() or np.any(ages <= 0)):
        raise ValueError('Finite factor rows and aligned positive full-window ages required')
    correlation_cov = covariance_with_weights(values, .5 ** (ages / settings.correlation_half_life))
    variance_cov = covariance_with_weights(values, .5 ** (ages / settings.variance_half_life))
    sd = np.sqrt(np.maximum(np.diag(correlation_cov), 0.))
    outer = np.outer(sd, sd)
    correlation = np.divide(correlation_cov, outer, out=np.zeros_like(outer), where=outer > 0)
    sd = np.sqrt(np.maximum(np.diag(variance_cov), 0.))
    fitted = correlation * np.outer(sd, sd)
    return (fitted + fitted.T) / 2


def _eigenbasis(covariance):
    values, vectors = np.linalg.eigh((covariance + covariance.T) / 2)
    if values.min() < -1e-9 * max(float(np.max(np.abs(values))), 1e-20):
        raise ValueError('Training factor covariance is materially indefinite')
    order = np.argsort(-values, kind='stable')
    return np.maximum(values[order], 0.), vectors[:, order]


def _project(values, vectors, fraction):
    total = float(values.sum())
    count = 0 if total == 0 else min(int(np.count_nonzero(values > 0)),
        int(np.searchsorted(np.cumsum(values), fraction * total)) + 1)
    retained = (vectors[:, :count] * values[:count]) @ vectors[:, :count].T
    return retained, count


def select_retained_variance(values, dates, cutoff, settings=RISK_SETTINGS, *, candidates=PCA_GRID):
    """Select only by five completed historical factor-covariance MSE folds.

    Loss excludes stock marginal-diagonal restoration and portfolio utility.
    The untruncated and fixed90 estimates are diagnostic controls, not candidates.
    """
    values = np.asarray(values, dtype=float); dates = list(dates)
    cutoff = date.fromisoformat(cutoff) if isinstance(cutoff, str) else cutoff
    candidates = tuple(sorted(set(map(float, candidates))))
    if (values.ndim != 2 or values.shape[1] == 0 or len(values) != len(dates)
            or not np.isfinite(values).all() or dates != sorted(set(dates))
            or dates and dates[-1] > cutoff):
        raise ValueError('Finite chronological unique factor history completed by outer cutoff required')
    if len(values) > settings.covariance_days:
        raise ValueError('PCA selection cannot expand the original covariance window')
    if not candidates or not all(np.isfinite(c) and 0 < c <= 1 for c in candidates):
        raise ValueError('Predeclared retained fractions in (0,1] required')
    months = sorted(set((d.year, d.month) for d in dates))
    record = dict(experiment='R04_TUNED', candidate_retained_fractions=list(candidates),
        tuning_information_cutoff=str(cutoff), tuning_first_factor_day=str(dates[0]) if dates else None,
        tuning_last_factor_day=str(dates[-1]) if dates else None, tuning_days=len(values), tuning_months=len(months),
        validation_policy='five contiguous historical calendar blocks; four train and one validate',
        validation_target='uniform centered unbiased held-out factor covariance; denominator n-1',
        validation_loss='mean squared factor covariance entries; equal fold weight',
        loss_excludes='stock diagonal restoration and portfolio utility',
        weighting_age_policy='full original trading-observation ages; held-out gaps retained',
        selected_retained_fraction=None, selection_fallback=None, candidates=[],
        untruncated_control_folds=[], fixed90_control_folds=[])
    if len(months) < 5 or len(values) < 10:
        record['selection_fallback'] = 'original full F; insufficient five-block history'
        return None, record
    month_index = {m: i for i, m in enumerate(months)}
    row_month = np.array([month_index[(d.year, d.month)] for d in dates])
    ages = np.arange(len(values), 0, -1, dtype=float)
    folds = []
    for block in np.array_split(np.arange(len(months)), 5):
        validation = np.isin(row_month, block); training = ~validation
        if validation.sum() < 2 or training.sum() < 2:
            record['selection_fallback'] = 'original full F; insufficient observations in a calendar fold'
            return None, record
        held_out = values[validation]; centered = held_out - held_out.mean(axis=0)
        target = centered.T @ centered / (len(held_out) - 1)
        fitted = fold_covariance(values[training], ages[training], settings)
        eigenvalues, eigenvectors = _eigenbasis(fitted)
        train_dates = [d for d, included in zip(dates, training) if included]
        valid_dates = [d for d, included in zip(dates, validation) if included]
        metadata = dict(training_days=int(training.sum()), validation_days=int(validation.sum()),
            training_first_day=str(train_dates[0]), training_last_day=str(train_dates[-1]),
            validation_start=str(valid_dates[0]), validation_end=str(valid_dates[-1]),
            training_dates=list(map(str, train_dates)), validation_dates=list(map(str, valid_dates)))
        folds.append((eigenvalues, eigenvectors, target, metadata))
        record['untruncated_control_folds'].append(dict(metadata,
            mean_squared_covariance_error=float(np.mean((fitted - target) ** 2))))
        fixed, count = _project(eigenvalues, eigenvectors, .9)
        record['fixed90_control_folds'].append(dict(metadata, retained_components=count,
            mean_squared_covariance_error=float(np.mean((fixed - target) ** 2))))
    for fraction in candidates:
        scores = []
        for eigenvalues, eigenvectors, target, metadata in folds:
            fitted, count = _project(eigenvalues, eigenvectors, fraction)
            scores.append(dict(metadata, retained_components=count,
                mean_squared_covariance_error=float(np.mean((fitted - target) ** 2))))
        record['candidates'].append(dict(retained_fraction=fraction, folds=scores,
            mean_validation_loss=float(np.mean([r['mean_squared_covariance_error'] for r in scores]))))
    winner = min(record['candidates'], key=lambda r: (r['mean_validation_loss'], r['retained_fraction']))
    selected = winner['retained_fraction']; record['selected_retained_fraction'] = selected
    record['grid_boundary_choice'] = selected in (candidates[0], candidates[-1])
    return selected, record


def _selection_identity(values, dates, cutoff, names, settings, candidates, identity):
    sources = ('tuned_covariance_pca.py', 'batch_risk.py', 'comparison_models.py',
               'extended_risk.py', 'research_resources.py')
    base = Path(__file__).resolve().parent
    return dict(experiment='R04_TUNED', caller=identity, cutoff=str(cutoff), factor_order=names,
        settings=asdict(settings), candidates=list(candidates),
        factor_values_sha256=hashlib.sha256(np.asarray(values, dtype='<f8').tobytes()).hexdigest(),
        factor_dates=list(map(str, dates)),
        sources_sha256={name: hashlib.sha256((base/name).read_bytes()).hexdigest() for name in sources})


def _annual_selection(values, dates, cutoff, names, settings, candidates, directory, identity):
    expected = _selection_identity(values, dates, cutoff, names, settings, candidates, identity)
    path = None if directory is None else Path(directory)/f'{cutoff}.json'
    if path is not None and path.exists():
        saved = json.loads(path.read_text())
        proof = hashlib.sha256(json.dumps(saved['selection'], sort_keys=True).encode()).hexdigest()
        if saved['identity'] != expected or saved.get('selection_sha256') != proof:
            raise ValueError('Annual PCA cache input, settings, cutoff or source identity changed')
        return saved['selection']['selected_retained_fraction'], saved['selection']
    selected, record = select_retained_variance(values, dates, cutoff, settings, candidates=candidates)
    if path is not None:
        write_json(path, dict(identity=expected, selection=record,
            selection_sha256=hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()))
    return selected, record


def _selection_receipt(directory, cutoff, year):
    path = Path(directory)/f'{cutoff}.json'
    saved = json.loads(path.read_text()); record = saved['selection']
    fields = ('experiment','candidate_retained_fractions','tuning_information_cutoff',
        'tuning_first_factor_day','tuning_last_factor_day','tuning_days','tuning_months',
        'selected_retained_fraction','selection_fallback','grid_boundary_choice')
    return dict({key:record.get(key) for key in fields}, forecast_year=year, annual_anchor=str(cutoff),
        curve_path=str(path), selection_sha256=saved['selection_sha256'],
        selection_identity_sha256=hashlib.sha256(json.dumps(saved['identity'],sort_keys=True).encode()).hexdigest())


def transform_tuned_covariance_pca(artifact_dir, *, cutoff=None, candidates=PCA_GRID,
                                 threads=None, checkpoint_dir=None, identity=None):
    """Select annually, then truncate each saved original monthly F and restore D.

    First available scored formation anchors each forecast year, including a
    partial initial year. A fallback also freezes for that year. Checkpoints
    contain only historical selection metadata; no future covariance state.
    """
    root, manifest, metadata, directories = _parent_months(artifact_dir, cutoff)
    settings = type(RISK_SETTINGS)(**manifest['settings'])
    execution_threads = settings.threads if threads is None else threads
    if not isinstance(execution_threads, int) or execution_threads < 1:
        raise ValueError('Execution thread override must be a positive integer')
    if checkpoint_dir is None or identity is None:
        raise ValueError('Explicit checkpoint directory and caller identity required for compact annual PCA receipts')
    if Path(checkpoint_dir).resolve().is_relative_to(root.resolve()):
        raise ValueError('PCA checkpoints must be outside the read-only original risk cache')
    candidates = tuple(sorted(set(map(float, candidates))))
    names = manifest['factor_order']; records = {r['formation_date']: r for r in metadata['records']}
    maximum = min(max(records), str(cutoff)) if cutoff is not None else max(records)
    directories = [p for p in directories if p.name <= maximum]
    parent_sha = hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest()
    history = deque(maxlen=settings.covariance_days); dates = deque(maxlen=settings.covariance_days)
    selections = {}; last_date = None
    with threadpool_limits(limits=execution_threads):
        for directory in directories:
            formation = date.fromisoformat(directory.name)
            stream, _ = _factor_stream(directory, names, formation)
            if stream.height:
                if last_date is not None and stream['date'].min() <= last_date:
                    raise ValueError('Original factor dates overlap or move backward')
                last_date = stream['date'].max()
                history.extend(stream.select(names).to_numpy()); dates.extend(stream['date'].to_list())
            if str(formation) not in records:
                continue
            year = formation.year + int(formation.month == 12)
            if year not in selections:
                selected, selection = _annual_selection(np.asarray(history).reshape(-1, len(names)), list(dates), formation, names,
                    settings, candidates, checkpoint_dir, identity)
                selections[year] = selected, _selection_receipt(checkpoint_dir,formation,year)
            selected, selection = selections[year]
            with np.load(directory/'risk.npz', allow_pickle=False) as saved:
                ids = saved['ids'].copy(); exposures = saved['B'].copy()
                reference = saved['F'].copy(); original_diagonal = saved['D'].copy()
                if str(saved['eom']) != str(formation):
                    raise ValueError('Saved covariance formation differs from its directory')
            if selected is None:
                covariance = reference.copy(); diagonal = original_diagonal.copy(); diagnostics = {}
            else:
                covariance, diagonal, diagnostics = pca_factor_covariance(
                    exposures, reference, original_diagonal, selected)
            record = dict(records[str(formation)], **diagnostics, covariance_variant='tuned_covariance_pca',
                experiment='R04_TUNED', covariance_information_cutoff=str(formation),
                factor_returns_through=str(last_date), factor_return_days=len(history),
                parent_manifest_sha256=parent_sha, covariance_history='original bounded2520 daily history',
                retained_variance_rule=selected, pca_selection=selection, execution_threads=execution_threads,
                discarded_risk_policy='restore omitted stock marginal variance to D',
                basis_information_cutoff=str(formation),
                basis_availability_semantics='unchanged original exposure availability; no basis fit')
            yield dict(ids=ids, eom=formation, B=exposures, F=covariance, D=diagonal,
                reference_B=exposures.copy(), reference_F=reference, reference_D=original_diagonal, record=record)
