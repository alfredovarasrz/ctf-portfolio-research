"""Historical-only single-component risk variants and a cached-data market proxy."""
from collections import deque
from datetime import date
import hashlib
import json
from pathlib import Path

import numpy as np
import polars as pl

from comparison_models import factor_covariance, RISK_SETTINGS, weighted_covariance


def hyperbolic_weights(n, half_life, power=2.):
    """Positive normalized weights with g(half_life)/g(0)=1/2.

    Ages n..1 match the original benchmark's trading-observation convention.
    This fixed power/half-life rule is declared before inspecting outcomes.
    """
    if n < 2 or half_life <= 0 or power <= 0:
        raise ValueError('At least two observations and positive decay parameters required')
    tau = half_life / (2. ** (1. / power) - 1.)
    weights = (1. + np.arange(n, 0, -1, dtype=float) / tau) ** (-power)
    return weights / weights.sum()


def covariance_with_weights(values, weights):
    """Original unbiased weighted covariance with an explicit age-weight vector."""
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if (values.ndim != 2 or len(values) < 2 or weights.shape != (len(values),)
            or not np.isfinite(values).all() or not np.isfinite(weights).all()
            or np.any(weights < 0) or weights.sum() <= 0):
        raise ValueError('Finite factor history and matching nonnegative weights required')
    weights = weights / weights.sum()
    denominator = 1. - weights @ weights
    if denominator <= 0:
        raise ValueError('At least two positively weighted observations required')
    centered = values - weights @ values
    covariance = (centered.T * weights) @ centered / denominator
    return (covariance + covariance.T) / 2


def hyperbolic_factor_covariance(values, settings=RISK_SETTINGS, *, component, power=2.):
    """Change either correlation age weights or variance age weights, separately."""
    if component not in ('correlation', 'variance'):
        raise ValueError('Hyperbolic component must be correlation or variance')
    correlation_covariance = (covariance_with_weights(values, hyperbolic_weights(
        len(values), settings.correlation_half_life, power)) if component == 'correlation'
        else weighted_covariance(values, settings.correlation_half_life))
    variance_covariance = (covariance_with_weights(values, hyperbolic_weights(
        len(values), settings.variance_half_life, power)) if component == 'variance'
        else weighted_covariance(values, settings.variance_half_life))
    standard_deviation = np.sqrt(np.maximum(np.diag(correlation_covariance), 0.))
    denominator = np.outer(standard_deviation, standard_deviation)
    correlation = np.divide(correlation_covariance, denominator,
                            out=np.zeros_like(correlation_covariance), where=denominator > 0)
    variance_sd = np.sqrt(np.maximum(np.diag(variance_covariance), 0.))
    covariance = correlation * np.outer(variance_sd, variance_sd)
    return (covariance + covariance.T) / 2


def factor_downside_semimoment(values, threshold=0.):
    """Uniform uncentered negative-part factor second moment, including all days.

    A PSD systematic-risk proxy. It is not portfolio/stock return semivariance:
    negative-part clipping does not commute with arbitrary signed B or weights.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or len(values) < 2 or not np.isfinite(values).all() or not np.isfinite(threshold):
        raise ValueError('At least two finite factor-return rows required')
    negative = np.minimum(values - threshold, 0.)
    moment = negative.T @ negative / len(values)
    return (moment + moment.T) / 2


def conditional_factor_covariances(values, market_returns, *, minimum_days=63):
    """Matched uniform MLE all/down/up F; predetermined small-sample fallback.

    All regimes use the identical bounded trading-date window. Missing proxy
    returns are errors; negative is strictly <0 and nonnegative includes zero.
    """
    values = np.asarray(values, dtype=float)
    market_returns = np.asarray(market_returns, dtype=float)
    if (values.ndim != 2 or len(values) < 2 or market_returns.shape != (len(values),)
            or not np.isfinite(values).all() or not np.isfinite(market_returns).all() or minimum_days < 2):
        raise ValueError('Finite aligned factor/proxy history and at least two minimum days required')
    def empirical(rows):
        centered = rows - rows.mean(axis=0)
        covariance = centered.T @ centered / len(rows)
        return (covariance + covariance.T) / 2
    pooled = empirical(values)
    covariances = {'uniform_all_days': pooled}
    diagnostics = {}
    for name, mask in (('negative_day_covariance', market_returns < 0),
                       ('nonnegative_day_covariance', market_returns >= 0)):
        count = int(mask.sum())
        fallback = count < minimum_days
        covariances[name] = pooled.copy() if fallback else empirical(values[mask])
        diagnostics[name + '_selected_days'] = count
        diagnostics[name + '_pooled_fallback'] = fallback
    covariances['balanced_conditional_covariance'] = (
        covariances['negative_day_covariance'] + covariances['nonnegative_day_covariance']) / 2
    diagnostics['conditional_minimum_days'] = minimum_days
    return covariances, diagnostics


def fit_state_blends(pairs, *, minimum_pairs=24, minimum_group_pairs=2):
    """Fit bounded scalar covariance-loss minima and original five-block CV.

    Pair a,b,c store ||target-up||², <down-up,target-up>, ||down-up||².
    No full covariance matrices or outer evaluation returns enter the fit.
    """
    def fit(rows):
        estimates = {}; fallback = {}
        for state in (False, True):
            selected = [row for row in rows if row['negative_signal'] == state]
            denominator = sum(row['c'] for row in selected)
            fallback[state] = len(rows) < minimum_pairs or len(selected) < minimum_group_pairs or denominator <= 0
            estimates[state] = .5 if fallback[state] else float(np.clip(
                sum(row['b'] for row in selected) / denominator, 0., 1.))
        return estimates, fallback
    final, fallback = fit(pairs)
    folds = []
    if len(pairs) >= 5:
        for indices in np.array_split(np.arange(len(pairs)), 5):
            held_out = set(indices.tolist())
            train = [row for i, row in enumerate(pairs) if i not in held_out]
            validate = [pairs[i] for i in indices]
            estimates, fold_fallback = fit(train)
            loss = sum(row['a'] - 2 * estimates[row['negative_signal']] * row['b']
                       + estimates[row['negative_signal']] ** 2 * row['c'] for row in validate)
            folds.append(dict(training_pairs=len(train), validation_pairs=len(validate),
                              validation_start=validate[0]['target_month'], validation_end=validate[-1]['target_month'],
                              q_negative=estimates[True], q_nonnegative=estimates[False],
                              negative_fallback=fold_fallback[True], nonnegative_fallback=fold_fallback[False],
                              mean_frobenius_squared_loss=float(max(loss, 0.) / len(validate))))
    diagnostics = dict(q_negative=final[True], q_nonnegative=final[False],
        negative_blend_fallback=fallback[True], nonnegative_blend_fallback=fallback[False],
        blend_training_pairs=len(pairs), blend_minimum_pairs=minimum_pairs,
        blend_minimum_group_pairs=minimum_group_pairs, blend_cv_policy='five historical blocked folds',
        blend_validation_folds=folds,
        blend_cv_mean_loss=float(np.mean([fold['mean_frobenius_squared_loss'] for fold in folds])) if folds else None)
    return final, diagnostics


def _parent_months(artifact_dir, cutoff=None):
    """Read checkpoint metadata only. Never load future-fitted checkpoint state."""
    root = Path(artifact_dir)
    manifest = json.loads((root / 'manifest.json').read_text())
    with np.load(root / 'checkpoint.npz', allow_pickle=False) as checkpoint:
        metadata = json.loads(str(checkpoint['metadata']))
    last_month = date.fromisoformat(metadata['last_month'])
    if cutoff is not None:
        cutoff = date.fromisoformat(cutoff) if isinstance(cutoff, str) else cutoff
        last_month = min(last_month, cutoff)
    directories = sorted(p for p in (root / 'months').iterdir()
                         if p.is_dir() and not p.name.endswith('.tmp') and p.name <= str(last_month))
    return root, manifest, metadata, directories


def _factor_stream(directory, names, d):
    marker = json.loads((directory / 'complete.json').read_text())
    if marker['available_through'] != str(d) or marker['factor_order'] != names:
        raise ValueError('Risk month cutoff or factor order differs from its parent')
    stream = pl.read_parquet(directory / 'factor_returns.parquet').sort('date')
    if (stream.columns != ['date'] + names or stream['date'].n_unique() != stream.height
            or (stream.height and (stream['date'].max() > d
                                   or any(v.year != d.year or v.month != d.month for v in stream['date'])))):
        raise ValueError('Factor stream dates or factor basis differ from their month')
    if stream.height and not np.isfinite(stream.select(names).to_numpy()).all():
        raise ValueError('Factor returns must be finite')
    return stream, marker


def iter_market_proxy_months(artifact_dir, *, cutoff=None):
    """Yield date/ret equal-weight excess-return proxies on original regression stocks.

    Reconstruct each observed daily return as preceding-month B @ f + residual.
    Empty fitting months yield empty frames. Missing returns are never filled.
    Only complete calendar months at or before cutoff are read. For a mid-month
    cutoff the current uncompleted month is omitted, matching formation usage.
    """
    root, manifest, _, directories = _parent_months(artifact_dir, cutoff)
    names = manifest['factor_order']
    for directory in directories:
        d = date.fromisoformat(directory.name)
        stream, marker = _factor_stream(directory, names, d)
        residuals = pl.read_parquet(directory / 'residuals.parquet').sort('date', 'id')
        if not stream.height:
            if residuals.height:
                raise ValueError('Residual observations exist without factor fits')
            yield pl.DataFrame(schema={'date': pl.Date, 'ret': pl.Float64})
            continue
        preceding = marker.get('preceding_exposure_date')
        if not preceding or date.fromisoformat(preceding) >= d:
            raise ValueError('Daily proxy needs preceding exposure date')
        with np.load(root / 'months' / preceding / 'exposures.npz', allow_pickle=False) as saved:
            ids = saved['ids']; exposures = saved['B']
            if str(saved['eom_ret']) != str(d) or str(saved['eom']) != preceding:
                raise ValueError('Daily proxy exposures do not precede this return month')
        if np.any(np.diff(ids) <= 0) or exposures.shape != (len(ids), len(names)):
            raise ValueError('Preceding exposure security order or factor basis is invalid')
        if (residuals.columns != ['id', 'date', 'res'] or residuals.select('id', 'date').is_duplicated().any()
                or not residuals['res'].is_finite().all()):
            raise ValueError('Daily residual stream is invalid')
        groups = residuals.partition_by('date', maintain_order=True)
        if len(groups) != stream.height or [g['date'][0] for g in groups] != stream['date'].to_list():
            raise ValueError('Daily residual and factor dates do not match')
        rows = []
        for group, coef in zip(groups, stream.select(names).to_numpy()):
            day_ids = group['id'].to_numpy()
            positions = np.searchsorted(ids, day_ids)
            if np.any(positions >= len(ids)) or np.any(ids[positions] != day_ids) or len(day_ids) < 2:
                raise ValueError('Daily proxy residual securities lack original exposures')
            returns = exposures[positions] @ coef + group['res'].to_numpy()
            if not np.isfinite(returns).all():
                raise ValueError('Reconstructed daily returns are invalid')
            rows.append((group['date'][0], float(returns.mean())))
        yield pl.DataFrame(rows, schema={'date': pl.Date, 'ret': pl.Float64}, orient='row')


def load_market_proxy(artifact_dir, *, cutoff=None):
    """Return the small date/ret proxy stream. No external data or disk mutation."""
    frames = list(iter_market_proxy_months(artifact_dir, cutoff=cutoff))
    return pl.concat(frames) if frames else pl.DataFrame(schema={'date': pl.Date, 'ret': pl.Float64})


def transform_extended_risk(artifact_dir, variant, *, cutoff=None,
                            market_proxy=None, proxy_parent_sha256=None, minimum_conditional_days=63):
    """Yield the batch_risk B/F/D/reference protocol, changing F only.

    R07 expanding EW history; R08 hyperbolic correlation-only or variance-only
    on the original bounded history. Settings come from the immutable parent.
    """
    if variant == 'learned_state_covariance':
        yield from transform_state_dependent_risk(artifact_dir, cutoff=cutoff, market_proxy=market_proxy,
            proxy_parent_sha256=proxy_parent_sha256, minimum_conditional_days=minimum_conditional_days)
        return
    variants = ('expanding_ew', 'hyperbolic_correlation', 'hyperbolic_variance',
                'factor_downside_semimoment', 'factor_second_moment_control',
                'uniform_all_days', 'negative_day_covariance', 'nonnegative_day_covariance',
                'balanced_conditional_covariance')
    if variant not in variants:
        raise ValueError(f'Unknown extended risk variant {variant}')
    root, manifest, metadata, directories = _parent_months(artifact_dir, cutoff)
    parent_manifest_sha256 = hashlib.sha256((root / 'manifest.json').read_bytes()).hexdigest()
    conditional = variant in ('uniform_all_days', 'negative_day_covariance',
                              'nonnegative_day_covariance', 'balanced_conditional_covariance')
    proxy_iterator = None
    if conditional:
        if market_proxy is None:
            proxy_iterator = iter(iter_market_proxy_months(root, cutoff=cutoff))
        elif (proxy_parent_sha256 != parent_manifest_sha256
              or market_proxy.columns != ['date', 'ret'] or market_proxy['date'].n_unique() != market_proxy.height
              or market_proxy['date'].null_count() or not market_proxy['ret'].is_finite().all()):
            raise ValueError('Supplied market proxy identity or observations differ from parent')
    names = manifest['factor_order']
    settings = type(RISK_SETTINGS)(**manifest['settings'])
    records = {r['formation_date']: r for r in metadata['records']}
    history = deque(maxlen=None if variant == 'expanding_ew' else settings.covariance_days)
    market_history = deque(maxlen=settings.covariance_days)
    last_date = None
    for directory in directories:
        d = date.fromisoformat(directory.name)
        stream, _ = _factor_stream(directory, names, d)
        if conditional:
            proxy = next(proxy_iterator) if proxy_iterator is not None else market_proxy.filter(
                (pl.col('date').dt.year() == d.year) & (pl.col('date').dt.month() == d.month))
            proxy = stream.select('date').join(proxy, on='date', how='left').sort('date')
            if proxy['ret'].null_count() or not proxy['ret'].is_finite().all():
                raise ValueError('Original factor dates lack a finite market proxy')
            market_history.extend(proxy['ret'].to_numpy())
        if stream.height:
            if last_date is not None and stream['date'].min() <= last_date:
                raise ValueError('Factor history dates overlap or move backward')
            last_date = stream['date'].max()
            history.extend(stream.select(names).to_numpy())
        if str(d) not in records:
            continue
        with np.load(directory / 'risk.npz', allow_pickle=False) as saved:
            ids = saved['ids'].copy(); exposures = saved['B'].copy()
            reference = saved['F'].copy(); diagonal = saved['D'].copy()
            if str(saved['eom']) != str(d):
                raise ValueError('Scored covariance formation differs from directory')
        values = np.array(history)
        if variant == 'expanding_ew':
            covariance = factor_covariance(values, settings)
        elif variant.startswith('hyperbolic_'):
            covariance = hyperbolic_factor_covariance(values, settings,
                            component='correlation' if variant.endswith('correlation') else 'variance')
        elif variant == 'factor_downside_semimoment':
            covariance = factor_downside_semimoment(values)
        elif variant == 'factor_second_moment_control':
            covariance = values.T @ values / len(values)
        else:
            conditional_covariances, conditional_diagnostics = conditional_factor_covariances(
                values, np.array(market_history), minimum_days=minimum_conditional_days)
            covariance = conditional_covariances[variant]
        record = dict(records[str(d)], covariance_variant=variant,
                      parent_manifest_sha256=parent_manifest_sha256,
                      covariance_information_cutoff=str(d), factor_returns_through=str(last_date),
                      factor_return_days=len(values), history_policy='expanding' if variant == 'expanding_ew' else 'original bounded',
                      correlation_half_life=settings.correlation_half_life,
                      variance_half_life=settings.variance_half_life,
                      hyperbolic_power=2. if variant.startswith('hyperbolic_') else None,
                      changed_component='factor covariance history' if variant == 'expanding_ew' else variant,
                      downside_threshold=0. if variant == 'factor_downside_semimoment' else None,
                      covariance_normalization='uniform uncentered denominator n' if variant.startswith('factor_')
                      else 'uniform centered MLE denominator selected n' if conditional else 'unbiased age weighted')
        if conditional:
            record.update(conditional_diagnostics, market_proxy_threshold=0.,
                          market_proxy_definition='equal-weight original daily regression securities',
                          conditional_blend_weight=.5 if variant == 'balanced_conditional_covariance' else None)
        for name, half_life in (('correlation', settings.correlation_half_life), ('variance', settings.variance_half_life)):
            weights = (np.ones(len(values)) if variant.startswith('factor_') or conditional else
                       hyperbolic_weights(len(values), half_life) if variant == 'hyperbolic_' + name
                       else .5 ** (np.arange(len(values), 0, -1, dtype=float) / half_life))
            weights /= weights.sum()
            record[name + '_effective_observations'] = float(1. / (weights @ weights))
        yield dict(ids=ids, eom=d, B=exposures, F=covariance, D=diagonal,
                   reference_F=reference, reference_D=diagonal.copy(), record=record)


def transform_state_dependent_risk(artifact_dir, *, cutoff=None, market_proxy=None,
        proxy_parent_sha256=None, minimum_conditional_days=63,
        learning_months=120, minimum_pairs=24, minimum_group_pairs=2):
    """R11 historical-only learned conditional-F blend with a monthly lagged signal.

    All daily history/B/D comes from the original baseline. Signal is the latest
    completed month's proxy sign, targeting next month's covariance. Each target
    is completed before it enters the current outer formation's training pairs.
    """
    root, manifest, metadata, directories = _parent_months(artifact_dir, cutoff)
    parent_sha = hashlib.sha256((root / 'manifest.json').read_bytes()).hexdigest()
    if learning_months < 5 or minimum_pairs < 1 or minimum_group_pairs < 1:
        raise ValueError('Positive learning/fit counts required')
    names = manifest['factor_order']; settings = type(RISK_SETTINGS)(**manifest['settings'])
    records = {r['formation_date']: r for r in metadata['records']}
    if market_proxy is None:
        proxy_iterator = iter(iter_market_proxy_months(root, cutoff=cutoff))
    else:
        proxy_iterator = None
        if (proxy_parent_sha256 != parent_sha or market_proxy.columns != ['date', 'ret']
                or market_proxy['date'].n_unique() != market_proxy.height
                or market_proxy['date'].null_count() or not market_proxy['ret'].is_finite().all()):
            raise ValueError('Supplied market proxy identity or observations differ from parent')
    history = deque(maxlen=settings.covariance_days)
    market_history = deque(maxlen=settings.covariance_days)
    pairs = deque(maxlen=learning_months)
    pending = None; last_date = None
    for directory in directories:
        d = date.fromisoformat(directory.name)
        stream, _ = _factor_stream(directory, names, d)
        proxy = next(proxy_iterator) if proxy_iterator is not None else market_proxy.filter(
            (pl.col('date').dt.year() == d.year) & (pl.col('date').dt.month() == d.month))
        proxy = stream.select('date').join(proxy, on='date', how='left').sort('date')
        if proxy['ret'].null_count() or not proxy['ret'].is_finite().all():
            raise ValueError('Original factor dates lack a finite market proxy')
        monthly_values = stream.select(names).to_numpy()
        if pending is not None and stream.height >= 2:
            expected_month = pending['formation'].year * 12 + pending['formation'].month + 1
            if d.year * 12 + d.month == expected_month:
                centered = monthly_values - monthly_values.mean(axis=0)
                target = centered.T @ centered / len(centered)
                difference = pending['down'] - pending['up']; error = target - pending['up']
                pairs.append(dict(formation_date=str(pending['formation']), target_month=str(d),
                    signal_month=str(pending['formation']), negative_signal=pending['negative_signal'],
                    a=float(np.sum(error ** 2)), b=float(np.sum(difference * error)), c=float(np.sum(difference ** 2))))
        if stream.height:
            if last_date is not None and stream['date'].min() <= last_date:
                raise ValueError('Factor history dates overlap or move backward')
            last_date = stream['date'].max()
            history.extend(monthly_values); market_history.extend(proxy['ret'].to_numpy())
        if len(history) < 2:
            pending = None
            continue
        covariances, conditional_record = conditional_factor_covariances(
            np.array(history), np.array(market_history), minimum_days=minimum_conditional_days)
        monthly_proxy_return = float(np.prod(1. + proxy['ret'].to_numpy()) - 1.) if proxy.height else None
        negative_signal = monthly_proxy_return is not None and monthly_proxy_return < 0
        estimates, fit_record = fit_state_blends(list(pairs), minimum_pairs=minimum_pairs,
                                                minimum_group_pairs=minimum_group_pairs)
        q = estimates[negative_signal] if monthly_proxy_return is not None else .5
        down = covariances['negative_day_covariance']; up = covariances['nonnegative_day_covariance']
        pending = dict(formation=d, down=down, up=up, negative_signal=negative_signal) if proxy.height else None
        if str(d) not in records:
            continue
        with np.load(directory / 'risk.npz', allow_pickle=False) as saved:
            ids = saved['ids'].copy(); exposures = saved['B'].copy()
            reference = saved['F'].copy(); diagonal = saved['D'].copy()
            if str(saved['eom']) != str(d):
                raise ValueError('Scored covariance formation differs from directory')
        covariance = q * down + (1. - q) * up
        record = dict(records[str(d)], **conditional_record, **fit_record,
            covariance_variant='learned_state_covariance', parent_manifest_sha256=parent_sha,
            covariance_information_cutoff=str(d), factor_returns_through=str(last_date), factor_return_days=len(history),
            signal_month=str(d) if proxy.height else None, signal_monthly_proxy_return=monthly_proxy_return,
            selected_blend_weight=q, signal_missing_fallback=monthly_proxy_return is None,
            blend_learning_window_pairs=learning_months, blend_training_information_cutoff=str(d),
            blend_pair_ids=[dict(formation_date=row['formation_date'], target_month=row['target_month'],
                                 signal_month=row['signal_month']) for row in pairs],
            covariance_normalization='learned blend of uniform centered conditional MLE',
            covariance_interpretation='expected within-state factor covariance; no between-state mean dispersion')
        yield dict(ids=ids, eom=d, B=exposures, F=(covariance + covariance.T) / 2, D=diagonal,
                   reference_F=reference, reference_D=diagonal.copy(), record=record)
