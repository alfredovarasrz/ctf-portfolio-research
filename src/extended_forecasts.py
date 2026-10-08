"""Atomic forecast-only experiments with original historical blocked validation."""
from collections import deque
from dataclasses import asdict, replace
from datetime import date
from hashlib import sha256
import gc
import json
from pathlib import Path

import numpy as np
import polars as pl
from threadpoolctl import threadpool_limits
import xgboost as xgb

from scipy.linalg import cho_factor, cho_solve

from baseline import DEFAULT_SETTINGS, MonthlySource, canonical_features, month_number, month_end
from comparison_models import FORECAST_SETTINGS, INDUSTRIES, _xgb_params, training_folds
from research_forecasts import _checkpoint_directory, _write_json


# Section 2: Fixed specifications and causal history ---------------------------
SPECIFICATIONS = {
    'P03': {'history': 'original rolling window', 'pca_mass_candidates': [.5, .7, .9, .95],
            'pca_scope': 'each four-block training set; full history for final fit'},
    'P04': {'history': 'all completed history', 'validation': 'five calendar blocks'},
    'P05': {'history': 'original rolling window', 'half_life_month_candidates': [12., 36., 60., 120.], 'power': 1.,
            'weight_normalization': 'training-fit mean1', 'validation_loss': 'unweighted MSE'},
    'P05_EXPANDING': {'history': 'all completed history', 'half_life_month_candidates': [12., 36., 60., 120.],
            'power': 1., 'weight_normalization': 'training-fit mean1',
            'validation_loss': 'unweighted MSE', 'joint_experiment': 'expanding history plus tuned hyperbolic fit weights'},
    'P06': {'history': 'original rolling window', 'bags': 5, 'block_months': 3, 'seed': 1,
            'selection': 'reuse original P01 selected configuration and tree count',
            'resampling': 'circular contiguous month blocks; complete cross-sections'},
    'P07': {'factor_days': 2520, 'monthly_multiplier': 21., 'factor_estimator': 'daily mean',
            'horizon': 'linear monthly approximation; not compounded factor assets'},
    'P08': {'history': '120 completed monthly factor targets', 'lag': 1,
            'ridge_lambdas': list(DEFAULT_SETTINGS.ridge_lambdas),
            'monthly_target': 'sum daily coefficients; linear aggregate, not compounded asset'},
}


def forecast_history(source, first, settings, variant):
    cutoff = source.return_to_formation[first]
    lower = month_number(first) - (120 if variant == 'P08' else settings.train_years * 12)
    return sorted(d for d in source.return_to_formation if d < first and d <= cutoff
                  and (variant in ('P04', 'P05_EXPANDING') or month_number(d) >= lower))


def historical_blocks(dates, first, settings, variant):
    if variant not in ('P04', 'P05_EXPANDING'):
        return training_folds(dates, first, settings)
    # Expanding history needs the same five-block policy over its longer span.
    if not dates:
        return [[] for _ in range(settings.folds)]
    start = month_number(dates[0])
    span = month_number(first) - start
    return [[d for d in dates if min(settings.folds - 1,
            (month_number(d) - start) * settings.folds // max(1, span - 1)) == i]
            for i in range(settings.folds)]


def observation_weights(row_dates, cutoff, tau=60., power=1.):
    """Positive fit weights on the unchanged history, normalized to mean one."""
    age = month_number(cutoff) - np.array([month_number(d) for d in row_dates])
    if tau <= 0 or power <= 0 or np.any(age < 0):
        raise ValueError('Positive decay parameters and completed historical dates required')
    weights = (1. + age / tau) ** -power
    return weights / weights.mean() if len(weights) else weights


def predictor_pca(x, mass=.90):
    """Unwhitened predictor PCA learned only from its current training fold."""
    if not 0 < mass <= 1 or not len(x):
        raise ValueError('Nonempty training predictors and PCA mass in (0,1] required')
    mean = x.mean(axis=0, dtype=np.float64)
    centered = x.astype(np.float64) - mean
    covariance = centered.T @ centered
    eigen, vectors = np.linalg.eigh((covariance + covariance.T)/2)
    eigen = np.maximum(eigen[::-1], 0.)
    vectors = vectors[:, ::-1]
    total = float(eigen.sum())
    count = max(1, min(len(eigen), int(np.searchsorted(np.cumsum(eigen), mass*total))+1)) if total else 1
    components = vectors[:, :count].copy()
    # Fix eigenvector signs so deterministic ordering does not depend on signs.
    for j in range(count):
        pivot = int(np.argmax(np.abs(components[:, j])))
        if components[pivot, j] < 0:
            components[:, j] *= -1
    return mean, components, dict(retained_components=count,
                                 retained_predictor_variance_fraction=float(eigen[:count].sum()/total) if total else 0.)


def bootstrap_months(dates, anchor, bag, block_months=3, seed=1):
    """Fixed circular month-block draws; order is retained within each block."""
    if not dates:
        return []
    rng = np.random.default_rng(np.random.SeedSequence([seed, month_number(anchor), bag]))
    indices = []
    for start in rng.integers(0, len(dates), size=(len(dates)+block_months-1)//block_months):
        indices.extend((int(start)+offset) % len(dates) for offset in range(block_months))
    return [dates[i] for i in indices[:len(dates)]]


def _identity(variant, names, settings, identity, artifact_manifest=None):
    files = ('extended_forecasts.py', 'baseline.py', 'comparison_models.py', 'research_forecasts.py')
    return json.loads(json.dumps(dict(variant=variant, features=names, settings=asdict(settings),
        specification=SPECIFICATIONS[variant], validation='historical_blocked', parent=identity,
        sources={name: sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in files},
        numpy=np.__version__, polars=pl.__version__, xgboost=xgb.__version__,
        factor_manifest=artifact_manifest), sort_keys=True))


def _load_anchor(directory, first, metadata, threads):
    if directory is None:
        return None
    anchor = directory / str(first)
    marker = anchor / 'fit.json'
    if not marker.exists():
        return None
    saved = json.loads(marker.read_text())
    if saved['metadata'] != metadata:
        raise ValueError('Extended forecast checkpoint identity mismatch')
    models = []
    for i in range(saved['model_count']):
        model = xgb.Booster(params={'nthread': threads})
        model.load_model(anchor / f'model-{i}.json')
        models.append(model)
    arrays = {}
    if saved['arrays']:
        with np.load(anchor / 'arrays.npz', allow_pickle=False) as values:
            arrays = {name: values[name].copy() for name in values.files}
    return models, arrays, saved['fit']


def _save_anchor(directory, first, metadata, models, arrays, fit):
    if directory is None:
        return
    anchor = directory / str(first)
    anchor.mkdir(parents=True, exist_ok=True)
    for i, model in enumerate(models):
        path = anchor / f'model-{i}.json'
        temporary = anchor / f'temporary-model-{i}.json'
        model.save_model(temporary)
        temporary.replace(path)
    if arrays:
        temporary = anchor / 'arrays.npz.tmp'
        with temporary.open('wb') as stream:
            np.savez(stream, **arrays)
        temporary.replace(anchor / 'arrays.npz')
    # Only this marker commits the complete model bundle.
    _write_json(anchor / 'fit.json', dict(metadata=metadata, model_count=len(models),
                                        arrays=bool(arrays), fit=fit))


def _month_prediction(frame, values, directory, first, metadata, return_date):
    pred = frame.select('id', 'eom', 'excntry').with_columns(pl.Series('pred', values).cast(pl.Float64))
    if not pred['pred'].is_finite().all():
        raise ValueError('Nonfinite extended return forecast')
    if directory is None:
        return pred
    path = directory / str(first) / (str(frame['eom'][0]) + '.parquet')
    marker = path.with_suffix('.json')
    expected = dict(metadata=metadata, formation=str(frame['eom'][0]), return_date=str(return_date))
    if marker.exists():
        if json.loads(marker.read_text()) != expected:
            raise ValueError('Extended prediction checkpoint identity mismatch')
        saved = pl.read_parquet(path)
        if not saved.select('id', 'eom', 'excntry').equals(pred.select('id', 'eom', 'excntry')):
            raise ValueError('Extended prediction checkpoint coverage mismatch')
        return saved
    temporary = path.with_name(path.name + '.tmp')
    pred.write_parquet(temporary)
    temporary.replace(path)
    _write_json(marker, expected)
    return pred


# Section 3: Original XGBoost selection, with one declared change ---------------
def _training_arrays(source, dates):
    xs, ys, row_dates = [], [], []
    for d in dates:
        frame = source.load(source.return_to_formation[d]).filter(pl.col('ret_exc_lead1m').is_finite())
        xs.append(frame.select(source.features).to_numpy().astype(np.float32))
        ys.append(frame['ret_exc_lead1m'].to_numpy().astype(np.float32))
        row_dates.extend([d] * frame.height)
    if not ys or not sum(len(y) for y in ys):
        return np.empty((0, len(source.features)), dtype=np.float32), np.array([], dtype=np.float32), []
    return np.concatenate(xs), np.concatenate(ys), row_dates


def _fit_xgb(source, dates, first, cutoff, settings, variant):
    x, y, row_dates = _training_arrays(source, dates)
    if not len(y):
        return [], {}, dict(train_rows=0, selected=None, trees=0, reason='no historical labels')
    blocks = historical_blocks(dates, first, settings, variant)
    row_array = np.array(row_dates, dtype='datetime64[D]')
    folds, pca_folds = [], []

    weighted_variant = variant in ('P05', 'P05_EXPANDING')
    options = (SPECIFICATIONS['P03']['pca_mass_candidates'] if variant == 'P03' else
               SPECIFICATIONS[variant]['half_life_month_candidates'] if weighted_variant else [None])

    def search(candidates, eta, rounds, option):
        results = [[] for _ in candidates]
        for block, validation in enumerate(blocks):
            mask = np.isin(row_array, np.array(validation, dtype='datetime64[D]'))
            if not mask.any() or mask.all():
                continue
            weights = observation_weights(np.array(row_dates, dtype=object)[~mask], cutoff, tau=option) if weighted_variant else None
            train_x, val_x = x[~mask], x[mask]
            if variant == 'P03':
                mean, components, record = predictor_pca(train_x, option)
                train_x = ((train_x-mean) @ components).astype(np.float32)
                val_x = ((val_x-mean) @ components).astype(np.float32)
                if not any(row['fold'] == block+1 and row['retained_mass'] == option for row in pca_folds):
                    pca_folds.append(dict(fold=block+1, retained_mass=option, **record))
            train = xgb.DMatrix(train_x, label=y[~mask], weight=weights, nthread=settings.threads)
            val = xgb.DMatrix(val_x, label=y[mask], nthread=settings.threads)
            for j, candidate in enumerate(candidates):
                model = xgb.train(_xgb_params({k: v for k, v in candidate.items() if k != 'hp_set'}, eta, settings),
                    train, num_boost_round=rounds, evals=[(val, 'val')],
                    early_stopping_rounds=settings.early_stopping, verbose_eval=False)
                results[j].append(dict(fold=block + 1, mse=float(model.best_score) ** 2,
                                       trees=int(model.best_iteration) + 1))
            del train, val, model, train_x, val_x
            gc.collect()
        return results

    for i, validation in enumerate(blocks):
        training = [d for d in dates if d not in validation]
        if training and validation:
            folds.append(dict(fold=i + 1, training_return_dates=[str(d) for d in training],
                              validation_return_dates=[str(d) for d in validation],
                              available_through=str(cutoff)))
    searches = [search(settings.candidates, settings.stage1_eta, settings.stage1_rounds, option) for option in options]
    valid = [(i, j) for i, results in enumerate(searches) for j, result in enumerate(results) if result]
    if valid:
        choice, chosen = min(valid, key=lambda ij: (np.mean([r['mse'] for r in searches[ij[0]][ij[1]]]),
                                                     ij[0], settings.candidates[ij[1]]['hp_set']))
        option, candidate = options[choice], settings.candidates[chosen]
        stage2 = search([candidate], settings.stage2_eta, settings.stage2_rounds, option)[0]
        trees = max(1, int(np.floor(np.mean([r['trees'] for r in stage2]))))
    else:
        choice, option, candidate, stage2, trees = 0, options[0], settings.candidates[0], [], 1
    stage1 = searches[choice]
    weights = observation_weights(row_dates, cutoff, tau=option) if weighted_variant else None
    arrays = {}
    final_pca = None
    if variant == 'P03':
        mean, components, final_pca = predictor_pca(x, option)
        arrays = {'pca_mean': mean, 'pca_components': components}
        x = ((x-mean) @ components).astype(np.float32)
    training = xgb.DMatrix(x, label=y, weight=weights, nthread=settings.threads)
    model = xgb.train(_xgb_params({k: v for k, v in candidate.items() if k != 'hp_set'}, settings.stage2_eta, settings),
                      training, num_boost_round=trees, verbose_eval=False)
    fit = dict(train_rows=len(y), train_months=len(dates), selected=dict(candidate), trees=trees,
               stage1=stage1, stage2=stage2, fold_records=folds)
    if variant == 'P03' or weighted_variant:
        fit.update(selected_retained_mass=option if variant == 'P03' else None,
            selected_half_life_months=option if weighted_variant else None,
            transformation_search=[dict(option=value, tree_candidate_scores=[
                dict(hp_set=settings.candidates[j]['hp_set'], mse=float(np.mean([r['mse'] for r in result])) if result else None,
                     folds=result) for j, result in enumerate(results)]) for value, results in zip(options, searches)],
            transformation_search_scope='predeclared finite candidate grid; no unrestricted optimum claim',
            transformation_grid_boundary=choice in (0, len(options)-1))
    if weights is not None:
        fit['weight_minimum'], fit['weight_maximum'] = float(weights.min()), float(weights.max())
    if final_pca is not None:
        fit.update(pca_folds=pca_folds, final_pca=final_pca)
    return [model], arrays, fit


def _fit_bags(source, dates, first, cutoff, settings, baseline_fit):
    if baseline_fit is None:
        raise ValueError('P06 requires the original P01 annual selected configuration and tree count')
    if (baseline_fit['test_return_start'] != str(first)
            or (baseline_fit['training_return_last'] is not None and baseline_fit['training_return_last'] > str(cutoff))):
        raise ValueError('P06 parent selection exceeds its actual formation cutoff')
    if (baseline_fit['training_return_first'] != (str(dates[0]) if dates else None)
            or baseline_fit['training_return_last'] != (str(dates[-1]) if dates else None)):
        raise ValueError('P06 parent selection uses a different historical window')
    if baseline_fit.get('validation_policy', 'historical_blocked') != 'historical_blocked':
        raise ValueError('P06 parent must use original historical blocked CV')
    models, bag_records = [], []
    for bag in range(SPECIFICATIONS['P06']['bags']):
        sampled = bootstrap_months(dates, cutoff, bag,
            block_months=SPECIFICATIONS['P06']['block_months'], seed=SPECIFICATIONS['P06']['seed'])
        x, y, _ = _training_arrays(source, sampled)
        if not len(y):
            continue
        if baseline_fit['selected'] is None or not baseline_fit['trees']:
            raise ValueError('P06 nonempty history lacks an original selected model')
        candidate = baseline_fit['selected']
        train = xgb.DMatrix(x, label=y, nthread=settings.threads)
        model = xgb.train(_xgb_params({k: v for k, v in candidate.items() if k != 'hp_set'}, settings.stage2_eta, settings),
                          train, num_boost_round=baseline_fit['trees'], verbose_eval=False)
        models.append(model)
        bag_records.append(dict(bag=bag, sampled_return_dates=[str(d) for d in sampled], train_rows=len(y)))
        del train, x, y
        gc.collect()
    return models, {}, dict(train_rows=baseline_fit['train_rows'], train_months=len(dates),
        selected=baseline_fit['selected'], trees=baseline_fit['trees'], bag_records=bag_records,
        selection='original P01 completed-history blocked CV, unchanged', parent_fit=baseline_fit)


# Section 4: Causal original factor-artifact forecasts -------------------------
def daily_factor_history(artifact_dir, cutoff, factor_order, maximum_days=2520):
    """Read committed past months only; never use the final rolling checkpoint."""
    root = Path(artifact_dir)
    values, dates = deque(maxlen=maximum_days), deque(maxlen=maximum_days)
    months = sorted(p for p in (root / 'months').iterdir()
                    if p.is_dir() and len(p.name) == 10 and p.name <= str(cutoff))
    for directory in months:
        marker = json.loads((directory / 'complete.json').read_text())
        if marker['available_through'] > str(cutoff) or marker['factor_order'] != factor_order:
            raise ValueError('Factor artifacts exceed cutoff or use inconsistent coordinates')
        frame = pl.read_parquet(directory / 'factor_returns.parquet').sort('date')
        if frame.columns != ['date'] + factor_order or (frame.height and frame['date'].max() > cutoff):
            raise ValueError('Factor history schema or completed-date cutoff mismatch')
        for d, row in zip(frame['date'].to_list(), frame.select(factor_order).to_numpy()):
            if dates and d <= dates[-1]:
                raise ValueError('Factor history dates must be unique and increasing')
            if not np.isfinite(row).all():
                raise ValueError('Nonfinite historical factor return')
            values.append(row); dates.append(d)
    if not values:
        raise ValueError('No completed factor history available')
    return np.array(values), list(dates)


def _factor_mean(artifact_dir, cutoff, factor_order):
    values, dates = daily_factor_history(artifact_dir, cutoff, factor_order)
    mean = values.mean(axis=0) * SPECIFICATIONS['P07']['monthly_multiplier']
    return [], {'factor_forecast': mean}, dict(train_rows=len(values), train_months=None,
        selected=None, trees=None, factor_history_first=str(dates[0]), factor_history_last=str(dates[-1]),
        factor_order=factor_order, factor_estimator='daily mean times21', horizon='linear monthly approximation')


def monthly_factor_targets(artifact_dir, cutoff, factor_order, first_month):
    """SUM daily explanatory coefficients into completed monthly linear targets."""
    targets = {}
    root = Path(artifact_dir)/'months'
    lower = str(month_end(first_month))
    for directory in sorted(p for p in root.iterdir() if p.is_dir() and len(p.name) == 10
                            and lower <= p.name <= str(cutoff)):
        marker = json.loads((directory/'complete.json').read_text())
        if marker['available_through'] > str(cutoff) or marker['factor_order'] != factor_order:
            raise ValueError('Monthly factor target exceeds cutoff or changes factor coordinates')
        frame = pl.read_parquet(directory/'factor_returns.parquet').sort('date')
        if frame.columns != ['date']+factor_order:
            raise ValueError('Monthly factor target schema mismatch')
        if not frame.height:
            continue
        d = date.fromisoformat(directory.name)
        if frame['date'].max() > cutoff or any(month_number(v) != month_number(d) for v in frame['date']):
            raise ValueError('Monthly factor target contains dates outside its completed month')
        values = frame.select(factor_order).to_numpy()
        if not np.isfinite(values).all():
            raise ValueError('Nonfinite monthly factor target')
        targets[d] = values.sum(axis=0)
    return targets


def multivariate_ridge(x, y, penalty):
    """Same mean-MSE Ridge and unpenalized intercept, shared solve for all factors."""
    if not len(x) or penalty <= 0:
        raise ValueError('Positive Ridge penalty and nonempty completed factor targets required')
    mx, my = x.mean(axis=0), y.mean(axis=0)
    centered = x-mx
    # Dual form saves a K-by-K factorization when monthly history is shorter than K.
    matrix = centered @ centered.T
    matrix[np.diag_indices_from(matrix)] += len(x)*penalty
    coef = centered.T @ cho_solve(cho_factor(matrix, lower=True, check_finite=False), y-my, check_finite=False)
    return my-mx@coef, coef


def _factor_lag_fit(artifact_dir, first, cutoff, factor_order, settings):
    lower = month_number(first)-120
    targets = monthly_factor_targets(artifact_dir, cutoff, factor_order, lower-1)
    eligible = sorted(d for d in targets if lower <= month_number(d) < month_number(first) and d <= cutoff)
    dates = [d for d in eligible if month_end(month_number(d)-1) in targets]
    missing_lags = [str(d) for d in eligible if d not in dates]
    if not dates:
        raise ValueError('No completed monthly factor target with an observed preceding-month lag')
    x = np.array([targets[month_end(month_number(d)-1)] for d in dates])
    y = np.array([targets[d] for d in dates])
    blocks = training_folds(dates, first, replace(settings, train_years=10))
    scores, records = [], []
    masks = []
    for i, validation in enumerate(blocks):
        mask = np.array([d in validation for d in dates])
        if mask.any() and not mask.all():
            masks.append(mask)
            records.append(dict(fold=i+1, training_return_dates=[str(d) for d,m in zip(dates, mask) if not m],
                                validation_return_dates=[str(d) for d,m in zip(dates, mask) if m]))
    for penalty in DEFAULT_SETTINGS.ridge_lambdas:
        loss = []
        for mask in masks:
            intercept, coef = multivariate_ridge(x[~mask], y[~mask], penalty)
            loss.append(float(np.mean((y[mask]-intercept-x[mask]@coef)**2)))
        scores.append(float(np.mean(loss)) if loss else None)
    chosen = min(range(len(scores)), key=lambda i: (scores[i], i)) if masks else 0
    penalty = DEFAULT_SETTINGS.ridge_lambdas[chosen]
    intercept, coef = multivariate_ridge(x, y, penalty)
    arrays = {'factor_intercept': intercept, 'factor_coef': coef}
    return [], arrays, dict(train_rows=len(dates), train_months=len(dates), selected=None, trees=None,
        factor_order=factor_order, factor_history_first=str(dates[0]), factor_history_last=str(dates[-1]),
        factor_estimator='monthly linear coefficient sums with lag1 multivariate Ridge',
        ridge_lambda=penalty, cv_mse=scores[chosen], grid_cv_mse=scores, fold_records=records,
        eligible_target_dates=[str(d) for d in eligible], usable_target_dates=[str(d) for d in dates],
        missing_lag_target_dates=missing_lags, missing_lag_target_count=len(missing_lags), target_month_window=120,
        reason=None if masks else 'predeclared first original penalty; no two usable blocked folds')


# Section 5: Annual anchors and monthly prediction streams --------------------
def extended_forecast_returns(chars, features, variant, settings=FORECAST_SETTINGS,
        source_factory=MonthlySource, *, artifact_dir=None, baseline_fits=None,
        checkpoint_dir=None, identity=None):
    """Return (predictions, fit records, component identity), with fixed baseline risk."""
    if variant not in SPECIFICATIONS:
        raise ValueError('Unsupported extended forecast experiment')
    if settings.chunk_months < 1 or settings.folds < 2 or not settings.candidates:
        raise ValueError('Invalid benchmark forecast settings')
    names = canonical_features(features)
    source = source_factory(chars, names, settings.train_years * 12 + 24)
    tests = sorted(source.metadata.filter(pl.col('ctff_test'))['eom_ret'].unique().to_list())
    if not tests:
        raise ValueError('No test return months')
    factor_variant = variant in ('P07', 'P08')
    if factor_variant and artifact_dir is None:
        raise ValueError('Factor forecast variants require original risk artifacts')
    manifest = json.loads((Path(artifact_dir) / 'manifest.json').read_text()) if factor_variant else None
    factor_order = list(INDUSTRIES) + names
    if manifest is not None and manifest['factor_order'] != factor_order:
        raise ValueError('Original factor artifact order differs from supplied characteristics')
    component = _identity(variant, names, settings, identity, manifest)
    directory = _checkpoint_directory(checkpoint_dir, identity, variant)
    predictions, fits = [], []
    with threadpool_limits(limits=settings.threads):
        for offset in range(0, len(tests), settings.chunk_months):
            chunk = tests[offset:offset + settings.chunk_months]
            first = chunk[0]
            cutoff = source.return_to_formation[first]
            dates = forecast_history(source, first, settings, variant)
            metadata = dict(component=component, first_test_return=str(first), available_through=str(cutoff),
                            training_return_dates=[str(d) for d in dates])
            baseline_fit = None
            if variant == 'P06':
                matched = [fit for fit in (baseline_fits or []) if fit['test_return_start'] == str(first)]
                if len(matched) != 1:
                    raise ValueError('P06 requires one original P01 fit per annual anchor')
                baseline_fit = matched[0]
                metadata['original_selection'] = {key: baseline_fit[key] for key in
                    ('selected', 'trees', 'training_return_first', 'training_return_last')}
            cached = _load_anchor(directory, first, metadata, settings.threads)
            if cached is None:
                if variant == 'P07':
                    models, arrays, fit = _factor_mean(artifact_dir, cutoff, factor_order)
                elif variant == 'P08':
                    models, arrays, fit = _factor_lag_fit(artifact_dir, first, cutoff, factor_order, settings)
                elif variant == 'P06':
                    models, arrays, fit = _fit_bags(source, dates, first, cutoff, settings, baseline_fit)
                else:
                    models, arrays, fit = _fit_xgb(source, dates, first, cutoff, settings, variant)
                fit = dict(fit, formation_cutoff=str(cutoff), validation_policy='historical_blocked', variant=variant)
                _save_anchor(directory, first, metadata, models, arrays, fit)
            else:
                models, arrays, fit = cached
            for d in chunk:
                formation = source.return_to_formation[d]
                frame = source.load(formation).filter(pl.col('ctff_test'))
                if factor_variant:
                    path = Path(artifact_dir) / 'months' / str(formation)
                    marker = json.loads((path / 'complete.json').read_text())
                    if marker['available_through'] > str(formation) or marker['factor_order'] != factor_order:
                        raise ValueError('Current factor exposures exceed formation cutoff or change coordinates')
                    with np.load(path / 'risk.npz', allow_pickle=False) as risk:
                        if not np.array_equal(risk['ids'], frame['id'].to_numpy()):
                            raise ValueError('Current factor exposures and forecast universe differ')
                        if variant == 'P07':
                            factor_forecast = arrays['factor_forecast']
                        else:
                            observed = monthly_factor_targets(artifact_dir, formation, factor_order, month_number(formation))
                            if formation not in observed:
                                raise ValueError('P08 current formation lacks an observed completed factor lag')
                            factor_forecast = arrays['factor_intercept'] + observed[formation] @ arrays['factor_coef']
                        values = risk['B'] @ factor_forecast
                else:
                    x = frame.select(names).to_numpy().astype(np.float32)
                    if 'pca_mean' in arrays:
                        x = ((x-arrays['pca_mean']) @ arrays['pca_components']).astype(np.float32)
                    values = np.mean([m.inplace_predict(x) for m in models], axis=0) if models else np.zeros(frame.height)
                predictions.append(_month_prediction(frame, values, directory, first, metadata, d))
            fits.append(dict(fit, test_return_start=str(first), test_return_end=str(chunk[-1]),
                              training_return_first=str(dates[0]) if dates else None,
                              training_return_last=str(dates[-1]) if dates else None))
            print(f"{variant} {first}: {fit['train_rows']:,} training observations", flush=True)
    return pl.concat(predictions).sort(['eom', 'id']), fits, component
