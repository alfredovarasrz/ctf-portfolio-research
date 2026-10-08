"""V01: benchmark-shaped forecasts with chronological historical validation.

The original benchmark ports remain unchanged. Optional local checkpoints are
separate from submission execution and require a caller-supplied input identity.
"""
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
import gc
import json

import numpy as np
import polars as pl
from threadpoolctl import threadpool_limits
import xgboost as xgb

from baseline import (DEFAULT_SETTINGS, MonthlySource, SufficientStats,
                      canonical_features, fit_ridge, month_number, validation_mse,
                      run_pipeline)
from comparison_models import FORECAST_SETTINGS, _xgb_params, forecast_returns


# Section 2: Historical windows and local checkpoint utilities -----------------
def forward_folds(source, dates, folds=5, minimum_training_months=24):
    """Each validation block trains on completed labels at its first formation.

    The first 24 historical months supply initial training. The remainder forms
    consecutive, approximately equal blocks. All stocks of a month stay together.
    """
    if minimum_training_months < 1 or folds < 1:
        raise ValueError('Positive initial history and fold count required')
    dates = sorted(dates)
    result = []
    for block in np.array_split(np.array(dates[minimum_training_months:], dtype=object), folds):
        if not len(block):
            continue
        validation = block.tolist()
        cutoff = source.return_to_formation[validation[0]]
        training = [d for d in dates if d < validation[0] and d <= cutoff]
        if len(training) >= minimum_training_months:
            result.append(dict(training=training, validation=validation, cutoff=cutoff))
    return result


def _fold_record(fold, train_rows, validation_rows):
    return dict(training_return_first=str(fold['training'][0]),
                training_return_last=str(fold['training'][-1]),
                validation_return_first=str(fold['validation'][0]),
                validation_return_last=str(fold['validation'][-1]),
                validation_formation_cutoff=str(fold['cutoff']),
                train_rows=train_rows, validation_rows=validation_rows)


def _history(source, first, train_years):
    cutoff = source.return_to_formation[first]
    lower = month_number(first) - train_years * 12
    return sorted(d for d in source.return_to_formation
                  if lower <= month_number(d) and d < first and d <= cutoff)


def _checkpoint_metadata(kind, names, settings, minimum, identity, first, cutoff, dates):
    # Hash only forecast component sources. Risk/evaluation edits do not invalidate.
    component = sha256()
    for filename in ('research_forecasts.py', 'baseline.py', 'comparison_models.py'):
        component.update(Path(__file__).with_name(filename).read_bytes())
    metadata = dict(kind=kind, features=names, settings=asdict(settings),
                    validation='forward-blocks-v1', minimum_training_months=minimum,
                    identity=identity, code_sha256=component.hexdigest(),
                    first_test_return=str(first), available_through=str(cutoff),
                    training_return_dates=[str(d) for d in dates])
    return json.loads(json.dumps(metadata, sort_keys=True))


def _write_json(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2) + '\n')
    temporary.replace(path)


def _checkpoint_directory(checkpoint_dir, identity, kind):
    if checkpoint_dir is None:
        return None
    if identity is None:
        raise ValueError('Local checkpoints require an explicit input identity')
    directory = Path(checkpoint_dir) / kind
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _fit_paths(directory, first, kind):
    anchor = directory / str(first)
    anchor.mkdir(parents=True, exist_ok=True)
    return anchor, anchor / 'fit.json', anchor / ('model.npz' if kind == 'ridge' else 'model.json')


def _load_fit(directory, first, metadata, kind):
    if directory is None:
        return None
    anchor, marker, model_path = _fit_paths(directory, first, kind)
    if not marker.exists():
        return None
    saved = json.loads(marker.read_text())
    if saved['metadata'] != metadata:
        raise ValueError(f'Forecast checkpoint identity mismatch: {anchor}')
    if kind == 'ridge':
        with np.load(model_path, allow_pickle=False) as values:
            model = (float(values['intercept']), values['coef'].copy())
    elif saved['fit']['train_rows']:
        model = xgb.Booster(params={'nthread': metadata['settings']['threads']})
        model.load_model(model_path)
    else:
        model = None
    return model, saved['fit']


def _save_fit(directory, first, metadata, kind, model, fit):
    if directory is None:
        return
    _, marker, model_path = _fit_paths(directory, first, kind)
    temporary = model_path.with_name('temporary-' + model_path.name)
    if kind == 'ridge':
        with temporary.open('wb') as stream:
            np.savez(stream, intercept=model[0], coef=model[1])
        temporary.replace(model_path)
    elif model is not None:
        model.save_model(temporary)
        temporary.replace(model_path)
    _write_json(marker, dict(metadata=metadata, fit=fit))


def _predictions(source, names, chunk, model, kind, directory, first, metadata):
    forecasts = []
    for d in chunk:
        formation = source.return_to_formation[d]
        frame = source.load(formation).filter(pl.col('ctff_test'))
        marker = path = None
        if directory is not None:
            anchor, _, _ = _fit_paths(directory, first, kind)
            path = anchor / (str(formation) + '.parquet')
            marker = anchor / (str(formation) + '.json')
        # Only this month's file is opened, even when a future chunk is cached.
        expected = dict(metadata=metadata, formation=str(formation), return_date=str(d))
        if marker is not None and marker.exists():
            if json.loads(marker.read_text()) != expected:
                raise ValueError(f'Prediction checkpoint identity mismatch: {marker}')
            pred = pl.read_parquet(path)
            keys = frame.select('id', 'eom', 'excntry')
            if not pred.select('id', 'eom', 'excntry').equals(keys):
                raise ValueError('Prediction checkpoint security coverage mismatch')
        else:
            x = frame.select(names).to_numpy()
            if kind == 'ridge':
                values = x @ model[1] + model[0]
            else:
                values = np.zeros(frame.height) if model is None else model.inplace_predict(x.astype(np.float32))
            pred = frame.select('id', 'eom', 'excntry').with_columns(pl.Series('pred', values).cast(pl.Float64))
            if not pred['pred'].is_finite().all():
                raise ValueError('Nonfinite return forecasts')
            if path is not None:
                temporary = path.with_name(path.name + '.tmp')
                pred.write_parquet(temporary)
                temporary.replace(path)
                _write_json(marker, expected)
        forecasts.append(pred)
    return forecasts


def _setup(chars, features, settings, source_factory, chunk_months):
    names = canonical_features(features)
    source = source_factory(chars, names, settings.train_years * 12 + 24)
    tests = sorted(source.metadata.filter(pl.col('ctff_test'))['eom_ret'].unique().to_list())
    if not tests:
        raise ValueError('No test return months')
    if chunk_months < 1 or settings.folds < 1:
        raise ValueError('Invalid forecast chunk/fold settings')
    return names, source, tests


# Section 3: Ridge and XGBoost training ----------------------------------------
def _ridge_fit(source, dates, folds, settings):
    total = SufficientStats.empty(len(source.features))
    for d in dates:
        total.accumulate(source.stats(d))
    scores = [[] for _ in settings.ridge_lambdas]
    records = []
    for fold in folds:
        train = SufficientStats.empty(len(source.features))
        val = SufficientStats.empty(len(source.features))
        for d in fold['training']:
            train.accumulate(source.stats(d))
        for d in fold['validation']:
            val.accumulate(source.stats(d))
        if not train.n or not val.n:
            continue
        records.append(_fold_record(fold, train.n, val.n))
        for j, penalty in enumerate(settings.ridge_lambdas):
            scores[j].append(validation_mse(val, *fit_ridge(train, penalty)))
    means = [float(np.mean(v)) if v else None for v in scores]
    selected = min(range(len(means)), key=lambda j: (means[j], j)) if records else 0
    penalty = settings.ridge_lambdas[selected]
    fit = dict(train_rows=total.n, train_months=len(dates), folds=len(records),
               fold_records=records, ridge_lambda=penalty, cv_mse=means[selected], grid_cv_mse=means,
               reason=None if records else 'predeclared first penalty; no usable forward validation')
    return fit_ridge(total, penalty), fit


def _xgb_fit(source, names, dates, folds, settings):
    xs, ys, row_months = [], [], []
    for d in dates:
        frame = source.load(source.return_to_formation[d]).filter(pl.col('ret_exc_lead1m').is_finite())
        xs.append(frame.select(names).to_numpy().astype(np.float32))
        ys.append(frame['ret_exc_lead1m'].to_numpy().astype(np.float32))
        row_months.extend([d] * frame.height)
    if not ys or not sum(len(v) for v in ys):
        return None, dict(train_rows=0, train_months=len(dates), folds=0, fold_records=[],
                          selected=None, trees=0, reason='no completed historical labels')
    x, y = np.concatenate(xs), np.concatenate(ys)
    del xs, ys
    row_dates = np.array(row_months, dtype='datetime64[D]')
    usable = []
    records = []
    for fold in folds:
        train = np.isin(row_dates, np.array(fold['training'], dtype='datetime64[D]'))
        val = np.isin(row_dates, np.array(fold['validation'], dtype='datetime64[D]'))
        if train.any() and val.any():
            usable.append((train, val))
            records.append(_fold_record(fold, int(train.sum()), int(val.sum())))

    def search(candidates, eta, rounds):
        results = [[] for _ in candidates]
        for block, (train_mask, val_mask) in enumerate(usable):
            train = xgb.DMatrix(x[train_mask], label=y[train_mask], nthread=settings.threads)
            val = xgb.DMatrix(x[val_mask], label=y[val_mask], nthread=settings.threads)
            for j, candidate in enumerate(candidates):
                params = _xgb_params({k: v for k, v in candidate.items() if k != 'hp_set'}, eta, settings)
                model = xgb.train(params, train, num_boost_round=rounds, evals=[(val, 'val')],
                                  early_stopping_rounds=settings.early_stopping, verbose_eval=False)
                results[j].append(dict(fold=block + 1, mse=float(model.best_score)**2,
                                       trees=int(model.best_iteration) + 1))
            del train, val
            gc.collect()
        return results

    stage1 = search(settings.candidates, settings.stage1_eta, settings.stage1_rounds)
    if records:
        chosen = min(range(len(stage1)), key=lambda j: (np.mean([v['mse'] for v in stage1[j]]),
                                                       settings.candidates[j]['hp_set']))
        candidate = settings.candidates[chosen]
        stage2 = search([candidate], settings.stage2_eta, settings.stage2_rounds)[0]
        trees = max(1, int(np.floor(np.mean([v['trees'] for v in stage2]))))
    else:
        candidate, stage2, trees = settings.candidates[0], [], 1
    training = xgb.DMatrix(x, label=y, nthread=settings.threads)
    booster = xgb.train(_xgb_params({k: v for k, v in candidate.items() if k != 'hp_set'},
                                   settings.stage2_eta, settings), training,
                        num_boost_round=trees, verbose_eval=False)
    fit = dict(train_rows=len(y), train_months=len(dates), folds=len(records), fold_records=records,
               selected=dict(candidate), trees=trees, stage1=stage1, stage2=stage2,
               reason=None if records else 'predeclared first candidate and one tree; no usable forward validation')
    return booster, fit


# Section 4: Forecast pipelines ------------------------------------------------
def benchmark_forecasts(chars, features, settings, source_factory=MonthlySource,
                        *, kind, checkpoint_dir=None, identity=None):
    """Reuse the original four-block fit / one-block validation without changes.

    Both fitting and validation use labels completed by the actual outer
    formation date. Local persistence records the entire forecast stream;
    interrupted incomplete forecast builds are refitted on the next invocation.
    """
    directory = _checkpoint_directory(checkpoint_dir, identity, kind)
    identity = json.loads(json.dumps(identity, sort_keys=True))
    marker = directory / 'blocked-fit.json' if directory else None
    if marker is not None and marker.exists():
        saved = json.loads(marker.read_text())
        if saved['identity'] != identity:
            raise ValueError(f'Forecast checkpoint identity mismatch: {directory}')
        return pl.read_parquet(directory / 'blocked-predictions.parquet'), saved['fits']
    if kind == 'ridge':
        _, predictions, fits = run_pipeline(chars, features, settings,
                                            source_factory=source_factory)
    elif kind == 'xgboost':
        predictions, fits = forecast_returns(chars, features, settings, source_factory)
    else:
        raise ValueError('Unknown forecast kind')
    source = source_factory(chars, canonical_features(features))
    for fit in fits:
        first = next(d for d in source.return_to_formation
                     if str(d) == fit['test_return_start'])
        cutoff = str(source.return_to_formation[first])
        if fit['training_return_last'] is not None and fit['training_return_last'] > cutoff:
            raise ValueError('Historical labels extend beyond actual portfolio formation')
        fit.update(formation_cutoff=cutoff, validation_policy='historical_blocked')
    predictions = predictions.sort(['eom', 'id'])
    if directory is not None:
        temporary = directory / 'blocked-predictions.parquet.tmp'
        predictions.write_parquet(temporary)
        temporary.replace(directory / 'blocked-predictions.parquet')
        _write_json(marker, dict(identity=identity, fits=fits))
    return predictions, fits


def forward_ridge_returns(chars, features, settings=DEFAULT_SETTINGS, source_factory=MonthlySource,
                          *, minimum_training_months=24, checkpoint_dir=None, identity=None):
    names, source, tests = _setup(chars, features, settings, source_factory, settings.test_period_length)
    if not settings.ridge_lambdas or any(p <= 0 for p in settings.ridge_lambdas):
        raise ValueError('Ridge penalties must be positive')
    directory = _checkpoint_directory(checkpoint_dir, identity, 'ridge')
    forecasts, fits = [], []
    with threadpool_limits(limits=settings.blas_threads):
        for offset in range(0, len(tests), settings.test_period_length):
            chunk = tests[offset:offset + settings.test_period_length]
            first = chunk[0]
            dates = _history(source, first, settings.train_years)
            folds = forward_folds(source, dates, settings.folds, minimum_training_months)
            metadata = _checkpoint_metadata('ridge', names, settings, minimum_training_months, identity,
                                             first, source.return_to_formation[first], dates) if directory else None
            cached = _load_fit(directory, first, metadata, 'ridge')
            model, fit = cached if cached is not None else _ridge_fit(source, dates, folds, settings)
            if cached is None:
                _save_fit(directory, first, metadata, 'ridge', model, fit)
            fit = dict(fit, test_return_start=str(first), test_return_end=str(chunk[-1]),
                       formation_cutoff=str(source.return_to_formation[first]),
                       training_return_first=str(dates[0]) if dates else None,
                       training_return_last=str(dates[-1]) if dates else None)
            forecasts.extend(_predictions(source, names, chunk, model, 'ridge', directory, first, metadata))
            fits.append(fit)
            print(f"Forward Ridge {first}: {fit['train_rows']:,} rows, lambda {fit['ridge_lambda']:g}", flush=True)
    return pl.concat(forecasts).sort(['eom', 'id']), fits


def forward_forecast_returns(chars, features, settings=FORECAST_SETTINGS, source_factory=MonthlySource,
                             *, minimum_training_months=24, checkpoint_dir=None, identity=None):
    names, source, tests = _setup(chars, features, settings, source_factory, settings.chunk_months)
    if not settings.candidates or settings.stage1_rounds < 1 or settings.stage2_rounds < 1:
        raise ValueError('Invalid XGBoost candidates/rounds')
    directory = _checkpoint_directory(checkpoint_dir, identity, 'xgboost')
    forecasts, fits = [], []
    with threadpool_limits(limits=settings.threads):
        for offset in range(0, len(tests), settings.chunk_months):
            chunk = tests[offset:offset + settings.chunk_months]
            first = chunk[0]
            dates = _history(source, first, settings.train_years)
            folds = forward_folds(source, dates, settings.folds, minimum_training_months)
            metadata = _checkpoint_metadata('xgboost', names, settings, minimum_training_months, identity,
                                             first, source.return_to_formation[first], dates) if directory else None
            cached = _load_fit(directory, first, metadata, 'xgboost')
            model, fit = cached if cached is not None else _xgb_fit(source, names, dates, folds, settings)
            if cached is None:
                _save_fit(directory, first, metadata, 'xgboost', model, fit)
            fit = dict(fit, test_return_start=str(first), test_return_end=str(chunk[-1]),
                       formation_cutoff=str(source.return_to_formation[first]),
                       training_return_first=str(dates[0]) if dates else None,
                       training_return_last=str(dates[-1]) if dates else None)
            forecasts.extend(_predictions(source, names, chunk, model, 'xgboost', directory, first, metadata))
            fits.append(fit)
            print(f"Forward XGBoost {first}: {fit['train_rows']:,} rows, {fit['trees']} trees", flush=True)
    return pl.concat(forecasts).sort(['eom', 'id']), fits
