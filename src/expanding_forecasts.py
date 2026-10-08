"""P04 and explicit joint P05_EXPANDING, full-history capped blocked XGBoost."""
from dataclasses import asdict
from hashlib import sha256
import gc
import json
from pathlib import Path

import numpy as np
import polars as pl
from threadpoolctl import threadpool_limits
import xgboost as xgb

from baseline import MonthlySource, canonical_features, month_number
from comparison_models import FORECAST_SETTINGS, _xgb_params
from capped_calendar_validation import capped_calendar_blocks
from extended_forecasts import _load_anchor, _save_anchor
from research_forecasts import _checkpoint_directory, _write_json

SPECIFICATIONS = {
    'P04': {'history': 'all usable completed raw history including pre1980',
        'validation': 'all-history rotating calendar blocks', 'minimum_blocks': 5,
        'maximum_validation_calendar_months': 96, 'weighting': 'original equal observation weights'},
    'P05_EXPANDING': {'history': 'all usable completed raw history including pre1980',
        'validation': 'all-history rotating calendar blocks', 'minimum_blocks': 5,
        'maximum_validation_calendar_months': 96, 'half_life_month_candidates': [12., 36., 60., 120.],
        'power': 1., 'weight_normalization': 'mean1 within each training fit',
        'validation_loss': 'unweighted stock-return MSE',
        'joint_experiment': 'expanding history plus historically tuned hyperbolic fit weights'},
}
SOURCE_FILES = ('expanding_forecasts.py', 'capped_calendar_validation.py', 'baseline.py',
    'comparison_models.py', 'extended_forecasts.py', 'research_forecasts.py')


def completed_history(source, first):
    cutoff = source.return_to_formation[first]
    dates = sorted(d for d in source.return_to_formation if d <= cutoff and d < first)
    # Availability is established by eom_ret, not by the predictor formation date.
    if any(source.return_to_formation[d] >= d or d > cutoff for d in dates):
        raise ValueError('Historical outcomes not completed at outer formation')
    usable = source.metadata.filter(pl.col('eom_ret').is_in(dates) & pl.col('ret_exc_lead1m').is_finite())
    return sorted(usable['eom_ret'].unique().to_list()), cutoff


def training_arrays(source, dates):
    """Allocate once; one monthly Polars frame at a time, no panel concatenation."""
    counts = source.metadata.filter(pl.col('eom_ret').is_in(dates) & pl.col('ret_exc_lead1m').is_finite())
    rows = counts.height
    x, y = np.empty((rows, len(source.features)), np.float32), np.empty(rows, np.float32)
    row_month = np.empty(rows, np.int32)
    offset = 0
    for d in dates:
        frame = source.load(source.return_to_formation[d]).filter(pl.col('ret_exc_lead1m').is_finite())
        n = frame.height
        x[offset:offset+n] = frame.select(source.features).to_numpy().astype(np.float32)
        y[offset:offset+n] = frame['ret_exc_lead1m'].to_numpy().astype(np.float32)
        row_month[offset:offset+n] = month_number(d)
        if not np.isfinite(x[offset:offset+n]).all() or not np.isfinite(y[offset:offset+n]).all():
            raise ValueError('Nonfinite monthly expanding features/labels')
        offset += n
    if offset != rows:
        raise ValueError('Prepared expanding-history labels/features changed or are nonfinite')
    return x, y, row_month


def fit_expanding(source, dates, cutoff, settings, variant):
    x, y, row_month = training_arrays(source, dates)
    blocks, boundaries = capped_calendar_blocks(dates)
    folds = []
    for i, validation in enumerate(blocks):
        training = [d for d in dates if d not in set(validation)]
        if training and validation:
            folds.append(dict(fold=i+1, training_return_dates=[str(d) for d in training],
                validation_return_dates=[str(d) for d in validation], available_through=str(cutoff),
                training_rows=int(np.isin(row_month, [month_number(d) for d in training]).sum()),
                validation_rows=int(np.isin(row_month, [month_number(d) for d in validation]).sum())))
    base = dict(train_rows=len(y), train_months=len(dates), fold_records=folds,
        calendar_blocks=boundaries, folds=len(blocks), minimum_blocks=5,
        maximum_validation_calendar_months=96, training_return_dates=[str(d) for d in dates])
    if not len(y):
        return [], {}, dict(base, selected=None, trees=0, reason='no historical labels',
            transformation_search=[], stage1=[], stage2=[])
    options = SPECIFICATIONS[variant].get('half_life_month_candidates', [None])
    age = month_number(cutoff) - row_month
    if np.any(age < 0):
        raise ValueError('Future training labels')

    def weights(mask, option):
        if option is None:
            return None
        raw = (1. + age[mask] / option) ** -1.
        return raw / raw.mean()

    def search(candidates, eta, rounds, option):
        results = [[] for _ in candidates]
        for i, validation in enumerate(blocks):
            mask = np.isin(row_month, [month_number(d) for d in validation])
            if not mask.any() or mask.all():
                continue
            fit_weights = weights(~mask, option)
            train = xgb.DMatrix(x[~mask], label=y[~mask], weight=fit_weights, nthread=settings.threads)
            val = xgb.DMatrix(x[mask], label=y[mask], nthread=settings.threads)
            weight_record = dict(training_weight_mean=1. if fit_weights is None else float(fit_weights.mean()),
                training_weight_minimum=1. if fit_weights is None else float(fit_weights.min()),
                training_weight_maximum=1. if fit_weights is None else float(fit_weights.max()),
                validation_weighting='unweighted')
            for j, candidate in enumerate(candidates):
                model = xgb.train(_xgb_params({k: v for k, v in candidate.items() if k != 'hp_set'}, eta, settings),
                    train, num_boost_round=rounds, evals=[(val, 'val')],
                    early_stopping_rounds=settings.early_stopping, verbose_eval=False)
                results[j].append(dict(fold=i+1, mse=float(model.best_score)**2,
                    trees=int(model.best_iteration)+1, **weight_record))
            del train, val, model, fit_weights
            gc.collect()
        return results

    searches = [search(settings.candidates, settings.stage1_eta, settings.stage1_rounds, option) for option in options]
    valid = [(i, j) for i, result in enumerate(searches) for j, folds_result in enumerate(result) if folds_result]
    if valid:
        choice, j = min(valid, key=lambda ij: (float(np.mean([r['mse'] for r in searches[ij[0]][ij[1]]])),
            ij[0], settings.candidates[ij[1]]['hp_set']))
        option, candidate = options[choice], settings.candidates[j]
        stage2 = search([candidate], settings.stage2_eta, settings.stage2_rounds, option)[0]
        trees = max(1, int(np.floor(np.mean([r['trees'] for r in stage2]))))
        reason = None
    else:
        choice, j, option, candidate, stage2, trees = 0, 0, options[0], settings.candidates[0], [], 1
        reason = 'no usable validation folds: predeclared first candidate, one tree'
    final_weights = weights(np.ones(len(y), bool), option)
    train = xgb.DMatrix(x, label=y, weight=final_weights, nthread=settings.threads)
    model = xgb.train(_xgb_params({k: v for k, v in candidate.items() if k != 'hp_set'}, settings.stage2_eta, settings),
        train, num_boost_round=trees, verbose_eval=False)
    fit = dict(base, selected=dict(candidate), trees=trees, reason=reason,
        stage1=searches[choice], stage2=stage2, selected_half_life_months=option,
        transformation_search=[dict(option=value, tree_candidate_scores=[dict(hp_set=settings.candidates[k]['hp_set'],
            mse=float(np.mean([r['mse'] for r in result])) if result else None, folds=result)
            for k, result in enumerate(results)]) for value, results in zip(options, searches)],
        transformation_grid_boundary=(choice in (0, len(options)-1)) if variant == 'P05_EXPANDING' else None,
        final_weight_mean=1. if final_weights is None else float(final_weights.mean()),
        final_weight_minimum=1. if final_weights is None else float(final_weights.min()),
        final_weight_maximum=1. if final_weights is None else float(final_weights.max()))
    del train, x, y, row_month
    gc.collect()
    return [model], {}, fit


def month_prediction(frame, values, directory, first, metadata, return_date):
    pred = frame.select('id', 'eom', 'excntry').with_columns(pl.Series('pred', values).cast(pl.Float64))
    if not pred['pred'].is_finite().all():
        raise ValueError('Nonfinite expanding forecast')
    if directory is None:
        return pred
    path = directory / str(first) / (str(frame['eom'][0]) + '.parquet')
    marker = path.with_suffix('.json')
    expected = dict(metadata=metadata, formation=str(frame['eom'][0]), return_date=str(return_date))
    if marker.exists():
        receipt = json.loads(marker.read_text())
        if receipt['identity'] != expected or receipt['sha256'] != sha256(path.read_bytes()).hexdigest():
            raise ValueError('Expanding prediction identity or bytes changed')
        saved = pl.read_parquet(path)
        if not saved.select('id', 'eom', 'excntry').equals(pred.select('id', 'eom', 'excntry')):
            raise ValueError('Expanding prediction key mismatch')
        return saved
    temporary = path.with_name(path.name + '.tmp')
    pred.write_parquet(temporary)
    temporary.replace(path)
    _write_json(marker, dict(identity=expected, sha256=sha256(path.read_bytes()).hexdigest()))
    return pred


def check_model_bundle(anchor, metadata):
    """Validate committed model coverage before the original loader reads it."""
    receipt_path = anchor / 'bundle.json'
    if not receipt_path.exists():
        raise ValueError('Uncommitted expanding model bundle')
    receipt = json.loads(receipt_path.read_text())
    saved = json.loads((anchor / 'fit.json').read_text())
    count = saved['model_count']
    expected_count = 1 if saved['fit']['train_rows'] else 0
    expected_files = {'fit.json'} | {f'model-{i}.json' for i in range(expected_count)}
    if (type(count) is not int or count != expected_count or saved['arrays']
            or saved['metadata'] != metadata or receipt['metadata'] != metadata
            or set(receipt['files_sha256']) != expected_files):
        raise ValueError('Expanding model bundle coverage changed')
    if any(sha256((anchor / name).read_bytes()).hexdigest() != value
            for name, value in receipt['files_sha256'].items()):
        raise ValueError('Expanding model bundle changed')


def expanding_forecast_returns(chars, features, variant, settings=FORECAST_SETTINGS,
        source_factory=MonthlySource, *, checkpoint_dir=None, identity=None):
    """Return predictions, yearly fit records, component/source identity."""
    if variant not in SPECIFICATIONS or settings.chunk_months < 1 or settings.folds != 5 or not settings.candidates:
        raise ValueError('Accepted expanding variant and original forecast settings required')
    if checkpoint_dir is not None and identity is None:
        raise ValueError('Expanding checkpoints require caller input identity')
    names = canonical_features(features)
    source = source_factory(chars, names, 1)
    tests = sorted(source.metadata.filter(pl.col('ctff_test'))['eom_ret'].unique().to_list())
    if not tests:
        raise ValueError('No test months')
    component = json.loads(json.dumps(dict(variant=variant, features=names, settings=asdict(settings),
        specification=SPECIFICATIONS[variant], parent=identity, numpy=np.__version__, polars=pl.__version__,
        xgboost=xgb.__version__, sources={name: sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in SOURCE_FILES}), sort_keys=True))
    directory = _checkpoint_directory(checkpoint_dir, component, variant)
    predictions, fits = [], []
    with threadpool_limits(limits=settings.threads):
        for offset in range(0, len(tests), settings.chunk_months):
            chunk = tests[offset:offset+settings.chunk_months]
            first = chunk[0]
            dates, cutoff = completed_history(source, first)
            metadata = dict(component=component, first_test_return=str(first), available_through=str(cutoff),
                training_return_dates=[str(d) for d in dates])
            if directory is not None:
                if (directory / str(first) / 'fit.json').exists():
                    check_model_bundle(directory / str(first), metadata)
            cached = _load_anchor(directory, first, metadata, settings.threads)
            if cached is None:
                models, arrays, fit = fit_expanding(source, dates, cutoff, settings, variant)
                fit.update(formation_cutoff=str(cutoff), validation_policy='historical_blocked', variant=variant)
                _save_anchor(directory, first, metadata, models, arrays, fit)
                if directory is not None:
                    anchor = directory / str(first)
                    files = ['fit.json'] + [f'model-{i}.json' for i in range(len(models))]
                    _write_json(anchor / 'bundle.json', dict(metadata=metadata,
                        files_sha256={name: sha256((anchor / name).read_bytes()).hexdigest() for name in files}))
            else:
                models, arrays, fit = cached
            for d in chunk:
                frame = source.load(source.return_to_formation[d]).filter(pl.col('ctff_test'))
                values = models[0].inplace_predict(frame.select(names).to_numpy().astype(np.float32)) if models else np.zeros(frame.height)
                predictions.append(month_prediction(frame, values, directory, first, metadata, d))
            fits.append(dict(fit, test_return_start=str(first), test_return_end=str(chunk[-1]),
                training_return_first=str(dates[0]) if dates else None, training_return_last=str(dates[-1]) if dates else None))
            del models
            gc.collect()
    return pl.concat(predictions).sort(['eom', 'id']), fits, component
