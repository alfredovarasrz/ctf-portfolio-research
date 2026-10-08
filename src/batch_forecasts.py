"""Bounded batch forecast variants reusing the original historical benchmark CV."""
from hashlib import sha256
from pathlib import Path

import polars as pl
from threadpoolctl import threadpool_limits

from baseline import (DEFAULT_SETTINGS, MonthlySource, SufficientStats,
                      canonical_features, fit_ridge, month_number)
from research_forecasts import (_checkpoint_directory, _checkpoint_metadata,
                                _load_fit, _save_fit, _predictions)
from ridge_path import initial_ridge_lambdas, validate_ridge_path


def blocked_ridge_fold_stats(source, dates, folds, selection_cutoff):
    """Exact usable-month grouping and total-minus-fold arithmetic in train_ridge."""
    dates = sorted(dates)
    if any(d > selection_cutoff for d in dates):
        raise ValueError('Training labels exceed actual outer formation cutoff')
    total = SufficientStats.empty(len(source.features))
    usable = []
    for d in dates:
        stats = source.stats(d)
        total.accumulate(stats)
        if stats.n:
            usable.append(d)
    fold_count = min(folds, len(usable))
    blocks = [SufficientStats.empty(len(source.features)) for _ in range(fold_count)]
    block_dates = [[] for _ in range(fold_count)]
    for i, d in enumerate(usable):
        block = min(fold_count - 1, int(i * fold_count / max(1, len(usable) - 1)))
        blocks[block].accumulate(source.stats(d))
        block_dates[block].append(d)
    pairs, records = [], []
    for i, val in enumerate(blocks if fold_count > 1 else []):
        train = total.minus(val)
        if not train.n or not val.n:
            continue
        train_dates = [d for j, group in enumerate(block_dates) if j != i for d in group]
        records.append(dict(training_return_first=str(min(train_dates)),
                            training_return_last=str(max(train_dates)),
                            validation_return_first=str(block_dates[i][0]),
                            validation_return_last=str(block_dates[i][-1]),
                            training_return_dates=[str(d) for d in sorted(train_dates)],
                            validation_return_dates=[str(d) for d in block_dates[i]],
                            train_rows=train.n, validation_rows=val.n))
        pairs.append((train, val))
    return total, pairs, records, len(usable), fold_count


def dense_ridge_fit(source, dates, settings, selection_cutoff):
    total, pairs, records, usable, fold_count = blocked_ridge_fold_stats(
        source, dates, settings.folds, selection_cutoff)
    if pairs:
        penalty, path = validate_ridge_path(pairs, fold_records=records, selection_cutoff=selection_cutoff,
                                           initial_lambdas=initial_ridge_lambdas(settings.ridge_lambdas))
        control = dict(zip(path['initial']['lambdas'], path['initial']['mse']))
        scores = [control[p] for p in settings.ridge_lambdas]
        mse, reason = path['selected_mse'], None
    else:
        penalty, path = settings.ridge_lambdas[0], None
        scores, mse = [None] * len(settings.ridge_lambdas), None
        reason = 'predeclared first original penalty; fewer than two usable historical months'
    model = fit_ridge(total, penalty)
    fit = dict(train_rows=total.n, train_months=usable, folds=fold_count,
               ridge_lambda=penalty, cv_mse=mse, grid_cv_mse=scores,
               fold_records=records, dense_path=path, reason=reason,
               validation_policy='historical_blocked', formation_cutoff=str(selection_cutoff))
    return model, fit


def dense_ridge_returns(chars, features, settings=DEFAULT_SETTINGS, source_factory=MonthlySource,
                        *, checkpoint_dir=None, identity=None):
    """H01 forecasts; covariance and allocation remain unchanged downstream."""
    names = canonical_features(features)
    source = source_factory(chars, names, settings.train_years * 12 + 24)
    tests = sorted(source.metadata.filter(pl.col('ctff_test'))['eom_ret'].unique().to_list())
    if not tests:
        raise ValueError('No test return months')
    if settings.test_period_length < 1 or settings.folds < 2 or not settings.ridge_lambdas or any(p <= 0 for p in settings.ridge_lambdas):
        raise ValueError('Invalid Ridge forecast settings')
    directory = _checkpoint_directory(checkpoint_dir, identity, 'ridge')
    forecasts, fits = [], []
    with threadpool_limits(limits=settings.blas_threads):
        for offset in range(0, len(tests), settings.test_period_length):
            chunk = tests[offset:offset + settings.test_period_length]
            first = chunk[0]
            cutoff = source.return_to_formation[first]
            lower = month_number(first) - settings.train_years * 12
            dates = sorted(d for d in source.return_to_formation
                           if lower <= month_number(d) < month_number(first) and d <= cutoff)
            metadata = None
            if directory:
                metadata = _checkpoint_metadata('ridge', names, settings, 0, identity, first, cutoff, dates)
                metadata['validation'] = 'historical_blocked'
                metadata['search'] = 'spectral-blocked-ridge-v2'
                metadata['dense_source_sha256'] = {name: sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                                   for name in ('batch_forecasts.py', 'ridge_path.py')}
            cached = _load_fit(directory, first, metadata, 'ridge')
            model, fit = cached if cached is not None else dense_ridge_fit(source, dates, settings, cutoff)
            if cached is None:
                _save_fit(directory, first, metadata, 'ridge', model, fit)
            fit = dict(fit, test_return_start=str(first), test_return_end=str(chunk[-1]),
                       training_return_first=str(dates[0]) if dates else None,
                       training_return_last=str(dates[-1]) if dates else None)
            forecasts.extend(_predictions(source, names, chunk, model, 'ridge', directory, first, metadata))
            fits.append(fit)
            print(f"Dense Ridge {first}: {fit['train_rows']:,} rows, lambda {fit['ridge_lambda']:g}", flush=True)
    return pl.concat(forecasts).sort(['eom', 'id']), fits
