"""Regularized allocation using the selected covariance and original risk scaling."""
from datetime import date
import numpy as np
import polars as pl

from batch_allocations import (Q_GRID, select_penalty, _realized_labels,
                               factor_loading, solve_penalized)
from comparison_models import portfolio_variance


def stock_forecasts(chars, features, threads):
    """Annual stock fits used by the equal-thirds combined strategy."""
    from dataclasses import replace
    from baseline import MonthlySource, canonical_features, month_number, train_ridge, DEFAULT_SETTINGS
    from comparison_models import FORECAST_SETTINGS, training_folds
    from research_forecasts import _xgb_fit

    names = canonical_features(features)
    source = MonthlySource(chars, names, 144)
    tests = sorted(source.metadata.filter(pl.col('ctff_test'))['eom_ret'].unique().to_list())
    settings = replace(FORECAST_SETTINGS, threads=threads)
    ridge_settings = replace(DEFAULT_SETTINGS, blas_threads=threads)
    forecasts, fits = {'ridge': [], 'xgboost': []}, {'ridge': [], 'xgboost': []}
    for start in range(0, len(tests), 12):
        chunk = tests[start:start + 12]
        first = chunk[0]
        cutoff = source.return_to_formation[first]
        dates = sorted(d for d in source.return_to_formation
            if month_number(first) - 120 <= month_number(d) < month_number(first) and d <= cutoff)
        intercept, coef, ridge_fit = train_ridge(source, dates, ridge_settings)
        blocks = training_folds(dates, first, settings)
        folds = [dict(training=[d for d in dates if d not in block], validation=block, cutoff=cutoff)
                 for block in blocks if block]
        booster, xgb_fit = _xgb_fit(source, names, dates, folds, settings)
        for kind, fit in (('ridge', ridge_fit), ('xgboost', xgb_fit)):
            fits[kind].append(dict(fit, formation_cutoff=str(cutoff),
                                  test_return_start=str(first), test_return_end=str(chunk[-1])))
        for target in chunk:
            frame = source.load(source.return_to_formation[target]).filter(pl.col('ctff_test'))
            x = frame.select(names).to_numpy()
            # Column-major arrays retain the arithmetic order of the combined research fit.
            values = {'ridge': intercept + x @ coef,
                      'xgboost': np.zeros(len(frame)) if booster is None
                      else booster.inplace_predict(x.astype(np.float32))}
            for kind, pred in values.items():
                forecasts[kind].append(frame.select('id', 'eom', 'excntry').with_columns(
                    pl.Series('pred', pred).cast(pl.Float64)))
        print(f'Stock forecast fits completed for {first.year}', flush=True)
    ridge = pl.concat(forecasts['ridge']).sort('eom', 'id')
    xgboost = pl.concat(forecasts['xgboost']).sort('eom', 'id')
    return ridge, xgboost, fits


def candidate_weights(mu, b, diagonal, covariance, reference, settings):
    loading = factor_loading(diagonal, covariance, b)
    scale = float(np.median(diagonal + np.einsum('ij,jk,ik->i', b, reference, b)))
    weights = []
    for q in Q_GRID:
        direction = solve_penalized(diagonal, loading, mu, float(q) * scale)
        variance = portfolio_variance(direction, diagonal, reference, b)
        weights.append(direction * settings.target_annual_volatility / np.sqrt(252 * variance))
    return np.array(weights), scale


def allocate_combination(risks, predictions, metadata, settings):
    history, frames, records, bank = [], [], [], []
    selections = {}
    for risk in risks:
        ids, formation = risk['ids'], risk['eom']
        forecast = predictions.filter(pl.col('eom') == formation).sort('id')
        if not np.array_equal(forecast['id'].to_numpy(), ids):
            raise ValueError('Risk and forecast stock identifiers differ')
        return_date = metadata.filter(pl.col('eom') == formation)['eom_ret'].unique().item()
        anchor = date(return_date.year, 1, 31)
        if anchor not in selections:
            _, selections[anchor] = select_penalty(history, formation, Q_GRID, portfolio='markowitz')
        selection = selections[anchor]
        q = selection['selected_q']
        candidates, scale = candidate_weights(forecast['pred'].to_numpy(), risk['B'],
            risk['D'], risk['F'], risk['reference_F'], settings)
        frames.append(pl.DataFrame({'id': ids, 'eom': [formation] * len(ids),
                                    'w': candidates[Q_GRID.index(q)]}))
        records.append(dict(formation_date=str(formation), selected_q=q, gamma=q * scale,
                            selection=selection, risk_record=risk['record']))
        # Candidate outcomes become eligible only after their return month completes.
        return_date, actual = _realized_labels(metadata, ids, formation)
        payoffs = None if actual is None else candidates @ actual
        history.append(dict(eom=formation, eom_ret=return_date, minimum=None, markowitz=payoffs))
        bank.extend(dict(eom=str(formation), eom_ret=str(return_date), q=float(value),
                         payoff=None if payoffs is None else float(payoffs[i]))
                    for i, value in enumerate(Q_GRID))
    return pl.concat(frames).sort('eom', 'id'), records, bank
