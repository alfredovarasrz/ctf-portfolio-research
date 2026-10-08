"""Local, resumable Barra artifacts. The frozen benchmark risk math is reused."""
from collections import deque
from dataclasses import asdict
from datetime import date
import hashlib
import inspect
import json
from importlib.metadata import version
from pathlib import Path
import shutil

import numpy as np
import polars as pl
from threadpoolctl import threadpool_limits

from baseline import MonthlySource, as_lazy, canonical_chars, canonical_features, prepare_pred_data
from comparison_models import (
    INDUSTRIES, RISK_SETTINGS, DailySource, RiskSource, SpecificRiskState,
    factor_covariance, ff12_class, glmnet_ridge, optimize_portfolios, weighted_covariance,
)


# Shared utilities ------------------------------------------------------------
def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), default=str)


def _write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(_json(value) + '\n')
    temporary.replace(path)


def _input_identity(data, keys):
    if isinstance(data, pl.LazyFrame):
        raise ValueError('Partition/lazy inputs require an explicit input identity')
    frame = as_lazy(data).collect().sort(keys).select(sorted(as_lazy(data).collect_schema().names()))
    return dict(rows=frame.height, schema=str(frame.schema),
                sha256=hashlib.sha256(frame.hash_rows(seed=1).to_numpy().tobytes()).hexdigest())


def _manifest(chars, daily_ret, names, settings, identity, source_factory, daily_factory):
    if identity is None:
        identity = dict(chars=_input_identity(chars, ['eom', 'id']),
                        daily=_input_identity(daily_ret, ['date', 'id']))
    functions = (RiskSource, DailySource, SpecificRiskState, glmnet_ridge,
                 factor_covariance, weighted_covariance, ff12_class, prepare_pred_data,
                 MonthlySource, canonical_chars, canonical_features, as_lazy,
                 _json, _input_identity, _manifest, _save_checkpoint, _load_checkpoint,
                 _month_directory, build_risk_artifacts)
    code = '\n'.join(inspect.getsource(f) for f in functions)
    # Allocation policy does not invalidate risk estimates.
    risk_settings = asdict(settings)
    risk_settings.pop('target_annual_volatility')
    return json.loads(_json(dict(version=1, identity=identity, settings=risk_settings,
                                features=names, factor_order=list(INDUSTRIES) + names,
                                code_sha256=hashlib.sha256(code.encode()).hexdigest(),
                                source=source_factory.__qualname__, daily=daily_factory.__qualname__,
                                numpy=np.__version__, polars=pl.__version__,
                                scipy=version('scipy'), threadpoolctl=version('threadpoolctl'),
                                units='daily covariance; preceding-month exposures')))


def _save_checkpoint(path, state, returns, return_dates, risk_model, risk_model_date, d, records):
    metadata = dict(last_month=str(d), day_index=state.day_index,
                    factor_dates=[str(v) for v in return_dates],
                    specific_risk_model_date=str(risk_model_date) if risk_model_date else None,
                    records=records)
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as stream:
        np.savez(stream, metadata=np.array(_json(metadata)), ids=state.ids,
                 count=state.count, seed_squares=state.seed_squares, seed_count=state.seed_count,
                 variance=state.variance, previous=state.previous, history=state.history,
                 factors=np.array(returns), risk_model=np.array([]) if risk_model is None else risk_model)
    temporary.replace(path)


def _load_checkpoint(path, ids, settings):
    state = SpecificRiskState(ids, settings)
    if not path.exists():
        return state, deque(maxlen=settings.covariance_days), deque(maxlen=settings.covariance_days), None, None, None, []
    with np.load(path, allow_pickle=False) as saved:
        metadata = json.loads(str(saved['metadata']))
        if not np.array_equal(ids, saved['ids']):
            raise ValueError('Risk checkpoint security universe differs from its inputs')
        for field in ('count', 'seed_squares', 'seed_count', 'variance', 'previous', 'history'):
            setattr(state, field, saved[field].copy())
        state.day_index = metadata['day_index']
        returns = deque((row.copy() for row in saved['factors']), maxlen=settings.covariance_days)
        return_dates = deque((date.fromisoformat(v) for v in metadata['factor_dates']), maxlen=settings.covariance_days)
        risk_model = saved['risk_model'].copy() if saved['risk_model'].size else None
    model_date = metadata['specific_risk_model_date']
    return state, returns, return_dates, risk_model, date.fromisoformat(model_date) if model_date else None, date.fromisoformat(metadata['last_month']), metadata['records']


def _month_directory(root, d):
    return root / 'months' / str(d)


# Risk construction -----------------------------------------------------------
def build_risk_artifacts(chars, features, daily_ret, artifact_dir, *, settings=RISK_SETTINGS,
                         identity=None, source_factory=RiskSource, daily_factory=DailySource):
    """Save monthly exposures, daily factors/residuals and scored B/F/D; resume identically.

    A supplied identity must fingerprint the complete inputs and custom source code.
    Completed monthly directories publish before the single latest checkpoint. A
    month published just before interruption is safely recomputed on resumption.
    """
    names = canonical_features(features)
    manifest = _manifest(chars, daily_ret, names, settings, identity, source_factory, daily_factory)
    root = Path(artifact_dir)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / 'manifest.json'
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError('Risk artifact identity mismatch; use a separate artifact directory')
    else:
        if any(root.iterdir()):
            raise ValueError('Risk directory has files but no manifest; use a separate directory')
        _write_json(manifest_path, manifest)
    source = source_factory(chars, names, 144)
    days = daily_factory(daily_ret)
    months = sorted(source.metadata['eom'].unique().to_list())
    test_months = set(source.metadata.filter(pl.col('ctff_test'))['eom'].unique().to_list())
    if not test_months:
        raise ValueError('No test formation months')
    months = [d for d in months if d <= max(test_months)]
    ids = np.array(sorted(source.metadata['id'].unique().to_list()))
    state, returns, return_dates, risk_model, risk_model_date, last_month, records = _load_checkpoint(root / 'checkpoint.npz', ids, settings)
    previous = source.load_exposures(last_month) if last_month else None
    factor_names = list(INDUSTRIES) + names
    (root / 'months').mkdir(exist_ok=True)
    with threadpool_limits(limits=settings.threads):
        for d in months:
            if last_month and d <= last_month:
                continue
            specific = {}; factors = []; factor_dates = []; residual_frames = []
            if previous is not None and previous[0]['eom_ret'][0] == d:
                prev_frame, prev_x, _ = previous
                prev_ids = prev_frame['id'].to_numpy()
                daily = days.load(d).filter(pl.col('id').is_in(prev_ids.tolist()))
                for group in daily.partition_by('date', maintain_order=True):
                    day_ids = group['id'].to_numpy()
                    x = prev_x[np.searchsorted(prev_ids, day_ids)]
                    y = group['ret_exc'].to_numpy()
                    if len(y) < 2:
                        continue
                    coef = glmnet_ridge(x, y, settings.ridge_lambda)
                    residual = y - x @ coef
                    trading_date = group['date'][0]
                    returns.append(coef); return_dates.append(trading_date)
                    factors.append(coef); factor_dates.append(trading_date)
                    positions = np.searchsorted(ids, day_ids)
                    vol, eligible = state.advance(positions, residual)
                    for security, value in zip(day_ids[eligible], vol[eligible]):
                        specific[int(security)] = float(value)
                    residual_frames.append(group.select('id', 'date').with_columns(pl.Series('res', residual)))
            frame, x, constant = source.load_exposures(d)
            frame_ids = frame['id'].to_numpy()
            actual = np.array([specific.get(int(i), np.nan) for i in frame_ids])
            available = np.isfinite(actual) & (actual > 0)
            if available.sum() >= 2:
                risk_model = glmnet_ridge(x[available], np.log(actual[available]), settings.ridge_lambda)
                risk_model_date = d
            temporary = root / 'months' / (str(d) + '.tmp')
            if temporary.exists():
                shutil.rmtree(temporary)
            temporary.mkdir()
            np.savez(temporary / 'exposures.npz', ids=frame_ids, eom=np.array(str(d)),
                     eom_ret=np.array(str(frame['eom_ret'][0])), ctff_test=frame['ctff_test'].to_numpy(), B=x,
                     observed_volatility=actual, specific_risk_model=np.array([]) if risk_model is None else risk_model)
            daily_factors = pl.DataFrame(np.array(factors), schema=factor_names, orient='row') if factors else pl.DataFrame(schema={name: pl.Float64 for name in factor_names})
            daily_factors.insert_column(0, pl.Series('date', factor_dates, dtype=pl.Date))
            daily_factors.write_parquet(temporary / 'factor_returns.parquet')
            residuals = pl.concat(residual_frames) if residual_frames else pl.DataFrame(schema={'id': pl.Int64, 'date': pl.Date, 'res': pl.Float64})
            residuals.write_parquet(temporary / 'residuals.parquet')
            record = None
            if d in test_months:
                if risk_model is None:
                    raise ValueError(f'No specific-risk model available by {d}; need 200 residuals in 252 trading dates')
                vol = np.exp(x @ risk_model)
                vol[available] = actual[available]
                selection = frame['ctff_test'].to_numpy()
                diagonal = vol[selection] ** 2
                if not np.isfinite(diagonal).all() or np.any(diagonal <= 0):
                    raise ValueError('Invalid predicted specific risk')
                covariance = factor_covariance(np.array(returns), settings)
                np.savez(temporary / 'risk.npz', ids=frame_ids[selection], eom=np.array(str(d)), B=x[selection], F=covariance, D=diagonal)
                record = dict(formation_date=str(d), factor_return_days=len(returns),
                              factor_count=x.shape[1], constant_characteristics=constant,
                              specific_risk_observed=int(available[selection].sum()),
                              specific_risk_predicted=int((~available[selection]).sum()),
                              specific_risk_model_date=str(risk_model_date),
                              markowitz_predicted_annual_volatility=None)
                records.append(record)
            _write_json(temporary / 'complete.json', dict(formation_date=str(d),
                        available_through=str(d), factor_returns_through=str(return_dates[-1]) if return_dates else None,
                        preceding_exposure_date=str(last_month) if last_month else None,
                        specific_risk_model_date=str(risk_model_date) if risk_model_date else None,
                        factor_order=factor_names, record=record))
            final = _month_directory(root, d)
            # Only an uncommitted month can already exist after interruption.
            if final.exists():
                shutil.rmtree(final)
            temporary.replace(final)
            _save_checkpoint(root / 'checkpoint.npz', state, returns, return_dates, risk_model, risk_model_date, d, records)
            _write_json(root / 'records.json', records)
            previous = (frame, x, constant)
            last_month = d
            if factors and (d.month == 12 or d in test_months):
                print(f'Barra artifacts {d}: {len(factors)} daily fits, {available.sum()} observed specific risks', flush=True)
    # Repair the compact record file if the last checkpoint publication preceded
    # an interruption before its JSON mirror was written.
    _write_json(root / 'records.json', records)
    return records


# Allocation reuse ------------------------------------------------------------
def allocate_from_risk_artifacts(artifact_dir, predictions=None, *, settings=RISK_SETTINGS):
    """Reuse saved B/F/D with alternative forecasts, calling the original allocator."""
    root = Path(artifact_dir)
    with np.load(root / 'checkpoint.npz', allow_pickle=False) as checkpoint:
        records = json.loads(str(checkpoint['metadata']))['records']
    minimum_weights = []; markowitz_weights = []; allocated_records = []
    with threadpool_limits(limits=settings.threads):
        for record in records:
            d = date.fromisoformat(record['formation_date'])
            directory = _month_directory(root, d)
            if not (directory / 'complete.json').exists():
                raise ValueError(f'Incomplete risk artifact month {d}')
            with np.load(directory / 'risk.npz', allow_pickle=False) as saved:
                ids = saved['ids']; x = saved['B']; covariance = saved['F']; diagonal = saved['D']
                mu = None
                if predictions is not None:
                    keys = pl.DataFrame({'id': ids, 'eom': [d] * len(ids)})
                    matched = keys.join(predictions.select('id', 'eom', 'pred'), on=['id', 'eom'], how='left').sort('id')
                    if matched.height != len(ids) or matched['pred'].null_count() or not matched['pred'].is_finite().all():
                        raise ValueError('Expected returns missing, duplicate or invalid for risk-model securities')
                    mu = matched['pred'].to_numpy()
                minimum, markowitz, predicted_vol = optimize_portfolios(ids, d, x, diagonal, covariance, mu, settings)
            minimum_weights.append(minimum)
            if markowitz is not None:
                markowitz_weights.append(markowitz)
            allocated_records.append(dict(record, markowitz_predicted_annual_volatility=predicted_vol))
    if not minimum_weights:
        raise ValueError('No completed test-month risk artifacts')
    return pl.concat(minimum_weights), pl.concat(markowitz_weights) if markowitz_weights else None, allocated_records
