"""R01/R02 single-penalty causal replay over immutable original risk inputs."""
from dataclasses import asdict
from datetime import date, timedelta
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import polars as pl
from threadpoolctl import threadpool_limits

from comparison_models import RISK_SETTINGS, factor_covariance, glmnet_ridge
from extended_risk import _parent_months
from extended_risk_rebuilds import ProjectedArtifactSource, CachedArtifactDailySource
from research_risk import _write_json, _save_checkpoint, _load_checkpoint
from risk_penalty_paths import annual_risk_penalty


def _prepare(parent_dir, output_dir, component, settings, identity, selection):
    parent, manifest, metadata, directories = _parent_months(parent_dir)
    if component not in ('factor', 'specific'):
        raise ValueError('Risk penalty component must be factor or specific')
    model_settings = asdict(settings)
    for key, value in manifest['settings'].items():
        if key != 'threads' and model_settings[key] != value:
            raise ValueError('Penalty replay must retain original risk settings')
    root = Path(output_dir); root.mkdir(parents=True, exist_ok=True)
    specification = dict(component=component, parent_manifest_sha256=hashlib.sha256((parent/'manifest.json').read_bytes()).hexdigest(),
        adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        selector_sha256=hashlib.sha256(Path(__file__).with_name('risk_penalty_paths.py').read_bytes()).hexdigest(),
        path_evaluator_sha256=hashlib.sha256(Path(__file__).with_name('ridge_path.py').read_bytes()).hexdigest(),
        cached_inputs_sha256=hashlib.sha256(Path(__file__).with_name('extended_risk_rebuilds.py').read_bytes()).hexdigest(),
        caller_identity=identity, settings=model_settings, selection=selection,
        factor_order=manifest['factor_order'], daily_input='original preceding B*f+residual')
    # Normalize date/NumPy-compatible caller fields exactly as the JSON publisher.
    specification = json.loads(json.dumps(specification, sort_keys=True, default=str))
    marker = root/'manifest.json'
    if marker.exists():
        if json.loads(marker.read_text()) != specification:
            raise ValueError('Risk penalty replay identity mismatch; use separate directory')
    else:
        if any(root.iterdir()):
            raise ValueError('Risk penalty replay directory has no manifest')
        _write_json(marker, specification)
    (root/'months').mkdir(exist_ok=True)
    return parent, root, manifest, metadata, directories, specification


def _penalty(parent, root, component, year, cutoff, settings, selection):
    value, record = annual_risk_penalty(parent, root/'penalties', component, year, cutoff,
        baseline_lambda=settings.ridge_lambda, **selection)
    path = root/'penalties'/f'{year}.json'
    # Full dense curves are saved once per year, never duplicated in each month.
    return value, dict(penalty_component=component, selected_ridge_lambda=value,
        penalty_year=year, penalty_information_cutoff=str(cutoff),
        penalty_curve_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        penalty_selected_mse=record['selected_mse'], penalty_fallback=record['fallback'])


def _publish_month(root, temporary, d):
    final = root/'months'/str(d)
    # Only the month after the latest committed checkpoint may be replaced.
    if final.exists():
        shutil.rmtree(final)
    temporary.replace(final)


def _temporary(root, d):
    path = root/'months'/(str(d)+'.tmp')
    if path.exists():
        shutil.rmtree(path)
    path.mkdir()
    return path


def build_factor_penalty_risk(parent_dir, output_dir, *, settings=RISK_SETTINGS,
        identity=None, **selection):
    """R01: causal annual daily-factor lambda, original log-volatility lambda.

    Select from completed original historical stock cross-sections at the start
    of EVERY calendar year, including burn-in. Earlier f/residual histories keep
    the lambda available when estimated. Upstream residual changes require full
    original SpecificRiskState replay, not substitution of F alone.
    """
    parent, root, manifest, metadata, directories, _ = _prepare(parent_dir, output_dir, 'factor', settings, identity, selection)
    scored = {date.fromisoformat(r['formation_date']) for r in metadata['records']}
    if not scored:
        raise ValueError('No scored risk formation months')
    # Match original build_risk_artifacts: unscored appended future months are
    # outside the needed replay prefix and its global specific-risk id universe.
    directories = [directory for directory in directories if date.fromisoformat(directory.name)<=max(scored)]
    source = ProjectedArtifactSource(parent, np.eye(len(manifest['factor_order'])), scored)
    daily_source = CachedArtifactDailySource(parent)
    ids = np.array(sorted(source.metadata['id'].unique().to_list()))
    state, returns, return_dates, risk_model, model_date, last_month, records = _load_checkpoint(root/'checkpoint.npz', ids, settings)
    previous = source.load_exposures(last_month) if last_month else None
    # The cutoff is the prior completed calendar year, also on mid-year resume.
    # The first available calendar year has no history and uses original lambda.
    selections = {}
    with threadpool_limits(limits=settings.threads):
        for directory in directories:
            d = date.fromisoformat(directory.name)
            if last_month and d <= last_month:
                continue
            if d.year not in selections:
                cutoff = date(d.year, 1, 1)-timedelta(days=1)
                selections[d.year] = _penalty(parent, root, 'factor', d.year, cutoff, settings, selection)
            daily_lambda, penalty_record = selections[d.year]
            specific = {}; factors = []; factor_dates = []; residual_frames = []
            if previous is not None and previous[0]['eom_ret'][0] == d:
                prev_frame, prev_x, _ = previous; prev_ids = prev_frame['id'].to_numpy()
                daily = daily_source.load(d).filter(pl.col('id').is_in(prev_ids.tolist()))
                for group in daily.partition_by('date', maintain_order=True):
                    day_ids = group['id'].to_numpy(); x = prev_x[np.searchsorted(prev_ids, day_ids)]
                    y = group['ret_exc'].to_numpy()
                    if len(y) < 2:
                        continue
                    coef = glmnet_ridge(x, y, daily_lambda); residual = y-x@coef
                    trading_date = group['date'][0]
                    returns.append(coef); return_dates.append(trading_date)
                    factors.append(coef); factor_dates.append(trading_date)
                    vol, eligible = state.advance(np.searchsorted(ids, day_ids), residual)
                    specific.update((int(i), float(v)) for i, v in zip(day_ids[eligible], vol[eligible]))
                    residual_frames.append(group.select('id', 'date').with_columns(pl.Series('res', residual)))
            frame, x, constant = source.load_exposures(d); frame_ids = frame['id'].to_numpy()
            actual = np.array([specific.get(int(i), np.nan) for i in frame_ids])
            available = np.isfinite(actual)&(actual>0)
            if available.sum() >= 2:
                risk_model = glmnet_ridge(x[available], np.log(actual[available]), settings.ridge_lambda)
                model_date = d
            temporary = _temporary(root, d)
            np.savez(temporary/'exposures.npz', ids=frame_ids, eom=np.array(str(d)),
                eom_ret=np.array(str(frame['eom_ret'][0])), ctff_test=frame['ctff_test'].to_numpy(), B=x,
                observed_volatility=actual, specific_risk_model=np.array([]) if risk_model is None else risk_model)
            factor_frame = (pl.DataFrame(np.array(factors), schema=manifest['factor_order'], orient='row') if factors
                else pl.DataFrame(schema={name: pl.Float64 for name in manifest['factor_order']}))
            factor_frame.insert_column(0, pl.Series('date', factor_dates, dtype=pl.Date))
            factor_frame.write_parquet(temporary/'factor_returns.parquet')
            residuals = pl.concat(residual_frames) if residual_frames else pl.DataFrame(schema={'id':pl.Int64,'date':pl.Date,'res':pl.Float64})
            residuals.write_parquet(temporary/'residuals.parquet')
            record = None
            if d in scored:
                if risk_model is None:
                    raise ValueError(f'No specific-risk model available by {d}')
                vol = np.exp(x@risk_model); vol[available] = actual[available]
                selected = frame['ctff_test'].to_numpy(); diagonal = vol[selected]**2
                if not np.isfinite(diagonal).all() or np.any(diagonal <= 0):
                    raise ValueError('Invalid predicted specific risk')
                np.savez(temporary/'risk.npz', ids=frame_ids[selected], eom=np.array(str(d)), B=x[selected],
                    F=factor_covariance(np.array(returns), settings), D=diagonal)
                record = dict(formation_date=str(d), factor_return_days=len(returns), factor_count=x.shape[1],
                    constant_characteristics=constant, specific_risk_observed=int(available[selected].sum()),
                    specific_risk_predicted=int((~available[selected]).sum()), specific_risk_model_date=str(model_date),
                    covariance_variant='tuned_daily_factor_ridge', fixed_specific_ridge_lambda=settings.ridge_lambda,
                    covariance_information_cutoff=str(d), factor_returns_through=str(return_dates[-1]), **penalty_record)
                records.append(record)
            _write_json(temporary/'complete.json', dict(formation_date=str(d), available_through=str(d),
                factor_returns_through=str(return_dates[-1]) if return_dates else None,
                preceding_exposure_date=str(last_month) if last_month else None,
                specific_risk_model_date=str(model_date) if model_date else None,
                factor_order=manifest['factor_order'], record=record, **penalty_record))
            _publish_month(root, temporary, d)
            _save_checkpoint(root/'checkpoint.npz', state, returns, return_dates, risk_model, model_date, d, records)
            _write_json(root/'records.json', records)
            previous = (frame, x, constant); last_month = d
    _write_json(root/'records.json', records)
    return records


def build_specific_penalty_risk(parent_dir, output_dir, *, settings=RISK_SETTINGS,
        identity=None, **selection):
    """R02: tune monthly log-volatility regression penalty; original f/residuals.

    Immutable original observed residual-volatility cross-sections supply each
    fit and validation. Only the latest regression-model checkpoint is needed;
    original full residual state remains in the read-only parent artifact.
    """
    parent, root, manifest, metadata, directories, _ = _prepare(parent_dir, output_dir, 'specific', settings, identity, selection)
    scored = {r['formation_date']:r for r in metadata['records']}
    if not scored:
        raise ValueError('No scored risk formation months')
    directories = [directory for directory in directories if directory.name<=max(scored)]
    checkpoint = root/'specific_checkpoint.npz'; model = None; model_date = None; last_month = None; records = []
    if checkpoint.exists():
        with np.load(checkpoint, allow_pickle=False) as saved:
            status = json.loads(str(saved['metadata'])); model = saved['model'].copy() if saved['model'].size else None
        model_date = date.fromisoformat(status['model_date']) if status['model_date'] else None
        last_month = date.fromisoformat(status['last_month']); records = status['records']
    selections = {}
    with threadpool_limits(limits=settings.threads):
        for directory in directories:
            d = date.fromisoformat(directory.name)
            if last_month and d <= last_month:
                continue
            year = d.year+int(d.month==12)
            # Previous December is available at the annual formation cutoff.
            cutoff = date(year-1, 12, 31)
            if cutoff > d:
                cutoff = date(d.year-1, 12, 31)
            if year not in selections:
                selections[year] = _penalty(parent, root, 'specific', year, cutoff, settings, selection)
            penalty, penalty_record = selections[year]
            with np.load(directory/'exposures.npz', allow_pickle=False) as saved:
                ids = saved['ids'].copy(); x = saved['B'].copy(); actual = saved['observed_volatility'].copy()
                selected = saved['ctff_test'].copy()
            available = np.isfinite(actual)&(actual>0)
            if available.sum()>=2:
                model = glmnet_ridge(x[available], np.log(actual[available]), penalty); model_date = d
            temporary = _temporary(root,d); record = None
            if str(d) in scored:
                if model is None:
                    raise ValueError(f'No specific-risk model available by {d}')
                vol = np.exp(x@model); vol[available] = actual[available]; diagonal = vol[selected]**2
                if not np.isfinite(diagonal).all() or np.any(diagonal<=0):
                    raise ValueError('Invalid predicted specific risk')
                with np.load(directory/'risk.npz', allow_pickle=False) as saved:
                    if not np.array_equal(ids[selected], saved['ids']):
                        raise ValueError('Specific risk scored security universe mismatch')
                    covariance = saved['F'].copy()
                np.savez(temporary/'risk.npz', ids=ids[selected], eom=np.array(str(d)), B=x[selected], F=covariance,D=diagonal)
                record = dict(scored[str(d)], covariance_variant='tuned_specific_volatility_ridge',
                    specific_risk_model_date=str(model_date), fixed_daily_factor_ridge_lambda=settings.ridge_lambda,
                    covariance_information_cutoff=str(d), **penalty_record)
                records.append(record)
            _write_json(temporary/'complete.json',dict(formation_date=str(d), available_through=str(d),
                specific_risk_model_date=str(model_date) if model_date else None, record=record, **penalty_record))
            _publish_month(root,temporary,d)
            temporary_checkpoint = checkpoint.with_suffix('.tmp')
            with temporary_checkpoint.open('wb') as output:
                np.savez(output, model=np.array([]) if model is None else model,
                    metadata=np.array(json.dumps(dict(last_month=str(d),model_date=str(model_date) if model_date else None,records=records))))
            temporary_checkpoint.replace(checkpoint); _write_json(root/'records.json',records)
            last_month = d
    _write_json(root/'records.json',records)
    return records


def iter_penalty_risk(parent_dir, output_dir):
    """Common derived risk protocol with immutable original reference scaling."""
    parent = Path(parent_dir); root = Path(output_dir)
    specification = json.loads((root/'manifest.json').read_text())
    if specification['parent_manifest_sha256'] != hashlib.sha256((parent/'manifest.json').read_bytes()).hexdigest():
        raise ValueError('Penalty risk parent identity mismatch')
    for record in json.loads((root/'records.json').read_text()):
        d = date.fromisoformat(record['formation_date']); directory = root/'months'/str(d)
        marker = json.loads((directory/'complete.json').read_text())
        if marker['record'] != record or marker['available_through'] != str(d):
            raise ValueError('Penalty month commit/cutoff mismatch')
        with np.load(directory/'risk.npz',allow_pickle=False) as saved:
            risk = {name:saved[name].copy() for name in ('ids','B','F','D')}
            if str(saved['eom']) != str(d):
                raise ValueError('Penalty risk date mismatch')
        with np.load(parent/'months'/str(d)/'risk.npz',allow_pickle=False) as saved:
            if not np.array_equal(risk['ids'],saved['ids']):
                raise ValueError('Penalty/reference security universe differs')
            reference = {'reference_'+name:saved[name].copy() for name in ('B','F','D')}
        yield dict(**risk,**reference,eom=d,record=record)
