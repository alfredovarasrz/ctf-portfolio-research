"""Expanding forecasts inheriting authenticated original annual tree parameters."""
from dataclasses import asdict, replace
from datetime import date
from hashlib import sha256
import gc
import json
from math import isfinite
from pathlib import Path

import numpy as np
import polars as pl
from threadpoolctl import threadpool_limits
import xgboost as xgb

from baseline import MonthlySource, canonical_features, month_end, month_number
from comparison_models import FORECAST_SETTINGS
from expanding_forecasts import (SPECIFICATIONS as ORIGINAL_SPECIFICATIONS, completed_history,
    fit_expanding, month_prediction, check_model_bundle)
from extended_forecasts import _load_anchor, _save_anchor
from research_forecasts import _checkpoint_directory, _write_json
from artifact_utils import digest

BASE=Path(__file__).resolve().parent
BASE_VARIANTS={'P04_INHERITED':'P04','P05_EXPANDING_INHERITED':'P05_EXPANDING'}
SPECIFICATIONS={key:dict(ORIGINAL_SPECIFICATIONS[value],
    tree_configuration='exact original P01 annual selected parameter record',
    inherited_parameters=['max_depth','reg_lambda','subsample','colsample_bytree'],
    stage1_tree_candidates=1,stage2_tree_count='refined on current all-history folds',
    parent_annual_booster_models_persisted=False) for key,value in BASE_VARIANTS.items()}
SOURCE_FILES=('inherited_expanding_forecasts.py','expanding_forecasts.py',
    'capped_calendar_validation.py','baseline.py','comparison_models.py',
    'extended_forecasts.py','research_forecasts.py','artifact_utils.py')


def fit_digest(value):
    return sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode('utf-8')).hexdigest()


def original_annual_selection(row,settings):
    first=date.fromisoformat(row['test_return_start'])
    cutoff=date.fromisoformat(row['formation_cutoff'])
    end=date.fromisoformat(row['test_return_end'])
    candidate=row.get('selected')
    if (row.get('validation_policy')!='historical_blocked' or cutoff!=month_end(month_number(first)-1)
            or end<first or month_number(end)-month_number(first)>=settings.chunk_months
            or row.get('training_return_last') and date.fromisoformat(row['training_return_last'])>cutoff
            or candidate not in settings.candidates or type(row.get('train_rows')) is not int or row['train_rows']<=0):
        raise ValueError('Missing or incompatible original annual selected configuration')
    if (row['training_return_first'] is None or row['training_return_last'] is None
            or month_number(date.fromisoformat(row['training_return_first']))<month_number(first)-120):
        raise ValueError('Original annual training coverage differs from rolling benchmark')
    scores=row['stage1']
    if len(scores)!=len(settings.candidates):raise ValueError('Original20 candidate curve missing')
    if any(not isfinite(r['mse']) or r['mse']<0 or type(r['trees']) is not int
            or not 1<=r['trees']<=settings.stage1_rounds for rows in scores for r in rows):
        raise ValueError('Invalid original candidate validation losses or tree counts')
    valid=[i for i,v in enumerate(scores) if v]
    if not valid:raise ValueError('Original annual parameter choice lacks validation evidence')
    selected=min(valid,key=lambda i:(float(np.mean([r['mse'] for r in scores[i]])),settings.candidates[i]['hp_set']))
    if candidate!=settings.candidates[selected]:raise ValueError('Original selected candidate and original curve disagree')
    stage2=row['stage2']
    if any(not isfinite(r['mse']) or r['mse']<0 or type(r['trees']) is not int
            or not 1<=r['trees']<=settings.stage2_rounds for r in stage2):
        raise ValueError('Invalid original second-stage validation evidence')
    if not stage2 or row['trees']!=max(1,int(np.floor(sum(r['trees'] for r in stage2)/len(stage2)))):
        raise ValueError('Original stage2 tree count disagrees with saved folds')
    return dict(first_test_return=str(first),formation_cutoff=str(cutoff),test_return_end=str(end),
        selected=dict(candidate),original_training_return_first=row['training_return_first'],
        original_training_return_last=row['training_return_last'],original_train_rows=row['train_rows'],
        parent_fit_sha256=fit_digest(row))


def parent_selection_contract(parent_path, settings=FORECAST_SETTINGS):
    """Use annual tree choices from a locally generated benchmark fit record."""
    path = Path(parent_path) / 'forecast_fits.json'
    rows = json.loads(path.read_text())['xgboost']
    selections = [original_annual_selection(row, settings) for row in rows]
    anchors = [date.fromisoformat(row['first_test_return']) for row in selections]
    if not anchors or any(month_number(b) - month_number(a) != settings.chunk_months
                          for a, b in zip(anchors, anchors[1:])):
        raise ValueError('Original annual fit records need consecutive twelve-month anchors')
    return dict(path=str(path.parent.resolve()), files_sha256={path.name: digest(path)},
                annual_selections=selections, parent_annual_booster_models_persisted=False)


def check_parent_receipt(receipt):
    """Prevent changing the annual parameter file during an inherited fit."""
    for name, value in receipt['files_sha256'].items():
        if digest(Path(receipt['path']) / name) != value:
            raise ValueError('Original annual parameter file changed during fitting')


def inherited_expanding_forecast_returns(chars,features,variant,settings=FORECAST_SETTINGS,
        source_factory=MonthlySource,*,parent_selections,checkpoint_dir=None,identity=None):
    if variant not in SPECIFICATIONS or settings.folds!=5 or settings.chunk_months!=12:
        raise ValueError('Only inherited original annual expanding specifications supported')
    if checkpoint_dir is not None and identity is None:raise ValueError('Explicit inherited input identity required')
    names=canonical_features(features);source=source_factory(chars,names,1)
    tests=sorted(source.metadata.filter(pl.col('ctff_test'))['eom_ret'].unique().to_list())
    if not tests:raise ValueError('No test months')
    component=json.loads(json.dumps(dict(variant=variant,features=names,settings=asdict(settings),
        specification=SPECIFICATIONS[variant],parent=identity,parent_selection_receipt=parent_selections,
        numpy=np.__version__,polars=pl.__version__,xgboost=xgb.__version__,
        sources={name:digest(BASE/name) for name in SOURCE_FILES}),sort_keys=True))
    directory=_checkpoint_directory(checkpoint_dir,component,variant)
    predictions,fits=[],[]
    with threadpool_limits(limits=settings.threads):
        for offset in range(0,len(tests),settings.chunk_months):
            chunk=tests[offset:offset+settings.chunk_months];first=chunk[0]
            dates,cutoff=completed_history(source,first)
            matched=[r for r in parent_selections['annual_selections'] if r['first_test_return']==str(first)]
            if (len(matched)!=1 or matched[0]['formation_cutoff']!=str(cutoff)
                    or matched[0]['test_return_end']<str(chunk[-1])
                    or matched[0]['selected'] not in settings.candidates):
                raise ValueError('Exact original annual anchor/cutoff/chunk/candidate required')
            inherited=matched[0]
            check_parent_receipt(parent_selections)
            metadata=dict(component=component,first_test_return=str(first),available_through=str(cutoff),
                training_return_dates=[str(d) for d in dates],inherited_selection=inherited)
            if directory is not None and (directory/str(first)/'fit.json').exists():
                check_model_bundle(directory/str(first),metadata)
            cached=_load_anchor(directory,first,metadata,settings.threads)
            if cached is None:
                models,arrays,fit=fit_expanding(source,dates,cutoff,
                    replace(settings,candidates=(inherited['selected'],)),BASE_VARIANTS[variant])
                fit.update(formation_cutoff=str(cutoff),validation_policy='historical_blocked',variant=variant,
                    inherited_selection=inherited,tree_configuration_selection='original P01 authenticated annual record')
                _save_anchor(directory,first,metadata,models,arrays,fit)
                if directory is not None:
                    anchor=directory/str(first)
                    files=['fit.json']+[f'model-{i}.json' for i in range(len(models))]
                    _write_json(anchor/'bundle.json',dict(metadata=metadata,files_sha256={name:digest(anchor/name) for name in files}))
            else:models,arrays,fit=cached
            for d in chunk:
                frame=source.load(source.return_to_formation[d]).filter(pl.col('ctff_test'))
                values=models[0].inplace_predict(frame.select(names).to_numpy().astype(np.float32)) if models else np.zeros(frame.height)
                predictions.append(month_prediction(frame,values,directory,first,metadata,d))
            fits.append(dict(fit,test_return_start=str(first),test_return_end=str(chunk[-1]),
                training_return_first=str(dates[0]) if dates else None,training_return_last=str(dates[-1]) if dates else None))
            print(f"{variant} {first}: inherited HP{inherited['selected']['hp_set']}, {fit['train_rows']:,} observations",flush=True)
            del models;gc.collect()
    return pl.concat(predictions).sort(['eom','id']),fits,component
