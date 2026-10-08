"""Own-regression risk-penalty MSE paths, without changing forecast time-block CV."""
from datetime import date
import hashlib
import json
from pathlib import Path

import numpy as np
import polars as pl

from comparison_models import RISK_SETTINGS, glmnet_ridge
from extended_risk import _parent_months, _factor_stream
from extended_risk_rebuilds import cached_daily_returns
from research_risk import _write_json
from ridge_path import _basis_mse, initial_ridge_lambdas


def risk_ridge_basis(x, y, validation_x, validation_y):
    """Uncentered, no-intercept Gaussian glmnet-compatible spectral MSE basis.

    Training-only constant-column exclusion and RMS(y) reproduce frozen math.
    Reuse H01's numerical path evaluator, not its centered/intercept estimator.
    """
    x=np.asarray(x,dtype=float); y=np.asarray(y,dtype=float)
    validation_x=np.asarray(validation_x,dtype=float); validation_y=np.asarray(validation_y,dtype=float)
    if (len(y)<2 or len(validation_y)<1 or x.ndim!=2 or validation_x.shape!=(len(validation_y),x.shape[1])
            or len(x)!=len(y) or not all(np.isfinite(v).all() for v in (x,y,validation_x,validation_y))):
        raise ValueError('Finite matching training and validation cross-sections required')
    rms=float(np.sqrt(np.mean(y*y))); active=np.var(x,axis=0)>0
    if not rms or not active.any():
        return (1.,len(validation_y),np.empty(0),np.empty(0),np.empty((0,0)),np.empty(0),float(validation_y@validation_y))
    train=x[:,active]; validate=validation_x[:,active]
    gram=train.T@train
    eigenvalues,vectors=np.linalg.eigh((gram+gram.T)/2)
    tolerance=1e-10*max(1.,float(np.max(np.abs(eigenvalues))))
    if eigenvalues.min() < -tolerance:
        raise ValueError('Risk regression Gram is materially indefinite')
    eigenvalues=np.maximum(eigenvalues,0.)
    return (len(y)/rms,len(validation_y),eigenvalues,vectors.T@(train.T@y),
            vectors.T@(validate.T@validate)@vectors,vectors.T@(validate.T@validation_y),float(validation_y@validation_y))


def stock_fold_bases(ids,x,y,observed_date):
    """Five deterministic security folds; held-out stock returns never fit f."""
    ids=np.asarray(ids); x=np.asarray(x); y=np.asarray(y)
    bases=[]; records=[]
    for fold in range(5):
        validation=ids%5==fold; training=~validation
        if training.sum()<2 or not validation.any():
            continue
        bases.append(risk_ridge_basis(x[training],y[training],x[validation],y[validation]))
        records.append(dict(observed_date=str(observed_date),security_fold=fold,
            training_stocks=int(training.sum()),validation_stocks=int(validation.sum()),
            security_fold_rule='integer security id modulo5'))
    return bases,records


def select_risk_lambda(sections, *, selection_cutoff, baseline_lambda=1e-4,
        initial_lambdas=None, refinement_points=10000,batch_size=512,maximum_extensions=3):
    """Score exact own-cross-section MSE, then requested wide/local refinement.

    Each section is (date,ids,x,y), already available at actual selection cutoff.
    Equal cross-section weight; stock-count weighted folds within each section.
    No prediction coefficients are pooled across dates.
    """
    cutoff=date.fromisoformat(str(selection_cutoff)); banks=[]; fold_records=[]
    for d,ids,x,y in sections:
        if d>cutoff:
            raise ValueError('Risk penalty validation labels exceed actual selection cutoff')
        bases,records=stock_fold_bases(ids,x,y,d)
        if bases:
            banks.append(bases); fold_records.extend(records)
    diagnostics=dict(selection_cutoff=str(cutoff),validation_policy='five security folds inside historical cross-sections',
        date_aggregation='equal cross-section mean; held-out stock SSE weighted within cross-section',
        forecast_time_cv='unchanged original five historical time blocks',fold_records=fold_records,
        cross_sections=len(banks),baseline_lambda=baseline_lambda,selected_lambda=baseline_lambda,selected_mse=None,
        fallback=None)
    if not banks:
        diagnostics['fallback']='original penalty; no usable completed historical stock folds'
        return baseline_lambda,diagnostics
    grid=initial_ridge_lambdas((baseline_lambda,)) if initial_lambdas is None else np.unique(np.asarray(initial_lambdas,dtype=float))
    if (len(grid)<2 or not np.isfinite(grid).all() or np.any(grid<=0) or refinement_points<2
            or batch_size<1 or maximum_extensions<0):
        raise ValueError('Positive grid and valid path refinement settings required')
    def score(values):
        total=np.zeros(len(values))
        for bank in banks:
            count=sum(basis[1] for basis in bank)
            total+=sum(_basis_mse(basis,values,batch_size)*basis[1] for basis in bank)/count
        return total/len(banks)
    def curve(values,losses):
        return dict(lambdas=values.tolist(),mse=losses.tolist())
    losses=score(grid); initial=curve(grid,losses); extensions=[]
    for _ in range(maximum_extensions):
        best=int(np.argmin(losses))
        if best==0 and grid[0]>1e-12:
            added=np.geomspace(max(1e-12,grid[0]/100),grid[0],1000)[:-1]; side='lower'
        elif best==len(grid)-1:
            added=np.geomspace(grid[-1],grid[-1]*100,1000)[1:]; side='upper'
        else:
            break
        added_losses=score(added); extensions.append(dict(side=side,**curve(added,added_losses)))
        order=np.argsort(np.r_[grid,added],kind='stable')
        grid,losses=np.r_[grid,added][order],np.r_[losses,added_losses][order]
    best=int(np.argmin(losses)); left,right=grid[max(0,best-1)],grid[min(len(grid)-1,best+1)]
    refined=np.linspace(left,right,refinement_points); refined_losses=score(refined)
    values,all_losses=np.r_[grid,refined],np.r_[losses,refined_losses]
    order=np.argsort(values,kind='stable'); winner=int(order[np.argmin(all_losses[order])])
    selected=float(values[winner])
    diagnostics.update(method='glmnet-no-intercept-own-cross-section-spectral-v1',selected_lambda=selected,
        selected_mse=float(all_losses[winner]),initial=initial,extensions=extensions,refinement=curve(refined,refined_losses),
        refinement_interval=[float(left),float(right)],candidate_evaluations=len(values),
        coarse_boundary_status='lower' if best==0 else 'upper' if best==len(grid)-1 else 'bracketed',
        extension_policy=dict(decades=2,points=1000,maximum=maximum_extensions,lower_limit=1e-12),batch_size=batch_size)
    return selected,diagnostics


def historical_risk_sections(parent_dir,component,cutoff,*,history_months=120):
    """Original cached x/y only; R01 deterministic last fit day per completed month.

    R02 uses each month's observed eligible log-specific-volatility cross-section.
    Neither selection fits a pooled historical coefficient model.
    """
    cutoff=date.fromisoformat(str(cutoff))
    if not isinstance(history_months,int) or history_months<1:
        raise ValueError('Positive historical selection window required')
    root,manifest,_,directories=_parent_months(parent_dir,cutoff)
    for directory in directories[-history_months:]:
        d=date.fromisoformat(directory.name)
        if component=='factor':
            stream,marker=_factor_stream(directory,manifest['factor_order'],d)
            if not stream.height:
                continue
            last_day=stream['date'].max()
            daily=cached_daily_returns(root,d).filter(pl.col('date')==last_day).sort('id')
            with np.load(root/'months'/marker['preceding_exposure_date']/'exposures.npz',allow_pickle=False) as saved:
                ids=saved['ids']; exposures=saved['B']
            positions=np.searchsorted(ids,daily['id'].to_numpy())
            yield last_day,daily['id'].to_numpy(),exposures[positions],daily['ret_exc'].to_numpy()
        elif component=='specific':
            with np.load(directory/'exposures.npz',allow_pickle=False) as saved:
                ids=saved['ids'].copy(); exposures=saved['B'].copy(); actual=saved['observed_volatility'].copy()
            available=np.isfinite(actual)&(actual>0)
            if available.sum()>=2:
                yield d,ids[available],exposures[available],np.log(actual[available])
        else:
            raise ValueError('Risk penalty component must be factor or specific')


def historical_parent_receipt(parent_dir,cutoff,*,history_months=120):
    """Cutoff-bounded input bytes, including any preceding exposure dependency.

    Outer run provenance binds the complete original checkpoint. Historical
    selection binds only its completed monthly input stream, never future
    checkpoint factor windows, residual arrays or future monthly file bytes.
    """
    cutoff=date.fromisoformat(str(cutoff))
    if not isinstance(history_months,int) or history_months<1:
        raise ValueError('Positive historical selection window required')
    root,manifest,_,directories=_parent_months(parent_dir,cutoff)
    files={}; months=[]
    def bind(path):
        digest=hashlib.sha256()
        with path.open('rb') as source:
            for block in iter(lambda:source.read(1024*1024),b''):
                digest.update(block)
        files[str(path.relative_to(root))]=digest.hexdigest()
    for directory in directories[-history_months:]:
        d=date.fromisoformat(directory.name); months.append(str(d))
        marker=json.loads((directory/'complete.json').read_text())
        if marker['formation_date']!=str(d) or marker['available_through']!=str(d) or d>cutoff:
            raise ValueError('Historical selection month cutoff mismatch')
        for name in ('complete.json','exposures.npz','factor_returns.parquet','residuals.parquet'):
            bind(directory/name)
        preceding=marker['preceding_exposure_date']
        if preceding is not None:
            if date.fromisoformat(preceding)>=d:
                raise ValueError('Historical selection preceding exposure is not earlier')
            for name in ('exposures.npz','complete.json'):
                bind(root/'months'/preceding/name)
    return dict(selection_cutoff=str(cutoff),history_months=history_months,
        committed_months=months,files_sha256=files,
        checkpoint_policy='as-of complete monthly prefix only; full checkpoint bound in outer caller identity',
        factor_order=manifest['factor_order'])


def annual_risk_penalty(parent_dir,penalty_dir,component,year,cutoff,*,history_months=120,
        baseline_lambda=1e-4,initial_lambdas=None,refinement_points=10000,batch_size=512,maximum_extensions=3):
    """Immutable per-year curve publication; called from chronological risk replay."""
    root=Path(penalty_dir); root.mkdir(parents=True,exist_ok=True); path=root/f'{year}.json'
    parent=Path(parent_dir)
    identity=dict(parent_manifest_sha256=hashlib.sha256((parent/'manifest.json').read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),component=component,year=year,
        ridge_path_sha256=hashlib.sha256(Path(__file__).with_name('ridge_path.py').read_bytes()).hexdigest(),
        cached_return_source_sha256=hashlib.sha256(Path(__file__).with_name('extended_risk_rebuilds.py').read_bytes()).hexdigest(),
        original_math_sha256={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ('baseline.py','comparison_models.py','research_risk.py','extended_risk.py')},
        historical_parent_receipt=historical_parent_receipt(parent,cutoff,history_months=history_months),
        selection_cutoff=str(cutoff),history_months=history_months,baseline_lambda=baseline_lambda,
        initial_lambdas=None if initial_lambdas is None else np.asarray(initial_lambdas).tolist(),
        refinement_points=refinement_points,batch_size=batch_size,maximum_extensions=maximum_extensions)
    if path.exists():
        saved=json.loads(path.read_text())
        if saved['identity']!=identity:
            raise ValueError('Annual risk penalty identity mismatch')
        return saved['record']['selected_lambda'],saved['record']
    selected,record=select_risk_lambda(historical_risk_sections(parent,component,cutoff,history_months=history_months),
        selection_cutoff=cutoff,baseline_lambda=baseline_lambda,initial_lambdas=initial_lambdas,
        refinement_points=refinement_points,batch_size=batch_size,maximum_extensions=maximum_extensions)
    record.update(component=component,year=year,history_months=history_months,
        screening='last original factor-fit day per completed calendar month' if component=='factor' else 'all available monthly observed-log-volatility cross-sections')
    _write_json(path,dict(identity=identity,record=record))
    return selected,record
