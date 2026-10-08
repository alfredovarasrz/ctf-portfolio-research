"""Two independent cold exploratory combinations. Scientific imports are deferred."""
from calendar import monthrange
from datetime import date
from hashlib import sha256
import json
from pathlib import Path

from artifact_utils import digest

CANDIDATES=('E08_C021_R08V1','P08_C021_R08V1')
FIRST=date(1990,1,31)
LAST=date(2023,12,31)


def policy(candidate):
    if candidate not in CANDIDATES:raise ValueError('Exactly the two approved exploratory candidates required')
    return dict(forecast=candidate.split('_')[0],c02=True,r08_variance=True)


def phase_dates(dates,stage):
    dates=sorted(set(d for d in dates if FIRST<=d<=LAST))
    if stage=='diagnostic':return dates[:3]
    if stage=='full':return dates
    raise ValueError('Explicit diagnostic/full exploratory stage required')



def expected_return_dates(stage):
    if stage not in ('diagnostic','full'):raise ValueError('Explicit diagnostic/full stage required')
    count=3 if stage=='diagnostic' else 408
    return [date(1990+i//12,i%12+1,monthrange(1990+i//12,i%12+1)[1]) for i in range(count)]


def canonical_digest(value):
    return sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def check_files(root,files,*,exact=False):
    root=Path(root).resolve()
    if not isinstance(files,dict) or not files:raise ValueError('Nonempty own byte ledger required')
    for name,value in files.items():
        rel=Path(name);p=root/rel
        if rel.is_absolute() or '..' in rel.parts or not p.is_file() or p.is_symlink() or not p.resolve().is_relative_to(root) or digest(p)!=value:
            raise ValueError('Owned artifact receipt changed: '+name)
    if exact and {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}!=set(files):
        raise ValueError('Exact owned artifact coverage differs')


def file_ledger(root):
    root=Path(root)
    if any(p.is_symlink() for p in root.rglob('*')):raise ValueError('Symlink in candidate-owned cache')
    return {str(p.relative_to(root)):digest(p) for p in sorted(root.rglob('*')) if p.is_file()}


def annual_stock_forecasts(source,names,tests,kind,root,identity,forecast_settings,ridge_settings):
    """Original blocked fits, authentic own annual final models and predictions."""
    import numpy as np
    import polars as pl
    import xgboost as xgb
    from baseline import month_number,train_ridge
    from comparison_models import training_folds
    from research_forecasts import _xgb_fit
    from artifact_utils import write_json
    root=Path(root);root.mkdir(parents=True,exist_ok=True);frames=[];records=[]
    for start in range(0,len(tests),12):
        chunk=tests[start:start+12];first=chunk[0];cutoff=source.return_to_formation[first]
        dates=sorted(d for d in source.return_to_formation if month_number(first)-120<=month_number(d)<month_number(first) and d<=cutoff)
        contract=dict(identity_sha256=canonical_digest(identity),kind=kind,first_return=str(first),formation_cutoff=str(cutoff),
            training_return_dates=list(map(str,dates)),features=names)
        folder=root/str(first);folder.mkdir(exist_ok=True);marker=folder/'fit.json';modelpath=folder/('model.npz' if kind=='ridge' else 'model.ubj')
        if marker.exists():
            saved=json.loads(marker.read_text())
            if saved['contract']!=contract or digest(modelpath)!=saved['model_sha256']:raise ValueError('Own annual stock model receipt changed')
            fit=saved['fit']
            if kind=='ridge':
                with np.load(modelpath,allow_pickle=False) as z:
                    if set(z.files)!={'intercept','coef'} or z['coef'].shape!=(len(names),) or not np.isfinite(z['coef']).all() or not np.isfinite(z['intercept']).all():raise ValueError('Own annual Ridge array contract differs')
                    model=(float(z['intercept']),z['coef'].copy())
            elif fit['train_rows']:
                model=xgb.Booster(params={'nthread':forecast_settings.threads});model.load_model(modelpath)
            else:model=None
        else:
            if kind=='ridge':
                intercept,coef,fit=train_ridge(source,dates,ridge_settings);model=(intercept,coef)
                tmp=modelpath.with_suffix('.tmp')
                with tmp.open('wb') as handle:np.savez(handle,intercept=intercept,coef=coef)
                tmp.replace(modelpath)
            else:
                blocks=training_folds(dates,first,forecast_settings)
                folds=[dict(training=[d for d in dates if d not in block],validation=block,cutoff=cutoff) for block in blocks if block]
                model,fit=_xgb_fit(source,names,dates,folds,forecast_settings)
                if model is None:modelpath.write_bytes(b'NO_COMPLETED_HISTORICAL_LABELS')
                else:
                    tmp=folder/'model.tmp.ubj';model.save_model(tmp);tmp.replace(modelpath)
            fit.update(formation_cutoff=str(cutoff),training_return_first=str(dates[0]) if dates else None,
                training_return_last=str(dates[-1]) if dates else None,validation_policy='historical_blocked')
            write_json(marker,dict(contract=contract,fit=fit,model_sha256=digest(modelpath)))
        for target in chunk:
            frame=source.load(source.return_to_formation[target]).filter(pl.col('ctff_test'))
            x=frame.select(names).to_numpy()
            values=(model[0]+x@model[1] if kind=='ridge' else np.zeros(len(frame)) if model is None else model.inplace_predict(x.astype(np.float32)))
            result=frame.select('id','eom','excntry').with_columns(pl.Series('pred',values).cast(pl.Float64))
            path=folder/(str(target)+'.parquet');recordpath=folder/(str(target)+'.json')
            wanted=dict(contract_sha256=canonical_digest(contract),model_sha256=digest(modelpath),return_date=str(target),formation_date=str(source.return_to_formation[target]),rows=len(frame))
            if recordpath.exists():
                saved=json.loads(recordpath.read_text())
                if saved['metadata']!=wanted or digest(path)!=saved['sha256'] or not pl.read_parquet(path).equals(result):raise ValueError('Own stock monthly prediction changed')
            else:
                temp=path.with_suffix('.tmp');result.write_parquet(temp);temp.replace(path);write_json(recordpath,dict(metadata=wanted,sha256=digest(path)))
            frames.append(result)
        records.append(dict(fit,test_return_start=str(first),test_return_end=str(chunk[-1]),model_path=str(modelpath.resolve()),model_sha256=digest(modelpath),fit_sha256=digest(marker)))
        print(f'Cold {kind} {first}: annual fit/model committed',flush=True)
    return pl.concat(frames).sort('eom','id'),records


def blend_predictions(streams,forecast):
    import polars as pl
    if forecast not in ('E08','P08'):raise ValueError('Approved exploratory forecast required')
    names=['ridge','xgboost','factor'] if forecast=='E08' else ['factor']
    frames=[streams[n].sort('eom','id') for n in names];keys=frames[0].select('id','eom','excntry')
    if any(not frame.select('id','eom','excntry').equals(keys) for frame in frames):raise ValueError('Ensemble raw stock forecast keys differ')
    result=keys.with_columns((sum((frame['pred'] for frame in frames))/len(frames)).alias('pred'))
    if result['pred'].null_count() or not result['pred'].is_finite().all():raise ValueError('Finite equal-weight forecast mixture required')
    return result


def candidate_directions(mu,b,d,f,reference_f,settings,q_grid):
    """Gamma scale and ALL volatility budgets use the ORIGINAL covariance."""
    import numpy as np
    from batch_allocations import factor_loading,solve_penalized
    from comparison_models import portfolio_variance
    loading=factor_loading(d,f,b);scale=float(np.median(d+np.einsum('ij,jk,ik->i',b,reference_f,b)))
    if not np.isfinite(scale) or scale<=0:raise ValueError('Positive ORIGINAL stock variance median required')
    weights=[];directions=[]
    for q in q_grid:
        direction=solve_penalized(d,loading,mu,float(q)*scale)
        variance=portfolio_variance(direction,d,reference_f,b)
        if not np.isfinite(variance) or variance<=0:raise ValueError('No finite nonzero allocation direction')
        weights.append(direction*settings.target_annual_volatility/np.sqrt(252*variance));directions.append(direction)
    return np.array(weights),np.array(directions),scale


def allocate_candidate(risks,predictions,metadata,root,identity,settings,*,c02):
    """Resume own causal q bank; never access a current label before selection."""
    import numpy as np
    import polars as pl
    from batch_allocations import Q_GRID,select_penalty,_realized_labels
    from comparison_models import portfolio_variance
    from artifact_utils import write_json
    root=Path(root);root.mkdir(parents=True,exist_ok=True);frames=[];records=[];bank=[];history=[];selections={}
    for marker in sorted(root.glob('*/complete.json')):
        saved=json.loads(marker.read_text())
        if saved['identity_sha256']!=canonical_digest(identity):raise ValueError('Own allocation checkpoint identity differs')
        check_files(marker.parent,saved['files_sha256'])
        row=saved['bank'];row=dict(row,eom=date.fromisoformat(row['eom']),eom_ret=date.fromisoformat(row['eom_ret']))
        history.append(dict(eom=row['eom'],eom_ret=row['eom_ret'],minimum=None,markowitz=None if row['returns'] is None else np.array(row['returns'])))
        bank.extend(saved['bank_rows']);selections[saved['annual_anchor']]=saved['record']['selection']
    if history:
        committed=[r['eom'] for r in history]
        expected=metadata.filter(pl.col('ctff_test')&pl.col('eom').is_between(committed[0],committed[-1]))['eom'].unique().sort().to_list()
        if committed!=expected:raise ValueError('Committed allocation history has a gap')
    for value in risks:
        d=value['eom'];folder=root/str(d);marker=folder/'complete.json';ids,b,f,diagonal=value['ids'],value['B'],value['F'],value['D'];reference=value['reference_F']
        mu=predictions.filter(pl.col('eom')==d).sort('id')
        if not np.array_equal(mu['id'].to_numpy(),ids):raise ValueError('Own risk/forecast universe differs')
        if marker.exists():
            saved=json.loads(marker.read_text());record=saved['record'];frame=pl.read_parquet(folder/'weights.parquet')
            wanted=dict(mu_sha256=canonical_digest(mu['pred'].to_list()),risk_sha256=canonical_digest({'F':f.tolist(),'reference_F':reference.tolist(),'D':diagonal.tolist(),'B':b.tolist(),'ids':ids.tolist()}))
            if saved['inputs']!=wanted:raise ValueError('Owned monthly direction input changed')
            frames.append(frame);records.append(record);continue
        target=metadata.filter(pl.col('eom')==d)['eom_ret'].unique().to_list()
        if len(target)!=1:raise ValueError('Single own next return month required')
        annual_anchor=str(date(target[0].year,1,31))
        if annual_anchor not in selections:
            _,selection=select_penalty(history,d,Q_GRID,portfolio='markowitz') if c02 else (0,dict(selected_q=0.,selection_formation=str(d),fallback='C02 off'))
            selections[annual_anchor]=selection
        selection=selections[annual_anchor];q=selection['selected_q'];grid=Q_GRID if c02 else (0.,);chosen=grid.index(q)
        allweights,directions,scale=candidate_directions(mu['pred'].to_numpy(),b,diagonal,f,reference,settings,grid)
        weights=allweights[chosen];frame=pl.DataFrame(dict(id=ids,eom=[d]*len(ids),w=weights))
        record=dict(formation_date=str(d),annual_anchor=annual_anchor,formation_cutoff=str(d),selected_q=q,gamma=q*scale,
            original_variance_median=scale,selection=selection,original_predicted_annual_volatility=float(np.sqrt(252*portfolio_variance(weights,diagonal,reference,b))),
            allocation_predicted_annual_volatility=float(np.sqrt(252*portfolio_variance(weights,diagonal,f,b))),net=float(weights.sum()),gross=float(np.abs(weights).sum()),risk_record=value['record'])
        # Labels enter candidate-owned outcomes only AFTER direction/selected q commit.
        return_date,actual=_realized_labels(metadata,ids,d)
        payoffs=None if actual is None else allweights@actual
        row=dict(eom=str(d),eom_ret=str(return_date),returns=None if payoffs is None else payoffs.tolist())
        rows=[dict(eom=str(d),eom_ret=str(return_date),available_after=str(return_date),q=float(v),gamma=float(v*scale),
            payoff=None if payoffs is None else float(payoffs[j])) for j,v in enumerate(grid)]
        folder.mkdir(exist_ok=True);tmp=folder/'weights.tmp.parquet';frame.write_parquet(tmp);tmp.replace(folder/'weights.parquet')
        with (folder/'directions.tmp').open('wb') as handle:np.savez(handle,q=np.array(grid),directions=directions,weights=allweights,ids=ids)
        (folder/'directions.tmp').replace(folder/'directions.npz')
        write_json(folder/'complete.json',dict(identity_sha256=canonical_digest(identity),annual_anchor=annual_anchor,record=record,bank=row,bank_rows=rows,
            inputs=dict(mu_sha256=canonical_digest(mu['pred'].to_list()),risk_sha256=canonical_digest({'F':f.tolist(),'reference_F':reference.tolist(),'D':diagonal.tolist(),'B':b.tolist(),'ids':ids.tolist()})),
            files_sha256={n:digest(folder/n) for n in ('weights.parquet','directions.npz')}))
        history.append(dict(eom=d,eom_ret=return_date,minimum=None,markowitz=payoffs));bank.extend(rows);frames.append(frame);records.append(record)
    if not frames:raise ValueError('No selected phase allocations')
    return pl.concat(frames).sort('eom','id'),records,bank
