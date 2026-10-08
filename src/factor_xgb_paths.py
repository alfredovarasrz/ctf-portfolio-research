"""P09 joint factor XGBoost. Original P08 coordinates, timing and stock loss."""
from hashlib import sha256
import json
from pathlib import Path

from time import perf_counter

from artifact_utils import digest

EXPERIMENT='P09_FACTOR_XGB'
SPECIFICATION=dict(factor_stream='original daily Barra coefficients',monthly_target='SUM',lag_months=1,
    train_months=120,folds=5,annual_chunk_months=12,factor_basis='original12industries+402characteristics',
    model='joint hist multi_output_tree',objective='factor reg:squarederror',selection_loss='ACTUAL mapped stock MSE',
    tree_parameters='authenticated original P01 annual depth/lambda/subsample/colsample',eta=.01,
    early_stopping=25,maximum_rounds_ceiling=1000,seed=1,base_score=0,full_search20='deferred',
    final_trees='floor mean earliest best mapped-stock round across usable folds; at least1',
    training_units='original monthly coefficient sums; no standardization/PCA',validation_policy='historical_blocked')


def canonical_digest(value):
    return sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def mapped_stock_mse(vectors,statistics):
    """Unrounded actual stock SSE from sufficient statistics, not factor MSE."""
    import numpy as np
    values=np.asarray(vectors,dtype=float)
    if values.ndim!=2 or len(values)!=len(statistics) or not np.isfinite(values).all():
        raise ValueError('Finite factor forecast matrix aligned with actual held-out months required')
    sse=0.;rows=0
    for v,stats in zip(values,statistics):
        if stats['n']<0 or stats['G'].shape!=(len(v),len(v)) or stats['c'].shape!=(len(v),):
            raise ValueError('Actual mapped stock sufficient-statistic dimensions changed')
        loss=float(v@stats['G']@v-2*v@stats['c']+stats['s'])
        tolerance=1e-10*max(1.,abs(float(stats['s'])),abs(float(v@stats['G']@v)))
        if not np.isfinite(loss) or loss < -tolerance:raise ValueError('Invalid actual stock SSE')
        sse+=max(0.,loss);rows+=stats['n']
    if rows<1:raise ValueError('Held-out block has no actual finite stock labels')
    return sse/rows


def raw_stopping_callback(validation,statistics,patience,*,stop=True):
    """Evaluate directly after each update, bypassing XGBoost's %f eval formatting."""
    import xgboost as xgb
    class MappedStop(xgb.callback.TrainingCallback):
        def __init__(self):self.losses=[];self.best=float('inf');self.best_round=0
        def after_iteration(self,model,epoch,evals_log):
            value=mapped_stock_mse(model.predict(validation,iteration_range=(0,epoch+1),strict_shape=True),statistics)
            self.losses.append(value)
            if value<self.best:self.best=value;self.best_round=epoch+1
            return bool(stop and epoch+1-self.best_round>=patience)
    return MappedStop()


def xgb_parameters(candidate,settings):
    return dict({k:v for k,v in candidate.items() if k!='hp_set'},objective='reg:squarederror',
        tree_method='hist',multi_strategy='multi_output_tree',booster='gbtree',base_score=0,
        disable_default_eval_metric=1,eta=.01,min_child_weight=1,gamma=0,nthread=settings.threads,seed=1)


def pairs(targets,first,cutoff):
    import numpy as np
    from baseline import month_number,month_end
    lower=month_number(first)-120
    eligible=sorted(d for d in targets if lower<=month_number(d)<month_number(first) and d<=cutoff)
    missing=[d for d in eligible if month_end(month_number(d)-1) not in targets]
    dates=[d for d in eligible if month_end(month_number(d)-1) in targets]
    if not dates:raise ValueError('No original completed monthly factor targets with contiguous observed lag')
    x=np.asarray([targets[month_end(month_number(d)-1)] for d in dates],dtype=np.float32)
    y=np.asarray([targets[d] for d in dates],dtype=np.float32)
    if x.ndim!=2 or x.shape!=y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Original factor pair shape/units/finite values required')
    return dates,x,y,eligible,missing


def fit_factor_xgb(targets,first,cutoff,labels,candidate,settings,*,maximum_rounds,profile=False):
    import numpy as np
    import xgboost as xgb
    from comparison_models import training_folds
    if type(maximum_rounds) is not int or not 1<=maximum_rounds<=1000 or settings.folds!=5 or settings.chunk_months!=12:
        raise ValueError('Explicit <=1000 cap and original five-fold/twelve-month policy required')
    if not profile and candidate not in settings.candidates:raise ValueError('Original authenticated candidate required')
    dates,x,y,eligible,missing=pairs(targets,first,cutoff)
    blocks=training_folds(dates,first,settings);fold_models={};records=[];skipped=[]
    params=xgb_parameters(candidate,settings)
    for i,block in enumerate(blocks,1):
        mask=np.array([d in block for d in dates]);train=~mask
        if not mask.any() or not train.any():skipped.append(dict(fold=i,reason='empty fit or validation'));continue
        stats=[labels.statistics(d) for d in block]
        if not sum(s['n'] for s in stats):skipped.append(dict(fold=i,reason='no actual finite stock labels'));continue
        started=perf_counter();dtrain=xgb.DMatrix(x[train],label=y[train],nthread=settings.threads)
        dval=xgb.DMatrix(x[mask],label=y[mask],nthread=settings.threads)
        callback=raw_stopping_callback(dval,stats,25,stop=not profile)
        booster=xgb.train(params,dtrain,num_boost_round=maximum_rounds,callbacks=[callback],verbose_eval=False)
        record=dict(fold=i,training_return_dates=[str(d) for d,m in zip(dates,train) if m],
            validation_return_dates=list(map(str,block)),available_through=str(cutoff),actual_stock_rows=sum(s['n'] for s in stats),
            loss_curve=callback.losses,best_mse=callback.best,best_rounds=callback.best_round,
            stopped_rounds=booster.num_boosted_rounds(),hit_round_cap=booster.num_boosted_rounds()==maximum_rounds)
        if profile:record['elapsed_seconds']=perf_counter()-started
        records.append(record);fold_models[str(i)]=booster
        del dtrain,dval
    trees=max(1,int(np.floor(np.mean([r['best_rounds'] for r in records])))) if records else 1
    model=None
    if not profile:
        training=xgb.DMatrix(x,label=y,nthread=settings.threads)
        model=xgb.train(params,training,num_boost_round=trees,verbose_eval=False)
    fit=dict(train_months=len(dates),train_rows=len(dates),factor_dimensions=x.shape[1],
        training_return_first=str(dates[0]),training_return_last=str(dates[-1]),formation_cutoff=str(cutoff),
        observed_target_dates=list(map(str,eligible)),observed_target_count=len(eligible),
        usable_lag_target_dates=list(map(str,dates)),usable_pair_count=len(dates),
        missing_observed_contiguous_lag_dates=list(map(str,missing)),missing_lag_count=len(missing),selected=dict(candidate),trees=trees,
        maximum_rounds=maximum_rounds,fold_records=records,skipped_folds=skipped,
        reason=None if records else 'fixed one tree; no usable actual-stock validation',
        validation_policy='historical_blocked',training_objective='factor squared error',
        early_stopping_objective='raw ACTUAL mapped stock MSE',profile_only=profile)
    return model,fold_models,fit


def annual_input_receipt(risk,first,cutoff,dates,names):
    """Only consumed factor and label-exposure month bytes, never future bodies."""
    from baseline import month_number,month_end
    from factor_forecast_paths import factors
    files={};lower=month_end(month_number(cutoff)-120)
    for d,_ in factors(risk,cutoff,names,lower):
        for name in ('factor_returns.parquet','complete.json'):
            p=Path(risk)/'months'/str(d)/name;files[str(p.relative_to(risk))]=digest(p)
    for d in dates:
        p=Path(risk)/'months'/str(month_end(month_number(d)-1))/'exposures.npz'
        files[str(p.relative_to(risk))]=digest(p)
    return dict(path=str(Path(risk).resolve()),cutoff=str(cutoff),factor_order=names,files_sha256=dict(sorted(files.items())))


def check_annual_bundle(folder,contract):
    saved=json.loads((folder/'fit.json').read_text())
    if saved['contract']!=contract:raise ValueError('Factor XGB annual input/source/cap identity changed')
    expected={'model.ubj'}|{'fold-'+str(r['fold'])+'.ubj' for r in saved['fit']['fold_records']}
    if set(saved['files_sha256'])!=expected or any(digest(folder/n)!=h for n,h in saved['files_sha256'].items()):
        raise ValueError('Factor XGB final/fold model receipt changed')
    actual={p.name for p in folder.iterdir() if p.is_file() and p.suffix=='.ubj'}
    if actual!=expected:raise ValueError('Unexpected factor XGB model files')
    return saved


def forecast_xgb_family(chars,names,risk,checkpoint,identity,parent_selections,settings,*,maximum_rounds):
    import numpy as np
    import polars as pl
    import xgboost as xgb
    from threadpoolctl import threadpool_limits
    from baseline import MonthlySource
    from comparison_models import INDUSTRIES
    from factor_forecast_paths import monthly_targets,ActualStockLabels
    from inherited_expanding_forecasts import check_parent_receipt
    from artifact_utils import write_json
    from extended_risk import _factor_stream
    source=MonthlySource(chars,names);metadata=source.metadata
    tests=sorted(metadata.filter(pl.col('ctff_test'))['eom_ret'].unique().to_list())
    order=json.loads((Path(risk)/'manifest.json').read_text())['factor_order']
    if order!=list(INDUSTRIES)+names:raise ValueError('Original factor coordinates/order changed')
    labels=ActualStockLabels(chars,names,risk);frames=[];fits=[];root=Path(checkpoint);root.mkdir(parents=True,exist_ok=True)
    with threadpool_limits(limits=settings.threads):
        for start in range(0,len(tests),12):
            chunk=tests[start:start+12];first=chunk[0];cutoff=source.return_to_formation[first]
            chosen=[r for r in parent_selections['annual_selections'] if r['first_test_return']==str(first)]
            if len(chosen)!=1 or chosen[0]['formation_cutoff']!=str(cutoff) or chosen[0]['test_return_end']!=str(chunk[-1]):
                raise ValueError('Exact original annual anchor/cutoff/chunk required')
            check_parent_receipt(parent_selections)
            targets=monthly_targets(risk,cutoff,order);dates,_,_,_,_=pairs(targets,first,cutoff)
            contract=dict(identity_sha256=canonical_digest(identity),first_return=str(first),cutoff=str(cutoff),specification=SPECIFICATION,
                maximum_rounds=maximum_rounds,inherited_selection=chosen[0],
                annual_inputs=annual_input_receipt(risk,first,cutoff,dates,order),factor_order=order)
            folder=root/str(first);folder.mkdir(exist_ok=True)
            if (folder/'fit.json').exists():
                saved=check_annual_bundle(folder,contract);fit=saved['fit'];model=xgb.Booster(params={'nthread':settings.threads})
                model.load_model(folder/'model.ubj')
            else:
                model,fold_models,fit=fit_factor_xgb(targets,first,cutoff,labels,chosen[0]['selected'],settings,
                    maximum_rounds=maximum_rounds)
                fit['inherited_selection']=chosen[0]
                models={'model.ubj':model,**{'fold-'+k+'.ubj':v for k,v in fold_models.items()}}
                for n,m in models.items():
                    temporary=folder/(n+'.tmp.ubj');m.save_model(temporary);temporary.replace(folder/n)
                write_json(folder/'fit.json',dict(contract=contract,fit=fit,files_sha256={n:digest(folder/n) for n in models}))
                del fold_models,models
            for target in chunk:
                formation=source.return_to_formation[target]
                with np.load(Path(risk)/'months'/str(formation)/'risk.npz',allow_pickle=False) as data:
                    ids,b=data['ids'].copy(),data['B'].copy()
                frame=metadata.filter((pl.col('eom')==formation)&pl.col('ctff_test')).sort('id')
                if not np.array_equal(frame['id'].to_numpy(),ids):raise ValueError('Original B/test key mismatch')
                stream,_=_factor_stream(Path(risk)/'months'/str(formation),order,formation)
                if not stream.height:raise ValueError('No observed completed factor lag at formation')
                lag=stream.select(order).to_numpy().sum(axis=0).astype(np.float32)[None,:]
                vector=model.inplace_predict(lag,strict_shape=True)[0].astype(float);predicted=b@vector
                if vector.shape!=(len(order),) or not np.isfinite(predicted).all():raise ValueError('Finite unchanged factor mapping required')
                result=frame.select('id','eom','excntry').with_columns(pl.Series('pred',predicted))
                marker=folder/(str(formation)+'.json');path=folder/(str(formation)+'.parquet')
                monthly=dict(contract_sha256=canonical_digest(contract),model_sha256=digest(folder/'model.ubj'),formation_date=str(formation),
                    return_date=str(target),lag_factor_sha256=digest(Path(risk)/'months'/str(formation)/'factor_returns.parquet'),
                    original_risk_sha256=digest(Path(risk)/'months'/str(formation)/'risk.npz'),rows=result.height)
                if marker.exists():
                    saved=json.loads(marker.read_text())
                    if saved['metadata']!=monthly or digest(path)!=saved['sha256'] or not pl.read_parquet(path).equals(result):
                        raise ValueError('Saved factor XGB monthly prediction receipt changed')
                else:
                    temporary=folder/(str(formation)+'.tmp.parquet');result.write_parquet(temporary);temporary.replace(path)
                    write_json(marker,dict(metadata=monthly,sha256=digest(path)))
                frames.append(result)
            fits.append(dict(fit,test_return_start=str(first),test_return_end=str(chunk[-1]),factor_order=order))
            print(f'{EXPERIMENT} {first}: {fit["train_months"]} factor months, {fit["trees"]} trees',flush=True)
    if not frames:raise ValueError('No supplied test keys')
    return pl.concat(frames).sort('eom','id'),fits
