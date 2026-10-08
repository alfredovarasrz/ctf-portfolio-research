"""P07_R01/P08_R01 accepted tuned daily-factor forecasts, causal mapped-stock Ridge paths."""
from collections import deque, OrderedDict
from datetime import date
from pathlib import Path
import json
import numpy as np
import polars as pl
from threadpoolctl import threadpool_limits
from baseline import canonical_chars, month_number, month_end, DEFAULT_SETTINGS
from comparison_models import training_folds, FORECAST_SETTINGS, INDUSTRIES
from extended_risk import _factor_stream
from ridge_path import initial_ridge_lambdas

SPECIFICATIONS = {
 'P07_R01': {'factor_stream':'accepted R01 validation-selected daily Barra coefficients','daily_history':2520,
         'monthly_conversion':21.,'conversion':'approximate linear mean times21','annual_freeze':True},
 'P08_R01': {'factor_stream':'accepted R01 validation-selected daily Barra coefficients','target':'completed monthly SUM of daily coefficients',
         'lag_months':1,'train_months':120,'folds':5,'loss':'actual stock-return MSE mapped through preceding original B',
         'coarse':'integers1..10000 plus1000geomspace1e-8..1 and original six',
         'refinement_points':10000,'maximum_extensions':3,'normalization':'n_train*lambda; unpenalized intercept',
         'zero_lambda':'excluded: underidentified414-coordinate training design', 'certificate_rtol':1e-8,'certificate_atol':1e-12},
}


def factors(root, cutoff, names, lower=None):
    """Read committed completed month bodies only, sorted in original coordinates."""
    root=Path(root)
    for folder in sorted((root/'months').iterdir()):
        if not folder.is_dir() or folder.name.endswith('.tmp') or folder.name>str(cutoff) or (lower is not None and folder.name<str(lower)): continue
        d=date.fromisoformat(folder.name)
        frame,_=_factor_stream(folder,names,d)
        yield d,frame

def monthly_targets(root, cutoff, names):
    lower=month_end(month_number(cutoff)-120)
    return {d:frame.select(names).to_numpy().sum(axis=0)
            for d,frame in factors(root,cutoff,names,lower) if frame.height}

def daily_mean(root, cutoff, names):
    frames=[];n=0
    folders=sorted(p for p in (Path(root)/'months').iterdir() if p.is_dir() and not p.name.endswith('.tmp') and p.name<=str(cutoff))
    for folder in reversed(folders):
        frame,_=_factor_stream(folder,names,date.fromisoformat(folder.name))
        frames.append(frame);n+=frame.height
        if n>=2520:break
    rows=deque(maxlen=2520);days=deque(maxlen=2520)
    for frame in reversed(frames):
        rows.extend(frame.select(names).to_numpy());days.extend(frame['date'].to_list())
    if not rows: raise ValueError('P07 requires completed original daily factor history')
    return np.array(rows).mean(axis=0)*21.,dict(factor_history_first=str(days[0]),factor_history_last=str(days[-1]),factor_days=len(days))

class ActualStockLabels:
    """Bounded raw monthly projection, never reconstructed factor labels."""
    def __init__(self,chars,names,root):
        self.chars=canonical_chars(chars,names).select('id','eom','eom_ret','ret_exc_lead1m')
        self.root=Path(root);self.cache=OrderedDict()
    def arrays(self,target):
        formation=month_end(month_number(target)-1)
        frame=self.chars.filter(pl.col('eom')==formation).collect(engine='streaming').sort('id')
        with np.load(self.root/'months'/str(formation)/'exposures.npz',allow_pickle=False) as data:
            ids=data['ids'];b=data['B'].copy()
            if str(data['eom_ret'])!=str(target):raise ValueError('Actual label/exposure date misalignment')
        if (not np.array_equal(frame['id'].to_numpy(),ids) or frame['eom_ret'].unique().to_list()!=[target]
                or not np.isfinite(b).all()):raise ValueError('Actual stock labels must match original historical stock exposures')
        y=frame['ret_exc_lead1m'].to_numpy();good=np.isfinite(y)
        return b[good],y[good]
    def statistics(self,target):
        if target in self.cache:
            self.cache.move_to_end(target);return self.cache[target]
        b,y=self.arrays(target)
        result=dict(G=b.T@b,c=b.T@y,s=float(y@y),n=len(y));self.cache[target]=result
        while len(self.cache)>144:self.cache.popitem(last=False)
        return result

def spectral_fit(x,y):
    if not len(x) or x.ndim!=2 or y.ndim!=2 or len(y)!=len(x) or not all(np.isfinite(a).all() for a in (x,y)):
        raise ValueError('Finite aligned nonempty factor training matrices required')
    mx,my=x.mean(axis=0),y.mean(axis=0)
    u,s,vt=np.linalg.svd(x-mx,full_matrices=False)
    # No rank truncation: zero singular coordinates have zero numerators.
    return dict(mx=mx,my=my,V=vt.T,H=s[:,None]*(u.T@(y-my)),eigen=s*s,n=len(x))

def coefficients(bank,penalty):
    if not np.isfinite(penalty) or penalty<=0:raise ValueError('Positive ordinary Ridge lambda required')
    coef=bank['V']@(bank['H']/(bank['eigen'][:,None]+bank['n']*penalty))
    return bank['my']-bank['mx']@coef,coef

def mapped_bank(bank,validation_x,stock_stats):
    """Sum stock SSE quadratics before scoring any lambda, not factor SSE."""
    rank=len(bank['eigen']);qsum=np.zeros((rank,rank));linear=np.zeros(rank);constant=0.;n=0
    for x,stats in zip(validation_x,stock_stats):
        q=(x-bank['mx'])@bank['V'];h=bank['H'];m=bank['my'];g,c=stats['G'],stats['c']
        qsum+=(q[:,None]*(h@g@h.T))*q[None,:]
        linear+=q*(h@(g@m-c));constant+=float(m@g@m-2*m@c+stats['s']);n+=stats['n']
    if n<1:raise ValueError('No actual finite stock labels in held-out block')
    return dict(n=n,n_train=bank['n'],eigen=bank['eigen'],Q=qsum,b=linear,C=constant)

def score_banks(banks,lambdas,batch_size=256):
    lambdas=np.asarray(lambdas,dtype=float)
    if (not banks or lambdas.ndim!=1 or not len(lambdas) or not np.isfinite(lambdas).all()
            or np.any(lambdas<=0) or batch_size<1):raise ValueError('Nonempty mapped banks and positive candidate grid required')
    loss=np.zeros(len(lambdas))
    for bank in banks:
        for start in range(0,len(lambdas),batch_size):
            k=1/(bank['eigen'][:,None]+bank['n_train']*lambdas[None,start:start+batch_size])
            sse=np.sum(k*(bank['Q']@k),axis=0)+2*bank['b']@k+bank['C']
            # Only numerical cancellation-sized negatives can be clipped.
            if np.any(sse < -1e-8*max(1.,abs(bank['C']))):raise ValueError('Mapped stock SSE materially negative')
            loss[start:start+len(sse)]+=np.maximum(sse,0.)/bank['n']/len(banks)
    return loss

def select_path(banks,*,initial=None,refinement_points=10000,maximum_extensions=3):
    grid=initial_ridge_lambdas() if initial is None else np.unique(np.asarray(initial,dtype=float))
    if len(grid)<2 or refinement_points<2 or maximum_extensions<0:raise ValueError('Invalid declared penalty path')
    def curve(g,l):return dict(lambdas=g.tolist(),mse=l.tolist())
    loss=score_banks(banks,grid);coarse=curve(grid,loss);extensions=[]
    for _ in range(maximum_extensions):
        j=int(np.argmin(loss))
        if j==0 and grid[0]>1e-12:g=np.geomspace(max(1e-12,grid[0]/100),grid[0],1000)[:-1];side='lower'
        elif j==len(grid)-1:g=np.geomspace(grid[-1],grid[-1]*100,1000)[1:];side='upper'
        else:break
        l=score_banks(banks,g);extensions.append(dict(side=side,**curve(g,l)));order=np.argsort(np.r_[grid,g],kind='stable');grid,loss=np.r_[grid,g][order],np.r_[loss,l][order]
    j=int(np.argmin(loss));left,right=grid[max(0,j-1)],grid[min(len(grid)-1,j+1)]
    refined=np.linspace(left,right,refinement_points);rloss=score_banks(banks,refined)
    g,l=np.r_[grid,refined],np.r_[loss,rloss];order=np.argsort(g,kind='stable');winner=int(order[np.argmin(l[order])])
    penalty,mse=float(g[winner]),float(l[winner]);rescored=float(score_banks(banks,[penalty])[0]);tolerance=1e-12+1e-8*abs(mse)
    if abs(rescored-mse)>tolerance or rescored>float(l.min())+tolerance:raise ValueError('Mapped-loss argmin/regret certificate failed')
    return penalty,dict(initial=coarse,extensions=extensions,refinement=curve(refined,rloss),
        selected_lambda=penalty,selected_mse=mse,selection_regret=max(0.,rescored-float(l.min())),
        certificate_tolerance=tolerance,refinement_interval=[float(left),float(right)],
        coarse_boundary_status='lower' if j==0 else 'upper' if j==len(grid)-1 else 'bracketed',
        maximum_extensions=maximum_extensions,candidate_evaluations=len(g),scope='finite grid and local refinement; no unrestricted optimum')

def fit_mapped(targets,first,cutoff,labels,*,settings=FORECAST_SETTINGS,initial=None,refinement_points=10000,maximum_extensions=3,path='wide'):
    lower=month_number(first)-120
    eligible=sorted(d for d in targets if lower<=month_number(d)<month_number(first) and d<=cutoff)
    missing_lag=[d for d in eligible if month_end(month_number(d)-1) not in targets]
    dates=[d for d in eligible if month_end(month_number(d)-1) in targets]
    if not dates:raise ValueError('P08 has no completed monthly factor targets with observed lag')
    x=np.array([targets[month_end(month_number(d)-1)] for d in dates]);y=np.array([targets[d] for d in dates]);blocks=training_folds(dates,first,settings)
    banks=[];records=[]
    for block in blocks:
        val=np.array([d in block for d in dates]);train=~val
        if not val.any() or not train.any():continue
        stats=[labels.statistics(d) for d in block]
        if not sum(a['n'] for a in stats):continue
        bank=spectral_fit(x[train],y[train]);banks.append(mapped_bank(bank,x[val],stats))
        records.append(dict(training_return_dates=[str(d) for d,m in zip(dates,train) if m],
            validation_return_dates=list(map(str,block)),available_through=str(cutoff),actual_stock_rows=sum(a['n'] for a in stats)))
    if banks:
        if path=='six':
            grid=np.asarray(DEFAULT_SETTINGS.ridge_lambdas);loss=score_banks(banks,grid);j=int(np.argmin(loss));penalty=float(grid[j])
            curve=dict(initial=dict(lambdas=grid.tolist(),mse=loss.tolist()),extensions=[],refinement=dict(lambdas=[],mse=[]),
                selected_lambda=penalty,selected_mse=float(loss[j]),selection_regret=0.,certificate_tolerance=1e-12+1e-8*abs(float(loss[j])),
                candidate_evaluations=len(grid),scope='original-six finite control')
        elif path=='wide':penalty,curve=select_path(banks,initial=initial,refinement_points=refinement_points,maximum_extensions=maximum_extensions)
        else:raise ValueError('Only declared six/wide penalty paths supported')
    else:penalty,curve=DEFAULT_SETTINGS.ridge_lambdas[0],dict(selected_lambda=DEFAULT_SETTINGS.ridge_lambdas[0],reason='fixed original first lambda; no usable blocked validation')
    intercept,coef=coefficients(spectral_fit(x,y),penalty)
    return intercept,coef,dict(train_months=len(dates),training_return_first=str(dates[0]),training_return_last=str(dates[-1]),
        formation_cutoff=str(cutoff),observed_target_dates=list(map(str,eligible)),observed_target_count=len(eligible),
        usable_lag_target_dates=list(map(str,dates)),usable_pair_count=len(dates),missing_observed_contiguous_lag_dates=list(map(str,missing_lag)),missing_lag_count=len(missing_lag),
        window_return_first=str(month_end(lower)),window_return_last=str(cutoff),
        fold_records=records,ridge_lambda=penalty,curve=curve,validation_policy='historical_blocked',
        label_source='ACTUAL ret_exc_lead1m; original preceding B; finite labels only',lag_months=1)

def forecast_family(chars,names,variant,risk,checkpoint,identity,settings=FORECAST_SETTINGS,*,selected_return_dates=None):
    """Annual coefficient fits, current monthly exposures/observed lags, all test keys."""
    from hashlib import sha256
    from baseline import MonthlySource
    from research_resources import write_json
    if variant not in SPECIFICATIONS: raise ValueError('Only declared tuned factor variants supported')
    source=MonthlySource(chars,names);metadata=source.metadata
    tests=sorted(metadata.filter(pl.col('ctff_test'))['eom_ret'].unique().to_list())
    if selected_return_dates is not None:
        selected=list(selected_return_dates)
        if selected!=sorted(set(selected)) or not selected or any(d not in tests for d in selected):
            raise ValueError('Selected scored dates must be a sorted nonempty supplied test subset')
        tests=selected
    factor_names=json.loads((Path(risk)/'manifest.json').read_text())['factor_order']
    if factor_names!=list(INDUSTRIES)+names:raise ValueError('Original factor coordinates changed')
    label_source=ActualStockLabels(chars,names,risk);root=Path(checkpoint);root.mkdir(parents=True,exist_ok=True)
    frames=[];fits=[]
    with threadpool_limits(limits=settings.threads):
        for start in range(0,len(tests),12):
            chunk=tests[start:start+12];first=chunk[0];cutoff=source.return_to_formation[first]
            folder=root/str(first);folder.mkdir(exist_ok=True);marker=folder/'fit.json';model_path=folder/'model.npz'
            contract=dict(identity_sha256=sha256(json.dumps(identity,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest(),variant=variant,first_return=str(first),cutoff=str(cutoff),specification=SPECIFICATIONS[variant])
            targets=None
            if marker.exists():
                saved=json.loads(marker.read_text())
                if saved['contract']!=contract or sha256(model_path.read_bytes()).hexdigest()!=saved['model_sha256']:raise ValueError('Factor forecast annual cache changed')
                with np.load(model_path,allow_pickle=False) as model:
                    if set(model.files)!=({'factor_forecast'} if variant=='P07_R01' else {'intercept','coef'}):raise ValueError('Exact annual model array coverage changed')
                    arrays={k:model[k].copy() for k in model.files}
                fit=saved['fit']
                if fit.get('formation_cutoff')!=str(cutoff):raise ValueError('Annual model formation cutoff changed')
            else:
                if variant=='P07_R01':
                    vector,fit=daily_mean(risk,cutoff,factor_names);arrays={'factor_forecast':vector}
                    fit.update(formation_cutoff=str(cutoff),validation_policy='historical_blocked',training_return_first=None,training_return_last=None)
                elif variant=='P08_R01':
                    targets=monthly_targets(risk,cutoff,factor_names)
                    intercept,coef,fit=fit_mapped(targets,first,cutoff,label_source,settings=settings,path='wide')
                    arrays={'intercept':intercept,'coef':coef}
                else:raise ValueError('Only P07_R01/P08_R01 accepted tuned daily-factor family supported')
                temporary=model_path.with_suffix('.tmp')
                with temporary.open('wb') as handle:np.savez(handle,**arrays)
                temporary.replace(model_path)
                write_json(marker,dict(contract=contract,model_sha256=sha256(model_path.read_bytes()).hexdigest(),fit=fit))
            expected_shapes={'factor_forecast':(len(factor_names),)} if variant=='P07_R01' else {'intercept':(len(factor_names),),'coef':(len(factor_names),len(factor_names))}
            if any(arrays[key].shape!=shape or not np.isfinite(arrays[key]).all() for key,shape in expected_shapes.items()):
                raise ValueError('Annual factor model dimensions or finite coefficients changed')
            for target in chunk:
                formation=source.return_to_formation[target]
                with np.load(Path(risk)/'months'/str(formation)/'risk.npz',allow_pickle=False) as data:
                    ids,b=data['ids'],data['B']
                    if str(data['eom'])!=str(formation):raise ValueError('Current R01 exposure formation differs')
                frame=metadata.filter((pl.col('eom')==formation)&pl.col('ctff_test')).sort('id')
                if b.shape!=(len(ids),len(factor_names)) or not np.array_equal(frame['id'].to_numpy(),ids):raise ValueError('Original current B and required test keys differ')
                if variant=='P07_R01':vector=arrays['factor_forecast']
                else:
                    # Current completed month's observed factor sum supplies the lag.
                    stream,_=_factor_stream(Path(risk)/'months'/str(formation),factor_names,formation)
                    if not stream.height:raise ValueError('P08 formation has no completed observed daily-factor lag')
                    vector=arrays['intercept']+stream.select(factor_names).to_numpy().sum(axis=0)@arrays['coef']
                predicted=b@vector
                if not np.isfinite(predicted).all():raise ValueError('Nonfinite mapped factor forecast')
                frames.append(frame.select('id','eom','excntry').with_columns(pl.Series('pred',predicted)))
            fits.append(dict(fit,variant=variant,test_return_start=str(first),test_return_end=str(chunk[-1]),factor_order=factor_names))
            print(f'{variant} {first}: completed accepted R01 factor forecast',flush=True)
    if not frames:raise ValueError('No supplied test months')
    return pl.concat(frames).sort('eom','id'),fits
