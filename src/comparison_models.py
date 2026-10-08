"""Python ports of the supplied Factor ML, Minimum Variance and Markowitz ML.

Shared monthly forecasting and Barra-style risk estimation. Local runners stage
Parquet separately; model logic uses only supplied frames and numerical arrays.
The original ridge baseline remains unchanged. See COMPARISON_MODELS.md.
"""
from collections import deque
from dataclasses import dataclass
from datetime import date
from time import perf_counter
import gc

import numpy as np
import pandas as pd
import polars as pl
from scipy.linalg import cho_factor, cho_solve
from threadpoolctl import threadpool_limits
import xgboost as xgb

from baseline import (MonthlySource, as_lazy, canonical_features, finalize_output,
                      month_number, month_end, prepare_pred_data, predictions_to_weights)

# Exported once from the R benchmark recipe: seed 1, dials 1.4.2, 402 features.
# A fixed design, independent of any returns or competition-period performance.
XGB_GRID = ({'colsample_bytree': 0.960199004975124,
  'hp_set': 1,
  'max_depth': 1,
  'reg_lambda': 6.79290624977431,
  'subsample': 0.792944552004337},
 {'colsample_bytree': 0.119402985074627,
  'hp_set': 2,
  'max_depth': 6,
  'reg_lambda': 4.36883814483563,
  'subsample': 0.839221168868244},
 {'colsample_bytree': 0.0124378109452736,
  'hp_set': 3,
  'max_depth': 1,
  'reg_lambda': 8.95821662650399,
  'subsample': 0.281950866617262},
 {'colsample_bytree': 0.082089552238806,
  'hp_set': 4,
  'max_depth': 6,
  'reg_lambda': 33.7013757815395,
  'subsample': 0.23539318125695},
 {'colsample_bytree': 0.880597014925373,
  'hp_set': 5,
  'max_depth': 6,
  'reg_lambda': 0.175136472721016,
  'subsample': 0.606159858591855},
 {'colsample_bytree': 0.308457711442786,
  'hp_set': 6,
  'max_depth': 7,
  'reg_lambda': 0.0104059298394788,
  'subsample': 0.282029826566577},
 {'colsample_bytree': 0.487562189054726,
  'hp_set': 7,
  'max_depth': 7,
  'reg_lambda': 2.69786783063962,
  'subsample': 0.273632002435625},
 {'colsample_bytree': 0.915422885572139,
  'hp_set': 8,
  'max_depth': 3,
  'reg_lambda': 0.0161590590154502,
  'subsample': 0.536597859859467},
 {'colsample_bytree': 0.283582089552239,
  'hp_set': 9,
  'max_depth': 2,
  'reg_lambda': 4.71612234358155,
  'subsample': 0.803678817115724},
 {'colsample_bytree': 0.0621890547263682,
  'hp_set': 10,
  'max_depth': 7,
  'reg_lambda': 0.543820024894272,
  'subsample': 0.350324857234955},
 {'colsample_bytree': 0.181592039800995,
  'hp_set': 11,
  'max_depth': 2,
  'reg_lambda': 0.631114491480233,
  'subsample': 0.400126792304218},
 {'colsample_bytree': 0.380597014925373,
  'hp_set': 12,
  'max_depth': 4,
  'reg_lambda': 6.07586861650063,
  'subsample': 0.422225289791822},
 {'colsample_bytree': 0.721393034825871,
  'hp_set': 13,
  'max_depth': 4,
  'reg_lambda': 0.191837168860237,
  'subsample': 0.21493322532624},
 {'colsample_bytree': 0.932835820895522,
  'hp_set': 14,
  'max_depth': 3,
  'reg_lambda': 2.51031514919226,
  'subsample': 0.22102674394846},
 {'colsample_bytree': 0.977611940298508,
  'hp_set': 15,
  'max_depth': 6,
  'reg_lambda': 4.12665637626617,
  'subsample': 0.988919165730476},
 {'colsample_bytree': 0.644278606965174,
  'hp_set': 16,
  'max_depth': 3,
  'reg_lambda': 97.3280110678184,
  'subsample': 0.824435774795711},
 {'colsample_bytree': 0.659203980099503,
  'hp_set': 17,
  'max_depth': 6,
  'reg_lambda': 47.6447723610298,
  'subsample': 0.670239540748298},
 {'colsample_bytree': 0.885572139303483,
  'hp_set': 18,
  'max_depth': 1,
  'reg_lambda': 0.168539364780137,
  'subsample': 0.302476367726922},
 {'colsample_bytree': 0.654228855721393,
  'hp_set': 19,
  'max_depth': 5,
  'reg_lambda': 88.6329188111331,
  'subsample': 0.277405075542629},
 {'colsample_bytree': 0.90547263681592,
  'hp_set': 20,
  'max_depth': 4,
  'reg_lambda': 5.96527272203017,
  'subsample': 0.58861793782562})


@dataclass(frozen=True)
class ForecastSettings:
    train_years: int = 10
    chunk_months: int = 12
    folds: int = 5
    stage1_rounds: int = 1000
    stage2_rounds: int = 10000
    stage1_eta: float = 0.15
    stage2_eta: float = 0.01
    early_stopping: int = 25
    threads: int = 4
    seed: int = 1
    candidates: tuple = XGB_GRID


@dataclass(frozen=True)
class RiskSettings:
    ridge_lambda: float = 1e-4
    covariance_days: int = 2520
    correlation_half_life: int = 504
    variance_half_life: int = 84
    specific_half_life: int = 84
    initial_variance_observations: int = 63
    observation_window: int = 252
    minimum_residual_observations: int = 200
    threads: int = 4
    target_annual_volatility: float = 0.10


FORECAST_SETTINGS = ForecastSettings()
RISK_SETTINGS = RiskSettings()
INDUSTRIES = ('BusEq','Chems','Durbl','Enrgy','Hlth','Manuf',
              'Money','NoDur','Other','Shops','Telcm','Utils')


def training_folds(dates, first_test, settings):
    # Same boundaries as R cut(1:120, 5), keeping each return month together.
    first = month_number(first_test) - settings.train_years * 12
    length = settings.train_years * 12
    return [[d for d in dates if min(settings.folds - 1,
             (month_number(d)-first)*settings.folds//max(1,length-1)) == i]
            for i in range(settings.folds)]


def _xgb_params(candidate, eta, settings):
    return dict(candidate, objective='reg:squarederror', tree_method='hist',
                booster='gbtree', base_score=0, eval_metric='rmse',
                eta=eta, min_child_weight=1, gamma=0,
                nthread=settings.threads, seed=settings.seed)


def forecast_returns(chars, features, settings=FORECAST_SETTINGS, source_factory=MonthlySource):
    """The exact same expected-return stream feeds Factor ML and Markowitz ML."""
    names = canonical_features(features)
    source = source_factory(chars, names, 144)
    tests = sorted(source.metadata.filter(pl.col('ctff_test'))['eom_ret'].unique().to_list())
    if not tests:
        raise ValueError('No test return months')
    if not settings.candidates or settings.folds < 2 or settings.chunk_months < 1:
        raise ValueError('Invalid forecast settings')
    forecasts, fits = [], []
    with threadpool_limits(limits=settings.threads):
        for offset in range(0,len(tests),settings.chunk_months):
            chunk = tests[offset:offset+settings.chunk_months]
            first = chunk[0]
            lower = month_number(first)-settings.train_years*12
            dates = sorted(d for d in source.return_to_formation
                           if lower <= month_number(d) < month_number(first))
            xs, ys, row_months = [], [], []
            for d in dates:
                frame = source.load(source.return_to_formation[d]).filter(pl.col('ret_exc_lead1m').is_finite())
                xs.append(frame.select(names).to_numpy().astype(np.float32))
                ys.append(frame['ret_exc_lead1m'].to_numpy().astype(np.float32))
                row_months.extend([d]*frame.height)
            fit = dict(test_return_start=str(first),test_return_end=str(chunk[-1]),
                       training_return_first=str(dates[0]) if dates else None,
                       training_return_last=str(dates[-1]) if dates else None)
            if not ys or not sum(len(y) for y in ys):
                booster = None
                fit.update(train_rows=0,selected=None,trees=0,reason='no historical labels')
            else:
                x = np.concatenate(xs)
                y = np.concatenate(ys)
                del xs, ys
                row_dates = np.array(row_months,dtype='datetime64[D]')
                folds = training_folds(dates,first,settings)
                def search(candidates, eta, rounds):
                    results = [[] for _ in candidates]
                    for block, val_dates in enumerate(folds):
                        mask = np.isin(row_dates,np.array(val_dates,dtype='datetime64[D]'))
                        if not mask.any() or mask.all():
                            continue
                        train = xgb.DMatrix(x[~mask],label=y[~mask],nthread=settings.threads)
                        val = xgb.DMatrix(x[mask],label=y[mask],nthread=settings.threads)
                        for j, candidate in enumerate(candidates):
                            params = _xgb_params({k:v for k,v in candidate.items() if k!='hp_set'},eta,settings)
                            model = xgb.train(params,train,num_boost_round=rounds,
                                              evals=[(val,'val')],early_stopping_rounds=settings.early_stopping,
                                              verbose_eval=False)
                            results[j].append(dict(fold=block+1,mse=float(model.best_score)**2,
                                                   trees=int(model.best_iteration)+1))
                        del train,val
                        gc.collect()
                    return results
                stage1 = search(settings.candidates,settings.stage1_eta,settings.stage1_rounds)
                valid = [j for j,result in enumerate(stage1) if result]
                if valid:
                    chosen = min(valid,key=lambda j:(np.mean([v['mse'] for v in stage1[j]]),settings.candidates[j]['hp_set']))
                    candidate = settings.candidates[chosen]
                    stage2 = search([candidate],settings.stage2_eta,settings.stage2_rounds)[0]
                    trees = max(1,int(np.floor(np.mean([v['trees'] for v in stage2]))))
                else:
                    chosen=0; candidate=settings.candidates[0]; stage2=[]; trees=1
                training = xgb.DMatrix(x,label=y,nthread=settings.threads)
                booster=xgb.train(_xgb_params({k:v for k,v in candidate.items() if k!='hp_set'},settings.stage2_eta,settings),
                                  training,num_boost_round=trees,verbose_eval=False)
                fit.update(train_rows=len(y),selected=dict(candidate),trees=trees,
                           stage1=stage1,stage2=stage2)
                del x,y,training,row_dates
                gc.collect()
            print(f"XGBoost {first}: {fit['train_rows']:,} rows, {fit['trees']} trees",flush=True)
            for d in chunk:
                frame=source.load(source.return_to_formation[d]).filter(pl.col('ctff_test'))
                pred=np.zeros(frame.height) if booster is None else booster.inplace_predict(
                    frame.select(names).to_numpy().astype(np.float32))
                forecasts.append(frame.select('id','eom','excntry').with_columns(pl.Series('pred',pred).cast(pl.Float64)))
            fits.append(fit)
            del booster
    return pl.concat(forecasts).sort(['eom','id']),fits


def ff12_class(sic):
    groups=[('NoDur',[(100,999),(2000,2399),(2700,2749),(2770,2799),(3100,3199),(3940,3989)]),
            ('Durbl',[(2500,2519),(3630,3659),(3710,3711),(3714,3714),(3716,3716),(3750,3751),(3792,3792),(3900,3939),(3990,3999)]),
            ('Manuf',[(2520,2589),(2600,2699),(2750,2769),(3000,3099),(3200,3569),(3580,3629),(3700,3709),(3712,3713),(3715,3715),(3717,3749),(3752,3791),(3793,3799),(3830,3839),(3860,3899)]),
            ('Enrgy',[(1200,1399),(2900,2999)]),('Chems',[(2800,2829),(2840,2899)]),
            ('BusEq',[(3570,3579),(3660,3692),(3694,3699),(3810,3829),(7370,7379)]),
            ('Telcm',[(4800,4899)]),('Utils',[(4900,4949)]),
            ('Shops',[(5000,5999),(7200,7299),(7600,7699)]),
            ('Hlth',[(2830,2839),(3693,3693),(3840,3859),(8000,8099)]),('Money',[(6000,6999)])]
    values=np.array([int(v) if v is not None and str(v).isdigit() else -1 for v in sic])
    labels=np.full(len(values),'Other',dtype='U5')
    assigned=np.zeros(len(values),bool)
    for name,ranges in groups:
        mask=np.zeros(len(values),bool)
        for lower,upper in ranges: mask |= (values>=lower)&(values<=upper)
        mask &= ~assigned
        labels[mask]=name; assigned |= mask
    return labels


class RiskSource(MonthlySource):
    def __init__(self,chars,features,cache_size=144):
        super().__init__(chars,features,cache_size)
        self.sic=as_lazy(chars).select(pl.col('id').cast(pl.Int64),pl.col('eom').cast(pl.Date),
                                     pl.col('sic').cast(pl.String)).collect(engine='streaming')
        self.exposures={}
    def load_exposures(self,d):
        if d in self.exposures: return self.exposures[d]
        raw=self.connection.compile(self.table.filter(self.table.eom==d)).collect(engine='streaming')
        frame=prepare_pred_data(raw,self.features,min_obs=10).join(self.sic.filter(pl.col('eom')==d),on=['id','eom']).sort('id')
        values=frame.select(self.features).to_numpy()
        mean=values.mean(axis=0); sd=values.std(axis=0,ddof=1) if len(values)>1 else np.zeros(values.shape[1])
        # User choice: no artificial noise when a factor is constant.
        active=np.isfinite(sd)&(np.round(sd,6)!=0)
        standardized=np.zeros_like(values)
        standardized[:,active]=(values[:,active]-mean[active])/sd[active]
        labels=ff12_class(frame['sic'].to_list())
        industry=np.column_stack([labels==label for label in INDUSTRIES]).astype(float)
        result=(frame,np.column_stack([industry,standardized]),int((~active).sum()))
        self.exposures[d]=result
        if len(self.exposures)>2: del self.exposures[next(iter(self.exposures))]
        return result


class DailySource:
    def __init__(self,daily):
        self.frame=as_lazy(daily).select(pl.col('id').cast(pl.Int64),pl.col('date').cast(pl.Date),
                                      pl.col('ret_exc').cast(pl.Float64)).filter(pl.col('ret_exc').is_finite())
    def load(self,d):
        start=date(d.year,d.month,1)
        return self.frame.filter(pl.col('date').is_between(start,d)).sort(['date','id']).collect(engine='streaming')


def glmnet_ridge(x,y,penalty):
    """Match Gaussian glmnet alpha=0, standardize=FALSE, intercept=FALSE.

    glmnet scales y by its uncentered RMS for this no-intercept fit, so the
    original-unit diagonal regularizer is n*lambda/RMS(y). Zero-variance columns
    are excluded even when standardize=FALSE, matching glmnet's preprocessing.
    """
    if len(y)<2: raise ValueError('Risk ridge needs at least two observations')
    active=np.var(x,axis=0)>0
    result=np.zeros(x.shape[1])
    rms=float(np.sqrt(np.mean(y*y)))
    if not rms or not active.any(): return result
    a=x[:,active]; regularizer=len(y)*penalty/rms
    if a.shape[1]<=a.shape[0]:
        gram=a.T@a; gram[np.diag_indices_from(gram)]+=regularizer
        result[active]=cho_solve(cho_factor(gram,lower=True,check_finite=False),a.T@y,check_finite=False)
    else:
        gram=a@a.T; gram[np.diag_indices_from(gram)]+=regularizer
        result[active]=a.T@cho_solve(cho_factor(gram,lower=True,check_finite=False),y,check_finite=False)
    return result


def weighted_covariance(values,half_life):
    if len(values)<2: raise ValueError('At least two historical factor-return dates are required')
    w=0.5**(np.arange(len(values),0,-1)/half_life); w/=w.sum()
    centered=values-w@values
    return (centered.T*w)@centered/(1-w@w)


def factor_covariance(values,settings=RISK_SETTINGS):
    cor_cov=weighted_covariance(values,settings.correlation_half_life)
    denom=np.sqrt(np.maximum(np.diag(cor_cov),0))
    outer=np.outer(denom,denom)
    cor=np.divide(cor_cov,outer,out=np.zeros_like(cor_cov),where=outer>0)
    var_cov=weighted_covariance(values,settings.variance_half_life)
    sd=np.sqrt(np.maximum(np.diag(var_cov),0))
    covariance=cor*np.outer(sd,sd)
    return (covariance+covariance.T)/2


class SpecificRiskState:
    """Vector updates preserve the reference's previous-residual EWMA timing."""
    def __init__(self,ids,settings):
        self.settings=settings; self.ids=np.asarray(ids); n=len(ids)
        self.count=np.zeros(n,dtype=np.int64); self.seed_squares=np.zeros(n)
        self.seed_count=np.zeros(n,dtype=np.int64)
        self.variance=np.zeros(n); self.previous=np.zeros(n)
        self.history=np.full((n,settings.minimum_residual_observations+1),-1,dtype=np.int32)
        self.day_index=-1
    def advance(self,positions,residuals):
        self.day_index+=1
        c=self.count[positions]; s=self.settings
        seeded=c>=s.initial_variance_observations
        updating=c>s.initial_variance_observations
        u=positions[updating & np.isfinite(self.previous[positions])]; decay=0.5**(1/s.specific_half_life)
        self.variance[u]=decay*self.variance[u]+(1-decay)*self.previous[u]**2
        vol=np.full(len(positions),np.nan); vol[seeded]=np.sqrt(self.variance[positions[seeded]])
        initializing=~seeded
        valid_initial=initializing & np.isfinite(residuals)
        init=positions[valid_initial]
        self.seed_squares[init]+=residuals[valid_initial]**2
        self.seed_count[init]+=1
        ending=positions[c==s.initial_variance_observations-1]
        self.variance[ending]=self.seed_squares[ending]/np.maximum(1,self.seed_count[ending]-1)
        valid=c>=s.minimum_residual_observations
        past=np.full(len(positions),-1,dtype=np.int32)
        width=s.minimum_residual_observations+1
        past[valid]=self.history[positions[valid],(c[valid]-s.minimum_residual_observations)%width]
        eligible=valid&(self.day_index>=s.observation_window)&(past>=self.day_index-s.observation_window)&np.isfinite(vol)&(vol>0)
        self.history[positions,c%width]=self.day_index
        self.previous[positions]=residuals; self.count[positions]+=1
        return vol,eligible


def covariance_solve(diagonal,factor_cov,x,b):
    """Woodbury via a factor square root, also valid for singular factor covariance."""
    if np.any(~np.isfinite(diagonal)) or np.any(diagonal<=0):
        raise ValueError('Specific variances must be finite and strictly positive')
    eigen,vectors=np.linalg.eigh(factor_cov)
    scale=max(float(np.max(np.abs(eigen))),1e-20)
    if eigen.min() < -1e-9*scale: raise ValueError('Factor covariance is materially indefinite')
    loading=x@(vectors*np.sqrt(np.maximum(eigen,0)))
    dinv_b=b/diagonal; dinv_loading=loading/diagonal[:,None]
    small=np.eye(loading.shape[1])+loading.T@dinv_loading
    correction=cho_solve(cho_factor(small,lower=True,check_finite=False),loading.T@dinv_b,check_finite=False)
    return dinv_b-dinv_loading@correction


def portfolio_variance(w,diagonal,factor_cov,x):
    exposure=x.T@w
    return float(np.sum(w*w*diagonal)+exposure@factor_cov@exposure)


def optimize_portfolios(ids,d,x,diagonal,factor_cov,mu,settings=RISK_SETTINGS):
    ones=np.ones(len(ids)); raw=covariance_solve(diagonal,factor_cov,x,ones)
    if not np.isfinite(raw.sum()) or raw.sum()<=0: raise ValueError('Invalid minimum-variance normalization')
    minimum=raw/raw.sum()
    minimum_frame=pl.DataFrame({'id':ids,'eom':[d]*len(ids),'w':minimum})
    markowitz_frame=None; predicted_vol=None
    if mu is not None:
        direction=covariance_solve(diagonal,factor_cov,x,mu)
        variance=portfolio_variance(direction,diagonal,factor_cov,x)
        if not np.isfinite(variance) or variance<=0:
            raise ValueError('Markowitz direction has zero or invalid variance; cannot scale to target volatility')
        w=direction*settings.target_annual_volatility/np.sqrt(252*variance)
        predicted_vol=np.sqrt(252*portfolio_variance(w,diagonal,factor_cov,x))
        markowitz_frame=pl.DataFrame({'id':ids,'eom':[d]*len(ids),'w':w})
    return minimum_frame,markowitz_frame,predicted_vol


def risk_portfolios(chars,features,daily_ret,predictions=None,settings=RISK_SETTINGS,
                    source_factory=RiskSource,daily_factory=DailySource):
    """Process one month of daily cross sections; never form the full wide daily join."""
    names=canonical_features(features); source=source_factory(chars,names,144)
    days=daily_factory(daily_ret)
    months=sorted(source.metadata['eom'].unique().to_list())
    test_months=set(source.metadata.filter(pl.col('ctff_test'))['eom'].unique().to_list())
    if not test_months: raise ValueError('No test formation months')
    last=max(test_months); months=[d for d in months if d<=last]
    ids=np.array(sorted(source.metadata['id'].unique().to_list()))
    state=SpecificRiskState(ids,settings)
    returns=deque(maxlen=settings.covariance_days)
    minimum_weights=[]; markowitz_weights=[]; records=[]
    previous=None; risk_model=None; risk_model_date=None
    with threadpool_limits(limits=settings.threads):
        for d in months:
            specific={}; regressions=0
            if previous is not None and previous[0]['eom_ret'][0]==d:
                prev_frame,prev_x,_=previous
                prev_ids=prev_frame['id'].to_numpy()
                daily=days.load(d).filter(pl.col('id').is_in(prev_ids.tolist()))
                for group in daily.partition_by('date',maintain_order=True):
                    day_ids=group['id'].to_numpy(); idx=np.searchsorted(prev_ids,day_ids)
                    x=prev_x[idx]; y=group['ret_exc'].to_numpy()
                    if len(y)<2: continue
                    coef=glmnet_ridge(x,y,settings.ridge_lambda)
                    residual=y-x@coef; returns.append(coef)
                    positions=np.searchsorted(ids,day_ids)
                    vol,eligible=state.advance(positions,residual)
                    for security,v in zip(day_ids[eligible],vol[eligible]): specific[int(security)]=float(v)
                    regressions+=1
            frame,x,constant=source.load_exposures(d)
            frame_ids=frame['id'].to_numpy()
            actual=np.array([specific.get(int(i),np.nan) for i in frame_ids])
            available=np.isfinite(actual)&(actual>0)
            if available.sum()>=2:
                risk_model=glmnet_ridge(x[available],np.log(actual[available]),settings.ridge_lambda)
                risk_model_date=d
            if d in test_months:
                if risk_model is None: raise ValueError(f'No specific-risk model available by {d}; need 200 residuals in 252 trading dates')
                vol=np.exp(x@risk_model)
                vol[available]=actual[available]
                selection=frame['ctff_test'].to_numpy()
                selected_ids=frame_ids[selection]; selected_x=x[selection]; diagonal=vol[selection]**2
                if not np.isfinite(diagonal).all() or np.any(diagonal<=0): raise ValueError('Invalid predicted specific risk')
                covariance=factor_covariance(np.array(returns),settings)
                mu=None
                if predictions is not None:
                    matched=frame.filter(pl.col('ctff_test')).select('id','eom').join(predictions.select('id','eom','pred'),on=['id','eom'],how='left').sort('id')
                    if matched['pred'].null_count(): raise ValueError('Expected returns missing for risk-model securities')
                    mu=matched['pred'].to_numpy()
                min_w,mark_w,ex_ante=optimize_portfolios(selected_ids,d,selected_x,diagonal,covariance,mu,settings)
                minimum_weights.append(min_w)
                if mark_w is not None: markowitz_weights.append(mark_w)
                records.append(dict(formation_date=str(d),factor_return_days=len(returns),
                                    factor_count=x.shape[1],constant_characteristics=constant,
                                    specific_risk_observed=int(available[selection].sum()),
                                    specific_risk_predicted=int((~available[selection]).sum()),
                                    specific_risk_model_date=str(risk_model_date),
                                    markowitz_predicted_annual_volatility=ex_ante))
            previous=(frame,x,constant)
            if regressions and (d.month==12 or d in test_months):
                print(f'Barra {d}: {regressions} daily fits, {available.sum()} observed specific risks',flush=True)
    return pl.concat(minimum_weights),pl.concat(markowitz_weights) if markowitz_weights else None,records


def run_comparisons(chars,features,daily_ret,models=('factor_ml','minimum_variance','markowitz_ml'),
                    forecast_settings=FORECAST_SETTINGS,risk_settings=RISK_SETTINGS,
                    forecast_source=MonthlySource,risk_source=RiskSource,daily_source=DailySource):
    started=perf_counter()
    unknown=set(models)-{'factor_ml','minimum_variance','markowitz_ml'}
    if unknown: raise ValueError(f'Unknown portfolios: {unknown}')
    predictions=None; fits=[]; risks=[]; outputs={}
    if 'factor_ml' in models or 'markowitz_ml' in models:
        predictions,fits=forecast_returns(chars,features,forecast_settings,forecast_source)
        if 'factor_ml' in models: outputs['factor_ml']=predictions_to_weights(predictions)
    if 'minimum_variance' in models or 'markowitz_ml' in models:
        minimum,markowitz,risks=risk_portfolios(chars,features,daily_ret,
                    predictions if 'markowitz_ml' in models else None,risk_settings,risk_source,daily_source)
        if 'minimum_variance' in models: outputs['minimum_variance']=minimum
        if 'markowitz_ml' in models: outputs['markowitz_ml']=markowitz
    metadata=MonthlySource(chars,canonical_features(features)).metadata
    expected=metadata.filter(pl.col('ctff_test')).select('id','eom')
    for name,w in outputs.items():
        w=finalize_output(w,started)
        if w.height!=expected.height or w.join(expected,on=['id','eom'],how='anti').height:
            raise ValueError(f'{name} does not cover exact test keys')
        outputs[name]=w
    return outputs,predictions,fits,risks
