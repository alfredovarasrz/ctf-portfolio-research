"""Isolated expanding F history and lagged drawdown-conditioned original F."""
from bisect import bisect_right
from calendar import monthrange
from datetime import date
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from comparison_models import RISK_SETTINGS
from extended_risk import _parent_months, _factor_stream
from tuned_risk_decay import decay_factor_covariance
from artifact_utils import write_json

VARIANTS = {'R07': None,
    'R08_EXPANDING_CORRELATION': 'correlation',
    'R08_EXPANDING_VARIANCE': 'variance',
    'R10_DD10': .10, 'R10_DD15': .15, 'R10_DD20': .20}
MINIMUM_STATE_DAYS = 63


def combine_correlation_variance(correlation_covariance, variance_covariance):
    sd=np.sqrt(np.maximum(np.diag(correlation_covariance),0.))
    outer=np.outer(sd,sd)
    correlation=np.divide(correlation_covariance,outer,out=np.zeros_like(outer),where=outer>0)
    var_sd=np.sqrt(np.maximum(np.diag(variance_covariance),0.))
    result=correlation*np.outer(var_sd,var_sd)
    return (result+result.T)/2


class ExpandingEW:
    """Merge weighted centered moments once per newly completed factor block."""
    def __init__(self, width, half_life):
        if width<1 or half_life<=0:
            raise ValueError('Positive factor width and half-life required')
        self.width=width; self.half_life=half_life; self.days=0
        self.weight=self.squared_weight=0.
        self.mean=np.zeros(width); self.scatter=np.zeros((width,width))

    def append(self, values):
        values=np.asarray(values,dtype=float)
        if values.ndim!=2 or values.shape[1]!=self.width or not np.isfinite(values).all():
            raise ValueError('Aligned finite expanding factor block required')
        if not len(values):
            return
        weights=.5**(np.arange(len(values),0,-1)/self.half_life)
        block_weight=float(weights.sum()); block_mean=weights@values/block_weight
        centered=values-block_mean
        block_scatter=(centered.T*weights)@centered
        decay=.5**(len(values)/self.half_life)
        old_weight=self.weight*decay; total=old_weight+block_weight
        delta=block_mean-self.mean
        self.scatter=(self.scatter*decay+block_scatter
            +np.outer(delta,delta)*(old_weight*block_weight/total))
        self.mean=self.mean+delta*(block_weight/total)
        self.weight=total; self.squared_weight=self.squared_weight*decay**2+float(weights@weights)
        self.days+=len(values)

    def covariance(self):
        denominator=self.weight-self.squared_weight/self.weight if self.weight else 0.
        if self.days<2 or denominator<=0:
            raise ValueError('At least two expanding factor dates required')
        result=self.scatter/denominator
        return (result+result.T)/2


def drawdown_path(proxy):
    """Observed excess-index peak and drawdown BEFORE and AFTER each return."""
    dates=proxy['date'].to_list(); returns=proxy['ret'].to_numpy()
    if (dates!=sorted(set(dates)) or not np.isfinite(returns).all() or np.any(returns<=-1)):
        raise ValueError('Unique chronological proxy returns with positive gross factors required')
    wealth=peak=1.; before=[]; after=[]; peaks=[]; levels=[]
    for value in returns:
        before.append(max(0.,1.-wealth/peak))
        wealth*=1.+float(value); peak=max(peak,wealth)
        if not np.isfinite(wealth) or not np.isfinite(peak) or wealth<=0:
            raise ValueError('Excess-return proxy index must remain finite and positive')
        after.append(max(0.,1.-wealth/peak)); peaks.append(peak); levels.append(wealth)
    return dict(dates=dates,pre_return=np.array(before),completed=np.array(after),
        observed_peak=np.array(peaks),excess_index=np.array(levels))


def conditional_covariance(values, ages, pre_return_drawdowns, current_drawdown, reference,
        threshold, settings=RISK_SETTINGS, minimum_days=MINIMUM_STATE_DAYS):
    """One common conditional EW estimator, with original full-window ages."""
    values=np.asarray(values,dtype=float); ages=np.asarray(ages,dtype=float)
    drawdowns=np.asarray(pre_return_drawdowns,dtype=float)
    if (values.ndim!=2 or len(values)<2 or ages.shape!=(len(values),) or drawdowns.shape!=ages.shape
            or not np.isfinite(values).all() or not np.isfinite(ages).all() or np.any(ages<=0)
            or not np.isfinite(drawdowns).all() or np.any((drawdowns<0)|(drawdowns>=1))
            or not np.isfinite(current_drawdown) or not 0<=current_drawdown<1
            or threshold not in (.10,.15,.20) or minimum_days<2
            or reference.shape!=(values.shape[1],values.shape[1]) or not np.isfinite(reference).all()):
        raise ValueError('Finite aligned conditional history and approved drawdown threshold required')
    # Compare the remaining peak fraction. Algebraically this is drawdown>=threshold,
    # while an exact ten-percent decline avoids cancellation in1-.9.
    bear=(1.-drawdowns)<=1.-threshold; current_bear=(1.-current_drawdown)<=1.-threshold
    selected=bear if current_bear else ~bear; count=int(selected.sum())
    fallback=count<minimum_days
    covariance=(reference.copy() if fallback else decay_factor_covariance(
        values[selected],ages[selected],settings,'correlation',None))
    return covariance,dict(drawdown_threshold=threshold,current_drawdown=float(current_drawdown),
        current_state='bear' if current_bear else 'normal',conditional_factor_days=count,
        bear_factor_days=int(bear.sum()),normal_factor_days=int((~bear).sum()),
        conditional_minimum_days=minimum_days,original_pooled_F_fallback=fallback,
        daily_state_timing='observed drawdown immediately BEFORE each factor return',
        allocation_state_timing='completed proxy state at formation for next-month weights',
        conditional_age_policy='original2520 observation ages retained before state filtering')


def select_expanding_decay(values, dates, settings=RISK_SETTINGS, *, component):
    """Rotate all expanding calendar blocks, each spanning at most96 months."""
    from capped_calendar_validation import capped_calendar_blocks
    values=np.asarray(values,dtype=float); dates=list(dates)
    if (component not in ('correlation','variance') or values.ndim!=2 or len(values)!=len(dates)
            or dates!=sorted(set(dates)) or not np.isfinite(values).all()):
        raise ValueError('Named component and unique aligned expanding factor dates required')
    original=settings.correlation_half_life if component=='correlation' else settings.variance_half_life
    candidates=[float(original*.5),float(original),float(original*2.)]
    months=sorted({date(d.year,d.month,monthrange(d.year,d.month)[1]) for d in dates})
    blocks,boundaries=capped_calendar_blocks(months)
    row_month=[date(d.year,d.month,monthrange(d.year,d.month)[1]) for d in dates]
    ages=np.arange(len(values),0,-1,dtype=float)
    record=dict(component=component,hyperbolic_power=1.,candidate_half_weight_days=candidates,
        validation_policy='all completed expanding calendar blocks; each held-out block at most96months',
        validation_target='uniform centered unbiased held-out factor covariance, denominator n-1',
        validation_loss='mean squared factor covariance entries; equal usable fold weight',
        weighting_age_policy='FULL expanding original observation ages; held-out gaps retained',
        tuning_days=len(values),tuning_months=len(months),calendar_blocks=boundaries,
        selected_half_weight_days=None,selection_fallback=None,candidates=[],exponential_control_folds=[],
        skipped_folds=[],minimum_usable_folds=2)
    folds=[]
    for block,boundary in zip(blocks,boundaries):
        held=set(block); validation=np.array([m in held for m in row_month]); training=~validation
        if validation.sum()<2 or training.sum()<2:
            record['skipped_folds'].append(dict(boundary,training_days=int(training.sum()),
                validation_days=int(validation.sum()),reason='fewer than two fit or validation factor rows'))
            continue
        v=values[validation]; centered=v-v.mean(axis=0); target=centered.T@centered/(len(v)-1)
        fit_dates=[d for d,t in zip(dates,training) if t]; held_dates=[d for d,t in zip(dates,validation) if t]
        folds.append((training,target,dict(boundary,training_days=int(training.sum()),validation_days=int(validation.sum()),
            training_first_day=str(fit_dates[0]),training_last_day=str(fit_dates[-1]),
            validation_start=str(held_dates[0]),validation_end=str(held_dates[-1]))))
    if len(folds)<2:
        record['selection_fallback']='expanding original exponential; fewer than two usable calendar folds'
        return None,record
    for candidate in [None]+candidates:
        scores=[]
        for training,target,metadata in folds:
            fitted=decay_factor_covariance(values[training],ages[training],settings,component,candidate)
            scores.append(dict(metadata,mean_squared_covariance_error=float(np.mean((fitted-target)**2))))
        if candidate is None:
            record['exponential_control_folds']=scores
        else:
            record['candidates'].append(dict(half_weight_days=candidate,folds=scores,
                mean_validation_loss=float(np.mean([r['mean_squared_covariance_error'] for r in scores]))))
    winner=min(record['candidates'],key=lambda row:(row['mean_validation_loss'],row['half_weight_days']))
    record['selected_half_weight_days']=winner['half_weight_days']
    return winner['half_weight_days'],record


def factor_prefix_receipt(root, cutoff):
    """Actual consumed factor bytes, including pre1980, excluding future bodies."""
    from artifact_utils import digest
    root=Path(root); manifest=json.loads((root/'manifest.json').read_text()); files={}
    months=[]
    for folder in sorted((root/'months').iterdir()):
        if not folder.is_dir() or folder.name.endswith('.tmp') or folder.name>str(cutoff):
            continue
        marker=json.loads((folder/'complete.json').read_text())
        if (marker['formation_date']!=folder.name or marker['available_through']!=folder.name
                or marker['factor_order']!=manifest['factor_order']):
            raise ValueError('Expanding consumed-prefix factor cutoff or order differs')
        months.append(folder.name)
        for name in ('complete.json','factor_returns.parquet'):
            files[str((folder/name).relative_to(root))]=digest(folder/name)
    if not months:
        raise ValueError('No original factor prefix available')
    return dict(path=str(root.resolve()),information_cutoff=str(cutoff),committed_months=months,
        factor_order=manifest['factor_order'],manifest_sha256=digest(root/'manifest.json'),files_sha256=files)


def annual_expanding_decay(root, cache, d, values, dates, settings, component, identity):
    from artifact_utils import digest
    cache=Path(cache); cache.mkdir(parents=True,exist_ok=True); year=d.year+int(d.month==12)
    wanted=dict(component=component,forecast_year=year,information_cutoff=str(d),
        source_sha256=digest(Path(__file__)),decay_math_sha256=digest(Path(__file__).with_name('tuned_risk_decay.py')),
        calendar_source_sha256=digest(Path(__file__).with_name('capped_calendar_validation.py')),
        parent_prefix=factor_prefix_receipt(root,d),caller_identity=identity,
        half_lives=dict(correlation=settings.correlation_half_life,variance=settings.variance_half_life))
    path=cache/f'{year}.json'
    if path.exists():
        saved=json.loads(path.read_text())
        if saved['identity']!=wanted:
            raise ValueError('Expanding annual decay cache identity differs; preserve evidence')
        record=saved['record']; half_weight=record['selected_half_weight_days']
    else:
        half_weight,record=select_expanding_decay(values,dates,settings,component=component)
        record.update(tuning_information_cutoff=str(d),tuning_last_factor_day=str(dates[-1]) if dates else None,
            forecast_year=year)
        write_json(path,dict(identity=wanted,record=record))
    compact={key:record[key] for key in ('component','hyperbolic_power','candidate_half_weight_days',
        'selected_half_weight_days','selection_fallback','tuning_information_cutoff','tuning_last_factor_day','forecast_year')}
    compact.update(annual_curve_sha256=digest(path),annual_curve_file=str(path.resolve()))
    return half_weight,compact


def iter_history_risk(artifact_dir, experiment, *, cutoff=None, proxy=None, threads=1,
        selection_cache=None, identity=None):
    """Yield original B/D/reference coordinates with only the declared F change."""
    if experiment not in VARIANTS or not isinstance(threads,int) or not 1<=threads<=10:
        raise ValueError('Approved isolated risk-history cell and one to ten threads required')
    root,manifest,metadata,directories=_parent_months(artifact_dir,cutoff)
    settings=type(RISK_SETTINGS)(**manifest['settings']); names=manifest['factor_order']
    records={r['formation_date']:r for r in metadata['records']}
    history=[]; dates=[]; selections={}; last_date=None
    expanding=experiment=='R07' or experiment.startswith('R08_EXPANDING_')
    corr=ExpandingEW(len(names),settings.correlation_half_life)
    variance=ExpandingEW(len(names),settings.variance_half_life)
    path=drawdown_path(proxy) if experiment.startswith('R10_DD') and proxy is not None else None
    if experiment.startswith('R10_DD') and path is None:
        raise ValueError('Verified excess-return proxy prefix required')
    if experiment.startswith('R08_EXPANDING_') and selection_cache is None:
        raise ValueError('Separate annual selection cache required')
    proxy_index={d:i for i,d in enumerate(path['dates'])} if path is not None else {}
    with threadpool_limits(limits=threads):
        for folder in directories:
            d=date.fromisoformat(folder.name); stream,_=_factor_stream(folder,names,d)
            values=stream.select(names).to_numpy(); new_dates=stream['date'].to_list()
            if new_dates and last_date is not None and new_dates[0]<=last_date:
                raise ValueError('Original factor dates overlap or move backward')
            if new_dates:
                last_date=new_dates[-1]; history.extend(values); dates.extend(new_dates)
                if expanding:
                    corr.append(values); variance.append(values)
                elif len(history)>settings.covariance_days:
                    del history[:-settings.covariance_days]; del dates[:-settings.covariance_days]
            if str(d) not in records:
                continue
            if len(history)<2:
                raise ValueError('Scored covariance requires at least two completed original factor rows')
            with np.load(folder/'risk.npz',allow_pickle=False) as saved:
                risk={name:saved[name].copy() for name in ('ids','B','F','D')}
                if str(saved['eom'])!=str(d):
                    raise ValueError('Original scored covariance date differs')
            reference=risk['F']; record=dict(records[str(d)],covariance_variant=experiment,
                covariance_information_cutoff=str(d),basis_information_cutoff=str(d),
                risk_exposure_basis='unchanged original B',factor_returns_through=str(last_date),
                factor_return_days=len(history),factor_history_first_day=str(dates[0]),
                covariance_history='ALL usable completed original factor history' if expanding else 'original2520 factor window')
            if experiment=='R07':
                covariance=combine_correlation_variance(corr.covariance(),variance.covariance())
            elif experiment.startswith('R08_EXPANDING_'):
                component=VARIANTS[experiment]; year=d.year+int(d.month==12)
                full=np.array(history)
                if year not in selections:
                    selections[year]=annual_expanding_decay(root,selection_cache,d,full,dates,settings,component,identity)
                half_weight,selection=selections[year]
                covariance=(combine_correlation_variance(corr.covariance(),variance.covariance()) if half_weight is None
                    else decay_factor_covariance(full,np.arange(len(full),0,-1),settings,component,half_weight))
                record['decay_selection']=selection
            else:
                missing=[day for day in dates if day not in proxy_index]
                position=bisect_right(path['dates'],d)-1
                if missing or position<0:
                    raise ValueError('Conditional factor dates or completed formation lack proxy observations')
                state=path['completed'][position]
                covariance,conditional=conditional_covariance(np.array(history),np.arange(len(history),0,-1),
                    np.array([path['pre_return'][proxy_index[day]] for day in dates]),state,reference,VARIANTS[experiment],settings)
                record.update(conditional,proxy_information_cutoff=str(path['dates'][position]),
                    proxy_index_level=float(path['excess_index'][position]),proxy_observed_peak=float(path['observed_peak'][position]))
            yield dict(ids=risk['ids'],eom=d,B=risk['B'],F=covariance,D=risk['D'],
                reference_B=risk['B'].copy(),reference_F=reference,reference_D=risk['D'].copy(),record=record)
