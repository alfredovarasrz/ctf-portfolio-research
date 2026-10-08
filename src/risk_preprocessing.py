"""Separate risk-only percentile, median and benchmark-noise exposure adapters."""
from collections import OrderedDict
import hashlib
from pathlib import Path

import numpy as np
import polars as pl
from scipy.special import ndtri

from baseline import MonthlySource, as_lazy, prepare_pred_data
from comparison_models import INDUSTRIES, RISK_SETTINGS, ff12_class
from research_risk import build_risk_artifacts, _input_identity

VARIANTS = ('percentile_exposures', 'benchmark_constant_noise', 'country_median', 'industry_median')


class BenchmarkNormalStream:
    """R seed initialization + MT19937 + two-uniform INVERSION normal recipe.

    Primary R sources document the initialization and inversion input. NumPy
    supplies MT19937; SciPy supplies inverse normal. Last-bit qnorm equality is
    a fixture acceptance boundary, not an untested exact-parity claim.
    """
    def __init__(self, seed=1):
        value = int(seed)&0xffffffff
        for _ in range(50):
            value = (69069*value+1)&0xffffffff
        state = []
        for _ in range(625):
            value = (69069*value+1)&0xffffffff; state.append(value)
        self.generator = np.random.MT19937(0)
        self.generator.state = dict(bit_generator='MT19937', state=dict(key=np.array(state[1:],dtype=np.uint32),pos=624))

    def normal(self, n):
        uniform = self.generator.random_raw(2*n).astype(float).reshape(n,2)/2.**32
        epsilon = .5/(2.**32-1.)
        uniform = np.clip(uniform,epsilon,1.-epsilon)
        probability = (np.floor(2.**27*uniform[:,0])+uniform[:,1])/2.**27
        return ndtri(probability)


def prepare_risk_medians(raw, features, group):
    """Only replace missing usable ranks; retain original min_obs=10 safeguard."""
    if group not in ('country','industry'):
        raise ValueError('Risk median group must be country or industry')
    raw = raw.with_columns(pl.col(features).cast(pl.Float64).fill_nan(None)).sort(['id','eom'])
    ranked = prepare_pred_data(raw,features,min_obs=10)
    country = ['excntry','eom']
    # Under min_obs10, the entire unavailable country-month characteristic stays
    # original zero. Missing cells are median-filled only when ranks were usable.
    counts = raw.select([pl.col(name).count().over(country).alias(name) for name in features])
    ranked = ranked.with_columns([
        pl.when(raw[name].is_null() & (counts[name]>=10)).then(None).otherwise(pl.col(name)).alias(name)
        for name in features])
    groups = country
    if group=='industry':
        ranked = ranked.with_columns(pl.Series('__risk_industry',ff12_class(raw['sic'].to_list())))
        groups = country+['__risk_industry']
    ranked = ranked.with_columns([
        pl.col(name).fill_null(pl.col(name).median().over(groups)
            .fill_null(pl.col(name).median().over(country)).fill_null(0.)).alias(name)
        for name in features])
    return ranked.drop('__risk_industry') if group=='industry' else ranked


def exposure_values(frame, features, variant, *, normal_stream=None):
    """Original industry block; change exactly the declared characteristic step."""
    values = frame.select(features).to_numpy()
    mean = values.mean(axis=0)
    sd = values.std(axis=0,ddof=1) if len(values)>1 else np.zeros(values.shape[1])
    active = np.isfinite(sd)&(np.round(sd,6)!=0)
    constants = int((~active).sum())
    if variant=='percentile_exposures':
        # The original midpoint/raw-zero/minimum-count convention stays intact.
        characteristics = values.copy()
        characteristics[:,~active] = 0.
    else:
        if variant=='benchmark_constant_noise':
            if normal_stream is None or len(values)<2:
                raise ValueError('Benchmark noise needs seeded stream and at least two stocks')
            values = values.copy()
            for j in range(len(features)):
                if not active[j]:
                    values[:,j] += normal_stream.normal(len(values))
            mean = values.mean(axis=0); sd = values.std(axis=0,ddof=1)
            active = np.isfinite(sd)&(sd>0)
            if not active.all():
                raise ValueError('Benchmark noise did not produce finite nonzero sample SD')
        characteristics = np.zeros_like(values)
        characteristics[:,active] = (values[:,active]-mean[active])/sd[active]
    labels = ff12_class(frame['sic'].to_list())
    industry = np.column_stack([labels==label for label in INDUSTRIES]).astype(float)
    return np.column_stack([industry,characteristics]),constants


class RiskPreprocessingSource(MonthlySource):
    """Bounded staged monthly reads; seeded noise advances only through requested d."""
    def __init__(self, chars, features, cache_size=144, *, variant, seed=1):
        super().__init__(chars,features,cache_size)
        if variant not in VARIANTS:
            raise ValueError('Unknown risk preprocessing variant')
        self.variant = variant; self.seed = seed; self.cache = OrderedDict()
        self.sic = as_lazy(chars).select(pl.col('id').cast(pl.Int64),pl.col('eom').cast(pl.Date),pl.col('sic').cast(pl.String))
        self.months = sorted(self.metadata['eom'].unique().to_list())
        self.stream = BenchmarkNormalStream(seed); self.last_noise_month = None

    def _one_month(self,d):
        raw = self.connection.compile(self.table.filter(self.table.eom==d)).collect(engine='streaming').sort(['id','eom'])
        sic = self.sic.filter(pl.col('eom')==d).collect(engine='streaming')
        raw = raw.join(sic,on=['id','eom']).sort(['id','eom'])
        if self.variant in ('country_median','industry_median'):
            frame = prepare_risk_medians(raw,self.features,'country' if self.variant=='country_median' else 'industry')
        else:
            frame = prepare_pred_data(raw,self.features,min_obs=10)
        x,constant = exposure_values(frame,self.features,self.variant,normal_stream=self.stream)
        result = frame,x,constant
        self.cache[d] = result
        while len(self.cache)>2:
            self.cache.popitem(last=False)
        return result

    def load_exposures(self,d):
        if d in self.cache:
            return self.cache[d]
        if d not in self.months:
            raise ValueError('Risk preprocessing requested absent formation month')
        if self.variant!='benchmark_constant_noise':
            return self._one_month(d)
        if self.last_noise_month is not None and d<self.last_noise_month:
            self.stream = BenchmarkNormalStream(self.seed); self.last_noise_month = None; self.cache.clear()
        # Replaying the prefix restores the exact causal RNG position on resume.
        # Never draw using any future-month feature or observed constant count.
        for month in self.months:
            if month>d:
                break
            if self.last_noise_month is None or month>self.last_noise_month:
                self._one_month(month); self.last_noise_month = month
        return self.cache[d]


def build_preprocessed_risk(chars, features, daily_ret, output_dir, variant, *,
        settings=RISK_SETTINGS, identity=None, seed=1):
    """Rebuild f/residual/F/D only because upstream risk B changed; forecasts fixed."""
    if variant not in VARIANTS:
        raise ValueError('Unknown risk preprocessing variant')
    if identity is None:
        identity = dict(chars=_input_identity(chars,['eom','id']),daily=_input_identity(daily_ret,['date','id']))
    specification = dict(caller_identity=identity,variant=variant,seed=seed,
        adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        minimum_characteristic_observations=10, noise_order='chronological month, canonical feature, sorted stock id',
        rng='R seed initialization + MT19937 + INVERSION probability + SciPy ndtri',
        missing_rank_policy='original midpoint unless declared usable-rank group median')
    def source_factory(chars,names,cache_size=144):
        return RiskPreprocessingSource(chars,names,cache_size,variant=variant,seed=seed)
    return build_risk_artifacts(chars,features,daily_ret,output_dir,settings=settings,
        identity=specification,source_factory=source_factory)
