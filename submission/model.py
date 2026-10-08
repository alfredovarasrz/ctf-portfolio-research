"""Factor-return forecasting with expanding variance and regularized allocation.

Self-contained Common Task Framework model. The only model inputs are the three
provided tables. Historical tuning remains part of the strategy; outer-period
performance evaluation is deliberately absent. The competition host's R
benchmarks provide the preprocessing and Barra risk model.
"""

# Section 1: Libraries and settings
from collections import OrderedDict, deque
from dataclasses import dataclass
from datetime import date
from calendar import monthrange
from math import ceil
from time import perf_counter

import numpy as np
import pandas as pd
import polars as pl
from scipy.linalg import cho_factor, cho_solve
from threadpoolctl import threadpool_limits


@dataclass(frozen=True)
class ForecastSettings:
    train_years: int = 10
    folds: int = 5
    ridge_lambdas: tuple = (0.0001, 0.001, 0.01, 0.1, 1., 10.)


@dataclass(frozen=True)
class RiskSettings:
    ridge_lambda: float = 0.0001
    covariance_days: int = 2520
    correlation_half_life: int = 504
    variance_half_life: int = 84
    specific_half_life: int = 84
    initial_variance_observations: int = 63
    observation_window: int = 252
    minimum_residual_observations: int = 200
    threads: int = 10
    target_annual_volatility: float = 0.1


FORECAST_SETTINGS = ForecastSettings()
RISK_SETTINGS = RiskSettings()
META = ["id", "eom", "eom_ret", "excntry", "ctff_test", "ret_exc_lead1m"]
INDUSTRIES = ('BusEq', 'Chems', 'Durbl', 'Enrgy', 'Hlth', 'Manuf',
              'Money', 'NoDur', 'Other', 'Shops', 'Telcm', 'Utils')
Q_GRID = (0.,) + tuple(float(q) for q in np.logspace(-5, 2, 15))

# Section 2: Monthly exposures and original daily Barra regressions


def canonical_features(features: pd.DataFrame) -> list[str]:
    return sorted(set(features['features'].tolist()))


def canonical_chars(chars: pd.DataFrame, features: list[str]) -> pl.LazyFrame:
    return pl.from_pandas(chars).lazy().select(
        pl.col('id').cast(pl.Int64),
        pl.col('eom', 'eom_ret').cast(pl.Date),
        pl.col('excntry').cast(pl.String),
        pl.col('sic').cast(pl.Float64, strict=False).fill_nan(None),
        pl.col('ctff_test').cast(pl.String).str.to_lowercase().replace_strict(
            {'0': False, '1': True, 'false': False, 'true': True},
            default=None, return_dtype=pl.Boolean),
        pl.col('ret_exc_lead1m').cast(pl.Float64).fill_nan(None),
        pl.col(features).cast(pl.Float64).fill_nan(None))


def prepare_pred_data(data: pl.DataFrame, features: list[str], min_obs=None) -> pl.DataFrame:
    """Match R: maximum tie rank / nonmissing count, zero override, center, impute."""
    groups = ['excntry', 'eom']
    expressions = []
    for name in features:
        x = pl.col(name).cast(pl.Float64).fill_nan(None)
        count = x.count().over(groups)
        ranked = x.rank(method='max').over(groups) / count
        centered = pl.when(x == 0).then(-0.5).otherwise(ranked - 0.5)
        if min_obs is not None:
            centered = pl.when(count >= min_obs).then(centered).otherwise(None)
        expressions.append(centered.fill_null(0.0).alias(name))
    return data.with_columns(expressions).sort(['id', 'eom'])


def month_number(d: date) -> int:
    return d.year * 12 + d.month - 1


def month_end(number: int) -> date:
    year, zero_month = divmod(number, 12)
    following = date(year + (zero_month == 11), zero_month % 12 + 2 if zero_month < 11 else 1, 1)
    return date.fromordinal(following.toordinal() - 1)


class MonthlySource:
    """Keep the supplied monthly table lazy and materialize one month at a time."""

    def __init__(self, chars, features):
        self.features = features
        self.table = canonical_chars(chars, features)
        self.metadata = self.table.select(META).collect(
            engine='streaming').sort(['eom', 'id'])
        self.return_to_formation = dict(
            self.metadata.select('eom_ret', 'eom').unique().iter_rows())


def ff12_class(sic):
    groups = [('NoDur',
      [(100, 999), (2000, 2399), (2700, 2749), (2770, 2799), (3100, 3199), (3940, 3989)]),
     ('Durbl',
      [(2500, 2519), (3630, 3659), (3710, 3711), (3714, 3714), (3716, 3716), (3750, 3751),
       (3792, 3792), (3900, 3939), (3990, 3999)]),
     ('Manuf',
      [(2520, 2589), (2600, 2699), (2750, 2769), (3000, 3099), (3200, 3569), (3580, 3629),
       (3700, 3709), (3712, 3713), (3715, 3715), (3717, 3749), (3752, 3791), (3793, 3799),
       (3830, 3839), (3860, 3899)]),
     ('Enrgy', [(1200, 1399), (2900, 2999)]), ('Chems', [(2800, 2829), (2840, 2899)]),
     ('BusEq', [(3570, 3579), (3660, 3692), (3694, 3699), (3810, 3829), (7370, 7379)]),
     ('Telcm', [(4800, 4899)]), ('Utils', [(4900, 4949)]),
     ('Shops', [(5000, 5999), (7200, 7299), (7600, 7699)]),
     ('Hlth', [(2830, 2839), (3693, 3693), (3840, 3859), (8000, 8099)]),
     ('Money', [(6000, 6999)])]
    values = np.asarray(sic, dtype=float)
    valid = np.isfinite(values) & (values == np.floor(values))
    values = np.where(valid, values, -1)
    labels = np.full(len(values), 'Other', dtype='U5')
    assigned = np.zeros(len(values), bool)
    for name, ranges in groups:
        mask = np.zeros(len(values), bool)
        for lower, upper in ranges:
            mask |= (values >= lower) & (values <= upper)
        mask &= ~assigned
        labels[mask] = name
        assigned |= mask
    return labels


class RiskSource(MonthlySource):

    def __init__(self, chars, features):
        super().__init__(chars, features)
        self.sic = self.table.select('id', 'eom', 'sic').collect(
            engine='streaming')
        self.exposures = {}

    def load_exposures(self, d):
        if d in self.exposures:
            return self.exposures[d]
        raw = self.table.filter(pl.col('eom') == d).drop('sic').collect(
            engine='streaming')
        frame = prepare_pred_data(raw, self.features, min_obs=10).join(
            self.sic.filter(pl.col('eom') == d),
            on=['id', 'eom']).sort('id')
        values = frame.select(self.features).to_numpy()
        mean = values.mean(axis=0)
        sd = values.std(axis=0, ddof=1) if len(values) > 1 else np.zeros(values.shape[1])
        active = np.isfinite(sd) & (np.round(sd, 6) != 0)
        standardized = np.zeros_like(values)
        standardized[:, active] = (values[:, active] - mean[active]) / sd[active]
        labels = ff12_class(frame['sic'].to_list())
        industry = np.column_stack([labels == label for label in INDUSTRIES]).astype(float)
        result = (frame, np.column_stack([industry, standardized]))
        self.exposures[d] = result
        if len(self.exposures) > 2:
            del self.exposures[next(iter(self.exposures))]
        return result


class DailySource:

    def __init__(self, daily):
        self.frame = pl.from_pandas(daily).lazy().select(
            pl.col('id').cast(pl.Int64), pl.col('date').cast(pl.Date),
            pl.col('ret_exc').cast(pl.Float64)).filter(pl.col('ret_exc').is_finite())

    def load(self, d):
        start = date(d.year, d.month, 1)
        return self.frame.filter(pl.col('date').is_between(start, d)).sort(
            ['date', 'id']).collect(engine='streaming')


def glmnet_ridge(x, y, penalty):
    """Match Gaussian glmnet alpha=0, standardize=FALSE, intercept=FALSE.

    glmnet scales y by its uncentered RMS for this no-intercept fit, so the
    original-unit diagonal regularizer is n*lambda/RMS(y). Zero-variance columns
    are excluded even when standardize=FALSE, matching glmnet's preprocessing.
    """
    active = np.var(x, axis=0) > 0
    result = np.zeros(x.shape[1])
    rms = float(np.sqrt(np.mean(y * y)))
    if not rms or not active.any():
        return result
    a = x[:, active]
    regularizer = len(y) * penalty / rms
    if a.shape[1] <= a.shape[0]:
        gram = a.T @ a
        gram[np.diag_indices_from(gram)] += regularizer
        result[active] = cho_solve(
            cho_factor(gram, lower=True, check_finite=False),
            a.T @ y, check_finite=False)
    else:
        gram = a @ a.T
        gram[np.diag_indices_from(gram)] += regularizer
        result[active] = a.T @ cho_solve(
            cho_factor(gram, lower=True, check_finite=False),
            y, check_finite=False)
    return result


def weighted_covariance(values, half_life):
    weights = 0.5 ** (np.arange(len(values), 0, -1) / half_life)
    # This covariance is symmetrized after correlation/variance scaling.
    return covariance_with_weights(values, weights, symmetric=False)


def factor_covariance(values, settings=RISK_SETTINGS):
    correlation_covariance = weighted_covariance(
        values, settings.correlation_half_life)
    variance_covariance = weighted_covariance(values, settings.variance_half_life)
    return combine_correlation_variance(correlation_covariance, variance_covariance)


class SpecificRiskState:
    """Update specific volatility using the previous observed residual."""

    def __init__(self, ids, settings):
        self.settings = settings
        n = len(ids)
        self.count = np.zeros(n, dtype=np.int64)
        self.seed_squares = np.zeros(n)
        self.seed_count = np.zeros(n, dtype=np.int64)
        self.variance = np.zeros(n)
        self.previous = np.zeros(n)
        self.history = np.full((n, settings.minimum_residual_observations + 1), -1, dtype=np.int32)
        self.day_index = -1

    def advance(self, positions, residuals):
        self.day_index += 1
        c = self.count[positions]
        s = self.settings
        seeded = c >= s.initial_variance_observations
        updating = c > s.initial_variance_observations
        u = positions[updating & np.isfinite(self.previous[positions])]
        decay = 0.5 ** (1 / s.specific_half_life)
        self.variance[u] = decay * self.variance[u] + (1 - decay) * self.previous[u] ** 2
        vol = np.full(len(positions), np.nan)
        vol[seeded] = np.sqrt(self.variance[positions[seeded]])
        initializing = ~seeded
        valid_initial = initializing & np.isfinite(residuals)
        init = positions[valid_initial]
        self.seed_squares[init] += residuals[valid_initial] ** 2
        self.seed_count[init] += 1
        ending = positions[c == s.initial_variance_observations - 1]
        self.variance[ending] = self.seed_squares[ending] / np.maximum(1, self.seed_count[ending] - 1)
        valid = c >= s.minimum_residual_observations
        past = np.full(len(positions), -1, dtype=np.int32)
        width = s.minimum_residual_observations + 1
        past[valid] = self.history[positions[valid], (c[valid] - s.minimum_residual_observations) % width]
        eligible = (valid & (self.day_index >= s.observation_window)
                    & (past >= self.day_index - s.observation_window)
                    & np.isfinite(vol) & (vol > 0))
        self.history[positions, c % width] = self.day_index
        self.previous[positions] = residuals
        self.count[positions] += 1
        return (vol, eligible)


def portfolio_variance(w, diagonal, factor_cov, x):
    exposure = x.T @ w
    return float(np.sum(w * w * diagonal) + exposure @ factor_cov @ exposure)


# Section 3: Next-month factor-return forecasts


def training_folds(dates, first_test, settings):
    first = month_number(first_test) - settings.train_years * 12
    length = settings.train_years * 12
    return [[d for d in dates
             if min(settings.folds - 1,
                    (month_number(d) - first) * settings.folds
                    // max(1, length - 1)) == i]
            for i in range(settings.folds)]


def initial_ridge_lambdas(existing=FORECAST_SETTINGS.ridge_lambdas):
    """Positive Ridge candidates, including small penalties and integer values."""
    return np.unique(np.r_[np.arange(1, 10001, dtype=float), np.geomspace(1e-08, 1.0, 1000), existing])


def spectral_fit(x, y):
    mx, my = (x.mean(axis=0), y.mean(axis=0))
    u, s, vt = np.linalg.svd(x - mx, full_matrices=False)
    return dict(mx=mx, my=my, V=vt.T, H=s[:, None] * (u.T @ (y - my)), eigen=s * s, n=len(x))


def coefficients(bank, penalty):
    coef = bank['V'] @ (bank['H'] / (bank['eigen'][:, None] + bank['n'] * penalty))
    return (bank['my'] - bank['mx'] @ coef, coef)


def mapped_bank(bank, validation_x, stock_stats):
    """Sum stock SSE quadratics before scoring any lambda, not factor SSE."""
    rank = len(bank['eigen'])
    qsum = np.zeros((rank, rank))
    linear = np.zeros(rank)
    constant = 0.0
    n = 0
    for x, stats in zip(validation_x, stock_stats):
        q = (x - bank['mx']) @ bank['V']
        h = bank['H']
        m = bank['my']
        g, c = (stats['G'], stats['c'])
        qsum += q[:, None] * (h @ g @ h.T) * q[None, :]
        linear += q * (h @ (g @ m - c))
        constant += float(m @ g @ m - 2 * m @ c + stats['s'])
        n += stats['n']
    return dict(n=n, n_train=bank['n'], eigen=bank['eigen'], Q=qsum, b=linear, C=constant)


def score_banks(banks, lambdas, batch_size=256):
    lambdas = np.asarray(lambdas, dtype=float)
    loss = np.zeros(len(lambdas))
    for bank in banks:
        for start in range(0, len(lambdas), batch_size):
            k = 1 / (bank['eigen'][:, None] + bank['n_train'] * lambdas[None, start:start + batch_size])
            sse = np.sum(k * (bank['Q'] @ k), axis=0) + 2 * bank['b'] @ k + bank['C']
            loss[start:start + len(sse)] += np.maximum(sse, 0.0) / bank['n'] / len(banks)
    return loss


def select_path(banks):
    """Search the tested positive Ridge grid, then refine near its first minimum."""
    grid = initial_ridge_lambdas()
    loss = score_banks(banks, grid)
    for _ in range(3):
        j = int(np.argmin(loss))
        if j == 0 and grid[0] > 1e-12:
            extra = np.geomspace(max(1e-12, grid[0] / 100), grid[0], 1000)[:-1]
        elif j == len(grid) - 1:
            extra = np.geomspace(grid[-1], grid[-1] * 100, 1000)[1:]
        else:
            break
        extra_loss = score_banks(banks, extra)
        order = np.argsort(np.r_[grid, extra], kind='stable')
        grid = np.r_[grid, extra][order]
        loss = np.r_[loss, extra_loss][order]
    j = int(np.argmin(loss))
    left, right = grid[max(0, j - 1)], grid[min(len(grid) - 1, j + 1)]
    refined = np.linspace(left, right, 10000)
    g = np.r_[grid, refined]
    losses = np.r_[loss, score_banks(banks, refined)]
    order = np.argsort(g, kind='stable')
    winner = int(order[np.argmin(losses[order])])
    return float(g[winner])


class StockLabels:
    """Retain stock-loss summaries only after their return month completes."""

    def __init__(self):
        self.cache = OrderedDict()

    def append(self, target, frame, exposures):
        y = frame['ret_exc_lead1m'].to_numpy()
        good = np.isfinite(y)
        b, y = exposures[good], y[good]
        self.cache[target] = dict(
            G=b.T @ b, c=b.T @ y, s=float(y @ y), n=len(y))
        while len(self.cache) > 144:
            self.cache.popitem(last=False)

    def statistics(self, target):
        return self.cache[target]


def fit_factor_forecast(targets, first, cutoff, labels):
    lower = month_number(first) - 120
    dates = sorted(d for d in targets
                   if lower <= month_number(d) < month_number(first)
                   and d <= cutoff and month_end(month_number(d) - 1) in targets)
    x = np.array([targets[month_end(month_number(d) - 1)] for d in dates])
    y = np.array([targets[d] for d in dates])
    banks = []
    for block in training_folds(dates, first, FORECAST_SETTINGS):
        val = np.array([d in block for d in dates])
        train = ~val
        if not val.any() or not train.any():
            continue
        stats = [labels.statistics(d) for d in block]
        if not sum(a['n'] for a in stats):
            continue
        banks.append(mapped_bank(spectral_fit(x[train], y[train]), x[val], stats))
    penalty = select_path(banks) if banks else FORECAST_SETTINGS.ridge_lambdas[0]
    return coefficients(spectral_fit(x, y), penalty), penalty


# Section 4: Expanding-history factor variance


def covariance_with_weights(values, weights, *, symmetric=True):
    """Unbiased weighted covariance, with optional numerical symmetrization."""
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    centered = values - weights @ values
    covariance = centered.T * weights @ centered / (1.0 - weights @ weights)
    return (covariance + covariance.T) / 2 if symmetric else covariance


def decay_factor_covariance(values, ages, settings, half_weight=None):
    """Combine exponential correlations with the selected variance age weights."""
    values = np.asarray(values, dtype=float)
    ages = np.asarray(ages, dtype=float)
    covariances = {}
    for name, original in (('correlation', settings.correlation_half_life),
                           ('variance', settings.variance_half_life)):
        weights = (1.0 / (1.0 + ages / half_weight)
                   if name == 'variance' and half_weight is not None
                   else 0.5 ** (ages / original))
        covariances[name] = covariance_with_weights(values, weights)
    return combine_correlation_variance(
        covariances['correlation'], covariances['variance'])


def capped_calendar_blocks(dates, *, minimum_blocks=5, maximum_months=96):
    """Split completed history into calendar blocks, retaining gaps."""
    dates = sorted(set(dates))
    if not dates:
        return [[] for _ in range(minimum_blocks)]
    start, end = month_number(dates[0]), month_number(dates[-1]) + 1
    span = end - start
    count = max(minimum_blocks, ceil(span / maximum_months))
    edges = [start + i * span // count for i in range(count + 1)]
    return [[d for d in dates if left <= month_number(d) < right]
            for left, right in zip(edges, edges[1:])]


def combine_correlation_variance(correlation_covariance, variance_covariance):
    sd = np.sqrt(np.maximum(np.diag(correlation_covariance), 0.0))
    outer = np.outer(sd, sd)
    correlation = np.divide(correlation_covariance, outer, out=np.zeros_like(outer), where=outer > 0)
    var_sd = np.sqrt(np.maximum(np.diag(variance_covariance), 0.0))
    result = correlation * np.outer(var_sd, var_sd)
    return (result + result.T) / 2


class ExpandingEW:
    """Merge weighted centered moments once per newly completed factor block."""

    def __init__(self, width, half_life):
        self.half_life = half_life
        self.weight = self.squared_weight = 0.0
        self.mean = np.zeros(width)
        self.scatter = np.zeros((width, width))

    def append(self, values):
        values = np.asarray(values, dtype=float)
        if not len(values):
            return
        weights = 0.5 ** (np.arange(len(values), 0, -1) / self.half_life)
        block_weight = float(weights.sum())
        block_mean = weights @ values / block_weight
        centered = values - block_mean
        block_scatter = centered.T * weights @ centered
        decay = 0.5 ** (len(values) / self.half_life)
        old_weight = self.weight * decay
        total = old_weight + block_weight
        delta = block_mean - self.mean
        self.scatter = (self.scatter * decay + block_scatter
                        + np.outer(delta, delta)
                        * (old_weight * block_weight / total))
        self.mean = self.mean + delta * (block_weight / total)
        self.weight = total
        self.squared_weight = self.squared_weight * decay ** 2 + float(weights @ weights)

    def covariance(self):
        denominator = self.weight - self.squared_weight / self.weight if self.weight else 0.0
        result = self.scatter / denominator
        return (result + result.T) / 2


def select_expanding_decay(values, dates, settings=RISK_SETTINGS):
    """Select variance half-weight age using completed historical covariance blocks."""
    values = np.asarray(values, dtype=float)
    dates = list(dates)
    candidates = [float(settings.variance_half_life * .5),
                  float(settings.variance_half_life),
                  float(settings.variance_half_life * 2.)]
    months = sorted({date(d.year, d.month, monthrange(d.year, d.month)[1]) for d in dates})
    blocks = capped_calendar_blocks(months)
    row_month = [date(d.year, d.month, monthrange(d.year, d.month)[1]) for d in dates]
    ages = np.arange(len(values), 0, -1, dtype=float)
    folds = []
    for block in blocks:
        held = set(block)
        validation = np.array([m in held for m in row_month])
        training = ~validation
        if validation.sum() < 2 or training.sum() < 2:
            continue
        v = values[validation]
        centered = v - v.mean(axis=0)
        target = centered.T @ centered / (len(v) - 1)
        folds.append((training, target))
    if len(folds) < 2:
        return None
    losses = []
    for candidate in candidates:
        scores = []
        for training, target in folds:
            fitted = decay_factor_covariance(values[training], ages[training],
                                              settings, candidate)
            scores.append(float(np.mean((fitted - target) ** 2)))
        losses.append(float(np.mean(scores)))
    winner = min(range(len(candidates)), key=lambda i: (losses[i], candidates[i]))
    return candidates[winner]


# Section 5: Regularized Markowitz allocation


def factor_loading(covariance, exposures):
    """Prepare B sqrt(F), clipping only roundoff-sized negative eigenvalues."""
    eigen, vectors = np.linalg.eigh(covariance)
    scale = max(float(np.max(np.abs(eigen))), 1e-20)
    if eigen.min() < -1e-09 * scale:
        raise ValueError('Factor covariance is materially indefinite')
    return exposures @ (vectors * np.sqrt(np.maximum(eigen, 0)))


def solve_penalized(diagonal, loading, rhs, gamma=0.0):
    """Solve (D + L L' + gamma I) for one or several right-hand sides."""
    vector = np.ndim(rhs) == 1
    b = np.asarray(rhs)[:, None] if vector else np.asarray(rhs)
    variance = diagonal + gamma
    inverse_b = b / variance[:, None]
    inverse_loading = loading / variance[:, None]
    small = np.eye(loading.shape[1]) + loading.T @ inverse_loading
    correction = cho_solve(
        cho_factor(small, lower=True, check_finite=False),
        loading.T @ inverse_b, check_finite=False)
    result = inverse_b - inverse_loading @ correction
    return result[:, 0] if vector else result


def select_penalty(history, cutoff, q_grid=Q_GRID):
    """Maximize mean historical block Sharpe using released candidate outcomes."""
    available = [row for row in history
                 if row['eom_ret'] <= cutoff and row['markowitz'] is not None][-120:]
    blocks = np.array_split(np.arange(len(available)), 5)
    usable = len(available) >= 10 and all(len(block) >= 2 for block in blocks)
    curve = []
    for index, q in enumerate(q_grid):
        scores = []
        if usable:
            for block in blocks:
                values = np.array([available[int(i)]['markowitz'][index] for i in block])
                variance = float(np.var(values, ddof=1))
                scores.append(float(np.mean(values) * np.sqrt(12 / variance))
                              if variance > 0 else None)
        score = (float(np.mean(scores)) if scores
                 and all(v is not None and np.isfinite(v) for v in scores) else None)
        curve.append(score)
    valid = [i for i, score in enumerate(curve) if score is not None]
    return min(valid, key=lambda i: (-curve[i], i)) if valid else 0


def candidate_directions(mu, b, d, f, reference_f, settings, q_grid):
    """Use benchmark stock variances for penalty units and volatility scaling."""
    loading = factor_loading(f, b)
    scale = float(np.median(d + np.einsum('ij,jk,ik->i', b, reference_f, b)))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('Positive stock variance median required')
    weights = []
    for q in q_grid:
        direction = solve_penalized(d, loading, mu, float(q) * scale)
        variance = portfolio_variance(direction, d, reference_f, b)
        if not np.isfinite(variance) or variance <= 0:
            raise ValueError('No finite nonzero allocation direction')
        weights.append(direction * settings.target_annual_volatility / np.sqrt(252 * variance))
    return np.array(weights)


# Section 6: Main entry point

def main(chars: pd.DataFrame, features: pd.DataFrame,
         daily_ret: pd.DataFrame) -> pd.DataFrame:
    """Fit the selected strategy from the supplied data and return monthly weights."""
    started = perf_counter()
    names = canonical_features(features)
    source = RiskSource(chars, names)
    metadata = source.metadata
    tests = sorted(metadata.filter(pl.col('ctff_test'))['eom_ret'].unique().to_list())
    if not tests:
        raise ValueError('No test dates in the supplied data')
    test_formations = set(source.return_to_formation[d] for d in tests)
    months = sorted(d for d in metadata['eom'].unique().to_list()
                    if d <= max(test_formations))
    refit_dates = {source.return_to_formation[tests[i]]: tests[i]
                  for i in range(0, len(tests), 12)}
    daily = DailySource(daily_ret)
    all_ids = np.array(sorted(metadata['id'].unique().to_list()))
    state = SpecificRiskState(all_ids, RISK_SETTINGS)
    bounded_factors = deque(maxlen=RISK_SETTINGS.covariance_days)
    full_factors, factor_dates, targets = [], [], {}
    corr = ExpandingEW(len(names) + 12, RISK_SETTINGS.correlation_half_life)
    variance = ExpandingEW(len(names) + 12, RISK_SETTINGS.variance_half_life)
    risk_model = previous = forecast = None
    risk_selections, penalty_selections = {}, {}
    history, pending, output = [], [], []

    print(f'Estimating daily factors and risk over {len(months)} months; '
          f'numerical thread cap {RISK_SETTINGS.threads}', flush=True)
    with threadpool_limits(limits=RISK_SETTINGS.threads):
        labels = StockLabels()
        for month_count, d in enumerate(months, 1):
            specific, factors, new_dates = {}, [], []
            if previous is not None and previous[0]['eom_ret'][0] == d:
                old_frame, old_b = previous
                labels.append(d, old_frame, old_b)
                old_ids = old_frame['id'].to_numpy()
                observed = daily.load(d).filter(pl.col('id').is_in(old_ids.tolist()))
                for group in observed.partition_by('date', maintain_order=True):
                    ids = group['id'].to_numpy()
                    x = old_b[np.searchsorted(old_ids, ids)]
                    y = group['ret_exc'].to_numpy()
                    if len(y) < 2:
                        continue
                    coef = glmnet_ridge(x, y, RISK_SETTINGS.ridge_lambda)
                    residual = y - x @ coef
                    factors.append(coef)
                    new_dates.append(group['date'][0])
                    bounded_factors.append(coef)
                    vol, eligible = state.advance(np.searchsorted(all_ids, ids), residual)
                    for security, value in zip(ids[eligible], vol[eligible]):
                        specific[int(security)] = float(value)

            frame, b = source.load_exposures(d)
            ids = frame['id'].to_numpy()
            actual = np.array([specific.get(int(i), np.nan) for i in ids])
            available = np.isfinite(actual) & (actual > 0)
            if available.sum() >= 2:
                risk_model = glmnet_ridge(b[available], np.log(actual[available]),
                                          RISK_SETTINGS.ridge_lambda)
            previous = frame, b
            if factors:
                # Column-contiguous storage fixes the daily-to-monthly summation order.
                block = np.array(factors, order='F')
                targets[d] = block.sum(axis=0)
                full_factors.extend(block)
                factor_dates.extend(new_dates)
                corr.append(block)
                variance.append(block)
            if month_count % 120 == 0:
                print(f'Daily factors through {d}: {month_count}/{len(months)} '
                      'months completed', flush=True)
            if d not in test_formations:
                continue
            if risk_model is None or not factors:
                raise ValueError(f'Insufficient completed daily history at {d}')

            # Release candidate payoffs only after their return month completes.
            unreleased = []
            for return_date, formation, bank_ids, bank_weights in pending:
                if return_date > d:
                    unreleased.append((return_date, formation, bank_ids, bank_weights))
                    continue
                bank_frame = metadata.filter(pl.col('eom') == formation).sort('id')
                realized = bank_frame.filter(pl.col('id').is_in(bank_ids.tolist()))
                y = realized['ret_exc_lead1m'].to_numpy()
                payoffs = bank_weights @ y if np.isfinite(y).all() else None
                history.append(dict(eom_ret=return_date, markowitz=payoffs))
            pending = unreleased

            if d in refit_dates:
                first = refit_dates[d]
                lower = month_end(month_number(d) - 120)
                completed = {m: v for m, v in targets.items() if lower <= m <= d}
                forecast, penalty = fit_factor_forecast(completed, first, d, labels)
                print(f'Factor forecast refit for {first}: lambda {penalty:g}', flush=True)
            intercept, coef = forecast
            selection = frame['ctff_test'].to_numpy()
            test_ids, test_b = ids[selection], b[selection]
            mu = test_b @ (intercept + targets[d] @ coef)
            vol = np.exp(b @ risk_model)
            vol[available] = actual[available]
            diagonal = vol[selection] ** 2
            reference_f = factor_covariance(np.array(bounded_factors), RISK_SETTINGS)

            # Expanded factor risk sets allocation; benchmark risk sets penalty units and scale.
            year = d.year + int(d.month == 12)
            full = np.array(full_factors)
            if year not in risk_selections:
                risk_selections[year] = select_expanding_decay(
                    full, factor_dates, RISK_SETTINGS)
            half_weight = risk_selections[year]
            changed_f = (combine_correlation_variance(corr.covariance(), variance.covariance())
                         if half_weight is None else decay_factor_covariance(
                             full, np.arange(len(full), 0, -1), RISK_SETTINGS,
                             half_weight))
            return_date = frame['eom_ret'][0]
            annual_anchor = date(return_date.year, 1, 31)
            if annual_anchor not in penalty_selections:
                penalty_selections[annual_anchor] = select_penalty(
                    history, d, Q_GRID)
            all_weights = candidate_directions(
                mu, test_b, diagonal, changed_f, reference_f, RISK_SETTINGS, Q_GRID)
            if d in refit_dates:
                q = Q_GRID[penalty_selections[annual_anchor]]
                print(f'Risk and allocation for {return_date.year}: '
                      f'variance age {half_weight}, position penalty q={q:g}', flush=True)
            w = all_weights[penalty_selections[annual_anchor]]
            output.append(pl.DataFrame(dict(id=test_ids, eom=[d] * len(w), w=w)))
            pending.append((return_date, d, test_ids, all_weights))

    weights = pl.concat(output).sort('eom', 'id').select(
        pl.col('id').cast(pl.Int64), pl.col('eom').cast(pl.Date), pl.col('w').cast(pl.Float64))
    expected = metadata.filter(pl.col('ctff_test')).select('id', 'eom').sort('eom', 'id')
    if not weights.select('id', 'eom').equals(expected) or not weights['w'].is_finite().all():
        raise ValueError('Output must cover every test key with finite weights')
    result = weights.to_pandas()
    result['eom'] = result['eom'].dt.date
    print(f'Output: {len(result):,} weights; runtime {(perf_counter()-started)/60:.1f} minutes',
          flush=True)
    return result


# Section 7: Local example

if __name__ == "__main__":
    # A small reproducible example with synthetic inputs. The contest calls main.
    rng = np.random.default_rng(1)
    example_ids = np.arange(1, 25)
    example_features = pd.DataFrame({'features': ['size', 'value', 'momentum']})
    formation_dates = pd.date_range('1980-01-31', periods=123, freq='ME')
    example_chars = pd.DataFrame({
        'id': np.tile(example_ids, len(formation_dates)),
        'eom': np.repeat(formation_dates, len(example_ids)),
        'excntry': 'USA',
        'sic': 2000,
    })
    example_chars['eom_ret'] = example_chars['eom'] + pd.offsets.MonthEnd(1)
    example_chars['ctff_test'] = example_chars['eom'] >= formation_dates[120]
    example_chars['ret_exc_lead1m'] = rng.normal(0.005, 0.08, len(example_chars))
    for name in example_features['features']:
        example_chars[name] = rng.normal(size=len(example_chars))
    trading_dates = pd.bdate_range(formation_dates[0] + pd.Timedelta(days=1),
                                  formation_dates[-1])
    example_daily = pd.DataFrame({
        'id': np.tile(example_ids, len(trading_dates)),
        'date': np.repeat(trading_dates, len(example_ids)),
        'ret_exc': rng.normal(0.0002, 0.015,
                              len(trading_dates) * len(example_ids)),
    })
    example_weights = main(example_chars, example_features, example_daily)
    print(example_weights.head().to_string(index=False))
