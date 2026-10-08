"""Independent allocation variants using frozen forecasts and Barra risk."""
from calendar import monthrange
from datetime import date
from hashlib import sha256

import numpy as np
import polars as pl
from scipy.linalg import cho_factor
from scipy.optimize import minimize
from threadpoolctl import threadpool_limits

from baseline import as_lazy
from comparison_models import RISK_SETTINGS, INDUSTRIES, portfolio_variance
from batch_allocations import (_months, _risk, _forecasts, _frame, _scaled,
                               factor_loading, solve_penalized)


# C03 changes the original allocation's budget, not its forecasts or risk fit.
def time_weights(weights, market_proxy, artifact_dir, *, settings=RISK_SETTINGS):
    """Multiply each frozen parent month by a predeclared 1 or .5 budget.

    market_proxy has date/ret, the cached equal-weight daily regression-universe
    excess-return proxy. Require twelve consecutive completed calendar months;
    their compounded proxy below zero halves exposure, otherwise retain it.
    This is a fixed research heuristic, not a calibrated market forecast.
    """
    weights = weights.select('id', 'eom', 'w').sort(['eom', 'id'])
    if not weights.height or not 0 < settings.target_annual_volatility:
        raise ValueError('Nonempty parent weights and positive risk budget required')
    daily = (as_lazy(market_proxy).select(pl.col('date').cast(pl.Date),
                pl.col('ret').cast(pl.Float64)).filter(pl.col('date') <= weights['eom'].max())
                .collect(engine='streaming').sort('date'))
    if daily['date'].n_unique() != daily.height or daily['date'].null_count():
        raise ValueError('Market proxy dates must be unique and present')
    if daily['ret'].null_count() or not daily['ret'].is_finite().all() or (daily['ret'] <= -1).any():
        raise ValueError('Market proxy needs finite returns greater than minus one')
    monthly = daily.with_columns(pl.col('date').dt.month_end().alias('eom')).group_by('eom').agg(
        ((pl.col('ret') + 1).product() - 1).alias('ret')).sort('eom')
    lookup = dict(zip(monthly['eom'].to_list(), monthly['ret'].to_list()))
    frames, records = [], []
    for frame in weights.partition_by('eom', maintain_order=True):
        d = frame['eom'][0]
        if d.day != monthrange(d.year, d.month)[1]:
            raise ValueError('C03 requires month-end portfolio formation')
        number = d.year * 12 + d.month - 1
        expected = [date(i // 12, i % 12 + 1, monthrange(i // 12, i % 12 + 1)[1])
                    for i in range(number - 11, number + 1)]
        complete = all(x in lookup for x in expected)
        signal = float(np.prod([1 + lookup[x] for x in expected]) - 1) if complete else None
        multiplier = .5 if signal is not None and signal < 0 else 1.
        ids, exposures, diagonal, covariance = _risk(artifact_dir, d)
        if not np.array_equal(frame['id'].to_numpy(), ids):
            raise ValueError('Timing parent keys differ from original risk universe')
        parent_vol = float(np.sqrt(252 * portfolio_variance(frame['w'].to_numpy(), diagonal, covariance, exposures)))
        if not np.isclose(parent_vol, settings.target_annual_volatility, rtol=1e-8, atol=1e-10):
            raise ValueError('C03 parent does not have the original fixed volatility budget')
        frames.append(frame.with_columns((pl.col('w') * multiplier).alias('w')))
        records.append(dict(formation_date=str(d), multiplier=multiplier,
            signal='last twelve completed monthly compounded excess-proxy returns',
            signal_return=signal, signal_first=str(expected[0]) if complete else None,
            signal_last=str(d) if complete else None,
            signal_information_cutoff=str(d),
            fallback=None if complete else 'fewer than twelve consecutive observed proxy months; multiplier=1',
            annual_volatility_budget=settings.target_annual_volatility * multiplier,
            actual_predicted_annual_volatility=parent_vol * multiplier,
            base_weight_sha256=sha256(frame.write_csv().encode()).hexdigest()))
    return pl.concat(frames), records


# C04-C06 each replace the complete allocation rule, with common sleeve units.
def industry_sleeves(mu, exposures, diagonal, covariance, *, settings=RISK_SETTINGS):
    """Net-zero local Markowitz directions, each with 10% original-model risk.

    Absent, one-stock and zero-direction industries have an inactive zero sleeve.
    All original stock rows are kept. All inactive is an explicit scaling error.
    One factor root is prepared per month, and all risk remains factor algebra.
    """
    mu = np.asarray(mu, dtype=float)
    h = exposures[:, :12]
    if h.shape != (len(mu), 12) or not np.isin(h, [0., 1.]).all() or not np.all(h.sum(axis=1) == 1):
        raise ValueError('Industry sleeves require twelve complete FF12 indicators')
    if not np.isfinite(mu).all():
        raise ValueError('Sleeve forecasts must be finite')
    loading = factor_loading(diagonal, covariance, exposures)
    sleeves, active, diagnostics = [], [], []
    for j, name in enumerate(INDUSTRIES):
        members = np.flatnonzero(h[:, j])
        reason = 'absent' if not len(members) else 'one stock' if len(members) == 1 else None
        direction = None
        if reason is None:
            inverse = solve_penalized(diagonal[members], loading[members],
                                      np.column_stack([mu[members], np.ones(len(members))]))
            direction = inverse[:, 0] - inverse[:, 1] * inverse[:, 0].sum() / inverse[:, 1].sum()
            if np.linalg.norm(direction) <= 1e-10 * np.linalg.norm(inverse[:, 0]):
                reason = 'zero constrained direction'
        if reason is None:
            local, vol = _scaled(direction, diagonal[members], covariance, exposures[members], settings)
            column = np.zeros(len(mu)); column[members] = local
            sleeves.append(column); active.append(j)
            diagnostics.append(dict(industry=name, stocks=len(members), active=True, reason=None,
                                    standalone_original_risk=vol, net=float(local.sum())))
        else:
            diagnostics.append(dict(industry=name, stocks=len(members), active=False, reason=reason,
                                    standalone_original_risk=0., net=0.))
    if not sleeves:
        raise ValueError('All industry sleeves inactive; no positive-risk allocation direction')
    u = np.column_stack(sleeves)
    factor_exposure = exposures.T @ u
    sleeve_cov = u.T @ (diagonal[:, None] * u) + factor_exposure.T @ covariance @ factor_exposure
    return u, u.T @ mu, (sleeve_cov + sleeve_cov.T) / 2, dict(
        sleeve_industries=[INDUSTRIES[j] for j in active], sleeve_diagnostics=diagnostics,
        sleeve_unit='configured standalone annual risk under original covariance',
        maximum_sleeve_industry_constraint=float(np.max(np.abs(h.T @ u))))


def sleeve_coefficients(mean, covariance, mode):
    """Nonnegative equal/simplex-minimum/cone-tangency sleeve coefficients.

    Numerical scale factors change neither optimum direction nor constraints.
    No ridge, diagonal jitter, model selection or future returns are introduced.
    """
    if mode not in ('equal', 'minimum', 'markowitz'):
        raise ValueError('Sleeve mode must be equal, minimum or markowitz')
    count = len(mean)
    cho_factor(covariance)  # Original positive specific risk makes active sleeves PD.
    initial = np.full(count, 1 / count)
    if mode == 'equal':
        return initial, dict(solver='equal coefficients', iterations=0, optimality_error=0.)
    scaled = covariance / np.mean(np.diag(covariance))
    if mode == 'minimum':
        objective = lambda a: .5 * a @ scaled @ a
        gradient = lambda a: scaled @ a
        constraints = dict(type='eq', fun=lambda a: a.sum() - 1,
                           jac=lambda a: np.ones(count))
    else:
        if not np.isfinite(mean).all() or np.max(np.abs(mean)) == 0:
            raise ValueError('Sleeve means have no finite tangency direction')
        normalized_mean = mean / np.max(np.abs(mean))
        objective = lambda a: .5 * a @ scaled @ a - normalized_mean @ a
        gradient = lambda a: scaled @ a - normalized_mean
        constraints = ()
    result = minimize(objective, initial, jac=gradient, method='SLSQP',
                      bounds=[(0., None)] * count, constraints=constraints,
                      options=dict(ftol=1e-12, maxiter=2000))
    raw = np.maximum(result.x, 0.)
    if not result.success or not np.isfinite(raw).all() or raw.sum() <= 0:
        raise ValueError(f'Sleeve allocation failed: {result.message}')
    grad = gradient(raw)
    active = raw > 1e-7
    lagrange = float(np.mean(grad[active])) if mode == 'minimum' else 0.
    error = max(float(np.max(np.abs(grad[active] - lagrange))),
                float(np.max(np.maximum(lagrange - grad[~active], 0.))) if (~active).any() else 0.)
    if error > 2e-6:
        raise ValueError('Sleeve optimizer failed its predeclared scaled KKT tolerance')
    return raw / raw.sum(), dict(solver='nonnegative SLSQP', iterations=int(result.nit),
                                optimality_error=error, ftol=1e-12, kkt_tolerance=2e-6)


def allocate_sleeves(artifact_dir, predictions, *, modes=('equal', 'minimum', 'markowitz'), settings=RISK_SETTINGS):
    """Return {mode: stock weights}, {mode: monthly records}; one sleeve pass.

    Primary input is original XGBoost forecasts. Each output is independently
    compared to original Markowitz ML, not a cumulative experiment combination.
    """
    if not modes or len(set(modes)) != len(modes):
        raise ValueError('Distinct sleeve modes required')
    root, months = _months(artifact_dir)
    weights, records = {m: [] for m in modes}, {m: [] for m in modes}
    with threadpool_limits(limits=settings.threads):
        for row in months:
            d = date.fromisoformat(row['formation_date'])
            ids, exposures, diagonal, covariance = _risk(root, d)
            mu = _forecasts(predictions, ids, d)
            u, means, sleeve_cov, sleeve_record = industry_sleeves(mu, exposures, diagonal, covariance, settings=settings)
            for mode in modes:
                coefficients, solver = sleeve_coefficients(means, sleeve_cov, mode)
                direction = u @ coefficients
                w, vol = _scaled(direction, diagonal, covariance, exposures, settings)
                weights[mode].append(_frame(ids, d, w))
                records[mode].append(dict(sleeve_record, formation_date=str(d),
                    allocation_mode=mode, sleeve_coefficients=coefficients.tolist(), outer_nonnegative=True,
                    original_predicted_annual_volatility=vol,
                    maximum_absolute_constraint=float(np.max(np.abs(exposures[:, :12].T @ w))),
                    net=float(w.sum()), gross=float(np.abs(w).sum()), solver=solver,
                    minimum_variance_scope='sleeve budget, not native stock net-one MVP'))
    return ({m: pl.concat(frames).sort(['eom', 'id']) for m, frames in weights.items()}, records)
