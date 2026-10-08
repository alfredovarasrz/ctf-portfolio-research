"""C01/C02 allocation changes using unchanged, saved Barra covariance estimates."""
from datetime import date
import json
from pathlib import Path

import numpy as np
import polars as pl
from scipy.linalg import cho_factor, cho_solve
from threadpoolctl import threadpool_limits

from baseline import as_lazy
from comparison_models import RISK_SETTINGS, portfolio_variance

Q_GRID = (0.,) + tuple(float(q) for q in np.logspace(-5, 2, 15))


# Section 2: Shared artifact and covariance utilities ---------------------------
def _months(artifact_dir):
    root = Path(artifact_dir)
    with np.load(root / 'checkpoint.npz', allow_pickle=False) as saved:
        records = json.loads(str(saved['metadata']))['records']
    if not records:
        raise ValueError('No completed test-month risk artifacts')
    return root, sorted(records, key=lambda row: row['formation_date'])


def _risk(root, d):
    directory = root / 'months' / str(d)
    if not (directory / 'complete.json').exists():
        raise ValueError(f'Incomplete risk artifact month {d}')
    with np.load(directory / 'risk.npz', allow_pickle=False) as saved:
        return saved['ids'], saved['B'], saved['D'], saved['F']


def _forecasts(predictions, ids, d):
    keys = pl.DataFrame({'id': ids, 'eom': [d] * len(ids)})
    matched = keys.join(predictions.filter(pl.col('eom') == d).select('id', 'eom', 'pred'),
                        on=['id', 'eom'], how='left', validate='1:1').sort('id')
    if matched['pred'].null_count() or not matched['pred'].is_finite().all():
        raise ValueError('Expected returns missing or invalid for risk-model securities')
    return matched['pred'].to_numpy()


def factor_loading(diagonal, covariance, exposures):
    """Prepare B sqrt(F) once per month, using the baseline's PSD convention."""
    if np.any(~np.isfinite(diagonal)) or np.any(diagonal <= 0):
        raise ValueError('Specific variances must be finite and strictly positive')
    eigen, vectors = np.linalg.eigh(covariance)
    scale = max(float(np.max(np.abs(eigen))), 1e-20)
    if eigen.min() < -1e-9 * scale:
        raise ValueError('Factor covariance is materially indefinite')
    return exposures @ (vectors * np.sqrt(np.maximum(eigen, 0)))


def solve_penalized(diagonal, loading, rhs, gamma=0.):
    """Solve (D + L L' + gamma I) for one or several right-hand sides."""
    if not np.isfinite(gamma) or gamma < 0:
        raise ValueError('Allocation penalty must be finite and nonnegative')
    vector = np.ndim(rhs) == 1
    b = np.asarray(rhs)[:, None] if vector else np.asarray(rhs)
    variance = diagonal + gamma
    inverse_b = b / variance[:, None]
    inverse_loading = loading / variance[:, None]
    small = np.eye(loading.shape[1]) + loading.T @ inverse_loading
    correction = cho_solve(cho_factor(small, lower=True, check_finite=False),
                           loading.T @ inverse_b, check_finite=False)
    result = inverse_b - inverse_loading @ correction
    return result[:, 0] if vector else result


def _scaled(direction, diagonal, covariance, exposures, settings):
    variance = portfolio_variance(direction, diagonal, covariance, exposures)
    if not np.isfinite(variance) or variance <= 0:
        raise ValueError('Markowitz direction has zero or invalid variance; cannot scale to target volatility')
    weights = direction * settings.target_annual_volatility / np.sqrt(252 * variance)
    return weights, float(np.sqrt(252 * portfolio_variance(weights, diagonal, covariance, exposures)))


def _frame(ids, d, weights):
    if not np.isfinite(weights).all():
        raise ValueError('Nonfinite portfolio weights')
    return pl.DataFrame({'id': ids, 'eom': [d] * len(ids), 'w': weights})


# Section 3: C01 direct exposure constraints ----------------------------------
def neutral_weights(mu, exposures, diagonal, covariance, *, mode='industry', settings=RISK_SETTINGS):
    """Constrain H'w=0, then use the unchanged covariance to target volatility."""
    if mode == 'industry':
        h = exposures[:, :12]
        if not np.all((h == 0) | (h == 1)) or not np.all(h.sum(axis=1) == 1):
            raise ValueError('C01 requires the original complete FF12 industry indicators')
        h = h[:, h.any(axis=0)]
    elif mode == 'net':
        h = np.ones((len(mu), 1))
    else:
        raise ValueError('Neutrality mode must be industry or net')
    loading = factor_loading(diagonal, covariance, exposures)
    inverse = solve_penalized(diagonal, loading, np.column_stack([mu, h]))
    raw, inverse_h = inverse[:, 0], inverse[:, 1:]
    multipliers = cho_solve(cho_factor(h.T @ inverse_h, lower=True, check_finite=False),
                            h.T @ raw, check_finite=False)
    direction = raw - inverse_h @ multipliers
    # Homogeneous constraints can eliminate the entire forecast direction.
    if np.linalg.norm(direction) <= 1e-10 * np.linalg.norm(raw):
        raise ValueError('Constrained Markowitz direction has zero or invalid variance')
    weights, volatility = _scaled(direction, diagonal, covariance, exposures, settings)
    return weights, dict(mode=mode, maximum_absolute_constraint=float(np.max(np.abs(h.T @ weights))),
                         net=float(weights.sum()), gross=float(np.abs(weights).sum()),
                         predicted_annual_volatility=volatility,
                         minimum_variance_applicability='native net-one MVP not applicable')


def allocate_neutral(artifact_dir, predictions, *, mode='industry', settings=RISK_SETTINGS):
    """Return monthly neutral Markowitz weights and constraint/risk diagnostics."""
    root, months = _months(artifact_dir)
    frames, records = [], []
    with threadpool_limits(limits=settings.threads):
        for row in months:
            d = date.fromisoformat(row['formation_date'])
            ids, exposures, diagonal, covariance = _risk(root, d)
            mu = _forecasts(predictions, ids, d)
            weights, record = neutral_weights(mu, exposures, diagonal, covariance,
                                              mode=mode, settings=settings)
            frames.append(_frame(ids, d, weights))
            records.append(dict(formation_date=str(d), **record))
    return pl.concat(frames).sort(['eom', 'id']), records


# Section 4: C02 causal historical penalty selection ---------------------------
def penalty_candidates(mu, exposures, diagonal, covariance, *, q_grid=Q_GRID,
                       settings=RISK_SETTINGS):
    """Compute all candidate allocations with one factor-root preparation."""
    loading = factor_loading(diagonal, covariance, exposures)
    risk_scale = float(np.median(diagonal + np.sum(loading * loading, axis=1)))
    minimum, markowitz, volatilities = [], [], []
    rhs = np.column_stack([np.ones(len(mu)), mu])
    for q in q_grid:
        inverse = solve_penalized(diagonal, loading, rhs, float(q) * risk_scale)
        denominator = float(inverse[:, 0].sum())
        if not np.isfinite(denominator) or denominator <= 0:
            raise ValueError('Invalid minimum-variance normalization')
        minimum.append(inverse[:, 0] / denominator)
        weights, volatility = _scaled(inverse[:, 1], diagonal, covariance, exposures, settings)
        markowitz.append(weights)
        volatilities.append(volatility)
    return np.array(minimum), np.array(markowitz), risk_scale, volatilities


def select_penalty(history, cutoff, q_grid=Q_GRID, *, portfolio):
    """Select from at most 120 completed, previously generated monthly outcomes.

    This does not train a forecasting model or replace its benchmark blocked CV.
    Five historical score blocks contain at least two completed months each.
    """
    if portfolio not in ('minimum', 'markowitz'):
        raise ValueError('Penalty portfolio must be minimum or markowitz')
    available = [row for row in history if row['eom_ret'] <= cutoff
                 and row[portfolio] is not None][-120:]
    blocks = np.array_split(np.arange(len(available)), 5)
    usable = len(available) >= 10 and all(len(block) >= 2 for block in blocks)
    curve = []
    for index, q in enumerate(q_grid):
        scores = []
        if usable:
            for block in blocks:
                values = np.array([available[int(i)][portfolio][index] for i in block])
                variance = float(np.var(values, ddof=1))
                if portfolio == 'minimum':
                    scores.append(variance)
                elif variance > 0:
                    scores.append(float(np.mean(values) * np.sqrt(12 / variance)))
                else:
                    scores.append(None)
        score = float(np.mean(scores)) if scores and all(v is not None and np.isfinite(v)
                                                       for v in scores) else None
        curve.append(dict(q=float(q), score=score, block_scores=scores))
    valid = [i for i, row in enumerate(curve) if row['score'] is not None]
    if valid:
        sign = 1 if portfolio == 'minimum' else -1
        chosen = min(valid, key=lambda i: (sign * curve[i]['score'], i))
        fallback = None
    else:
        chosen = 0
        fallback = 'predeclared q=0; no five usable historical score blocks'
    record = dict(portfolio=portfolio, selected_q=float(q_grid[chosen]),
                  criterion='mean of five sample-variance blocks' if portfolio == 'minimum'
                  else 'mean of five annualized Sharpe blocks',
                  selection_formation=str(cutoff), history_months=len(available),
                  history_return_first=str(available[0]['eom_ret']) if available else None,
                  history_return_last=str(available[-1]['eom_ret']) if available else None,
                  history_available_through=str(cutoff), fallback=fallback,
                  candidate_validation_curve=curve)
    return chosen, record


def _realized_labels(metadata, ids, d):
    keys = pl.DataFrame({'id': ids, 'eom': [d] * len(ids)})
    matched = keys.join(metadata.filter(pl.col('eom') == d), on=['id', 'eom'],
                        how='left', validate='1:1').sort('id')
    if matched['eom_ret'].null_count() or matched['eom_ret'].n_unique() != 1:
        raise ValueError('Missing or inconsistent candidate-bank return dates')
    realized = matched['ret_exc_lead1m'].to_numpy()
    complete = np.isfinite(realized).all()
    return matched['eom_ret'][0], realized if complete else None


def allocate_penalized(artifact_dir, predictions, metadata, *, settings=RISK_SETTINGS, q_grid=Q_GRID):
    """Return MVP/MML weights, selection records, and an as-of candidate-return bank.

    Selection occurs before this month's future labels are accessed. Candidate
    labels are released to later selection only at their eom_ret, with no use
    of the current or future month's outcome. Missing labels remain unavailable.
    """
    q_grid = tuple(float(q) for q in q_grid)
    if not q_grid or q_grid[0] != 0 or any(not np.isfinite(q) or q < 0 for q in q_grid):
        raise ValueError('Penalty grid must start with zero and contain finite nonnegative values')
    root, months = _months(artifact_dir)
    metadata = as_lazy(metadata).select('id', 'eom', 'eom_ret', 'ret_exc_lead1m').collect(engine='streaming')
    history, bank_rows, records, minimum_frames, markowitz_frames = [], [], [], [], []
    chosen_minimum = chosen_markowitz = 0
    minimum_selection = markowitz_selection = None
    with threadpool_limits(limits=settings.threads):
        for offset, row in enumerate(months):
            d = date.fromisoformat(row['formation_date'])
            if offset % 12 == 0:
                chosen_minimum, minimum_selection = select_penalty(history, d, q_grid, portfolio='minimum')
                chosen_markowitz, markowitz_selection = select_penalty(history, d, q_grid, portfolio='markowitz')
            ids, exposures, diagonal, covariance = _risk(root, d)
            mu = _forecasts(predictions, ids, d)
            minimum, markowitz, risk_scale, volatilities = penalty_candidates(
                mu, exposures, diagonal, covariance, q_grid=q_grid, settings=settings)
            minimum_frames.append(_frame(ids, d, minimum[chosen_minimum]))
            markowitz_frames.append(_frame(ids, d, markowitz[chosen_markowitz]))
            records.append(dict(formation_date=str(d), daily_variance_scale=risk_scale,
                minimum_q=float(q_grid[chosen_minimum]), markowitz_q=float(q_grid[chosen_markowitz]),
                minimum_gamma=float(q_grid[chosen_minimum] * risk_scale),
                markowitz_gamma=float(q_grid[chosen_markowitz] * risk_scale),
                minimum_selection=minimum_selection, markowitz_selection=markowitz_selection,
                minimum_predicted_annual_volatility=float(np.sqrt(252 * portfolio_variance(
                    minimum[chosen_minimum], diagonal, covariance, exposures))),
                markowitz_predicted_annual_volatility=volatilities[chosen_markowitz],
                minimum_net=float(minimum[chosen_minimum].sum()),
                markowitz_net=float(markowitz[chosen_markowitz].sum())))
            # Current future outcomes enter the bank only AFTER current weights
            # and selected penalties are fixed; eom_ret gates every later use.
            return_date, realized = _realized_labels(metadata, ids, d)
            minimum_returns = minimum @ realized if realized is not None else None
            markowitz_returns = markowitz @ realized if realized is not None else None
            history.append(dict(eom=d, eom_ret=return_date, minimum=minimum_returns,
                                markowitz=markowitz_returns))
            for i, q in enumerate(q_grid):
                bank_rows.append(dict(eom=d, eom_ret=return_date, available_after=return_date,
                    q=float(q), daily_gamma=float(q * risk_scale), daily_variance_scale=risk_scale,
                    minimum_return=float(minimum_returns[i]) if realized is not None else None,
                    markowitz_return=float(markowitz_returns[i]) if realized is not None else None))
    bank = pl.DataFrame(bank_rows).with_columns(pl.col('minimum_return', 'markowitz_return').cast(pl.Float64))
    return (pl.concat(minimum_frames).sort(['eom', 'id']),
            pl.concat(markowitz_frames).sort(['eom', 'id']), records, bank)
