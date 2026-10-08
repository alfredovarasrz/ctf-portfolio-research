"""H01 ordinary Ridge penalty paths from historical sufficient statistics.

One eigendecomposition per training fold replaces thousands of repeated solves.
The objective and unpenalized intercept match baseline.fit_ridge exactly.
"""
from datetime import date

import numpy as np

from baseline import DEFAULT_SETTINGS


def initial_ridge_lambdas(existing=DEFAULT_SETTINGS.ridge_lambdas):
    """Requested integer path plus predetermined subunit coverage and controls."""
    return np.unique(np.r_[np.arange(1, 10001, dtype=float),
                            np.geomspace(1e-8, 1., 1000), existing])


def _ridge_basis(train, validation):
    if train.n < 1 or validation.n < 1:
        raise ValueError('Ridge paths require nonempty training and validation')
    mean_x, mean_y = train.sx / train.n, train.sy / train.n
    gram = train.xx - np.outer(train.sx, train.sx) / train.n
    rhs = train.xy - train.sx * train.sy / train.n
    eigenvalues, vectors = np.linalg.eigh((gram + gram.T) / 2)
    # Centering can leave roundoff-sized negative values in a singular Gram.
    tolerance = 1e-10 * max(1., float(np.max(np.abs(eigenvalues))))
    if eigenvalues.min() < -tolerance:
        raise ValueError('Training sufficient statistics have an indefinite Gram')
    eigenvalues = np.maximum(eigenvalues, 0.)
    val_gram = (validation.xx - np.outer(validation.sx, mean_x)
                - np.outer(mean_x, validation.sx) + validation.n * np.outer(mean_x, mean_x))
    val_rhs = (validation.xy - mean_x * validation.sy - mean_y * validation.sx
               + validation.n * mean_y * mean_x)
    constant = validation.yy - 2 * mean_y * validation.sy + validation.n * mean_y**2
    return (train.n, validation.n, eigenvalues, vectors.T @ rhs,
            vectors.T @ val_gram @ vectors, vectors.T @ val_rhs, constant)


def _basis_mse(basis, lambdas, batch_size):
    n_train, n_val, eigenvalues, rhs, gram, val_rhs, constant = basis
    losses = np.empty(len(lambdas))
    for start in range(0, len(lambdas), batch_size):
        batch = lambdas[start:start + batch_size]
        coef = rhs[:, None] / (eigenvalues[:, None] + n_train * batch[None, :])
        sse = constant - 2 * val_rhs @ coef + np.sum(coef * (gram @ coef), axis=0)
        losses[start:start + len(batch)] = np.maximum(0., sse) / n_val
    return losses


def ridge_path_mse(train, validation, lambdas, *, batch_size=512):
    """Numerical helper; callers supply an already valid historical fold."""
    lambdas = np.asarray(lambdas, dtype=float)
    if batch_size < 1 or lambdas.ndim != 1 or not len(lambdas) or not np.isfinite(lambdas).all() or (lambdas <= 0).any():
        raise ValueError('Positive finite penalties and batch size required')
    return _basis_mse(_ridge_basis(train, validation), lambdas, batch_size)


def validate_ridge_path(fold_stats, *, fold_records, selection_cutoff,
                        initial_lambdas=None, refinement_points=10000,
                        batch_size=512, maximum_extensions=3):
    """Return selected lambda and complete curves from historical blocked folds.

    fold_stats contains (training, validation) SufficientStats pairs. Its paired
    Fold records identify training/validation completed-return dates. A curve
    reports equal-weight mean fold MSE, matching the existing six-penalty search.
    """
    if not fold_stats or len(fold_stats) != len(fold_records):
        raise ValueError('Historical fold statistics and matching records required')
    cutoff = date.fromisoformat(str(selection_cutoff))
    for record in fold_records:
        train_last = date.fromisoformat(record['training_return_last'])
        train_first = date.fromisoformat(record['training_return_first'])
        val_first = date.fromisoformat(record['validation_return_first'])
        val_last = date.fromisoformat(record['validation_return_last'])
        if not train_first <= train_last <= cutoff or not val_first <= val_last <= cutoff:
            raise ValueError('Ridge path folds violate completed-label cutoffs')
    if refinement_points < 2 or batch_size < 1 or maximum_extensions < 0:
        raise ValueError('Invalid Ridge path refinement/batch/extension settings')
    grid = initial_ridge_lambdas() if initial_lambdas is None else np.unique(np.asarray(initial_lambdas, dtype=float))
    if grid.ndim != 1 or len(grid) < 2 or not np.isfinite(grid).all() or (grid <= 0).any():
        raise ValueError('At least two positive finite initial penalties required')
    bases = [_ridge_basis(train, validation) for train, validation in fold_stats]
    def score(values):
        return np.mean([_basis_mse(basis, values, batch_size) for basis in bases], axis=0)
    def curve(values, losses):
        return dict(lambdas=values.tolist(), mse=losses.tolist())

    losses = score(grid)
    initial = curve(grid, losses)
    extensions = []
    # Fixed policy: extend two decades at a time, at most three times. The lower
    # numerical coverage limit is 1e-12; this is not a statistically chosen bound.
    for _ in range(maximum_extensions):
        best = int(np.argmin(losses))
        if best == 0 and grid[0] > 1e-12:
            added = np.geomspace(max(1e-12, grid[0] / 100), grid[0], 1000)[:-1]
            side = 'lower'
        elif best == len(grid) - 1:
            added = np.geomspace(grid[-1], grid[-1] * 100, 1000)[1:]
            side = 'upper'
        else:
            break
        added_losses = score(added)
        extensions.append(dict(side=side, **curve(added, added_losses)))
        order = np.argsort(np.r_[grid, added], kind='stable')
        grid, losses = np.r_[grid, added][order], np.r_[losses, added_losses][order]
    best = int(np.argmin(losses))
    status = 'lower_boundary' if best == 0 else 'upper_boundary' if best == len(grid) - 1 else 'bracketed'
    left, right = grid[max(0, best - 1)], grid[min(len(grid) - 1, best + 1)]
    refined = np.linspace(left, right, refinement_points)
    refined_losses = score(refined)
    # Retain the coarse winner because an even-sized local grid can omit it.
    candidate_values, candidate_losses = np.r_[grid, refined], np.r_[losses, refined_losses]
    order = np.argsort(candidate_values, kind='stable')
    selected = int(order[np.argmin(candidate_losses[order])])
    penalty = float(candidate_values[selected])
    diagnostics = dict(method='spectral-blocked-ridge-v2', validation_policy='historical_blocked',
                       selection_cutoff=str(cutoff),
                       fold_records=fold_records, initial=initial, extensions=extensions,
                       refinement=curve(refined, refined_losses), refinement_interval=[float(left), float(right)],
                       coarse_boundary_status=status, selected_lambda=penalty,
                       selected_mse=float(candidate_losses[selected]), batch_size=batch_size,
                       candidate_evaluations=len(grid) + len(refined),
                       extension_policy=dict(decades=2, points=1000, maximum=maximum_extensions, lower_limit=1e-12))
    return penalty, diagnostics
