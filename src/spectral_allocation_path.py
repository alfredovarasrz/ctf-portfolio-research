"""Exact stock-coordinate C02 penalty path. No model selection or data access."""
import numpy as np


def prepare_spectrum(mu, exposures, diagonal, covariance):
    """Diagonalize ORIGINAL Sigma once, retaining every stock eigencomponent."""
    mu, b, d, f = (np.asarray(a, dtype=float) for a in (mu, exposures, diagonal, covariance))
    if (mu.ndim != 1 or not len(mu) or d.shape != mu.shape or b.ndim != 2
            or b.shape[0] != len(mu) or f.shape != (b.shape[1], b.shape[1])
            or not all(np.isfinite(a).all() for a in (mu, b, d, f)) or np.any(d <= 0)
            or not np.allclose(f, f.T, rtol=1e-12, atol=1e-15)):
        raise ValueError('Aligned finite inputs, positive specific variances and symmetric F required')
    # Unlike whitening by D, diagonalizing stock Sigma preserves gamma I with
    # unequal specific variances. No jitter, factor truncation or D replacement.
    sigma = (b @ f) @ b.T
    sigma[np.diag_indices_from(sigma)] += d
    sigma = (sigma + sigma.T) / 2
    scale = float(np.median(np.diag(sigma)))
    eigenvalues, vectors = np.linalg.eigh(sigma)
    if not np.isfinite(eigenvalues).all() or np.any(eigenvalues <= 0) or scale <= 0:
        raise ValueError('Original stock covariance must be positive definite, without jitter')
    return dict(eigenvalues=eigenvalues, vectors=vectors, mu_projection=vectors.T @ mu,
                ones_projection=vectors.sum(axis=0), daily_variance_scale=scale)


def candidate_weights(spectrum, q, *, target=.1):
    """Reconstruct only requested current weights, with original covariance risk."""
    if not np.isfinite(q) or q < 0 or not np.isfinite(target) or target <= 0:
        raise ValueError('Nonnegative q and positive risk target required')
    s = spectrum['eigenvalues']; a = spectrum['vectors']
    denominator = s + q * spectrum['daily_variance_scale']
    v_one = spectrum['ones_projection'] / denominator
    v_mu = spectrum['mu_projection'] / denominator
    budget = float(spectrum['ones_projection'] @ v_one)
    variance = float(s @ (v_mu**2))
    if not np.isfinite(budget) or budget <= 0 or not np.isfinite(variance) or variance <= 0:
        raise ValueError('Invalid MVP budget or zero Markowitz direction')
    return a @ (v_one / budget), a @ (v_mu * target / np.sqrt(252 * variance))


def payoff_batches(spectrum, q_grid, realised_projection, *, batch_size=512, target=.1):
    """Scalar candidate outcomes. Labels enter ONLY this future-released bank.

    O(NQ) arithmetic and O(N*batch_size) workspace. No candidate stock weights
    are constructed. The caller must release these payoffs only at eom_ret.
    """
    grid = np.asarray(q_grid, dtype=float); cy = np.asarray(realised_projection, dtype=float)
    s = spectrum['eigenvalues']; cm = spectrum['mu_projection']; c1 = spectrum['ones_projection']
    if (grid.ndim != 1 or not len(grid) or not np.isfinite(grid).all() or np.any(grid < 0)
            or cy.shape != s.shape or not np.isfinite(cy).all() or batch_size < 1
            or not np.isfinite(target) or target <= 0):
        raise ValueError('Finite nonnegative path, aligned label projection and positive batch/target required')
    for first in range(0, len(grid), batch_size):
        q = grid[first:first+batch_size]
        inverse = 1 / (s[:, None] + spectrum['daily_variance_scale'] * q[None, :])
        one = c1[:, None] * inverse
        mu = cm[:, None] * inverse
        budget = c1 @ one
        raw_variance = s @ (mu**2)
        if np.any(budget <= 0) or np.any(raw_variance <= 0):
            raise ValueError('Invalid MVP budget or zero Markowitz direction')
        scaling = target / np.sqrt(252 * raw_variance)
        yield dict(q=q, gamma=q*spectrum['daily_variance_scale'],
            minimum_return=(cy @ one)/budget, markowitz_return=(cy @ mu)*scaling,
            minimum_annual_volatility=np.sqrt(252 * (s @ (one**2)))/budget,
            markowitz_annual_volatility=np.sqrt(252*raw_variance)*scaling,
            markowitz_net=(c1 @ mu)*scaling)
