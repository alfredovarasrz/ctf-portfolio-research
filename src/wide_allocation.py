"""C02_WIDE scalar banks and causal annual allocation-grid refinement."""
from datetime import date
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import polars as pl
from threadpoolctl import threadpool_limits

def coarse_grid():
    import numpy as np
    from batch_allocations import Q_GRID
    return np.unique(np.r_[0., np.geomspace(1e-8, 1., 200), np.arange(1., 10001.), Q_GRID])

def refinement_grid(grid, chosen, *, points=10000):
    """Neighbor interval after a COMPLETED historical portfolio score selection."""
    import numpy as np
    grid = np.asarray(grid, dtype=float)
    if (grid.ndim != 1 or len(grid) < 2 or not np.isfinite(grid).all() or np.any(grid < 0)
            or not 0 <= chosen < len(grid) or points < 2 or np.any(np.diff(grid) <= 0)):
        raise ValueError('Increasing coarse grid, selected index and two refinement points required')
    lower, upper = grid[max(0, chosen-1)], grid[min(len(grid)-1, chosen+1)]
    return np.linspace(lower, upper, points), dict(lower=float(lower), upper=float(upper),
        boundary=chosen in (0, len(grid)-1), points=points,
        extension='none; report a grid boundary rather than claim an unrestricted optimum')
from batch_allocations import _months, _risk, _forecasts, _realized_labels, _frame
from comparison_models import RISK_SETTINGS, portfolio_variance
from artifact_utils import write_json
from spectral_allocation_path import prepare_spectrum, candidate_weights, payoff_batches

MODELS = ('penalized_minimum_variance','ridge_penalized_markowitz','xgboost_penalized_markowitz')


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def atomic_arrays(path, arrays):
    path = Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    with temporary.open('wb') as handle: np.savez(handle,**arrays)
    temporary.replace(path)


def scalar_returns(bank, values, portfolio, *, batch_size=512):
    """Replay ONLY completed scalar bank data. Eigenvectors are never persisted."""
    if not bank['label_complete']:
        raise ValueError('Cannot replay an unavailable completed outcome')
    projection = bank['mu_xgboost'] if portfolio=='minimum' else bank['mu_'+portfolio]
    spectrum = dict(eigenvalues=bank['eigenvalues'],ones_projection=bank['ones_projection'],
        mu_projection=projection,daily_variance_scale=float(bank['daily_variance_scale']))
    key = 'minimum_return' if portfolio=='minimum' else 'markowitz_return'
    return np.concatenate([row[key] for row in payoff_batches(spectrum,values,
        bank['realised_projection'],batch_size=batch_size)])


def block_scores(returns, *, portfolio):
    """Same five consecutive score blocks, unbiased variance and equal-fold mean."""
    count, candidates = returns.shape
    blocks = np.array_split(np.arange(count),5)
    if count<10 or any(len(block)<2 for block in blocks):
        return np.full((5,candidates),np.nan),np.full(candidates,np.nan)
    scores = []
    for block in blocks:
        values = returns[block]
        variance = np.var(values,axis=0,ddof=1)
        score = variance if portfolio=='minimum' else np.divide(np.mean(values,axis=0)*np.sqrt(12),
            np.sqrt(variance),out=np.full(candidates,np.nan),where=variance>0)
        scores.append(score)
    scores = np.asarray(scores)
    mean = np.mean(scores,axis=0)
    mean[~np.isfinite(scores).all(axis=0)] = np.nan
    return scores,mean


def best_index(scores, *, portfolio):
    valid = np.flatnonzero(np.isfinite(scores))
    if not len(valid): return None
    return int(valid[np.argmin(scores[valid] if portfolio=='minimum' else -scores[valid])])


def select_wide_penalty(history, cutoff, *, portfolio, grid=None, refinement_points=10000,
                        batch_size=512):
    """Same original C02 history and objective, changed grid and local procedure."""
    if portfolio not in ('minimum','ridge','xgboost'):
        raise ValueError('Native minimum, original Ridge or original XGBoost branch required')
    cutoff = date.fromisoformat(str(cutoff))
    grid = coarse_grid() if grid is None else np.asarray(grid,dtype=float)
    if (grid.ndim!=1 or len(grid)<2 or grid[0]!=0 or np.any(np.diff(grid)<=0)
            or not np.isfinite(grid).all() or refinement_points<2):
        raise ValueError('Increasing finite coarse grid starting q0 required')
    # EXACT original filter and last120 mechanics; no current-month payoff is
    # supplied by the outer loop until after its current allocations are fixed.
    available = [row for row in history if date.fromisoformat(row['eom_ret'])<=cutoff
                 and row['label_complete']][-120:]
    matrices = []
    for row in available:
        with np.load(row['bank_path'],allow_pickle=False) as saved:
            bank = {name:saved[name].copy() for name in saved.files}
        if (str(bank['eom'])!=row['eom'] or str(bank['eom_ret'])!=row['eom_ret']
                or not bool(bank['label_complete']) or not np.array_equal(bank['coarse_q'],grid)):
            raise ValueError('Historical scalar bank coordinates changed')
        matrices.append(bank['coarse_'+portfolio])
    returns = np.asarray(matrices).reshape(len(available),len(grid))
    folds,mean = block_scores(returns,portfolio='minimum' if portfolio=='minimum' else 'markowitz')
    chosen = best_index(mean,portfolio='minimum' if portfolio=='minimum' else 'markowitz')
    curves = dict(coarse_q=grid,coarse_fold_scores=folds,coarse_scores=mean,
        refinement_q=np.array([]),refinement_fold_scores=np.empty((5,0)),refinement_scores=np.array([]))
    fallback = None; interval = None; coarse_q = None; boundary = False; selected = 0.
    if chosen is None:
        fallback = 'predeclared q=0; no five usable historical score blocks'
    else:
        coarse_q = float(grid[chosen]); boundary = chosen in (0,len(grid)-1)
        values,interval = refinement_grid(grid,chosen,points=refinement_points)
        # The FULL eigenbasis is not needed for historical replay, just the
        # original-stock sufficient vectors released at the label's eom_ret.
        refined = []
        for row in available:
            with np.load(row['bank_path'],allow_pickle=False) as saved:
                bank = {name:saved[name].copy() for name in saved.files}
            refined.append(scalar_returns(bank,values,portfolio,batch_size=batch_size))
        refined_folds,refined_scores = block_scores(np.asarray(refined),
            portfolio='minimum' if portfolio=='minimum' else 'markowitz')
        curves.update(refinement_q=values,refinement_fold_scores=refined_folds,refinement_scores=refined_scores)
        # Retain ALL coarse candidates, including the winner. Sorted q supplies
        # the original smaller-grid-index tie rule when coarse/refinement tie.
        combined_q = np.r_[grid,values]; combined_score = np.r_[mean,refined_scores]
        order = np.argsort(combined_q,kind='stable')
        winner = best_index(combined_score[order],portfolio='minimum' if portfolio=='minimum' else 'markowitz')
        selected = float(combined_q[order[winner]])
    record = dict(portfolio=portfolio,selection_formation=str(cutoff),history_months=len(available),
        history_return_first=available[0]['eom_ret'] if available else None,
        history_return_last=available[-1]['eom_ret'] if available else None,
        history_available_through=str(cutoff),history_formation_dates=[r['eom'] for r in available],
        history_return_dates=[r['eom_ret'] for r in available],
        history_bank_sha256={r['eom']:digest(r['bank_path']) for r in available},
        criterion='mean of five sample-variance blocks' if portfolio=='minimum' else 'mean of five annualized Sharpe blocks',
        selected_q=selected,coarse_selected_q=coarse_q,coarse_grid_boundary=boundary,
        refinement_interval=interval,fallback=fallback,
        search_scope='predeclared broad grid and local refinement; no unrestricted optimum claim')
    return selected,record,curves


def annual_selection(history, cutoff, directory, identity, *, grid, refinement_points, batch_size):
    root = Path(directory)/'selection'/str(cutoff)
    selections,records = {},{}
    for portfolio in ('minimum','ridge','xgboost'):
        marker = root/(portfolio+'.json'); path = root/(portfolio+'.npz')
        if marker.exists():
            receipt = json.loads(marker.read_text())
            expected_history = [r for r in history if r['eom_ret']<=str(cutoff) and r['label_complete']][-120:]
            if (receipt['identity']!=identity or receipt['record']['selection_formation']!=str(cutoff)
                    or receipt['record']['history_bank_sha256']!={r['eom']:digest(r['bank_path']) for r in expected_history}
                    or receipt['curve_sha256']!=digest(path)):
                raise ValueError('Annual C02_WIDE selector identity/history/curve changed')
            record = receipt['record']
        else:
            _,record,curves = select_wide_penalty(history,cutoff,portfolio=portfolio,
                grid=grid,refinement_points=refinement_points,batch_size=batch_size)
            atomic_arrays(path,curves)
            write_json(marker,dict(identity=identity,record=record,curve_sha256=digest(path)))
        selections[portfolio] = record['selected_q']
        records[portfolio] = dict(selected_q=record['selected_q'],selection_formation=record['selection_formation'],
            history_months=record['history_months'],history_return_last=record['history_return_last'],
            coarse_selected_q=record['coarse_selected_q'],coarse_grid_boundary=record['coarse_grid_boundary'],
            fallback=record['fallback'],curve_path=str(path.resolve()),curve_sha256=digest(path),
            record_path=str(marker.resolve()),record_sha256=digest(marker))
    return selections,records


def allocate_wide(artifact_dir,predictions,metadata,*,cache_dir,identity,settings=RISK_SETTINGS,
                  grid=None,refinement_points=10000,batch_size=512):
    """One ORIGINAL covariance eigensystem/month, three streams, no cached A."""
    if set(predictions)!={'ridge','xgboost'}:
        raise ValueError('Both unchanged original P01 forecast streams required')
    root,months = _months(artifact_dir); cache = Path(cache_dir)
    if cache.resolve()==root.resolve() or cache.resolve().is_relative_to(root.resolve()):
        raise ValueError('Scalar cache must be separate from original risk artifacts')
    grid = coarse_grid() if grid is None else np.asarray(grid,dtype=float)
    cache.mkdir(parents=True,exist_ok=True); identity_path = cache/'identity.json'
    if identity_path.exists() and json.loads(identity_path.read_text())!=identity:
        raise ValueError('Scalar allocation cache identity changed')
    write_json(identity_path,identity)
    frames = {name:[] for name in MODELS}; records,history = [],[]
    with threadpool_limits(limits=settings.threads):
        for offset,row in enumerate(months):
            d = date.fromisoformat(row['formation_date'])
            # Preserve ORIGINAL C02 offset%12, not an invented January anchor.
            if offset%12==0:
                selected,selection = annual_selection(history,d,cache,identity,grid=grid,
                    refinement_points=refinement_points,batch_size=batch_size)
            month = cache/'months'/str(d); marker = month/'complete.json'
            if marker.exists():
                saved = json.loads(marker.read_text()); path = month/'bank.npz'
                if (saved['identity']!=identity or saved['formation_date']!=str(d)
                        or saved['selection']!=selection or saved['bank_sha256']!=digest(path)
                        or saved['weights_sha256']!=digest(month/'weights.npz')):
                    raise ValueError('Monthly C02_WIDE bank/weights/selection identity changed')
                with np.load(month/'weights.npz',allow_pickle=False) as weights:
                    for name in MODELS: frames[name].append(_frame(weights['ids'],d,weights[name]))
                records.append(saved['allocation'])
                history.append(dict(eom=str(d),eom_ret=saved['eom_ret'],label_complete=saved['label_complete'],bank_path=str(path)))
                continue
            ids,b,diagonal,f = _risk(root,d)
            mu = {label:_forecasts(pred,ids,d) for label,pred in predictions.items()}
            spectrum = prepare_spectrum(mu['xgboost'],b,diagonal,f)
            ridge_spectrum = dict(spectrum,mu_projection=spectrum['vectors'].T@mu['ridge'])
            minimum,_ = candidate_weights(spectrum,selected['minimum'],target=settings.target_annual_volatility)
            _,ridge = candidate_weights(ridge_spectrum,selected['ridge'],target=settings.target_annual_volatility)
            _,xgboost = candidate_weights(spectrum,selected['xgboost'],target=settings.target_annual_volatility)
            weights = dict(zip(MODELS,(minimum,ridge,xgboost)))
            allocation = dict(formation_date=str(d),daily_variance_scale=spectrum['daily_variance_scale'],
                minimum_q=selected['minimum'],ridge_q=selected['ridge'],xgboost_q=selected['xgboost'],selection=selection,
                minimum_net=float(minimum.sum()),minimum_predicted_annual_volatility=float(np.sqrt(252*portfolio_variance(minimum,diagonal,f,b))),
                ridge_predicted_annual_volatility=float(np.sqrt(252*portfolio_variance(ridge,diagonal,f,b))),
                xgboost_predicted_annual_volatility=float(np.sqrt(252*portfolio_variance(xgboost,diagonal,f,b))))
            for name,w in weights.items(): frames[name].append(_frame(ids,d,w))
            # Read current FUTURE labels only AFTER current weights/choices.
            available,y = _realized_labels(metadata,ids,d)
            bank = dict(eom=np.array(str(d)),eom_ret=np.array(str(available)),ids=ids,
                label_complete=np.array(y is not None),eigenvalues=spectrum['eigenvalues'],
                ones_projection=spectrum['ones_projection'],mu_xgboost=spectrum['mu_projection'],
                mu_ridge=ridge_spectrum['mu_projection'],daily_variance_scale=np.array(spectrum['daily_variance_scale']),
                coarse_q=grid)
            if y is not None:
                bank['realised_projection'] = spectrum['vectors'].T@y
                for portfolio in ('minimum','ridge','xgboost'):
                    bank['coarse_'+portfolio] = scalar_returns(bank,grid,portfolio,batch_size=batch_size)
            else:
                for portfolio in ('minimum','ridge','xgboost'): bank['coarse_'+portfolio] = np.full(len(grid),np.nan)
            atomic_arrays(month/'weights.npz',dict(ids=ids,**weights)); atomic_arrays(month/'bank.npz',bank)
            complete = dict(identity=identity,formation_date=str(d),eom_ret=str(available),label_complete=y is not None,
                weights_information_cutoff=str(d),labels_release_date=str(available),selection=selection,allocation=allocation,
                bank_sha256=digest(month/'bank.npz'),weights_sha256=digest(month/'weights.npz'))
            write_json(marker,complete); records.append(allocation)
            history.append(dict(eom=str(d),eom_ret=str(available),label_complete=y is not None,bank_path=str(month/'bank.npz')))
            print(f'C02_WIDE {d}: {len(ids)} stocks; selected q {selected}',flush=True)
            del spectrum,ridge_spectrum,bank
    return {name:pl.concat(parts).sort(['eom','id']) for name,parts in frames.items()},records
