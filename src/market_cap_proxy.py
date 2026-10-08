"""Market-cap-weighted excess returns from supplied, completed stock data."""
from datetime import date
import json
from pathlib import Path

def frame_digest(frame):
    from market_cap_value_digest import logical_cap_digest
    if frame.columns != ['id', 'eom', 'market_equity']:
        raise ValueError('Exact capitalization digest schema required')
    return logical_cap_digest(frame.iter_rows(), frame.height)

def raw_caps(chars, cutoff, capital_dates):
    """Collect only raw as-of sizes for the consumed preceding exposure dates."""
    import polars as pl
    if 'market_equity' not in chars.collect_schema().names():
        raise ValueError('Provided raw market_equity required for this cap-weighted specification')
    wanted = sorted(set(date.fromisoformat(str(x)) for x in capital_dates))
    if not wanted or max(wanted) >= cutoff:
        raise ValueError('Capitalizations must strictly precede last proxy return month')
    frame = (chars.filter(pl.col('eom').is_in(wanted))
        .select(pl.col('id').cast(pl.Int64), pl.col('eom').cast(pl.Date),
                pl.col('market_equity').cast(pl.Float64)).collect(engine='streaming')
        .sort(['eom', 'id']))
    if frame.select('id', 'eom').is_duplicated().any() or frame['id'].null_count() or frame['eom'].null_count():
        raise ValueError('Duplicate or missing raw cap keys')
    return frame, dict(rows=frame.height, dates=[str(x) for x in wanted],
        columns=frame.columns, consumed_prefix_sha256=frame_digest(frame),
        digest_format='logical-cap-v1',
        scope='raw preceding-month size rows only, excludes all future dates')

def daily_cap_mean(returns, caps):
    """Pure per-day aggregation; missing caps cannot acquire fabricated values."""
    import numpy as np
    returns = np.asarray(returns, dtype=float); caps = np.asarray(caps, dtype=float)
    if returns.ndim != 1 or caps.shape != returns.shape or len(returns) < 2 or not np.isfinite(returns).all():
        raise ValueError('Finite observed daily returns with aligned size vector required')
    valid = np.isfinite(caps) & (caps > 0)
    count = int(valid.sum())
    if count:
        # Scaling avoids overflow when monetary units are very large.
        units = caps[valid] / caps[valid].max()
        value = float(np.dot(units / units.sum(), returns[valid]))
        capital_sum = float(caps[valid].sum())
        if not np.isfinite(capital_sum):
            raise ValueError('Nonfinite aggregate raw capitalization')
        mode = 'market_cap'
    else:
        value = float(returns.mean()); capital_sum = 0.; mode = 'equal_no_valid_cap'
    if not np.isfinite(value) or value <= -1:
        raise ValueError('Proxy gross return must be finite and positive for drawdown index')
    return value, dict(observed_stocks=len(returns), valid_cap_stocks=count,
        valid_cap_count_fraction=count/len(returns), capital_sum=capital_sum, weighting=mode)

def build_proxy(caps, risk_root, cutoff):
    """Original Bf+residual reconstructs observed returns, not expected returns."""
    import numpy as np
    import polars as pl
    risk_root = Path(risk_root); cutoff = date.fromisoformat(str(cutoff))
    names = json.loads((risk_root/'manifest.json').read_text())['factor_order']
    cap_months = {str(x['eom'][0]): x for x in caps.partition_by('eom', maintain_order=True)}
    days, coverage, months = [], [], []
    for directory in sorted(p for p in (risk_root/'months').iterdir()
                            if p.is_dir() and not p.name.endswith('.tmp') and p.name <= str(cutoff)):
        month = date.fromisoformat(directory.name)
        marker = json.loads((directory/'complete.json').read_text())
        if marker['available_through'] != str(month) or marker['factor_order'] != names:
            raise ValueError('Original proxy month marker/cutoff/basis differs')
        factors = pl.read_parquet(directory/'factor_returns.parquet').sort('date')
        residuals = pl.read_parquet(directory/'residuals.parquet').sort(['date','id'])
        if factors.columns != ['date']+names or factors['date'].n_unique() != factors.height:
            raise ValueError('Original daily factor dates or coordinates differ')
        if residuals.columns != ['id','date','res'] or residuals.select('id','date').is_duplicated().any():
            raise ValueError('Original residual schema/keys differ')
        if not factors.height:
            if residuals.height:
                raise ValueError('Residuals without original factor fits')
            months.append(dict(month=str(month), days=0)); continue
        preceding = marker.get('preceding_exposure_date')
        if preceding is None or date.fromisoformat(preceding) >= month:
            raise ValueError('Missing preceding exposure information')
        cap_frame = cap_months.get(preceding)
        if cap_frame is None:
            raise ValueError('Preceding raw cap month absent')
        with np.load(risk_root/'months'/preceding/'exposures.npz', allow_pickle=False) as saved:
            ids, b = saved['ids'], saved['B']
            if str(saved['eom']) != preceding or str(saved['eom_ret']) != str(month):
                raise ValueError('Original exposures do not precede return month')
        if np.any(np.diff(ids) <= 0) or b.shape != (len(ids),len(names)) or not np.isfinite(b).all():
            raise ValueError('Original exposure coordinates/security order invalid')
        coef = factors.select(names).to_numpy()
        if not np.isfinite(coef).all() or not residuals['res'].is_finite().all():
            raise ValueError('Nonfinite original factor/residual state')
        fitted = b @ coef.T
        grouped = residuals.partition_by('date', maintain_order=True)
        if [x['date'][0] for x in grouped] != factors['date'].to_list():
            raise ValueError('Original factor/residual day coverage differs')
        cap_map = dict(zip(cap_frame['id'].to_list(),cap_frame['market_equity'].to_list()))
        for column, group in enumerate(grouped):
            d = group['date'][0]
            if d > month or (d.year,d.month) != (month.year,month.month):
                raise ValueError('Original daily inputs exceed completed return month')
            positions = np.searchsorted(ids,group['id'].to_numpy())
            if np.any(positions >= len(ids)) or np.any(ids[positions] != group['id'].to_numpy()):
                raise ValueError('Observed daily regression stocks lack preceding exposures')
            observed = fitted[positions,column] + group['res'].to_numpy()
            sizes = [cap_map.get(i) if cap_map.get(i) is not None else float('nan') for i in group['id'].to_list()]
            value, info = daily_cap_mean(observed,sizes)
            days.append((d,value)); coverage.append(dict(date=d,return_month=month,
                capitalization_date=date.fromisoformat(preceding),**info))
        months.append(dict(month=str(month),days=factors.height))
        print('Proxy completed '+str(month),flush=True)
    if not days:
        raise ValueError('No usable original daily regression history')
    frame = pl.DataFrame(days,schema={'date':pl.Date,'ret':pl.Float64},orient='row')
    if frame['date'].n_unique() != frame.height or not frame['date'].is_sorted():
        raise ValueError('Proxy day coverage must be unique and chronological')
    return frame,pl.DataFrame(coverage),months
