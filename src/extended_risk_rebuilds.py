"""Fixed annual risk projections with full causal state replay in one basis."""
from collections import deque
from dataclasses import asdict
from datetime import date
import hashlib
import json
from pathlib import Path

import numpy as np
import polars as pl
from scipy.cluster.hierarchy import cut_tree, linkage
from threadpoolctl import threadpool_limits

from comparison_models import INDUSTRIES, RISK_SETTINGS
from extended_risk import _parent_months, _factor_stream, load_market_proxy
from research_risk import build_risk_artifacts, _write_json


def characteristic_pca(gram, retained_variance=.90):
    """Fixed exposure projection; no factor-return or future-return selection."""
    gram = np.asarray(gram, dtype=float)
    if (gram.ndim != 2 or gram.shape[0] != gram.shape[1] or not gram.shape[0]
            or not np.isfinite(gram).all() or not 0 < retained_variance <= 1):
        raise ValueError('Finite square characteristic Gram and retention fraction required')
    eigenvalues, vectors = np.linalg.eigh((gram + gram.T) / 2)
    scale = max(float(np.max(np.abs(eigenvalues))), 1e-20)
    if eigenvalues.min() < -1e-9 * scale:
        raise ValueError('Characteristic Gram is materially indefinite')
    order = np.argsort(-eigenvalues, kind='stable')
    eigenvalues = np.maximum(eigenvalues[order], 0.); vectors = vectors[:, order]
    total = float(eigenvalues.sum())
    # Keep one zero-score coordinate if no history varies; no fabricated exposure.
    count = 1 if total == 0 else (len(eigenvalues) if retained_variance == 1 else
        int(np.searchsorted(np.cumsum(eigenvalues), retained_variance * total)) + 1)
    projection = vectors[:, :count].copy()
    for j in range(count):
        largest = int(np.argmax(np.abs(projection[:, j])))
        if projection[largest, j] < 0:
            projection[:, j] *= -1
    return projection, dict(retained_components=count, retained_variance_rule=retained_variance,
        retained_exposure_eigenvalue_fraction=1. if total == 0 else float(eigenvalues[:count].sum() / total))


def characteristic_groups(values, market_returns, groups=21):
    """Ward grouping of proxy-residualized factors, without altering stock exposures.

    Constant factor streams have correlation0 with every other factor and1
    with themselves. Orthogonal artificial DISTANCE coordinates realize that
    convention; they never enter B, regressions, or returns.
    """
    values = np.asarray(values, dtype=float); market_returns = np.asarray(market_returns, dtype=float)
    if (values.ndim != 2 or len(values) < 2 or not values.shape[1]
            or market_returns.shape != (len(values),) or not np.isfinite(values).all()
            or not np.isfinite(market_returns).all() or groups < 1):
        raise ValueError('Finite aligned factor/proxy rows and positive group count required')
    centered = values - values.mean(axis=0)
    market = market_returns - market_returns.mean()
    if market @ market > 0:
        centered -= np.outer(market, market @ centered / (market @ market))
    norm = np.sqrt(np.sum(centered ** 2, axis=0))
    active = norm > 1e-12 * np.maximum(np.sqrt(np.sum(values ** 2, axis=0)), 1e-20)
    observations = np.zeros((values.shape[1], len(values) + int((~active).sum())))
    observations[active, :len(values)] = (centered[:, active] / norm[active]).T
    for j, factor in enumerate(np.flatnonzero(~active)):
        observations[factor, len(values) + j] = 1.
    count = min(groups, values.shape[1])
    labels = (np.arange(count) if count == values.shape[1] else
              cut_tree(linkage(observations, method='ward', metric='euclidean'), n_clusters=[count])[:, 0])
    # Canonical group names follow their first canonical characteristic member.
    unique = sorted(set(labels.tolist()), key=lambda label: int(np.flatnonzero(labels == label)[0]))
    projection = np.zeros((values.shape[1], len(unique)))
    memberships = []
    for j, label in enumerate(unique):
        members = np.flatnonzero(labels == label)
        projection[members, j] = 1. / len(members)
        memberships.append(members.tolist())
    return projection, dict(requested_groups=groups, actual_groups=len(unique),
        constant_factor_streams=int((~active).sum()), memberships=memberships,
        distance='sqrt(2*(1-correlation)); Ward Euclidean',
        constant_stream_correlation='off-diagonal0 diagonal1; raw exposures retained')


def _full_projection(characteristics):
    count = len(INDUSTRIES)
    projection = np.zeros((count + characteristics.shape[0], count + characteristics.shape[1]))
    projection[:count, :count] = np.eye(count)
    projection[count:, count:] = characteristics
    return projection


def fit_annual_projection(parent_dir, variant, anchor, *, basis_months=120, retained_variance=.90, groups=21,
                          market_proxy=None):
    """Fit only the unchanged original risk history available at annual anchor."""
    root, manifest, _, directories = _parent_months(parent_dir, anchor)
    count = len(INDUSTRIES)
    if manifest['factor_order'][:count] != list(INDUSTRIES):
        raise ValueError('Parent does not use original fixed industry coordinates')
    names = manifest['factor_order'][count:]
    if variant == 'exposure_pca90':
        gram = np.zeros((len(names), len(names))); used = []
        for directory in directories[-basis_months:]:
            with np.load(directory / 'exposures.npz', allow_pickle=False) as saved:
                x = saved['B'][:, count:]
            if len(x) >= 2:
                gram += x.T @ x / (len(x) - 1)
                used.append(directory.name)
        if not used:
            raise ValueError('No historical exposure cross-sections for PCA')
        projected, record = characteristic_pca(gram / len(used), retained_variance)
        record.update(basis_window_months=basis_months, basis_training_months=used,
                      exposure_month_weighting='equal month average of original standardized Gram matrices')
        reduced_names = [f'pc_{i:03d}' for i in range(projected.shape[1])]
    elif variant == 'characteristic_groups21':
        history = deque(maxlen=manifest['settings']['covariance_days']); days = deque(maxlen=history.maxlen)
        for directory in directories:
            stream, _ = _factor_stream(directory, manifest['factor_order'], date.fromisoformat(directory.name))
            history.extend(stream.select(names).to_numpy()); days.extend(stream['date'].to_list())
        proxy = load_market_proxy(root, cutoff=anchor) if market_proxy is None else market_proxy
        matched = pl.DataFrame({'date': pl.Series(list(days), dtype=pl.Date)}).join(proxy, on='date', how='left')
        if matched.height != len(days) or matched['ret'].null_count() or not matched['ret'].is_finite().all():
            raise ValueError('Clustering factor dates lack finite original market proxy')
        projected, record = characteristic_groups(np.array(history), matched['ret'].to_numpy(), groups)
        record.update(basis_training_first_day=str(days[0]), basis_training_last_day=str(days[-1]),
                      basis_training_factor_days=len(days), basis_covariance_days=history.maxlen)
        record['member_names'] = [[names[i] for i in members] for members in record['memberships']]
        reduced_names = [f'group_{i:03d}' for i in range(projected.shape[1])]
    else:
        raise ValueError('Projection variant must be exposure_pca90 or characteristic_groups21')
    record.update(variant=variant, basis_information_cutoff=str(anchor), original_characteristics=names,
                  industries_policy='original12 separate indicators', historical_dimension_tuning=False)
    return _full_projection(projected), reduced_names, record


def cached_daily_returns(parent_dir, d):
    """Recover original finite daily stock excess returns without raw-data refits."""
    root = Path(parent_dir); directory = root / 'months' / str(d)
    manifest = json.loads((root / 'manifest.json').read_text())
    stream, marker = _factor_stream(directory, manifest['factor_order'], d)
    residuals = pl.read_parquet(directory / 'residuals.parquet').sort('date', 'id')
    if not stream.height:
        if residuals.height:
            raise ValueError('Residual rows exist without factor fits')
        return pl.DataFrame(schema={'id': pl.Int64, 'date': pl.Date, 'ret_exc': pl.Float64})
    preceding = marker['preceding_exposure_date']
    if not preceding or date.fromisoformat(preceding) >= d:
        raise ValueError('Daily reconstruction needs preceding-month exposure')
    with np.load(root / 'months' / preceding / 'exposures.npz', allow_pickle=False) as saved:
        ids = saved['ids']; x = saved['B']
        if str(saved['eom_ret']) != str(d) or np.any(np.diff(ids) <= 0):
            raise ValueError('Original exposure timing or security order differs')
    groups = residuals.partition_by('date', maintain_order=True)
    if len(groups) != stream.height or [g['date'][0] for g in groups] != stream['date'].to_list():
        raise ValueError('Factor and residual fit dates do not match')
    frames = []
    for group, coefficients in zip(groups, stream.drop('date').to_numpy()):
        positions = np.searchsorted(ids, group['id'].to_numpy())
        if np.any(positions >= len(ids)) or np.any(ids[positions] != group['id'].to_numpy()):
            raise ValueError('Original residual securities lack preceding exposures')
        values = x[positions] @ coefficients + group['res'].to_numpy()
        if not np.isfinite(values).all():
            raise ValueError('Original reconstructed returns are invalid')
        frames.append(group.select('id', 'date').with_columns(pl.Series('ret_exc', values)))
    return pl.concat(frames)


class ProjectedArtifactSource:
    """Original monthly B/metadata with one annual projection fixed for its prefix."""
    def __init__(self, parent_dir, projection, scored_dates):
        self.root = Path(parent_dir); self.projection = projection
        self.scored_dates = set(scored_dates)
        _, _, _, directories = _parent_months(self.root, max(scored_dates))
        frames = []
        for directory in directories:
            frames.append(self._frame(date.fromisoformat(directory.name)))
        self.metadata = pl.concat(frames)

    def _frame(self, d):
        with np.load(self.root / 'months' / str(d) / 'exposures.npz', allow_pickle=False) as saved:
            ids = saved['ids'].copy(); selection = saved['ctff_test'].copy()
            eom_ret = date.fromisoformat(str(saved['eom_ret']))
        return pl.DataFrame({'id': ids, 'eom': [d] * len(ids), 'eom_ret': [eom_ret] * len(ids),
                             'ctff_test': selection if d in self.scored_dates else np.zeros(len(ids), dtype=bool)})

    def load_exposures(self, d):
        with np.load(self.root / 'months' / str(d) / 'exposures.npz', allow_pickle=False) as saved:
            x = saved['B'].copy()
        projected = x @ self.projection
        constants = int(np.count_nonzero(np.all(projected[:, len(INDUSTRIES):] == 0., axis=0)))
        return self._frame(d), projected, constants


class CachedArtifactDailySource:
    def __init__(self, parent_dir):
        self.root = Path(parent_dir)
    def load(self, d):
        return cached_daily_returns(self.root, d)


def build_projected_risk_artifacts(parent_dir, output_dir, variant, *, settings=RISK_SETTINGS,
        identity=None, basis_months=120, retained_variance=.90, groups=21):
    """Annual fixed-basis full-prefix replay using the unchanged risk engine.

    Completed annual directories are resumable independently. Never reuse any
    state from another basis. The caller identity fingerprints original inputs.
    """
    parent, manifest, metadata, _ = _parent_months(parent_dir)
    root = Path(output_dir); root.mkdir(parents=True, exist_ok=True)
    specification = dict(parent_manifest_sha256=hashlib.sha256((parent / 'manifest.json').read_bytes()).hexdigest(),
        adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), caller_identity=identity,
        variant=variant, settings=asdict(settings), basis_months=basis_months,
        retained_variance=retained_variance, groups=groups,
        daily_input='original Bprev*f+residual', replay_history='full available prefix in fixed annual basis')
    marker = root / 'manifest.json'
    if marker.exists() and json.loads(marker.read_text()) != specification:
        raise ValueError('Projection artifact identity mismatch; use separate directory')
    if not marker.exists():
        if any(root.iterdir()):
            raise ValueError('Projection directory has no manifest')
        _write_json(marker, specification)
    years = {}
    for record in metadata['records']:
        d = date.fromisoformat(record['formation_date'])
        years.setdefault(d.year + int(d.month == 12), []).append(d)
    records = []
    with threadpool_limits(limits=settings.threads):
        for year, dates in sorted(years.items()):
            anchor = min(dates)
            projection, names, basis_record = fit_annual_projection(parent, variant, anchor,
                basis_months=basis_months, retained_variance=retained_variance, groups=groups)
            year_dir = root / str(year); year_dir.mkdir(exist_ok=True)
            basis_sha = hashlib.sha256(projection.tobytes()).hexdigest()
            basis_record['basis_sha256'] = basis_sha
            np.savez(year_dir / 'basis.npz', projection=projection)
            _write_json(year_dir / 'basis.json', basis_record)
            source = ProjectedArtifactSource(parent, projection, dates)
            daily = CachedArtifactDailySource(parent)
            def source_factory(chars, features, cache_size=144):
                return source
            def daily_factory(data):
                return daily
            annual_records = build_risk_artifacts(None, pl.DataFrame({'features': names}), None,
                year_dir / 'risk', settings=settings,
                identity=dict(**specification, basis_sha256=basis_sha, anchor=str(anchor), scored_dates=list(map(str, dates))),
                source_factory=source_factory, daily_factory=daily_factory)
            records.extend(dict(record, covariance_variant=variant, basis_sha256=basis_sha,
                basis_information_cutoff=str(anchor), replay_basis_year=year) for record in annual_records)
    _write_json(root / 'records.json', records)
    return records


def iter_projected_risk(parent_dir, output_dir):
    """Yield completed projected B/F/D with original same-date reference covariance."""
    parent = Path(parent_dir); root = Path(output_dir)
    specification = json.loads((root / 'manifest.json').read_text())
    if specification['parent_manifest_sha256'] != hashlib.sha256((parent / 'manifest.json').read_bytes()).hexdigest():
        raise ValueError('Projected risk parent identity mismatch')
    for record in json.loads((root / 'records.json').read_text()):
        d = date.fromisoformat(record['formation_date'])
        path = root / str(record['replay_basis_year']) / 'risk' / 'months' / str(d) / 'risk.npz'
        with np.load(path, allow_pickle=False) as saved:
            risk = {name: saved[name].copy() for name in ('ids', 'B', 'F', 'D')}
        with np.load(parent / 'months' / str(d) / 'risk.npz', allow_pickle=False) as saved:
            if not np.array_equal(risk['ids'], saved['ids']):
                raise ValueError('Projected/reference stock universe differs')
            reference_B = saved['B'].copy(); reference_F = saved['F'].copy(); reference_D = saved['D'].copy()
        # Different basis requires reference_B in reference scaling, not new B.
        yield dict(**risk, eom=d, reference_B=reference_B, reference_F=reference_F,
                   reference_D=reference_D, record=record)
