"""First CTF baseline: benchmark preprocessing, ridge forecasts, decile portfolios.

Structure and preprocessing follow the supplied Factor ML, Markowitz ML, and
Minimum Variance R submissions by Theis Ingerslev Jensen. See the source README for the public implementation guide.
This file is self-contained; main() uses only its supplied dataframes.
"""

# Section 1: Libraries and settings --------------------------------------------
from collections import OrderedDict
from dataclasses import dataclass
from datetime import date
from time import perf_counter

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
from scipy.linalg import cho_factor, cho_solve
from threadpoolctl import threadpool_limits


@dataclass(frozen=True)
class Settings:
    train_years: int = 10
    test_period_length: int = 12
    folds: int = 5
    ridge_lambdas: tuple[float, ...] = (0.0001, 0.001, 0.01, 0.1, 1.0, 10.0)
    n_pfs: int = 10
    cash_on_degenerate: bool = False
    blas_threads: int = 4


DEFAULT_SETTINGS = Settings()
META = ["id", "eom", "eom_ret", "excntry", "ctff_test", "ret_exc_lead1m"]


# Section 2: Shared utilities --------------------------------------------------
def as_lazy(data) -> pl.LazyFrame:
    if isinstance(data, pl.LazyFrame):
        return data
    if isinstance(data, pl.DataFrame):
        return data.lazy()
    if isinstance(data, pd.DataFrame):
        return pl.from_pandas(data).lazy()
    if isinstance(data, pa.Table):
        return pl.from_arrow(data).lazy()
    raise TypeError("Expected pandas, Polars, or Arrow input")


def canonical_features(features) -> list[str]:
    names = as_lazy(features).select("features").collect()["features"].to_list()
    if not names or any(not isinstance(name, str) or not name for name in names):
        raise ValueError("Feature names must be nonempty strings")
    if set(names) & set(META):
        raise ValueError("Metadata and return targets cannot be prediction features")
    return sorted(set(names))


def canonical_chars(chars, features: list[str]) -> pl.LazyFrame:
    data = as_lazy(chars)
    absent = set(META + features) - set(data.collect_schema().names())
    if absent:
        raise ValueError(f"Missing input columns: {sorted(absent)}")
    return data.select(
        pl.col("id").cast(pl.Int64),
        pl.col("eom", "eom_ret").cast(pl.Date),
        pl.col("excntry").cast(pl.String),
        pl.col("ctff_test").cast(pl.String).str.to_lowercase().replace_strict(
            {"0": False, "1": True, "false": False, "true": True},
            default=None, return_dtype=pl.Boolean,
        ),
        pl.col("ret_exc_lead1m").cast(pl.Float64).fill_nan(None),
        pl.col(features).cast(pl.Float64).fill_nan(None),
    )


def prepare_pred_data(data: pl.DataFrame, features: list[str], min_obs=None) -> pl.DataFrame:
    """Match R: maximum tie rank / nonmissing count, zero override, center, impute."""
    groups = ["excntry", "eom"]
    expressions = []
    for name in features:
        x = pl.col(name).cast(pl.Float64).fill_nan(None)
        count = x.count().over(groups)
        ranked = x.rank(method="max").over(groups) / count
        centered = pl.when(x == 0).then(-0.5).otherwise(ranked - 0.5)
        if min_obs is not None:
            centered = pl.when(count >= min_obs).then(centered).otherwise(None)
        expressions.append(centered.fill_null(0.0).alias(name))
    return data.with_columns(expressions).sort(["id", "eom"])


def month_number(d: date) -> int:
    return d.year * 12 + d.month - 1


def month_end(number: int) -> date:
    year, zero_month = divmod(number, 12)
    following = date(year + (zero_month == 11), zero_month % 12 + 2 if zero_month < 11 else 1, 1)
    return date.fromordinal(following.toordinal() - 1)


@dataclass
class SufficientStats:
    n: int
    sx: np.ndarray
    sy: float
    xx: np.ndarray
    xy: np.ndarray
    yy: float

    @classmethod
    def empty(cls, p):
        return cls(0, np.zeros(p), 0.0, np.zeros((p, p)), np.zeros(p), 0.0)

    def accumulate(self, other):
        self.n += other.n
        self.sx += other.sx
        self.sy += other.sy
        self.xx += other.xx
        self.xy += other.xy
        self.yy += other.yy

    def minus(self, other):
        return SufficientStats(self.n - other.n, self.sx - other.sx, self.sy - other.sy,
                               self.xx - other.xx, self.xy - other.xy, self.yy - other.yy)


def sufficient_stats(data: pl.DataFrame, features: list[str]) -> SufficientStats:
    data = data.filter(pl.col("ret_exc_lead1m").is_finite())
    if not data.height:
        return SufficientStats.empty(len(features))
    x = data.select(features).to_numpy(order="c")
    y = data["ret_exc_lead1m"].to_numpy()
    return SufficientStats(len(y), x.sum(axis=0), float(y.sum()), x.T @ x,
                           x.T @ y, float(y @ y))


class MonthlySource:
    """Materialize one monthly cross-section and cache its training statistics."""
    def __init__(self, chars, features, cache_size=144):
        self.features = features
        self.frame = canonical_chars(chars, features)
        self.cache_size = cache_size
        self.stats_cache = OrderedDict()
        self.metadata = self.frame.select(META).collect(engine="streaming").sort(["eom", "id"])
        required = self.metadata.select("id", "eom", "eom_ret", "excntry", "ctff_test")
        if required.null_count().to_numpy().sum():
            raise ValueError("Missing identifiers, dates, countries, or invalid test flags")
        if self.metadata.select(pl.struct("id", "eom").is_duplicated().any()).item():
            raise ValueError("Duplicate monthly security-date keys")
        invalid_dates = self.metadata.filter(
            (pl.col("eom") != pl.col("eom").dt.month_end())
            | (pl.col("eom_ret") != pl.col("eom").dt.offset_by("1mo").dt.month_end())
        )
        if invalid_dates.height:
            raise ValueError("Formation dates and next-month return dates are misaligned")
        self.return_to_formation = dict(
            self.metadata.select("eom_ret", "eom").unique().iter_rows()
        )

    def load_raw(self, formation_date: date) -> pl.DataFrame:
        return self.frame.filter(pl.col("eom") == formation_date).collect(engine="streaming")

    def load(self, formation_date: date) -> pl.DataFrame:
        return prepare_pred_data(self.load_raw(formation_date), self.features)

    def stats(self, return_date: date) -> SufficientStats:
        if return_date in self.stats_cache:
            self.stats_cache.move_to_end(return_date)
            return self.stats_cache[return_date]
        result = sufficient_stats(self.load(self.return_to_formation[return_date]), self.features)
        self.stats_cache[return_date] = result
        while len(self.stats_cache) > self.cache_size:
            self.stats_cache.popitem(last=False)
        return result


def fit_ridge(stats: SufficientStats, penalty: float):
    """Minimize mean squared error / 2 + lambda * squared coefficients / 2.

    The intercept is unpenalized. Equivalent sklearn alpha is n * lambda.
    """
    if stats.n == 0:
        return 0.0, np.zeros(len(stats.sx))
    centered_xx = stats.xx - np.outer(stats.sx, stats.sx) / stats.n
    centered_xy = stats.xy - stats.sx * stats.sy / stats.n
    matrix = (centered_xx + centered_xx.T) / 2
    matrix[np.diag_indices_from(matrix)] += stats.n * penalty
    coef = cho_solve(cho_factor(matrix, lower=True, check_finite=False), centered_xy, check_finite=False)
    intercept = float((stats.sy - stats.sx @ coef) / stats.n)
    return intercept, coef


def validation_mse(stats, intercept, coef):
    sse = (stats.yy - 2 * intercept * stats.sy - 2 * coef @ stats.xy
           + stats.n * intercept**2 + 2 * intercept * coef @ stats.sx
           + coef @ stats.xx @ coef)
    return max(0.0, float(sse)) / stats.n


def train_ridge(source, dates, settings):
    p = len(source.features)
    total = SufficientStats.empty(p)
    usable = []
    for d in dates:
        stats = source.stats(d)
        total.accumulate(stats)
        if stats.n:
            usable.append(d)
    fold_count = min(settings.folds, len(usable))
    folds = [SufficientStats.empty(p) for _ in range(fold_count)]
    # R cut(seq_along(train_dates), folds): contiguous, approximately equal blocks.
    for i, d in enumerate(usable):
        block = min(fold_count - 1, int(i * fold_count / max(1, len(usable) - 1)))
        folds[block].accumulate(source.stats(d))
    scores = []
    for penalty in settings.ridge_lambdas:
        errors = []
        for fold in folds if fold_count > 1 else []:
            train = total.minus(fold)
            if train.n:
                intercept, coef = fit_ridge(train, penalty)
                errors.append(validation_mse(fold, intercept, coef))
        scores.append(float(np.mean(errors)) if errors else None)
    selected = min(range(len(scores)), key=lambda i: (scores[i], i)) if scores[0] is not None else 0
    penalty = settings.ridge_lambdas[selected]
    intercept, coef = fit_ridge(total, penalty)
    return intercept, coef, {"train_rows": total.n, "train_months": len(usable),
                            "folds": fold_count, "ridge_lambda": penalty,
                            "cv_mse": scores[selected], "grid_cv_mse": scores}


# Section 3: Portfolio construction --------------------------------------------
def predictions_to_weights(preds: pl.DataFrame, settings=DEFAULT_SETTINGS) -> pl.DataFrame:
    grouped = ["excntry", "eom"]
    preds = preds.with_columns(
        pl.col("pred").quantile(1 / settings.n_pfs, interpolation="linear").over(grouped).alias("low"),
        pl.col("pred").quantile(1 - 1 / settings.n_pfs, interpolation="linear").over(grouped).alias("high"),
    ).with_columns(
        pl.when(pl.col("pred") <= pl.col("low")).then(-1)
        .when(pl.col("pred") >= pl.col("high")).then(1).otherwise(0).alias("side")
    ).with_columns(
        pl.len().over("eom", "side").alias("n_side"),
        (pl.col("side") == 1).sum().over("eom").alias("n_long"),
        (pl.col("side") == -1).sum().over("eom").alias("n_short"),
    ).with_columns((pl.col("side") / pl.col("n_side")).alias("w"))
    if settings.cash_on_degenerate:
        preds = preds.with_columns(
            pl.when((pl.col("n_long") == 0) | (pl.col("n_short") == 0))
            .then(0.0).otherwise(pl.col("w")).alias("w")
        )
    return preds.select("id", "eom", "w").sort(["eom", "id"])


def finalize_output(weights: pl.DataFrame, started):
    out = weights.select(pl.col("id").cast(pl.Int64), pl.col("eom").cast(pl.Date), pl.col("w").cast(pl.Float64))
    if not out.height or out.null_count().to_numpy().sum():
        raise ValueError("Output must be nonempty with no missing values")
    if not out["w"].is_finite().all():
        raise ValueError("Nonfinite portfolio weights")
    if out.select(pl.struct("id", "eom").is_duplicated().any()).item():
        raise ValueError("Duplicate output security-date keys")
    print(f"Output: {out.height:,} rows, {out['eom'].n_unique()} months; "
          f"runtime {(perf_counter() - started) / 60:.1f} minutes", flush=True)
    return out


# Section 4: Main entry point --------------------------------------------------
def run_pipeline(chars, features, settings=DEFAULT_SETTINGS, *, source_factory=MonthlySource):
    started = perf_counter()
    names = canonical_features(features)
    source = source_factory(chars, names, settings.train_years * 12 + 24)
    test_dates = sorted(source.metadata.filter(pl.col("ctff_test"))["eom_ret"].unique().to_list())
    if not test_dates:
        raise ValueError("No test dates in supplied data")
    if settings.test_period_length < 1 or settings.folds < 2 or settings.n_pfs < 2:
        raise ValueError("Invalid chunk, fold, or portfolio settings")
    if not settings.ridge_lambdas or any(p <= 0 for p in settings.ridge_lambdas):
        raise ValueError("Ridge penalties must be strictly positive")
    predictions, weights, fits = [], [], []
    with threadpool_limits(limits=settings.blas_threads):
        for offset in range(0, len(test_dates), settings.test_period_length):
            chunk = test_dates[offset:offset + settings.test_period_length]
            first = chunk[0]
            first_month = month_number(first) - settings.train_years * 12
            train_dates = sorted(d for d in source.return_to_formation
                                 if first_month <= month_number(d) < month_number(first))
            intercept, coef, fit = train_ridge(source, train_dates, settings)
            fit.update({"test_return_start": str(first), "test_return_end": str(chunk[-1]),
                        "training_return_first": str(train_dates[0]) if train_dates else None,
                        "training_return_last": str(train_dates[-1]) if train_dates else None})
            fits.append(fit)
            print(f"Ridge: {first} to {chunk[-1]}, {fit['train_rows']:,} training rows, "
                  f"lambda {fit['ridge_lambda']:g}", flush=True)
            for d in chunk:
                data = source.load(source.return_to_formation[d]).filter(pl.col("ctff_test"))
                pred = data.select(names).to_numpy(order="c") @ coef + intercept
                preds = data.select("id", "eom", "excntry").with_columns(pl.Series("pred", pred))
                if not preds["pred"].is_finite().all():
                    raise ValueError("Nonfinite return forecasts")
                predictions.append(preds)
                weights.append(predictions_to_weights(preds, settings))
    output = finalize_output(pl.concat(weights), started)
    expected = source.metadata.filter(pl.col("ctff_test")).select("id", "eom")
    if output.height != expected.height or output.join(expected, on=["id", "eom"], how="anti").height:
        raise ValueError("Output does not cover every test security-date")
    return output, pl.concat(predictions), fits


def main(chars: pd.DataFrame, features: pd.DataFrame, daily_ret: pd.DataFrame) -> pd.DataFrame:
    """CTF interface. Daily returns are accepted for later risk-model extensions.

    This initial rank-portfolio baseline uses monthly data only. No files,
    network connections, saved models, or test targets are read by main().
    """
    weights, _, _ = run_pipeline(chars, features)
    result = weights.to_pandas()
    result["eom"] = result["eom"].dt.date
    return result


# Section 5: Local execution and evaluation live in run_baseline.py. ------------
