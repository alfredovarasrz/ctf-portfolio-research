"""Local evaluation, separate from the competition's main() entry point."""
import math

import numpy as np
import polars as pl


def maximum_drawdown(returns):
    wealth = np.cumprod(1 + np.asarray(returns))
    peak = np.maximum.accumulate(np.concatenate(([1.0], wealth)))[1:]
    return float(np.max(1 - wealth / peak))


def evaluate(weights, predictions, raw_metadata):
    keys = ["id", "eom"]
    expected = raw_metadata.filter(pl.col("ctff_test")).select(
        *keys, "eom_ret", "ret_exc_lead1m"
    )
    if weights.height != expected.height or weights.join(expected, on=keys, how="anti").height:
        raise ValueError("Evaluation weights do not cover the exact test universe")
    joined = weights.join(expected, on=keys, validate="1:1")
    if joined.filter(~pl.col("ret_exc_lead1m").is_finite() | pl.col("ret_exc_lead1m").is_null()).height:
        raise ValueError("Test returns must be available and finite for evaluation")
    monthly = joined.group_by("eom", "eom_ret").agg(
        (pl.col("w") * pl.col("ret_exc_lead1m")).sum().alias("ret"),
        pl.col("w").sum().alias("net"),
        pl.col("w").abs().sum().alias("gross"),
        (pl.col("w") > 0).sum().alias("n_long"),
        (pl.col("w") < 0).sum().alias("n_short"),
        pl.len().alias("universe"),
    ).sort("eom")
    date_map = monthly.select("eom").with_columns(pl.col("eom").shift(-1).alias("next_eom"))
    drifted = (
        joined.join(monthly.select("eom", pl.col("ret").alias("pf_ret")), on="eom")
        .join(date_map, on="eom").filter(pl.col("next_eom").is_not_null())
        .select("id", pl.col("next_eom").alias("eom"),
                (pl.col("w") * (1 + pl.col("ret_exc_lead1m")) / (1 + pl.col("pf_ret"))).alias("drift"))
    )
    turnover = (
        weights.join(drifted, on=keys, how="full", coalesce=True)
        .filter(pl.col("eom") > monthly["eom"].min())
        .with_columns(pl.col("w", "drift").fill_null(0))
        .group_by("eom").agg((pl.col("w") - pl.col("drift")).abs().sum().alias("turnover"))
    )
    monthly = monthly.join(turnover, on="eom", how="left").sort("eom")
    prediction_errors = predictions.join(expected, on=keys, validate="1:1").select(
        ((pl.col("ret_exc_lead1m") - pl.col("pred")) ** 2).mean().alias("mse"),
        (pl.col("ret_exc_lead1m") ** 2).mean().alias("zero_mse"),
    ).row(0, named=True) if predictions is not None else None
    annual_mean = monthly["ret"].mean() * 12
    annual_sd = monthly["ret"].std(ddof=1) * math.sqrt(12)
    summary = {
        "formation_start": str(monthly["eom"].min()),
        "formation_end": str(monthly["eom"].max()),
        "return_start": str(monthly["eom_ret"].min()),
        "return_end": str(monthly["eom_ret"].max()),
        "months": monthly.height, "weight_rows": weights.height,
        "mean_annual": annual_mean, "volatility_annual": annual_sd,
        "sharpe_annual": annual_mean / annual_sd if annual_sd else None,
        "average_universe": monthly["universe"].mean(),
        "average_holdings": (monthly["n_long"] + monthly["n_short"]).mean(),
        "gross_leverage": monthly["gross"].mean(),
        "maximum_absolute_net_weight": monthly["net"].abs().max(),
        "turnover_monthly": monthly["turnover"].mean(),
        "maximum_drawdown": maximum_drawdown(monthly["ret"]),
        "maximum_drawdown_scaled_10pct": maximum_drawdown(monthly["ret"] * 0.10 / annual_sd) if annual_sd else None,
        "prediction_mse": prediction_errors["mse"] if prediction_errors else None,
        "prediction_r2_zero": 1 - prediction_errors["mse"] / prediction_errors["zero_mse"] if prediction_errors and prediction_errors["zero_mse"] else None,
        "trading_costs_included": False,
    }
    return monthly, summary
