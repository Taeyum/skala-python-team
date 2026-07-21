from time import perf_counter
import pandas as pd
from scipy.stats import ttest_ind
import polars as pl


def descriptive_stats(pandas_clean: pd.DataFrame, output_dir: str):
    eda_columns = [
        "trip_distance",
        "trip_duration_minutes",
        "fare_amount",
        "total_amount",
    ]
    descriptive_stats = pandas_clean[eda_columns].describe().T
    descriptive_stats = descriptive_stats[
        ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]
    ]
    descriptive_stats.to_csv(
        f"{output_dir}/descriptive_statistics.csv", encoding="utf-8-sig"
    )
    return descriptive_stats


def hourly_averages(pandas_clean: pd.DataFrame, polars_clean: pl.DataFrame):
    start = perf_counter()
    pandas_hourly = (
        pandas_clean.groupby("pickup_hour", as_index=False)["total_amount"].mean()
        .rename(columns={"total_amount": "average_total_amount"})
        .sort_values("pickup_hour")
    )
    pandas_groupby_time = perf_counter() - start

    start = perf_counter()
    polars_hourly = (
        polars_clean.group_by("pickup_hour").agg(pl.col("total_amount").mean().alias("average_total_amount")).sort("pickup_hour")
    )
    polars_groupby_time = perf_counter() - start

    return pandas_hourly, polars_hourly, pandas_groupby_time, polars_groupby_time


def correlation_and_save(pandas_clean: pd.DataFrame, output_dir: str):
    correlation_df = pandas_clean[["trip_distance", "trip_duration_minutes", "total_amount"]].corr(method="pearson")
    correlation_df.to_csv(f"{output_dir}/correlation_matrix.csv", encoding="utf-8-sig")
    return correlation_df


def ttest_rush(pandas_clean: pd.DataFrame, output_dir: str):
    rush_fares = pandas_clean.loc[pandas_clean["is_rush_hour"], "total_amount"]
    non_rush_fares = pandas_clean.loc[~pandas_clean["is_rush_hour"], "total_amount"]

    t_statistic, p_value = ttest_ind(rush_fares, non_rush_fares, equal_var=False, nan_policy="omit")

    rush_mean = rush_fares.mean()
    non_rush_mean = non_rush_fares.mean()

    ttest_result = pd.DataFrame(
        {
            "항목": [
                "혼잡 시간대 평균 총요금",
                "비혼잡 시간대 평균 총요금",
                "t 통계량",
                "p-value",
            ],
            "값": [rush_mean, non_rush_mean, t_statistic, p_value],
        }
    )
    ttest_result.to_csv(f"{output_dir}/ttest_result.csv", index=False, encoding="utf-8-sig")

    return {
        "rush_n": len(rush_fares),
        "non_rush_n": len(non_rush_fares),
        "rush_mean": rush_mean,
        "non_rush_mean": non_rush_mean,
        "t_statistic": t_statistic,
        "p_value": p_value,
    }
