from time import perf_counter
import pandas as pd
import polars as pl


def preprocess_pandas(pandas_df: pd.DataFrame):
    start = perf_counter()
    df = pandas_df.copy()
    df["trip_duration_minutes"] = (
        df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]
    ).dt.total_seconds() / 60
    df["pickup_hour"] = df["tpep_pickup_datetime"].dt.hour

    df = df.dropna(
        subset=[
            "trip_distance",
            "fare_amount",
            "total_amount",
            "trip_duration_minutes",
            "pickup_hour",
        ]
    )
    df = df.drop_duplicates()

    Q1 = df["trip_distance"].quantile(0.25)
    Q3 = df["trip_distance"].quantile(0.75)
    IQR = Q3 - Q1
    low = Q1 - 1.5 * IQR
    high = Q3 + 1.5 * IQR

    before_iqr_count = len(df)
    df = df[df["trip_distance"].between(low, high)]
    after_iqr_count = len(df)
    iqr_removed_count = before_iqr_count - after_iqr_count

    df = df[
        (df["trip_distance"] > 0)
        & (df["fare_amount"] >= 0)
        & (df["total_amount"] > 0)
        & (df["trip_duration_minutes"] > 0)
        & (df["trip_duration_minutes"] <= 240)
    ].copy()

    df["is_rush_hour"] = (
        df["pickup_hour"].between(8, 10) | df["pickup_hour"].between(17, 19)
    )

    return df, perf_counter() - start, low, high, iqr_removed_count


def preprocess_polars(polars_df: pl.DataFrame, low: float, high: float):
    start = perf_counter()
    polars_clean = (
        polars_df
        .with_columns(
            (
                (
                    pl.col("tpep_dropoff_datetime") - pl.col("tpep_pickup_datetime")
                ).dt.total_seconds()
                / 60
            ).alias("trip_duration_minutes"),
            pl.col("tpep_pickup_datetime").dt.hour().alias("pickup_hour"),
        )
        .drop_nulls(
            subset=[
                "trip_distance",
                "fare_amount",
                "total_amount",
                "trip_duration_minutes",
                "pickup_hour",
            ]
        )
        .unique()
        .filter(
            (pl.col("trip_distance") > 0)
            & (pl.col("trip_distance") >= low)
            & (pl.col("trip_distance") <= high)
            & (pl.col("fare_amount") >= 0)
            & (pl.col("total_amount") > 0)
            & (pl.col("trip_duration_minutes") > 0)
            & (pl.col("trip_duration_minutes") <= 240)
        )
        .with_columns(
            (
                pl.col("pickup_hour").is_between(8, 10, closed="both")
                | pl.col("pickup_hour").is_between(17, 19, closed="both")
            ).alias("is_rush_hour")
        )
    )
    return polars_clean, perf_counter() - start
