"""[Day2] Pandas vs Polars 로딩·전처리·집계 속도 비교.

동일한 3가지 작업(로딩 / 이상치·결측 필터링 / 시간대별 평균 총요금 집계)을
Pandas와 Polars 각각으로 구현해 timeit으로 소요시간을 측정한다. 필터링
임계값은 src.eda.clean_trip_data와 동일한 규칙을 사용해 두 도구의 결과가
동등하도록 맞춘다.
"""

import timeit
from pathlib import Path

import pandas as pd
import polars as pl

from src.config import DATE_MAX, DATE_MIN, OPERATIONS


def load_pandas(path: str | Path) -> pd.DataFrame:
    """Pandas로 parquet 파일을 로딩한다."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {path}")
    return pd.read_parquet(path)


def load_polars(path: str | Path) -> pl.DataFrame:
    """Polars로 parquet 파일을 로딩한다."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {path}")
    return pl.read_parquet(path)


def filter_pandas(df: pd.DataFrame) -> pd.DataFrame:
    """Pandas로 이상치·결측 필터링(요금>0, 거리>0, 날짜범위, 운행시간>0)을 수행한다."""
    duration = (df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]).dt.total_seconds() / 60
    mask = (
        (df["fare_amount"] > 0)
        & (df["total_amount"] > 0)
        & (df["trip_distance"] > 0)
        & (duration > 0)
        & df["tpep_pickup_datetime"].between(DATE_MIN, DATE_MAX)
    )
    return df[mask]


def filter_polars(df: pl.DataFrame) -> pl.DataFrame:
    """Polars로 동일한 이상치·결측 필터링을 수행한다."""
    return df.filter(
        (pl.col("fare_amount") > 0)
        & (pl.col("total_amount") > 0)
        & (pl.col("trip_distance") > 0)
        & ((pl.col("tpep_dropoff_datetime") - pl.col("tpep_pickup_datetime")).dt.total_minutes() > 0)
        & pl.col("tpep_pickup_datetime").is_between(DATE_MIN, DATE_MAX)
    )


def hourly_fare_pandas(df: pd.DataFrame) -> pd.DataFrame:
    """Pandas로 시간대(pickup hour)별 평균 total_amount를 집계한다."""
    hour = df["tpep_pickup_datetime"].dt.hour
    result = df.groupby(hour)["total_amount"].mean().reset_index()
    result.columns = ["pickup_hour", "avg_total_amount"]
    return result.sort_values("pickup_hour").reset_index(drop=True)


def hourly_fare_polars(df: pl.DataFrame) -> pl.DataFrame:
    """Polars로 시간대별 평균 total_amount를 집계한다."""
    return (
        df.with_columns(pl.col("tpep_pickup_datetime").dt.hour().alias("pickup_hour"))
        .group_by("pickup_hour")
        .agg(pl.col("total_amount").mean().alias("avg_total_amount"))
        .sort("pickup_hour")
    )


def run_benchmark(path: str | Path, number: int = 1) -> dict:
    """3개 작업(로딩/필터링/시간대별집계)을 Pandas·Polars 각각 timeit으로 측정한다."""
    rows = []

    t_pd_load = timeit.timeit(lambda: load_pandas(path), number=number) / number
    t_pl_load = timeit.timeit(lambda: load_polars(path), number=number) / number
    rows += [("로딩", "pandas", t_pd_load), ("로딩", "polars", t_pl_load)]

    pdf = load_pandas(path)
    pldf = load_polars(path)

    t_pd_filter = timeit.timeit(lambda: filter_pandas(pdf), number=number) / number
    t_pl_filter = timeit.timeit(lambda: filter_polars(pldf), number=number) / number
    rows += [("필터링", "pandas", t_pd_filter), ("필터링", "polars", t_pl_filter)]

    pdf_clean = filter_pandas(pdf)
    pldf_clean = filter_polars(pldf)

    t_pd_agg = timeit.timeit(lambda: hourly_fare_pandas(pdf_clean), number=number) / number
    t_pl_agg = timeit.timeit(lambda: hourly_fare_polars(pldf_clean), number=number) / number
    rows += [("시간대별집계", "pandas", t_pd_agg), ("시간대별집계", "polars", t_pl_agg)]

    timings = pd.DataFrame(rows, columns=["operation", "tool", "seconds"])
    hourly_fare = hourly_fare_pandas(pdf_clean)

    print(timings.to_string(index=False))

    return {
        "timings": timings,
        "hourly_fare": hourly_fare,
        "pandas_raw": pdf,
        "pandas_clean": pdf_clean,
    }
