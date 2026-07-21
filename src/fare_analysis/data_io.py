"""Pandas vs Polars 데이터 로딩·전처리 성능 비교"""

import timeit

import pandas as pd
import polars as pl

from . import config


def load_pandas(path=config.RAW_DATA_PATH):
    return pd.read_parquet(path)


def load_polars(path=config.RAW_DATA_PATH):
    return pl.read_parquet(path)


def preprocess_pandas(df):
    """결측치·중복·이상치를 제거하고 파생 컬럼(운행 시간, 승차 시각)을 추가"""
    df = df.drop_duplicates()
    df = df.dropna(subset=config.CORE_COLUMNS)

    duration_min = (df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]).dt.total_seconds() / 60
    df = df.assign(trip_duration_min=duration_min, pickup_hour=df["tpep_pickup_datetime"].dt.hour)

    df = df[
        df["trip_distance"].between(config.MIN_TRIP_DISTANCE, config.MAX_TRIP_DISTANCE)
        & df["fare_amount"].gt(0)
        & df["total_amount"].gt(0)
        & df["trip_duration_min"].between(config.MIN_TRIP_DURATION_MIN, config.MAX_TRIP_DURATION_MIN)
    ]
    return df.reset_index(drop=True)


def preprocess_polars(df):
    """Pandas와 동일한 조건으로 결측치·중복·이상치 제거 및 파생 컬럼 추가 (Polars eager API)"""
    df = df.unique()
    df = df.drop_nulls(subset=config.CORE_COLUMNS)

    df = df.with_columns([
        ((pl.col("tpep_dropoff_datetime") - pl.col("tpep_pickup_datetime")).dt.total_seconds() / 60)
        .alias("trip_duration_min"),
        pl.col("tpep_pickup_datetime").dt.hour().alias("pickup_hour"),
    ])

    return df.filter(
        pl.col("trip_distance").is_between(config.MIN_TRIP_DISTANCE, config.MAX_TRIP_DISTANCE)
        & (pl.col("fare_amount") > 0)
        & (pl.col("total_amount") > 0)
        & pl.col("trip_duration_min").is_between(config.MIN_TRIP_DURATION_MIN, config.MAX_TRIP_DURATION_MIN)
    )


def _average_runtime(func, *args, number=config.N_REPEAT):
    """func(*args)를 number회 반복 실행한 평균 소요 시간(초)을 timeit으로 측정"""
    return timeit.timeit(lambda: func(*args), number=number) / number


def compare_pandas_polars(path=config.RAW_DATA_PATH):
    """로딩·전처리 단계별 Pandas vs Polars 평균 소요 시간을 측정해 반환.
    반환값: (timings, 원본 Pandas DataFrame, 전처리 완료 Pandas DataFrame)"""
    pandas_load_sec = _average_runtime(load_pandas, path)
    polars_load_sec = _average_runtime(load_polars, path)

    pdf = load_pandas(path)
    pldf = load_polars(path)

    pandas_prep_sec = _average_runtime(preprocess_pandas, pdf)
    polars_prep_sec = _average_runtime(preprocess_polars, pldf)

    timings = {
        "load": {"Pandas": pandas_load_sec, "Polars": polars_load_sec},
        "preprocess": {"Pandas": pandas_prep_sec, "Polars": polars_prep_sec},
    }
    print("=== Pandas vs Polars 평균 소요 시간(초) ===")
    for stage, tool_times in timings.items():
        print(f"  {stage}: " + ", ".join(f"{tool}={sec:.4f}" for tool, sec in tool_times.items()))

    return timings, pdf, preprocess_pandas(pdf)
