import pytest

from src.io_compare import (
    filter_pandas,
    filter_polars,
    hourly_fare_pandas,
    hourly_fare_polars,
    load_pandas,
    load_polars,
    run_benchmark,
)


def test_load_pandas_and_polars_match_row_count(sample_trip_parquet):
    pdf = load_pandas(sample_trip_parquet)
    pldf = load_polars(sample_trip_parquet)
    assert len(pdf) == len(pldf) == 8


def test_load_pandas_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_pandas("does/not/exist.parquet")


def test_filter_pandas_and_polars_match(sample_trip_df):
    import polars as pl

    pldf = pl.from_pandas(sample_trip_df)

    pd_result = filter_pandas(sample_trip_df)
    pl_result = filter_polars(pldf)

    assert len(pd_result) == len(pl_result) == 5


def test_hourly_fare_pandas_and_polars_match(sample_trip_df):
    import polars as pl

    pldf = pl.from_pandas(sample_trip_df)
    cleaned_pd = filter_pandas(sample_trip_df)
    cleaned_pl = filter_polars(pldf)

    pd_hourly = hourly_fare_pandas(cleaned_pd)
    pl_hourly = hourly_fare_polars(cleaned_pl).sort("pickup_hour").to_pandas()

    pd_map = dict(zip(pd_hourly["pickup_hour"], pd_hourly["avg_total_amount"]))
    pl_map = dict(zip(pl_hourly["pickup_hour"], pl_hourly["avg_total_amount"]))
    for hour, value in pd_map.items():
        assert pl_map[hour] == pytest.approx(value)


def test_run_benchmark_returns_expected_structure(sample_trip_parquet):
    result = run_benchmark(sample_trip_parquet, number=1)

    assert set(result.keys()) == {"timings", "hourly_fare", "pandas_raw", "pandas_clean"}
    assert len(result["timings"]) == 6  # 3개 작업 x 2개 도구
    assert set(result["timings"]["tool"]) == {"pandas", "polars"}
    assert len(result["pandas_clean"]) == 5
