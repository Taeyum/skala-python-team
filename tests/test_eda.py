import pandas as pd
import pytest

from src.eda import clean_trip_data, drop_duplicates, fare_descriptive_stats, missing_summary


def test_missing_summary_reports_percentages():
    df = pd.DataFrame({"a": [1, None, 3, None], "b": [1, 2, 3, 4]})
    result = missing_summary(df)
    assert result["a"] == 50.0
    assert result["b"] == 0.0


def test_missing_summary_rejects_empty_df():
    with pytest.raises(ValueError):
        missing_summary(pd.DataFrame())


def test_drop_duplicates_removes_exact_duplicates():
    df = pd.DataFrame({"a": [1, 1, 2], "b": [1, 1, 2]})
    deduped, removed = drop_duplicates(df)
    assert removed == 1
    assert len(deduped) == 2


def test_clean_trip_data_removes_business_rule_violations(sample_trip_df):
    cleaned, stats = clean_trip_data(sample_trip_df)

    assert stats["rows_before"] == 8
    assert stats["rows_after_rules"] == 5  # fare<=0, distance<=0, 날짜범위 밖 3건 제거
    assert stats["rows_after_iqr"] <= stats["rows_after_rules"]
    assert (cleaned["fare_amount"] > 0).all()
    assert (cleaned["total_amount"] > 0).all()
    assert (cleaned["trip_distance"] > 0).all()
    assert cleaned["tpep_pickup_datetime"].min() >= pd.Timestamp("2026-05-01")


def test_clean_trip_data_missing_column_raises():
    df = pd.DataFrame({"fare_amount": [1.0]})
    with pytest.raises(KeyError):
        clean_trip_data(df)


def test_fare_descriptive_stats_returns_describe(sample_trip_df):
    result = fare_descriptive_stats(sample_trip_df)
    assert "fare_amount" in result.columns
    assert "total_amount" in result.columns
    assert "mean" in result.index


def test_fare_descriptive_stats_missing_column_raises():
    df = pd.DataFrame({"fare_amount": [1.0]})
    with pytest.raises(KeyError):
        fare_descriptive_stats(df)
