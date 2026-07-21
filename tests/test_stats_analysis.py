import numpy as np
import pandas as pd
import pytest

from src.stats_analysis import congestion_ttest, correlation_matrix


def test_correlation_matrix_supports_hypothesis1():
    rng = np.random.default_rng(0)
    distance = rng.uniform(1, 10, 100)
    duration = rng.uniform(5, 30, 100)
    total = distance * 10 + duration * 2 + rng.normal(0, 0.5, 100)
    df = pd.DataFrame({"trip_distance": distance, "trip_duration": duration, "total_amount": total})

    result = correlation_matrix(df)
    assert result["hypothesis1_supported"] is True
    assert result["distance_corr"] > 0.5


def test_correlation_matrix_rejects_hypothesis1_when_weak():
    rng = np.random.default_rng(1)
    df = pd.DataFrame(
        {
            "trip_distance": rng.uniform(1, 10, 100),
            "trip_duration": rng.uniform(5, 30, 100),
            "total_amount": rng.uniform(10, 50, 100),
        }
    )
    result = correlation_matrix(df)
    assert result["hypothesis1_supported"] is False


def test_correlation_matrix_missing_column_raises():
    df = pd.DataFrame({"trip_distance": [1.0]})
    with pytest.raises(KeyError):
        correlation_matrix(df)


def test_congestion_ttest_supports_hypothesis2():
    pickup = pd.to_datetime(
        [f"2026-05-01 {h:02d}:00:00" for h in [8, 8, 9, 9, 17, 17, 18, 18]]
        + [f"2026-05-01 {h:02d}:00:00" for h in [2, 2, 3, 3, 13, 13, 14, 14]]
    )
    total_amount = [50, 55, 52, 58, 51, 53, 56, 54] + [10, 12, 11, 9, 13, 10, 11, 12]
    df = pd.DataFrame({"tpep_pickup_datetime": pickup, "total_amount": total_amount})

    result = congestion_ttest(df)
    assert result["significant"] is True
    assert result["hypothesis2_supported"] is True


def test_congestion_ttest_rejects_hypothesis2_when_no_difference():
    pickup = pd.to_datetime(
        [f"2026-05-01 {h:02d}:00:00" for h in [8, 9, 10, 17, 18, 19]] * 3
        + [f"2026-05-01 {h:02d}:00:00" for h in [1, 2, 3, 4, 5, 6]] * 3
    )
    total_amount = [30] * 18 + [30] * 18
    df = pd.DataFrame({"tpep_pickup_datetime": pickup, "total_amount": total_amount})

    result = congestion_ttest(df)
    assert result["hypothesis2_supported"] is False


def test_congestion_ttest_missing_column_raises():
    df = pd.DataFrame({"total_amount": [1.0]})
    with pytest.raises(KeyError):
        congestion_ttest(df)


def test_congestion_ttest_insufficient_sample_raises():
    df = pd.DataFrame(
        {
            "tpep_pickup_datetime": pd.to_datetime(["2026-05-01 08:00", "2026-05-01 02:00"]),
            "total_amount": [10.0, 20.0],
        }
    )
    with pytest.raises(ValueError):
        congestion_ttest(df)
