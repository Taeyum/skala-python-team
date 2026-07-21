from src import config


def test_config_exposes_expected_constants():
    assert config.DATA_PATH.endswith(".parquet")
    assert set(config.OPERATIONS) == {"로딩", "필터링", "시간대별집계"}
    assert 0 < config.ALPHA < 1
    assert 0 < config.TARGET_QUANTILE < 1
    assert "trip_distance" in config.NUMERIC_FEATURES
    assert "payment_type" in config.CATEGORICAL_FEATURES
    assert set(config.CHART_PATHS.keys()) == {"benchmark_bar", "hourly_fare_line"}
