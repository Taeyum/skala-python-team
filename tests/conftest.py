"""테스트 공용 fixture.

실제 70MB parquet 대신, 정상값+이상치가 섞인 작은 합성 데이터로
정제 로직·Pandas/Polars 동등성을 빠르고 결정적으로 검증한다.
"""

import pandas as pd
import pytest


@pytest.fixture
def sample_trip_df() -> pd.DataFrame:
    pickup = pd.to_datetime(
        [
            "2026-05-01 08:00:00",  # 정상 (혼잡)
            "2026-05-01 09:30:00",  # 정상 (혼잡)
            "2026-05-01 14:00:00",  # 정상 (비혼잡)
            "2026-05-02 18:00:00",  # 정상 (혼잡)
            "2026-05-02 23:00:00",  # 정상 (비혼잡)
            "2026-05-03 12:00:00",  # 이상치: fare<=0
            "2026-05-03 13:00:00",  # 이상치: distance<=0
            "2026-04-30 12:00:00",  # 이상치: 날짜범위 밖
        ]
    )
    dropoff = pickup + pd.to_timedelta([10, 15, 20, 25, 12, 10, 10, 10], unit="m")
    return pd.DataFrame(
        {
            "tpep_pickup_datetime": pickup,
            "tpep_dropoff_datetime": dropoff,
            "trip_distance": [2.0, 3.5, 1.2, 4.0, 1.8, 2.0, -1.0, 2.0],
            "fare_amount": [10.0, 15.0, 8.0, 20.0, 9.0, -5.0, 10.0, 10.0],
            "total_amount": [12.5, 18.0, 9.5, 24.0, 11.0, -5.0, 12.0, 12.0],
            "passenger_count": [1, 2, 1, 1, 3, 1, 1, 1],
        }
    )


@pytest.fixture
def sample_trip_parquet(tmp_path, sample_trip_df) -> str:
    path = tmp_path / "sample_trips.parquet"
    sample_trip_df.to_parquet(path)
    return str(path)
