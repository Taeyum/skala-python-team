"""
요금 예측 & Pandas/Polars 성능 비교 프로젝트 공통 설정값

변경내역:
  - 2026-07-21: 최초 작성
"""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = ROOT_DIR / "data" / "raw" / "yellow_tripdata_2026-05.parquet"
OUTPUT_DIR = ROOT_DIR / "output"

# EDA / 전처리 기준값
CORE_COLUMNS = [
    "passenger_count",
    "trip_distance",
    "fare_amount",
    "total_amount",
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
]
MIN_TRIP_DISTANCE = 0.1   # mile
MAX_TRIP_DISTANCE = 100.0
MIN_TRIP_DURATION_MIN = 1.0
MAX_TRIP_DURATION_MIN = 180.0

# 통계/가설 검정 기준값
HIGH_FARE_QUANTILE = 0.75            # total_amount 상위 25% -> is_high_fare=1
RUSH_HOURS = {8, 9, 10, 17, 18, 19}   # 출퇴근 혼잡 시간대(Group A), 그 외는 Group B
CORRELATION_COLUMNS = ["trip_distance", "trip_duration_min", "total_amount"]

# Pandas vs Polars 성능 비교 반복 횟수
N_REPEAT = 3

# ML Pipeline 피처 구성 (요금 구성요소 컬럼은 is_high_fare와 직접 연산되므로 누수 방지를 위해 제외)
NUMERIC_FEATURES = ["trip_distance", "trip_duration_min", "passenger_count", "pickup_hour"]
CATEGORICAL_FEATURES = ["VendorID", "payment_type", "RatecodeID"]
TARGET_COL = "is_high_fare"
