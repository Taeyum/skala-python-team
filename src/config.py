# 작성자 : 이다은
# 작성일 : 2026-07-21
"""프로젝트 전역에서 쓰는 경로·상수·컬럼명을 모아두는 설정 모듈."""

from pathlib import Path

# ---------------------------------------------------------------------------
# 경로
# ---------------------------------------------------------------------------
DATA_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2026-05.parquet"
DATA_PATH = Path("data/raw/yellow_tripdata_2026-05.parquet")

OUTPUT_DIR = Path("output")
SPEED_FIG_PATH = OUTPUT_DIR / "pandas_vs_polars_speed.png"
HOURLY_FARE_HTML_PATH = OUTPUT_DIR / "hourly_avg_total_amount.html"
MODEL_PATH = OUTPUT_DIR / "high_fare_pipeline.joblib"
REPORT_PATH = Path("report.md")

# ---------------------------------------------------------------------------
# 원본 데이터 컬럼명 (NYC TLC Yellow Taxi 스키마 기준)
# ---------------------------------------------------------------------------
PICKUP_COL = "tpep_pickup_datetime"
DROPOFF_COL = "tpep_dropoff_datetime"
FARE_COL = "fare_amount"
TOTAL_COL = "total_amount"
DISTANCE_COL = "trip_distance"

# 혼잡 시간대 정의: 출근(08~10시), 퇴근(17~19시) - 양 끝 시각 포함
RUSH_HOURS = {8, 9, 10, 17, 18, 19}

# ML 분류에 사용할 피처 (fare_amount/total_amount는 타겟(is_high_fare)을 만드는
# 재료이므로 피처로 넣으면 정답을 미리 알려주는 데이터 누수가 되어 절대 포함하지 않는다)
NUMERIC_FEATURES = ["trip_distance", "trip_duration_min", "passenger_count"]
CATEGORICAL_FEATURES = ["payment_type"]
TARGET = "is_high_fare"
