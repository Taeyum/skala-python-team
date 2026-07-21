"""[Day2] 프로젝트 전역 설정값.

각 모듈에 흩어져 있던 임계값·경로 상수를 한 곳에 모은다(영민님 브랜치의
config.py 패턴을 참고). 값을 바꿀 때 이 파일만 보면 되도록 하기 위함이다.
"""

import pandas as pd

# 데이터 경로
DATA_PATH = "data/raw/yellow_tripdata_2026-05.parquet"

# 산출물 경로
CHART_PATHS = {
    "benchmark_bar": "output/benchmark_speed.png",
    "hourly_fare_line": "output/hourly_fare_line.html",
}
MODEL_PATH = "output/model.joblib"
REPORT_PATH = "report.md"

# io_compare: 벤치마크 대상 작업 및 반복 횟수
OPERATIONS = ["로딩", "필터링", "시간대별집계"]
BENCHMARK_NUMBER = 1

# eda: 기술통계 대상 컬럼, 유효 날짜 범위(2026-05)
FARE_COLUMNS = ["fare_amount", "total_amount"]
DATE_MIN = pd.Timestamp("2026-05-01")
DATE_MAX = pd.Timestamp("2026-06-01")

# stats_analysis: 유의수준, 강한 상관 기준, 혼잡시간대, 상관분석 대상 컬럼
ALPHA = 0.05
STRONG_CORR_THRESHOLD = 0.5
RUSH_HOURS = (8, 9, 10, 17, 18, 19)
CORRELATION_COLUMNS = ["trip_distance", "trip_duration", "total_amount"]

# ml_pipeline: is_high_fare 분위수, 피처셋(요금 구성요소는 누수 방지를 위해 절대 포함 안 함)
TARGET_QUANTILE = 0.75
NUMERIC_FEATURES = ["trip_distance", "trip_duration", "passenger_count"]
CATEGORICAL_FEATURES = ["pickup_hour", "pickup_weekday", "payment_type"]
