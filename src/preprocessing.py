# 작성자 : 이다은
# 작성일 : 2026-07-21
"""결측치/중복 제거 및 분석용 파생 컬럼(운행시간, 혼잡여부, 고요금 타겟) 생성."""

import pandas as pd

from src.config import DISTANCE_COL, DROPOFF_COL, FARE_COL, PICKUP_COL, RUSH_HOURS, TARGET, TOTAL_COL


def clean_and_engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """결측치·중복을 제거하고, 분석에 필요한 파생 컬럼을 추가한다."""
    before_rows = len(df)

    # 분석에 반드시 필요한 컬럼이 비어 있는 행은 제거
    required_cols = [PICKUP_COL, DROPOFF_COL, DISTANCE_COL, FARE_COL, TOTAL_COL]
    df = df.dropna(subset=required_cols).drop_duplicates().copy()

    # 운행 시간(분) 계산
    df[PICKUP_COL] = pd.to_datetime(df[PICKUP_COL])
    df[DROPOFF_COL] = pd.to_datetime(df[DROPOFF_COL])
    df["trip_duration_min"] = (
        df[DROPOFF_COL] - df[PICKUP_COL]
    ).dt.total_seconds() / 60

    # 비정상 값 제거: 거리/시간/요금이 0 이하이거나 비정상적으로 큰 이상치성 행
    df = df[
        (df[DISTANCE_COL] > 0)
        & (df["trip_duration_min"] > 0)
        & (df["trip_duration_min"] < 180)  # 3시간 이상 운행은 이상치로 간주
        & (df[FARE_COL] > 0)
        & (df[TOTAL_COL] > 0)
    ].copy()

    df = df[
        (df[DISTANCE_COL] > 0)
        & (df[DISTANCE_COL] < 100)  # 100마일 초과는 GPS 오류 등 이상치로 간주
        & (df["trip_duration_min"] > 0)
        & (df["trip_duration_min"] < 180)
        & (df[FARE_COL] > 0)
        & (df[TOTAL_COL] > 0)
    ].copy()

    if df.empty:
        raise ValueError("전처리 후 남은 데이터가 없습니다. 원본 데이터를 확인하세요.")

    # 파생 변수: 승차 시각(시), 혼잡 시간대 여부
    df["pickup_hour"] = df[PICKUP_COL].dt.hour
    df["is_rush_hour"] = df["pickup_hour"].isin(RUSH_HOURS)

    # 타겟 변수: 총요금 상위 25% 이상이면 고요금(1), 아니면 0
    high_fare_threshold = df[TOTAL_COL].quantile(0.75)
    df[TARGET] = (df[TOTAL_COL] >= high_fare_threshold).astype(int)

    print(f"전처리 전: {before_rows:,}행 -> 전처리 후: {len(df):,}행")
    print(f"고요금 기준(상위 25%): 총요금 {high_fare_threshold:,.2f} 이상")

    return df
