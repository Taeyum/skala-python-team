"""[Day2] 결측치·중복·이상치 처리 + 기초 EDA.

NYC Yellow Taxi(2026-05) 원본 데이터의 결측치 비율, 중복행,
이상치(음수 요금·0거리·날짜 범위 밖 등 비즈니스 규칙 + trip_distance/
trip_duration IQR)를 처리해 이후 시각화·통계·ML 단계의 공통 입력이 될
정제 DataFrame을 만든다.
"""

import pandas as pd

FARE_COLUMNS = ["fare_amount", "total_amount"]
DATE_MIN = pd.Timestamp("2026-05-01")
DATE_MAX = pd.Timestamp("2026-06-01")


def missing_summary(df: pd.DataFrame) -> pd.Series:
    """컬럼별 결측치 비율(%)을 반환한다."""
    if df.empty:
        raise ValueError("빈 데이터프레임은 결측치 요약을 계산할 수 없습니다.")
    return (df.isna().sum() / len(df) * 100).round(2)


def drop_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """완전히 동일한 중복행을 제거하고 (정제 DataFrame, 제거건수)를 반환한다."""
    before = len(df)
    deduped = df.drop_duplicates()
    return deduped, before - len(deduped)


def _iqr_bounds(series: pd.Series) -> tuple[float, float]:
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr


def clean_trip_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """비즈니스 규칙 + IQR로 이상치를 제거하고 trip_duration(분) 컬럼을 추가한다.

    규칙: fare_amount>0, total_amount>0, trip_distance>0, pickup이
    2026-05 범위 내, dropoff>pickup(양의 운행시간), 이후 trip_distance/
    trip_duration에 IQR 상한을 적용해 극단적 이상치(수백 마일 트립 등)를 제거.
    """
    required = {"fare_amount", "total_amount", "trip_distance", "tpep_pickup_datetime", "tpep_dropoff_datetime"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"정제에 필요한 컬럼이 없습니다: {sorted(missing)}")

    before = len(df)
    data = df.copy()
    data["trip_duration"] = (
        data["tpep_dropoff_datetime"] - data["tpep_pickup_datetime"]
    ).dt.total_seconds() / 60

    rule_mask = (
        (data["fare_amount"] > 0)
        & (data["total_amount"] > 0)
        & (data["trip_distance"] > 0)
        & (data["trip_duration"] > 0)
        & data["tpep_pickup_datetime"].between(DATE_MIN, DATE_MAX)
    )
    data = data[rule_mask]
    after_rules = len(data)

    dist_lo, dist_hi = _iqr_bounds(data["trip_distance"])
    dur_lo, dur_hi = _iqr_bounds(data["trip_duration"])
    data = data[
        data["trip_distance"].between(dist_lo, dist_hi) & data["trip_duration"].between(dur_lo, dur_hi)
    ]
    after_iqr = len(data)

    stats = {
        "rows_before": before,
        "rows_after_rules": after_rules,
        "rows_after_iqr": after_iqr,
        "rows_removed": before - after_iqr,
        "trip_distance_bounds": (dist_lo, dist_hi),
        "trip_duration_bounds": (dur_lo, dur_hi),
    }
    print(
        f"[정제] 원본={before} -> 규칙적용={after_rules} -> IQR적용={after_iqr} "
        f"(총 제거={stats['rows_removed']}건)"
    )
    return data, stats


def fare_descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    """fare_amount/total_amount 기술통계를 반환한다."""
    missing = set(FARE_COLUMNS) - set(df.columns)
    if missing:
        raise KeyError(f"기술통계에 필요한 컬럼이 없습니다: {sorted(missing)}")
    return df[FARE_COLUMNS].describe()
