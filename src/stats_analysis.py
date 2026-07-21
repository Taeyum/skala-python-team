"""[Day2] 통계 분석 — 상관계수 + 출퇴근 혼잡시간대 t-test.

src.eda.clean_trip_data로 정제된 DataFrame(trip_duration 포함)을 입력으로,
1) trip_distance/trip_duration/total_amount 상관계수를 계산해 가설1(운행
시간·거리와 총요금의 강한 양의 상관)을 검증하고,
2) 혼잡시간대(08-10, 17-19시) vs 비혼잡시간대의 평균 총요금을 t-test로
비교해 가설2(혼잡시간대 평균요금이 더 높다)를 검증한다.
"""

import pandas as pd
from scipy import stats

ALPHA = 0.05
STRONG_CORR_THRESHOLD = 0.5
RUSH_HOURS = (8, 9, 10, 17, 18, 19)
CORRELATION_COLUMNS = ["trip_distance", "trip_duration", "total_amount"]


def correlation_matrix(df: pd.DataFrame, columns: list[str] = CORRELATION_COLUMNS) -> dict:
    """상관행렬을 계산하고 가설1(거리·시간과 총요금의 강한 양의 상관)을 판정한다."""
    missing = set(columns) - set(df.columns)
    if missing:
        raise KeyError(f"상관분석에 필요한 컬럼이 없습니다: {sorted(missing)}")

    corr = df[columns].corr()
    dist_corr = corr.loc["trip_distance", "total_amount"]
    dur_corr = corr.loc["trip_duration", "total_amount"]

    hypothesis1_supported = bool(dist_corr >= STRONG_CORR_THRESHOLD and dur_corr >= STRONG_CORR_THRESHOLD)
    message = (
        f"[상관분석] trip_distance-total_amount r={dist_corr:.4f}, "
        f"trip_duration-total_amount r={dur_corr:.4f} -> "
        f"{'가설1 채택(강한 양의 상관 확인)' if hypothesis1_supported else '가설1 기각(강한 양의 상관으로 보기 어려움)'} "
        f"(기준: r>={STRONG_CORR_THRESHOLD})"
    )
    print(message)
    print(corr)

    return {
        "correlation": corr,
        "distance_corr": dist_corr,
        "duration_corr": dur_corr,
        "hypothesis1_supported": hypothesis1_supported,
        "message": message,
    }


def congestion_ttest(
    df: pd.DataFrame, rush_hours: tuple[int, ...] = RUSH_HOURS, column: str = "total_amount"
) -> dict:
    """혼잡시간대(rush_hours) vs 비혼잡시간대의 column 평균을 t-test로 비교하고 가설2를 판정한다."""
    required = {"tpep_pickup_datetime", column}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"t-test에 필요한 컬럼이 없습니다: {sorted(missing)}")

    hour = df["tpep_pickup_datetime"].dt.hour
    group_a = df.loc[hour.isin(rush_hours), column].dropna()
    group_b = df.loc[~hour.isin(rush_hours), column].dropna()
    if len(group_a) < 2 or len(group_b) < 2:
        raise ValueError(f"t-test를 수행하기엔 표본이 부족합니다: 혼잡={len(group_a)}, 비혼잡={len(group_b)}")

    t_stat, p_value = stats.ttest_ind(group_a, group_b, equal_var=False)
    significant = bool(p_value < ALPHA)
    mean_a, mean_b = group_a.mean(), group_b.mean()
    hypothesis2_supported = bool(significant and mean_a > mean_b)

    message = (
        f"[t-test] 혼잡시간대(n={len(group_a)}, mean={mean_a:.2f}) vs "
        f"비혼잡시간대(n={len(group_b)}, mean={mean_b:.2f}): "
        f"t={t_stat:.4f}, p={p_value:.4e} -> "
        f"{'p<0.05, 통계적으로 유의미한 차이 있음' if significant else 'p>=0.05, 통계적으로 유의미한 차이 없음'} | "
        f"{'가설2 채택(혼잡시간대 평균요금이 더 높음)' if hypothesis2_supported else '가설2 기각(혼잡시간대 평균요금이 더 높다고 보기 어려움)'}"
    )
    print(message)

    return {
        "t_stat": t_stat,
        "p_value": p_value,
        "significant": significant,
        "mean_rush": mean_a,
        "mean_non_rush": mean_b,
        "hypothesis2_supported": hypothesis2_supported,
        "message": message,
    }
