# 작성자 : 이다은
# 작성일 : 2026-07-21
"""상관분석(가설 1)과 혼잡/비혼잡 시간대 t-test(가설 2)."""

import pandas as pd
from scipy.stats import ttest_ind

from src.config import DISTANCE_COL, TOTAL_COL


def compute_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """운행 거리·운행 시간·총요금 간 상관계수를 계산한다 (가설 1 검증용)."""
    print("\n=== 상관분석: 거리 / 운행시간 / 총요금 ===")
    corr_cols = [DISTANCE_COL, "trip_duration_min", TOTAL_COL]
    corr = df[corr_cols].corr()
    print(corr)
    return corr


def run_ttest(df: pd.DataFrame) -> dict:
    """혼잡 시간대(Group A) vs 비혼잡 시간대(Group B) 평균 총요금 t-test (가설 2 검증용)."""
    print("\n=== t-test: 혼잡 시간대 vs 비혼잡 시간대 평균 총요금 ===")
    group_a = df.loc[df["is_rush_hour"], TOTAL_COL]
    group_b = df.loc[~df["is_rush_hour"], TOTAL_COL]

    if len(group_a) < 2 or len(group_b) < 2:
        raise ValueError("t-test를 수행하기에 두 그룹 중 하나 이상의 표본이 부족합니다.")

    # 두 집단의 표본 크기와 분산이 다를 수 있어 Welch's t-test(equal_var=False) 사용
    t_stat, p_value = ttest_ind(group_a, group_b, equal_var=False)

    print(f"혼잡 시간대 평균: {group_a.mean():,.2f} / 비혼잡 시간대 평균: {group_b.mean():,.2f}")
    print(f"t-statistic: {t_stat:.4f}, p-value: {p_value:.4g}")

    if p_value < 0.05:
        interpretation = "혼잡/비혼잡 시간대의 평균 총요금은 통계적으로 유의미하게 다르다 (p < 0.05)"
    else:
        interpretation = "혼잡/비혼잡 시간대의 평균 총요금 차이는 통계적으로 유의미하지 않다 (p >= 0.05)"
    print(f"결론: {interpretation}")

    return {
        "group_a_mean": group_a.mean(),
        "group_b_mean": group_b.mean(),
        "t_stat": t_stat,
        "p_value": p_value,
        "interpretation": interpretation,
    }
