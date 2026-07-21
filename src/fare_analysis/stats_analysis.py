"""상관분석 및 t-test 가설 검정"""

from scipy import stats


def correlation_matrix(df, columns):
    """운행 거리·운행 시간·총요금 간 상관계수 산출 (가설1 검증용)"""
    corr = df[list(columns)].corr()
    print("--- 상관계수 (운행거리 · 운행시간 · 총요금) ---")
    print(corr)
    return corr


def run_rush_hour_ttest(df, rush_hours):
    """출퇴근 혼잡 시간대(Group A) vs 비혼잡 시간대(Group B) 평균 총요금 차이를
    Welch's t-test(등분산 가정 없음)로 검정 (가설2 검증용)"""
    group_a = df.loc[df["pickup_hour"].isin(rush_hours), "total_amount"]
    group_b = df.loc[~df["pickup_hour"].isin(rush_hours), "total_amount"]

    t_stat, p_value = stats.ttest_ind(group_a, group_b, equal_var=False)
    is_significant = p_value < 0.05
    supports_hypothesis = is_significant and group_a.mean() > group_b.mean()

    if not is_significant:
        verdict = "통계적으로 유의미한 차이 없음"
    elif supports_hypothesis:
        verdict = "가설 지지: 혼잡시간대 평균 총요금이 유의미하게 높음"
    else:
        verdict = "가설 기각: 유의미한 차이는 있으나 방향이 반대(혼잡시간대가 오히려 낮음)"

    print(f"[t-test] 혼잡시간대(n={len(group_a):,}, 평균={group_a.mean():.2f}) vs "
          f"비혼잡시간대(n={len(group_b):,}, 평균={group_b.mean():.2f})")
    print(f"  t통계량={t_stat:.4f}, p-value={p_value:.6f} -> {verdict}")

    return {
        "t_stat": t_stat,
        "p_value": p_value,
        "group_a_mean": group_a.mean(),
        "group_b_mean": group_b.mean(),
        "group_a_n": len(group_a),
        "group_b_n": len(group_b),
        "verdict": verdict,
    }
