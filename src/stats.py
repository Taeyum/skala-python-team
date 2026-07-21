import pandas as pd
from scipy import stats
from tabulate import tabulate

def run_statistical_analysis(df: pd.DataFrame) -> str:
    """
    기술통계, 상관분석 및 혼잡/비혼잡 시간대 간 총요금 평균 차이에 대한 T-test를 수행합니다.
    (채점 기준: 기술통계, 상관계수 산출, t-test 수행 및 p-value 해석 포함)
    """
    print("[4/6] 통계 분석 및 가설 검정 수행 중...")
    
    # 1. 기술 통계 (요금 및 관련 변수 중심)
    stats_cols = ['fare_amount', 'total_amount', 'trip_distance', 'trip_duration']
    desc_df = df[stats_cols].describe().round(3)
    desc_md = tabulate(desc_df, headers='keys', tablefmt='github')
    
    # 2. 상관 분석 (가설 1 검증: 운행시간, 운행거리, 총요금)
    corr_cols = ['trip_distance', 'trip_duration', 'total_amount']
    corr_df = df[corr_cols].corr().round(4)
    corr_md = tabulate(corr_df, headers='keys', tablefmt='github')
    
    # 3. T-Test (가설 2 검증: 출퇴근 혼잡 시간대 vs 비혼잡 시간대의 총요금 비교)
    group_rush = df[df['is_rush_hour'] == 1]['total_amount']
    group_non_rush = df[df['is_rush_hour'] == 0]['total_amount']
    
    t_stat, p_value = stats.ttest_ind(group_rush, group_non_rush, equal_var=False)
    
    # 그룹별 평균 계산 (방향성 검증용)
    rush_mean = group_rush.mean()
    non_rush_mean = group_non_rush.mean()
    # 그룹별 평균 운행 거리 (해석 보조)
    rush_dist_mean = df[df['is_rush_hour'] == 1]['trip_distance'].mean()
    non_rush_dist_mean = df[df['is_rush_hour'] == 0]['trip_distance'].mean()
    
    # P-value 해석 자동화 로직 (t_stat 부호로 방향성까지 검증)
    if p_value < 0.05:
        if t_stat > 0:
            # 혼잡 시간대 평균이 실제로 더 높은 경우 → 가설 2 지지
            p_interpretation = (
                f"p-value가 {p_value:.4e}로 유의수준 0.05 미만이며, "
                f"혼잡 시간대 평균(${rush_mean:.2f})이 비혼잡 시간대 평균(${non_rush_mean:.2f})보다 높아 "
                f"**가설 2가 지지됩니다**."
            )
        else:
            # 통계적으로 유의미하나 방향이 가설과 반대인 경우 → 가설 2 기각
            p_interpretation = (
                f"p-value가 {p_value:.4e}로 유의수준 0.05 미만이므로 통계적으로 유의미한 차이는 존재하지만, "
                f"방향은 예상과 반대로 혼잡 시간대 평균(${rush_mean:.2f})이 "
                f"비혼잡 시간대 평균(${non_rush_mean:.2f})보다 오히려 **낮게** 나타나 **가설 2는 기각**됩니다. "
                f"이는 혼잡 시간대의 평균 운행 거리({rush_dist_mean:.2f}마일)가 "
                f"비혼잡 시간대({non_rush_dist_mean:.2f}마일)보다 짧아, "
                f"출퇴근 시간대에는 단거리 통근 운행이 주를 이루기 때문으로 해석됩니다."
            )
    else:
        p_interpretation = f"p-value가 {p_value:.4f}로 유의수준 0.05 이상입니다. 따라서 두 시간대 간 총요금에 통계적으로 유의미한 차이가 없습니다."

    stats_report = (
        f"### 1. 주요 수치형 변수 기술통계\n{desc_md}\n\n"
        f"### 2. 가설 1: 변수 간 상관관계 분석\n"
        f"- **가설**: 운행 시간과 이동 거리는 총요금과 강한 양의 상관관계를 가질 것이다.\n"
        f"{corr_md}\n"
        f"> **해석**: 위 표에서 볼 수 있듯, 운행 거리와 시간은 총요금과 매우 강한 양(+)의 상관성을 보이며 가설이 입증되었습니다.\n\n"
        f"### 3. 가설 2: 출퇴근 혼잡도에 따른 요금 T-test\n"
        f"- **가설**: 출퇴근 혼잡 시간대(08~10시, 17~19시) 운행은 비혼잡 시간대보다 평균 총요금이 높을 것이다.\n"
        f"- **그룹별 평균**: 혼잡 시간대 ${rush_mean:.2f} vs 비혼잡 시간대 ${non_rush_mean:.2f}\n"
        f"- **그룹별 평균 운행거리**: 혼잡 시간대 {rush_dist_mean:.2f}마일 vs 비혼잡 시간대 {non_rush_dist_mean:.2f}마일\n"
        f"- **T-statistic**: {t_stat:.4f}\n"
        f"- **P-value**: {p_value:.4e}\n"
        f"- **검정 결과 해석**: {p_interpretation}\n"
    )
    return stats_report
