"""EDA: 결측치·중복 확인, 기술통계, 고요금(High-Fare) 라벨링"""


def summarize_missing_and_duplicates(df):
    """원본 데이터의 결측치 개수와 중복 행 수를 출력하고 반환"""
    null_counts = df.isnull().sum()
    dup_count = df.duplicated().sum()

    print("--- 결측치 개수 (0건 컬럼 제외) ---")
    print(null_counts[null_counts > 0])
    print(f"\n중복 행 수: {dup_count:,}건")

    return null_counts, dup_count


def descriptive_stats(df, columns=("fare_amount", "total_amount", "trip_distance", "trip_duration_min")):
    """fare_amount·total_amount 등 핵심 컬럼의 평균·표준편차·분위수 등 기술통계 산출"""
    stats_df = df[list(columns)].describe(percentiles=[0.25, 0.5, 0.75]).T
    print("--- 기술통계 (fare_amount / total_amount / trip_distance / trip_duration_min) ---")
    print(stats_df)
    return stats_df


def add_high_fare_label(df, quantile):
    """total_amount 상위 (1-quantile) 비율을 고요금(1)/일반(0)으로 라벨링한 DataFrame과
    기준 임계값(threshold)을 반환"""
    threshold = df["total_amount"].quantile(quantile)
    labeled = df.assign(is_high_fare=(df["total_amount"] >= threshold).astype(int))

    print(f"고요금 기준: total_amount >= {threshold:.2f} (상위 {(1 - quantile) * 100:.0f}%)")
    print(labeled["is_high_fare"].value_counts(normalize=True).rename("비율"))

    return labeled, threshold
