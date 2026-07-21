# 프로그램 : 뉴욕 옐로 택시 데이터 분석
# 작성자 : 김지영
# 작성일 : 2026-07-21
# 기능 :
# 1. Pandas vs Polars 처리 성능 비교
# 2. 기본 EDA 및 기술통계
# 3. 시간대별 평균 총요금 시각화
# 4. 거리·시간·총요금 상관분석
# 5. 혼잡/비혼잡 시간대 평균 총요금 t-test
# 변경이력 :
# 2026-07-21 | 최초 작성 및 데이터 분석·성능 비교·예외 처리 기능 구현
# ==========================================

from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use("Agg")  # 그래프 창 대신 이미지 파일로 저장

import pandas as pd
import polars as pl
import matplotlib.pyplot as plt

from src.io import ensure_data, load_pandas, load_polars
from src.preprocess import preprocess_pandas, preprocess_polars
from src.analysis import (
    descriptive_stats,
    correlation_and_save,
    ttest_rush,
)
from src.viz import save_performance_plot, save_hourly_plot
from src.viz import save_seaborn_distribution, save_correlation_heatmap
from src.report import generate_report

from src.model import train_fare_model

# ==========================
# 한글 폰트 설정 (macOS)
# ==========================

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

# 기본 설정
DATA_URL = (
    "https://d37ci6vzurychx.cloudfront.net/"
    "trip-data/yellow_tripdata_2026-05.parquet"
)
DATA_PATH = Path("yellow_tripdata_2026-05.parquet")
OUTPUT_DIR = Path("analysis_output")
OUTPUT_DIR.mkdir(exist_ok=True)

ANALYSIS_COLUMNS = [
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "trip_distance",
    "fare_amount",
    "total_amount",
]


# 1. 데이터 파일 준비
# 다운로드 시간은 Pandas와 Polars의 로딩 성능 비교에서 제외합니다.
print("[1/10] 데이터 파일을 준비합니다.")
if not DATA_PATH.exists():
    print("데이터 파일을 다운로드합니다.")
ensure_data(DATA_PATH, DATA_URL)


# 2. Pandas와 Polars 데이터 로딩 시간 비교
print("\n[2/10] 데이터를 불러옵니다.")
pandas_df, pandas_load_time = load_pandas(DATA_PATH, ANALYSIS_COLUMNS)
polars_df, polars_load_time = load_polars(DATA_PATH, ANALYSIS_COLUMNS)
print(f"Pandas 로딩 시간: {pandas_load_time:.3f}초")
print(f"Polars 로딩 시간: {polars_load_time:.3f}초")
print(f"원본 행 수: {len(pandas_df):,}")


# 3. 결측치·중복 처리 및 파생변수 생성
print("\n[3/10] 데이터를 전처리합니다.")

pandas_missing_before = pandas_df.isna().sum()
pandas_duplicates_before = pandas_df.duplicated().sum()

(
    pandas_clean,
    pandas_preprocess_time,
    low,
    high,
    iqr_removed_count,
) = preprocess_pandas(pandas_df)
polars_clean, polars_preprocess_time = preprocess_polars(polars_df, low, high)

print(f"IQR 기준 이동 거리 정상 범위: {low:.3f} ~ {high:.3f}")
print(f"IQR로 제거된 이동 거리 이상치 수: {iqr_removed_count:,}")
print(f"Pandas 전처리 시간: {pandas_preprocess_time:.3f}초")
print(f"Polars 전처리 시간: {polars_preprocess_time:.3f}초")
print(f"전처리 후 행 수(Pandas): {len(pandas_clean):,}")
print(f"전처리 후 행 수(Polars): {polars_clean.height:,}")

print("\n전처리 전 컬럼별 결측치 수")
print(pandas_missing_before.to_string())
print(f"\n전처리 전 중복 행 수: {pandas_duplicates_before:,}")
print(f"전처리 후 결측치 총합: {pandas_clean.isna().sum().sum():,}")
print(f"전처리 후 중복 행 수: {pandas_clean.duplicated().sum():,}")


# 4. 기본 EDA 및 기술통계
print("\n[4/10] 기본 EDA와 기술통계를 확인합니다.")
desc_stats = descriptive_stats(pandas_clean, OUTPUT_DIR)
print("\n기술통계")
print(desc_stats.round(3).to_string())


# 5. 시간대별 집계 및 Pandas vs Polars 성능 비교
print("\n[5/10] 시간대별 평균 총요금을 계산합니다.")

start = perf_counter()
pandas_hourly = (
    pandas_clean
    .groupby("pickup_hour", as_index=False)["total_amount"]
    .mean()
    .rename(columns={"total_amount": "average_total_amount"})
    .sort_values("pickup_hour")
)
pandas_groupby_time = perf_counter() - start

start = perf_counter()
polars_hourly = (
    polars_clean
    .group_by("pickup_hour")
    .agg(
        pl.col("total_amount")
        .mean()
        .alias("average_total_amount")
    )
    .sort("pickup_hour")
)
polars_groupby_time = perf_counter() - start

print(f"Pandas groupby 시간: {pandas_groupby_time:.3f}초")
print(f"Polars groupby 시간: {polars_groupby_time:.3f}초")


# 6. 시각화
print("\n[6/10] 그래프를 저장합니다.")

performance_df = pd.DataFrame(
    {
        "작업": [
            "데이터 로딩", "데이터 로딩",
            "전처리", "전처리",
            "시간대별 집계", "시간대별 집계",
        ],
        "도구": [
            "Pandas", "Polars",
            "Pandas", "Polars",
            "Pandas", "Polars",
        ],
        "소요 시간(초)": [
            pandas_load_time,
            polars_load_time,
            pandas_preprocess_time,
            polars_preprocess_time,
            pandas_groupby_time,
            polars_groupby_time,
        ],
    }
)

save_performance_plot(performance_df, OUTPUT_DIR)
performance_df.to_csv(OUTPUT_DIR / "performance_comparison.csv", index=False, encoding="utf-8-sig")

save_hourly_plot(pandas_hourly, OUTPUT_DIR)
pandas_hourly.to_csv(OUTPUT_DIR / "hourly_average_total_amount.csv", index=False, encoding="utf-8-sig")

# Seaborn static charts
# seaborn plots will be generated after correlation and t-test are computed


# 7. 상관분석
print("\n[7/10] 변수 간 상관계수를 계산합니다.")
correlation_df = correlation_and_save(pandas_clean, OUTPUT_DIR)
print("\n상관계수")
print(correlation_df.round(3).to_string())


# 8. 독립표본 t-test
print("\n[8/10] 혼잡 시간대와 비혼잡 시간대의 평균 총요금을 비교합니다.")
ttest_info = ttest_rush(pandas_clean, OUTPUT_DIR)

print(f"\n혼잡 시간대 표본 수: {ttest_info['rush_n']:,}")
print(f"비혼잡 시간대 표본 수: {ttest_info['non_rush_n']:,}")
print(f"혼잡 시간대 평균 총요금: ${ttest_info['rush_mean']:.3f}")
print(f"비혼잡 시간대 평균 총요금: ${ttest_info['non_rush_mean']:.3f}")
print(f"t 통계량: {ttest_info['t_statistic']:.4f}")
print(f"p-value: {ttest_info['p_value']:.6g}")

alpha = 0.05
if ttest_info["p_value"] < alpha:
    print("\n해석: p-value가 0.05보다 작으므로 두 집단의 평균 총요금에는")
    print("통계적으로 유의한 차이가 있다고 판단합니다.")
    if ttest_info["rush_mean"] > ttest_info["non_rush_mean"]:
        print("또한 혼잡 시간대의 평균 총요금이 더 높아 가설 2와 같은 방향입니다.")
    else:
        print("다만 비혼잡 시간대의 평균 총요금이 더 높아 가설 2의 방향과는 다릅니다.")
else:
    print("\n해석: p-value가 0.05 이상이므로 두 집단의 평균 총요금에")
    print("유의한 차이가 있다고 보기 어렵습니다.")
# Seaborn static charts (generate after correlation/t-test computed)
save_seaborn_distribution(pandas_clean, OUTPUT_DIR)
save_correlation_heatmap(pandas_clean, OUTPUT_DIR)

# 9. 머신러닝 Pipeline
print("\n[9/10] 총요금 예측 모델을 학습합니다.")

ml_metrics, model_path = train_fare_model(
    pandas_clean,
    OUTPUT_DIR,
)

print(f"MAE: {ml_metrics['MAE']:.3f}")
print(f"MSE: {ml_metrics['MSE']:.3f}")
print(f"R²: {ml_metrics['R2']:.3f}")
print(f"모델 저장 위치: {model_path.resolve()}")


# 10. 자동 리포트 생성
print("\n[10/10] 자동 리포트를 생성합니다.")
report_path = generate_report(
    OUTPUT_DIR,
    desc_stats,
    correlation_df,
    ttest_info,
    performance_df,
    ml_metrics,
    model_path,
)


print("\n분석이 완료되었습니다.")
print(f"결과 파일 위치: {OUTPUT_DIR.resolve()}")
print(f"자동 리포트 생성: {report_path.resolve()}")
