"""
요금 예측 & 도구 성능 비교 - 전체 파이프라인 실행

주제: Pandas vs Polars 처리 성능 비교 기반 고요금(High-Fare) 운행 예측
데이터: NYC Yellow Taxi Trip Data (2026-05, data/raw/yellow_tripdata_2026-05.parquet)

전체 흐름:
  1) Pandas vs Polars 로딩·전처리 성능 비교 (data_io)
  2) EDA - 결측치·중복 확인, 기술통계, is_high_fare 라벨링 (eda)
  3) 상관분석 + t-test 가설 검정 (stats_analysis)
  4) Seaborn 처리속도 비교 바 차트 + Plotly 시간대별 요금 라인 차트 (visualization)
  5) sklearn Pipeline으로 is_high_fare 분류 학습·평가·저장 (ml_pipeline)
  6) report.md 자동 생성 (report)

실행: python src/run_fare_analysis.py

변경내역:
  - 2026-07-21: 최초 작성 (실습3+4 통합형 파이프라인)
  - 2026-07-21: 반복되는 섹션 출력을 _print_header로 통합, 파이프라인 실행 전체에
    대한 예외 처리(FileNotFoundError/ValueError) 추가
"""

import sys
from datetime import datetime

from fare_analysis import config, data_io, eda, ml_pipeline, report, stats_analysis, visualization


def _print_header(title):
    """섹션 제목을 구분선과 함께 출력 (반복되는 출력 포맷을 한 곳에서 관리)"""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def _run_pipeline():
    """분석 파이프라인의 각 단계를 순서대로 실행"""
    config.OUTPUT_DIR.mkdir(exist_ok=True)

    _print_header("[1] Pandas vs Polars 로딩·전처리 성능 비교")
    timings, raw_df, clean_df = data_io.compare_pandas_polars()

    _print_header("[2] EDA: 결측치·중복·기술통계·고요금 라벨링")
    eda.summarize_missing_and_duplicates(raw_df)
    eda.descriptive_stats(clean_df)
    labeled_df, threshold = eda.add_high_fare_label(clean_df, config.HIGH_FARE_QUANTILE)

    _print_header("[3] 상관분석 & t-test")
    corr = stats_analysis.correlation_matrix(labeled_df, config.CORRELATION_COLUMNS)
    ttest_result = stats_analysis.run_rush_hour_ttest(labeled_df, config.RUSH_HOURS)

    _print_header("[4] 시각화")
    speed_chart_path = config.OUTPUT_DIR / "speed_comparison.png"
    trend_chart_path = config.OUTPUT_DIR / "hourly_fare_trend.html"
    visualization.plot_speed_comparison(timings, speed_chart_path)
    visualization.plot_hourly_fare_trend(labeled_df, trend_chart_path)

    _print_header("[5] ML Pipeline: is_high_fare 분류")
    model_path = config.OUTPUT_DIR / "high_fare_pipeline.joblib"
    _, metrics = ml_pipeline.train_evaluate_save(labeled_df, model_path)

    _print_header("[6] report.md 생성")
    context = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "n_rows": f"{len(labeled_df):,}",
        "timings": timings,
        "correlation_table": corr.to_string(),
        "ttest": ttest_result,
        "metrics": metrics,
        "model_filename": model_path.name,
    }
    report.generate_report(context, config.OUTPUT_DIR / "report.md")

    print("\n[완료] 전체 파이프라인 실행 성공")


def main():
    """파이프라인 전체를 실행하고, 실패 시 원인을 안내한 뒤 종료 코드 1로 종료"""
    try:
        _run_pipeline()
    except FileNotFoundError as e:
        print(f"[ERROR] 데이터 파일 문제로 파이프라인을 중단합니다: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"[ERROR] 데이터/설정 값 문제로 파이프라인을 중단합니다: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
