"""
요금 예측 & 도구 성능 비교 - 전체 파이프라인 실행

주제: Pandas vs Polars 처리 성능 비교 기반 고요금(High-Fare) 운행 예측
  1) Pandas vs Polars 로딩·전처리 성능 비교 (data_io)
  2) EDA - 결측치·중복 확인, 기술통계, is_high_fare 라벨링 (eda)
  3) 상관분석 + t-test 가설 검정 (stats_analysis)
  4) Seaborn 처리속도 비교 바 차트 + Plotly 시간대별 요금 라인 차트 (visualization)
  5) sklearn Pipeline으로 is_high_fare 분류 학습·평가·저장 (ml_pipeline)
  6) report.md 자동 생성 (report)
"""

from datetime import datetime

from fare_analysis import config, data_io, eda, ml_pipeline, report, stats_analysis, visualization


def main():
    config.OUTPUT_DIR.mkdir(exist_ok=True)

    print("=" * 60)
    print("[1] Pandas vs Polars 로딩·전처리 성능 비교")
    print("=" * 60)
    timings, raw_df, clean_df = data_io.compare_pandas_polars()

    print("\n" + "=" * 60)
    print("[2] EDA: 결측치·중복·기술통계·고요금 라벨링")
    print("=" * 60)
    eda.summarize_missing_and_duplicates(raw_df)
    eda.descriptive_stats(clean_df)
    labeled_df, threshold = eda.add_high_fare_label(clean_df, config.HIGH_FARE_QUANTILE)

    print("\n" + "=" * 60)
    print("[3] 상관분석 & t-test")
    print("=" * 60)
    corr = stats_analysis.correlation_matrix(labeled_df, config.CORRELATION_COLUMNS)
    ttest_result = stats_analysis.run_rush_hour_ttest(labeled_df, config.RUSH_HOURS)

    print("\n" + "=" * 60)
    print("[4] 시각화")
    print("=" * 60)
    speed_chart_path = config.OUTPUT_DIR / "speed_comparison.png"
    trend_chart_path = config.OUTPUT_DIR / "hourly_fare_trend.html"
    visualization.plot_speed_comparison(timings, speed_chart_path)
    visualization.plot_hourly_fare_trend(labeled_df, trend_chart_path)

    print("\n" + "=" * 60)
    print("[5] ML Pipeline: is_high_fare 분류")
    print("=" * 60)
    model_path = config.OUTPUT_DIR / "high_fare_pipeline.joblib"
    _, metrics = ml_pipeline.train_evaluate_save(labeled_df, model_path)

    print("\n" + "=" * 60)
    print("[6] report.md 생성")
    print("=" * 60)
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


if __name__ == "__main__":
    main()
