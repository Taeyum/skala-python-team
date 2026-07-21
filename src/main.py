"""[Day2] End2End 분석 파이프라인 — NYC Taxi 고요금 예측.

변경내역:
- 2026-07-21: 초기 구현. 데이터 준비(Pandas/Polars 벤치마크+정제) → 시각화
  → 통계분석(상관+t-test) → ML Pipeline → report.md 생성까지 한 번에 실행.

실행: python -m src.main
"""

import sys

from src.eda import clean_trip_data, drop_duplicates, fare_descriptive_stats, missing_summary
from src.io_compare import load_pandas, run_benchmark
from src.ml_pipeline import train_evaluate_save
from src.report import build_context, generate_report
from src.stats_analysis import congestion_ttest, correlation_matrix
from src.visualize import plot_benchmark_bar, plot_hourly_fare_line

DATA_PATH = "data/raw/yellow_tripdata_2026-05.parquet"
BENCHMARK_NUMBER = 1
CHART_PATHS = {
    "benchmark_bar": "output/benchmark_speed.png",
    "hourly_fare_line": "output/hourly_fare_line.html",
}
MODEL_PATH = "output/model.joblib"
REPORT_PATH = "report.md"


def main() -> None:
    try:
        raw = load_pandas(DATA_PATH)
    except FileNotFoundError as exc:
        print(f"[오류] 데이터 로딩 실패: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    print("=" * 60)
    print("1) 데이터 준비 — 결측치/중복/기술통계 + Pandas vs Polars 벤치마크")
    print("=" * 60)
    na = missing_summary(raw)
    print(na[na > 0])
    _, dup_removed = drop_duplicates(raw)
    print(f"중복행 제거: {dup_removed}건")
    fare_stats = fare_descriptive_stats(raw)
    print(fare_stats)

    bench = run_benchmark(DATA_PATH, number=BENCHMARK_NUMBER)
    cleaned, clean_stats = clean_trip_data(raw)

    print()
    print("=" * 60)
    print("2) 시각화 — Seaborn 처리속도 바차트 + Plotly 시간대별 요금 라인차트")
    print("=" * 60)
    plot_benchmark_bar(bench["timings"], save_path=CHART_PATHS["benchmark_bar"])
    plot_hourly_fare_line(bench["hourly_fare"], save_path=CHART_PATHS["hourly_fare_line"])
    print(f"저장: {CHART_PATHS['benchmark_bar']}, {CHART_PATHS['hourly_fare_line']}")

    print()
    print("=" * 60)
    print("3) 통계 분석 — 상관계수(가설1) + 혼잡시간대 t-test(가설2)")
    print("=" * 60)
    corr_result = correlation_matrix(cleaned)
    ttest_result = congestion_ttest(cleaned)

    print()
    print("=" * 60)
    print("4) ML Pipeline — is_high_fare 분류")
    print("=" * 60)
    pipeline_result = train_evaluate_save(cleaned, model_path=MODEL_PATH)

    print()
    print("=" * 60)
    print("5) report.md 자동 생성")
    print("=" * 60)
    context = build_context(
        dataset_path=DATA_PATH,
        rows_before=clean_stats["rows_before"],
        rows_after=clean_stats["rows_after_iqr"],
        missing_summary=na,
        duplicates_removed=dup_removed,
        fare_stats=fare_stats,
        benchmark_timings=bench["timings"],
        correlation_result=corr_result,
        ttest_result=ttest_result,
        pipeline_result=pipeline_result,
        chart_paths=CHART_PATHS,
    )
    generate_report(context, template_path="templates/report_template.md.j2", output_path=REPORT_PATH)


if __name__ == "__main__":
    main()
