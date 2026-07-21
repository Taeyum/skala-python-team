"""[Day2] End2End 분석 파이프라인 — NYC Taxi 고요금 예측.

변경내역:
- 2026-07-21: 초기 구현. 데이터 준비(Pandas/Polars 벤치마크+정제) → 시각화
  → 통계분석(상관+t-test) → ML Pipeline → report.md 생성까지 한 번에 실행.
- 2026-07-21: 경로/임계값 상수를 src/config.py로 이동. 각 단계를 개별
  try/except로 감싸 한 단계가 실패해도 나머지 단계는 계속 진행하고
  report.md에 실패 사실을 남기도록 변경(기준님 브랜치의 단계별 graceful
  degradation 패턴 반영).

실행: python -m src.main
"""

import sys

from src.config import CHART_PATHS, DATA_PATH, MODEL_PATH, REPORT_PATH
from src.eda import clean_trip_data, drop_duplicates, fare_descriptive_stats, missing_summary
from src.io_compare import load_pandas, run_benchmark
from src.ml_pipeline import train_evaluate_save
from src.report import build_context, generate_report
from src.stats_analysis import congestion_ttest, correlation_matrix
from src.visualize import plot_benchmark_bar, plot_hourly_fare_line


def main() -> None:
    try:
        raw = load_pandas(DATA_PATH)
    except FileNotFoundError as exc:
        print(f"[오류] 데이터 로딩 실패: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    print("=" * 60)
    print("1) 데이터 준비 — 결측치/중복/기술통계 + Pandas vs Polars 벤치마크")
    print("=" * 60)

    na = dup_removed = fare_stats = None
    try:
        na = missing_summary(raw)
        print(na[na > 0])
        _, dup_removed = drop_duplicates(raw)
        print(f"중복행 제거: {dup_removed}건")
        fare_stats = fare_descriptive_stats(raw)
        print(fare_stats)
    except (KeyError, ValueError) as exc:
        print(f"[ERROR] 1) 데이터 준비(EDA 요약) 실패, 다음 단계로 계속 진행: {exc}")

    bench = None
    try:
        bench = run_benchmark(DATA_PATH, number=1)
    except (FileNotFoundError, KeyError) as exc:
        print(f"[ERROR] 1) Pandas/Polars 벤치마크 실패, 다음 단계로 계속 진행: {exc}")

    cleaned = clean_stats = None
    try:
        cleaned, clean_stats = clean_trip_data(raw)
    except (KeyError, TypeError) as exc:
        print(f"[ERROR] 1) 데이터 정제 실패 — 통계/ML 단계는 건너뜁니다: {exc}")

    print()
    print("=" * 60)
    print("2) 시각화 — Seaborn 처리속도 바차트 + Plotly 시간대별 요금 라인차트")
    print("=" * 60)
    if bench is not None:
        try:
            plot_benchmark_bar(bench["timings"], save_path=CHART_PATHS["benchmark_bar"])
            plot_hourly_fare_line(bench["hourly_fare"], save_path=CHART_PATHS["hourly_fare_line"])
            print(f"저장: {CHART_PATHS['benchmark_bar']}, {CHART_PATHS['hourly_fare_line']}")
        except (KeyError, ValueError) as exc:
            print(f"[ERROR] 2) 시각화 실패, 다음 단계로 계속 진행: {exc}")
    else:
        print("[SKIP] 2) 벤치마크 결과가 없어 시각화를 건너뜁니다.")

    print()
    print("=" * 60)
    print("3) 통계 분석 — 상관계수(가설1) + 혼잡시간대 t-test(가설2)")
    print("=" * 60)
    corr_result = ttest_result = None
    if cleaned is not None:
        try:
            corr_result = correlation_matrix(cleaned)
        except (KeyError, ValueError) as exc:
            print(f"[ERROR] 3) 상관분석 실패, 다음 단계로 계속 진행: {exc}")
        try:
            ttest_result = congestion_ttest(cleaned)
        except (KeyError, ValueError) as exc:
            print(f"[ERROR] 3) t-test 실패, 다음 단계로 계속 진행: {exc}")
    else:
        print("[SKIP] 3) 정제된 데이터가 없어 통계 분석을 건너뜁니다.")

    print()
    print("=" * 60)
    print("4) ML Pipeline — is_high_fare 분류")
    print("=" * 60)
    pipeline_result = None
    if cleaned is not None:
        try:
            pipeline_result = train_evaluate_save(cleaned, model_path=MODEL_PATH)
        except (KeyError, ValueError) as exc:
            print(f"[ERROR] 4) ML Pipeline 실패, 다음 단계로 계속 진행: {exc}")
    else:
        print("[SKIP] 4) 정제된 데이터가 없어 ML Pipeline을 건너뜁니다.")

    print()
    print("=" * 60)
    print("5) report.md 자동 생성")
    print("=" * 60)
    context = build_context(
        dataset_path=DATA_PATH,
        rows_before=clean_stats["rows_before"] if clean_stats else None,
        rows_after=clean_stats["rows_after_iqr"] if clean_stats else None,
        missing_summary=na,
        duplicates_removed=dup_removed,
        fare_stats=fare_stats,
        benchmark_timings=bench["timings"] if bench else None,
        correlation_result=corr_result,
        ttest_result=ttest_result,
        pipeline_result=pipeline_result,
        chart_paths=CHART_PATHS,
    )
    generate_report(context, template_path="templates/report_template.md.j2", output_path=REPORT_PATH)


if __name__ == "__main__":
    main()
