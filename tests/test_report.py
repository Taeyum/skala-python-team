import pandas as pd
import pytest

from src.report import build_context, generate_report


@pytest.fixture
def sample_context() -> dict:
    missing_summary = pd.Series({"passenger_count": 23.35, "trip_distance": 0.0})
    fare_stats = pd.DataFrame(
        {
            "fare_amount": [100, 20.0, 5.0, 1.0, 15.0, 20.0, 25.0, 50.0],
            "total_amount": [100, 25.0, 6.0, 1.5, 18.0, 24.0, 30.0, 60.0],
        },
        index=["count", "mean", "std", "min", "25%", "50%", "75%", "max"],
    )
    benchmark_timings = pd.DataFrame(
        {
            "operation": ["로딩", "로딩", "필터링", "필터링", "시간대별집계", "시간대별집계"],
            "tool": ["pandas", "polars", "pandas", "polars", "pandas", "polars"],
            "seconds": [0.3, 0.1, 0.2, 0.25, 0.08, 0.02],
        }
    )
    correlation_result = {
        "correlation": pd.DataFrame(
            [[1.0, 0.68, 0.67], [0.68, 1.0, 0.70], [0.67, 0.70, 1.0]],
            index=["trip_distance", "trip_duration", "total_amount"],
            columns=["trip_distance", "trip_duration", "total_amount"],
        ),
        "message": "가설1 채택(강한 양의 상관 확인)",
        "hypothesis1_supported": True,
    }
    ttest_result = {
        "mean_rush": 24.71,
        "mean_non_rush": 24.88,
        "n_rush": 1000,
        "n_non_rush": 2000,
        "t_stat": -12.2,
        "p_value": 2.9e-34,
        "message": "가설2 기각",
        "hypothesis2_supported": False,
    }
    pipeline_result = {
        "threshold": 29.82,
        "accuracy": 0.896,
        "precision": 0.738,
        "recall": 0.908,
        "f1": 0.814,
        "confusion_matrix": [[458867, 55596], [15804, 156639]],
        "n_train": 2747620,
        "n_test": 686906,
        "model_path": "output/model.joblib",
        "reload_matches": True,
    }
    chart_paths = {"benchmark_bar": "output/benchmark_speed.png", "hourly_fare_line": "output/hourly_fare_line.html"}

    return build_context(
        dataset_path="data/raw/yellow_tripdata_2026-05.parquet",
        rows_before=4090836,
        rows_after=3434526,
        missing_summary=missing_summary,
        duplicates_removed=0,
        fare_stats=fare_stats,
        benchmark_timings=benchmark_timings,
        correlation_result=correlation_result,
        ttest_result=ttest_result,
        pipeline_result=pipeline_result,
        chart_paths=chart_paths,
    )


def test_build_context_has_expected_top_level_keys(sample_context):
    assert set(sample_context.keys()) == {
        "generated_at",
        "dataset",
        "missing_summary",
        "duplicates_removed",
        "fare_stats",
        "benchmark",
        "correlation",
        "ttest",
        "pipeline",
        "charts",
    }
    assert len(sample_context["benchmark"]["operations"]) == 3


def test_generate_report_renders_markdown_file(sample_context, tmp_path):
    output_path = tmp_path / "report.md"
    result_path = generate_report(
        sample_context, template_path="templates/report_template.md.j2", output_path=output_path
    )

    assert result_path == output_path
    content = output_path.read_text(encoding="utf-8")
    assert "# NYC Yellow Taxi" in content
    assert "가설1 채택" in content
    assert "가설2 기각" in content


def test_generate_report_missing_template_raises(sample_context, tmp_path):
    with pytest.raises(FileNotFoundError):
        generate_report(sample_context, template_path="templates/does_not_exist.md.j2", output_path=tmp_path / "r.md")


def test_build_context_handles_all_stages_failed():
    """모든 단계가 실패(None)해도 build_context는 예외 없이 fallback 컨텍스트를 만든다."""
    context = build_context(
        dataset_path="data/raw/yellow_tripdata_2026-05.parquet",
        rows_before=None,
        rows_after=None,
        missing_summary=None,
        duplicates_removed=None,
        fare_stats=None,
        benchmark_timings=None,
        correlation_result=None,
        ttest_result=None,
        pipeline_result=None,
        chart_paths={"benchmark_bar": "output/benchmark_speed.png", "hourly_fare_line": "output/hourly_fare_line.html"},
    )

    assert context["missing_summary"] == {}
    assert context["fare_stats"] == {}
    assert "error" in context["benchmark"]
    assert "error" in context["correlation"]
    assert "error" in context["ttest"]
    assert "error" in context["pipeline"]


def test_generate_report_renders_when_some_stages_failed(tmp_path):
    """일부 단계만 실패해도 report.md 생성 자체는 죽지 않고 오류 메시지를 남긴다."""
    context = build_context(
        dataset_path="data/raw/yellow_tripdata_2026-05.parquet",
        rows_before=100,
        rows_after=90,
        missing_summary=pd.Series({"a": 1.0}),
        duplicates_removed=0,
        fare_stats=pd.DataFrame(
            {"fare_amount": [1, 2, 3, 4, 5, 6, 7, 8], "total_amount": [1, 2, 3, 4, 5, 6, 7, 8]},
            index=["count", "mean", "std", "min", "25%", "50%", "75%", "max"],
        ),
        benchmark_timings=None,  # 벤치마크 단계 실패
        correlation_result=None,  # 상관분석 단계 실패
        ttest_result=None,  # t-test 단계 실패
        pipeline_result=None,  # ML 단계 실패
        chart_paths={"benchmark_bar": "output/benchmark_speed.png", "hourly_fare_line": "output/hourly_fare_line.html"},
    )

    output_path = tmp_path / "report.md"
    generate_report(context, template_path="templates/report_template.md.j2", output_path=output_path)

    content = output_path.read_text(encoding="utf-8")
    assert "벤치마크 단계 실패" in content
    assert "상관분석 단계 실패" in content
    assert "t-test 단계 실패" in content
    assert "ML Pipeline 단계 실패" in content
    assert "검증 실패" in content  # 결론 섹션의 가설1/2 판정
