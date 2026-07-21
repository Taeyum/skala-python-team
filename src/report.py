"""[Day2] 분석 결과를 report.md로 자동 생성.

1~4단계(io_compare/eda/visualize/stats_analysis/ml_pipeline)의 결과 dict를
받아 Jinja2 템플릿에 채워 넣어 report.md를 생성한다.
"""

from datetime import datetime
from pathlib import Path

import pandas as pd
from jinja2 import Environment, FileSystemLoader

from src.io_compare import OPERATIONS


def build_context(
    dataset_path: str,
    rows_before: int | None,
    rows_after: int | None,
    missing_summary: pd.Series | None,
    duplicates_removed: int | None,
    fare_stats: pd.DataFrame | None,
    benchmark_timings: pd.DataFrame | None,
    correlation_result: dict | None,
    ttest_result: dict | None,
    pipeline_result: dict | None,
    chart_paths: dict,
) -> dict:
    """각 단계 결과를 report 템플릿이 바로 쓸 수 있는 dict로 변환한다.

    파이프라인의 한 단계가 실패해 결과가 None으로 전달되면, 해당 섹션은
    빈 값 대신 오류 메시지로 대체해 report.md 생성 자체는 계속되도록 한다
    (한 단계 실패가 전체를 막지 않는 graceful degradation).
    """
    if missing_summary is not None:
        missing_nonzero = missing_summary[missing_summary > 0].sort_values(ascending=False).to_dict()
    else:
        missing_nonzero = {}

    if benchmark_timings is not None:
        benchmark_wide = benchmark_timings.pivot(index="operation", columns="tool", values="seconds")
        benchmark_wide = benchmark_wide.reindex(OPERATIONS)
        benchmark_ctx = {
            "operations": [
                {"operation": op, "pandas": row["pandas"], "polars": row["polars"]}
                for op, row in benchmark_wide.iterrows()
            ]
        }
    else:
        benchmark_ctx = {"operations": [], "error": "벤치마크 단계 실패"}

    if correlation_result is not None:
        corr_matrix = correlation_result["correlation"]
        correlation_ctx = {
            "matrix": [{"label": label, **corr_matrix.loc[label].to_dict()} for label in corr_matrix.index],
            "message": correlation_result["message"],
            "hypothesis1_supported": correlation_result["hypothesis1_supported"],
        }
    else:
        correlation_ctx = {"matrix": [], "message": None, "hypothesis1_supported": None, "error": "상관분석 단계 실패"}

    if ttest_result is not None:
        ttest_ctx = {
            "mean_rush": ttest_result["mean_rush"],
            "mean_non_rush": ttest_result["mean_non_rush"],
            "n_rush": ttest_result["n_rush"],
            "n_non_rush": ttest_result["n_non_rush"],
            "t_stat": ttest_result["t_stat"],
            "p_value": ttest_result["p_value"],
            "message": ttest_result["message"],
            "hypothesis2_supported": ttest_result["hypothesis2_supported"],
        }
    else:
        ttest_ctx = {"hypothesis2_supported": None, "error": "t-test 단계 실패"}

    pipeline_ctx = pipeline_result if pipeline_result is not None else {"error": "ML Pipeline 단계 실패"}

    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": {"path": dataset_path, "rows_before": rows_before, "rows_after": rows_after},
        "missing_summary": missing_nonzero,
        "duplicates_removed": duplicates_removed,
        "fare_stats": fare_stats.to_dict() if fare_stats is not None else {},
        "benchmark": benchmark_ctx,
        "correlation": correlation_ctx,
        "ttest": ttest_ctx,
        "pipeline": pipeline_ctx,
        "charts": chart_paths,
    }


def generate_report(
    context: dict,
    template_path: str | Path = "templates/report_template.md.j2",
    output_path: str | Path = "report.md",
) -> Path:
    """context를 Jinja2 템플릿에 렌더링해 report.md 파일로 저장한다."""
    template_path = Path(template_path)
    if not template_path.exists():
        raise FileNotFoundError(f"템플릿 파일을 찾을 수 없습니다: {template_path}")

    env = Environment(loader=FileSystemLoader(str(template_path.parent)))
    template = env.get_template(template_path.name)
    rendered = template.render(**context)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered, encoding="utf-8")
    print(f"[report] 생성 완료: {output_path}")

    return output_path
