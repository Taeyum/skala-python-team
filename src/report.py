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
    rows_before: int,
    rows_after: int,
    missing_summary: pd.Series,
    duplicates_removed: int,
    fare_stats: pd.DataFrame,
    benchmark_timings: pd.DataFrame,
    correlation_result: dict,
    ttest_result: dict,
    pipeline_result: dict,
    chart_paths: dict,
) -> dict:
    """각 단계 결과를 report 템플릿이 바로 쓸 수 있는 dict로 변환한다."""
    missing_nonzero = missing_summary[missing_summary > 0].sort_values(ascending=False)

    benchmark_wide = benchmark_timings.pivot(index="operation", columns="tool", values="seconds")
    benchmark_wide = benchmark_wide.reindex(OPERATIONS)
    operations = [
        {"operation": op, "pandas": row["pandas"], "polars": row["polars"]}
        for op, row in benchmark_wide.iterrows()
    ]

    corr_matrix = correlation_result["correlation"]
    matrix_rows = [
        {"label": label, **corr_matrix.loc[label].to_dict()} for label in corr_matrix.index
    ]

    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": {"path": dataset_path, "rows_before": rows_before, "rows_after": rows_after},
        "missing_summary": missing_nonzero.to_dict(),
        "duplicates_removed": duplicates_removed,
        "fare_stats": fare_stats.to_dict(),
        "benchmark": {"operations": operations},
        "correlation": {
            "matrix": matrix_rows,
            "message": correlation_result["message"],
            "hypothesis1_supported": correlation_result["hypothesis1_supported"],
        },
        "ttest": {
            "mean_rush": ttest_result["mean_rush"],
            "mean_non_rush": ttest_result["mean_non_rush"],
            "n_rush": ttest_result["n_rush"],
            "n_non_rush": ttest_result["n_non_rush"],
            "t_stat": ttest_result["t_stat"],
            "p_value": ttest_result["p_value"],
            "message": ttest_result["message"],
            "hypothesis2_supported": ttest_result["hypothesis2_supported"],
        },
        "pipeline": pipeline_result,
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
