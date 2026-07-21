# 작성자 : 이다은
# 작성일 : 2026-07-21
"""분석 결과(성능비교/EDA/상관/검정/ML지표/가설검증)를 report.md로 자동 생성."""

from pathlib import Path

import pandas as pd

from src.config import (
    DISTANCE_COL,
    HOURLY_FARE_HTML_PATH,
    MODEL_PATH,
    SPEED_FIG_PATH,
    TOTAL_COL,
)


def generate_report_md(
    timing_rows: list[dict],
    eda_summary: pd.DataFrame,
    corr: pd.DataFrame,
    ttest_result: dict,
    ml_metrics: dict,
    report_path: Path,
) -> None:
    """분석 결과를 report.md 파일로 자동 생성한다."""
    timing_table = pd.DataFrame(timing_rows).pivot(
        index="task", columns="library", values="seconds"
    )

    # 가설 1: 거리/운행시간과 총요금의 상관계수가 충분히 큰 양의 값인지 확인
    distance_corr = corr.loc[DISTANCE_COL, TOTAL_COL]
    duration_corr = corr.loc["trip_duration_min", TOTAL_COL]
    h1_supported = distance_corr > 0.5 and duration_corr > 0.5
    h1_conclusion = "지지됨" if h1_supported else "지지되지 않음"

    # 가설 2: 혼잡 시간대 평균 총요금이 더 높고, 그 차이가 통계적으로 유의미한지 확인
    h2_supported = (
        ttest_result["group_a_mean"] > ttest_result["group_b_mean"]
        and ttest_result["p_value"] < 0.05
    )
    h2_conclusion = "지지됨" if h2_supported else "지지되지 않음"

    report_content = f"""# Day2 종합실습 - 고요금 운행 예측 분석 리포트

## 1. 데이터 로딩 성능 비교 (Pandas vs Polars)

{timing_table.to_markdown()}

![처리 속도 비교]({SPEED_FIG_PATH.as_posix()})

## 2. 기술통계

{eda_summary.to_markdown()}

## 3. 시간대별 평균 총요금

인터랙티브 차트: [{HOURLY_FARE_HTML_PATH.name}]({HOURLY_FARE_HTML_PATH.as_posix()})

## 4. 상관분석

{corr.to_markdown()}

## 5. t-test: 혼잡 시간대 vs 비혼잡 시간대 총요금

- 혼잡 시간대 평균: {ttest_result["group_a_mean"]:,.2f}
- 비혼잡 시간대 평균: {ttest_result["group_b_mean"]:,.2f}
- t-statistic: {ttest_result["t_stat"]:.4f}
- p-value: {ttest_result["p_value"]:.4g}
- 해석: {ttest_result["interpretation"]}

## 6. ML Pipeline 성능 (고요금 운행 분류)

- Accuracy: {ml_metrics["accuracy"]:.4f}
- F1-score: {ml_metrics["f1_score"]:.4f}
- 저장된 모델: `{MODEL_PATH.as_posix()}`

## 7. 가설 검증 결과

| 가설 | 내용 | 결과 |
|---|---|---|
| H1 | 운행 시간·이동 거리와 총요금은 강한 양의 상관관계를 가질 것이다 | {h1_conclusion} (거리 상관계수={distance_corr:.3f}, 시간 상관계수={duration_corr:.3f}) |
| H2 | 혼잡 시간대 운행이 비혼잡 시간대보다 평균 총요금이 높을 것이다 | {h2_conclusion} (p-value={ttest_result["p_value"]:.4g}) |
"""

    report_path.write_text(report_content, encoding="utf-8")
    print(f"리포트 생성 완료: {report_path}")
