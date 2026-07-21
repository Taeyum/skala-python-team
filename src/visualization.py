# 작성자 : 이다은
# 작성일 : 2026-07-21
"""Seaborn 정적 차트(작업별 속도 비교)와 Plotly 인터랙티브 차트(시간대별 요금)."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import seaborn as sns

from src.config import TOTAL_COL

# 한글 폰트 설정 (macOS: AppleGothic, Windows: Malgun Gothic, Linux: NanumGothic)
plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False


def plot_speed_comparison_seaborn(timing_rows: list[dict], save_path: Path) -> None:
    """작업별(로딩/필터링/그룹집계) Pandas vs Polars 소요 시간 바 차트."""
    timing_df = pd.DataFrame(timing_rows)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=timing_df, x="task", y="seconds", hue="library", ax=ax)
    ax.set_title("작업별 Pandas vs Polars 처리 속도 비교")
    ax.set_xlabel("작업 종류")
    ax.set_ylabel("소요 시간 (초)")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"속도 비교 차트 저장 완료: {save_path}")


def plot_hourly_fare_plotly(df: pd.DataFrame, save_path: Path) -> None:
    """시간대별 평균 총요금 변화를 Plotly 인터랙티브 라인 차트로 저장한다."""
    hourly_avg = df.groupby("pickup_hour")[TOTAL_COL].mean().reset_index()

    fig = px.line(
        hourly_avg,
        x="pickup_hour",
        y=TOTAL_COL,
        markers=True,
        title="시간대별 평균 총요금(total_amount) 변화",
        labels={"pickup_hour": "승차 시각(시)", TOTAL_COL: "평균 총요금($)"},
    )
    fig.write_html(save_path)
    print(f"시간대별 요금 차트 저장 완료: {save_path}")
