"""[Day2] 시각화 — Seaborn 처리속도 바차트 + Plotly 시간대별 평균요금 라인차트.

1단계(io_compare.run_benchmark) 결과를 그대로 받아 두 개의 인터랙티브/정적
차트를 만든다. matplotlib은 headless 환경에서도 항상 저장되도록 Agg
백엔드를 고정한다.
"""

from pathlib import Path
import platform

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns


def setup_korean_font() -> None:
    """OS 환경에 맞춰 matplotlib 한글 폰트를 자동으로 설정한다."""
    system_name = platform.system()
    
    if system_name == "Windows":
        font_name = "Malgun Gothic"
    elif system_name == "Darwin":
        font_name = "AppleGothic"
    else:  # Linux 및 기타 환경
        font_name = "NanumGothic"
        
    from matplotlib import font_manager

    available = {f.name for f in font_manager.fontManager.ttflist}
    if font_name not in available:
        print(f"[경고] '{font_name}' 폰트를 찾을 수 없어 한글이 깨질 수 있습니다.")
        return
    plt.rcParams["font.family"] = font_name
    plt.rcParams["axes.unicode_minus"] = False


def plot_benchmark_bar(
    timings: pd.DataFrame, save_path: str | Path = "output/benchmark_speed.png"
) -> plt.Figure:
    """작업별 Pandas vs Polars 처리속도를 그룹 바차트로 그려 저장한다."""
    required = {"operation", "tool", "seconds"}
    missing = required - set(timings.columns)
    if missing:
        raise KeyError(f"시각화에 필요한 컬럼이 없습니다: {sorted(missing)}")
    if timings.empty:
        raise ValueError("빈 데이터프레임은 시각화할 수 없습니다.")

    setup_korean_font()

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.barplot(data=timings, x="operation", y="seconds", hue="tool", ax=ax)
    ax.set_title("Pandas vs Polars 작업별 처리 속도 비교")
    ax.set_xlabel("작업")
    ax.set_ylabel("소요 시간 (초)")
    ax.legend(title="도구")
    fig.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_hourly_fare_line(
    hourly_fare: pd.DataFrame, save_path: str | Path = "output/hourly_fare_line.html"
) -> go.Figure:
    """시간대별 평균 총요금을 인터랙티브 라인차트로 만들어 HTML로 저장한다."""
    required = {"pickup_hour", "avg_total_amount"}
    missing = required - set(hourly_fare.columns)
    if missing:
        raise KeyError(f"시각화에 필요한 컬럼이 없습니다: {sorted(missing)}")
    if hourly_fare.empty:
        raise ValueError("빈 데이터프레임은 시각화할 수 없습니다.")

    fig = px.line(
        hourly_fare.sort_values("pickup_hour"),
        x="pickup_hour",
        y="avg_total_amount",
        markers=True,
        title="시간대별 평균 총요금 변화",
        labels={"pickup_hour": "승차 시각(시)", "avg_total_amount": "평균 총요금($)"},
    )
    fig.update_xaxes(dtick=1)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(save_path)

    return fig
