"""
Seaborn 정적 차트 + Plotly 인터랙티브 차트

변경내역:
  - 2026-07-21: 최초 작성
  - 2026-07-21: 차트 저장 실패에 대한 예외 처리 추가
"""

import matplotlib

matplotlib.use("Agg")  # 디스플레이 없는 환경에서도 이미지로 저장 가능하도록 백엔드 지정
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import seaborn as sns

plt.rcParams["font.family"] = "AppleGothic"  # windows 환경에서는 windows가 지원하는 font로 변경 가능
plt.rcParams["axes.unicode_minus"] = False


def plot_speed_comparison(timings, save_path):
    """로딩/전처리 단계별 Pandas vs Polars 평균 처리 시간 비교 바 차트"""
    rows = [
        {"stage": stage, "tool": tool, "seconds": sec}
        for stage, tool_times in timings.items()
        for tool, sec in tool_times.items()
    ]
    plot_df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=plot_df, x="stage", y="seconds", hue="tool", ax=ax)
    ax.set_title("Pandas vs Polars 처리 속도 비교")
    ax.set_xlabel("작업 단계")
    ax.set_ylabel("평균 소요 시간 (초)")
    fig.tight_layout()

    try:
        fig.savefig(save_path, dpi=120)
    except OSError as e:
        print(f"[ERROR] 속도 비교 차트 저장 실패: {e}")
        raise
    finally:
        plt.close(fig)
    print(f"[OK] 처리 속도 비교 바 차트 저장: {save_path}")


def plot_hourly_fare_trend(df, save_path):
    """시간대별 평균 총요금 변화 인터랙티브 라인 차트"""
    hourly = df.groupby("pickup_hour", as_index=False)["total_amount"].mean()
    fig = px.line(
        hourly, x="pickup_hour", y="total_amount", markers=True,
        title="시간대별 평균 총요금 변화",
        labels={"pickup_hour": "승차 시각(시)", "total_amount": "평균 총요금($)"},
    )
    try:
        fig.write_html(save_path)
    except OSError as e:
        print(f"[ERROR] 시간대별 요금 차트 저장 실패: {e}")
        raise
    print(f"[OK] 시간대별 평균 총요금 인터랙티브 차트 저장: {save_path}")
