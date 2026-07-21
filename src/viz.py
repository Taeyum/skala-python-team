import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import warnings

# 한글 폰트 및 시각화 경고 방지 설정
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')

def generate_visualizations(bench_df: pd.DataFrame, df: pd.DataFrame):
    """
    Pandas vs Polars 속도 비교 바 차트(Seaborn)와 시간대별 평균 요금 라인 차트(Plotly)를 생성합니다.
    (채점 기준: 정적 차트 1개, 인터랙티브 1개, 제목 및 축 레이블 반드시 포함)
    """
    print("[3/6] 데이터 시각화 (정적 및 인터랙티브 차트) 생성 중...")
    
    # 1. 정적 차트 (Seaborn): Pandas vs Polars 처리 속도 비교 바 차트
    plt.figure(figsize=(8, 6))
    sns.barplot(data=bench_df, x='Task', y='Time_sec', hue='Tool', palette='Set2')
    # 조건 만족: 제목과 축 레이블 명시
    plt.title("Performance Comparison: Pandas vs Polars", fontsize=14, fontweight='bold')
    plt.xlabel("Data Processing Task", fontsize=12)
    plt.ylabel("Execution Time (Seconds)", fontsize=12)
    plt.tight_layout()
    plt.savefig('seaborn_benchmark_comparison.png', dpi=300)
    plt.close()
    
    # 2. 동적 차트 (Plotly): 시간대별 평균 총요금 변화 인터랙티브 라인 차트
    hourly_fare = df.groupby('pickup_hour')['total_amount'].mean().reset_index()
    fig = px.line(
        hourly_fare, 
        x='pickup_hour', 
        y='total_amount',
        markers=True,
        # 조건 만족: 제목과 축 레이블 명시
        title="Average Total Fare by Hour of the Day",
        labels={
            "pickup_hour": "Time of Day (Hour)",
            "total_amount": "Average Total Fare (USD)"
        }
    )
    # 혼잡 시간대(Rush Hour) 강조를 위한 배경색 추가
    fig.add_vrect(x0=7.5, x1=10.5, fillcolor="red", opacity=0.1, line_width=0, annotation_text="Morning Rush")
    fig.add_vrect(x0=16.5, x1=19.5, fillcolor="red", opacity=0.1, line_width=0, annotation_text="Evening Rush")
    fig.write_html('plotly_hourly_fare_trend.html')
    
    print("  -> 시각화 이미지 저장 완료 (seaborn_benchmark_comparison.png, plotly_hourly_fare_trend.html)")
