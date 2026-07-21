import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


def save_seaborn_distribution(pandas_clean, output_dir):
    # Distribution + KDE for total_amount and a boxplot below
    plt.figure(figsize=(10, 6))
    sns.histplot(pandas_clean['total_amount'], kde=True, bins=80)
    plt.title('Total Amount Distribution')
    plt.xlabel('Total Amount ($)')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/total_amount_distribution.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 3))
    sns.boxplot(x=pandas_clean['total_amount'])
    plt.title('Total Amount Boxplot')
    plt.xlabel('Total Amount ($)')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/total_amount_boxplot.png", dpi=150)
    plt.close()


def save_correlation_heatmap(pandas_clean, output_dir):
    corr = pandas_clean[["trip_distance", "trip_duration_minutes", "total_amount"]].corr()
    plt.figure(figsize=(6, 5))
    sns.heatmap(corr, annot=True, fmt='.3f', cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/correlation_heatmap.png", dpi=150)
    plt.close()


def save_performance_plot(performance_df, output_dir):
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=performance_df,
        x="작업",
        y="소요 시간(초)",
        hue="도구",
    )
    plt.title("Pandas와 Polars 작업별 처리 시간 비교")
    plt.xlabel("작업")
    plt.ylabel("소요 시간(초)")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/pandas_vs_polars_performance.png", dpi=150)
    plt.close()


def save_hourly_plot(pandas_hourly, output_dir):
    fig = px.line(
        pandas_hourly,
        x="pickup_hour",
        y="average_total_amount",
        markers=True,
        title="시간대별 평균 총요금 변화",
        labels={"pickup_hour": "탑승 시간", "average_total_amount": "평균 총요금($)"},
    )
    fig.update_xaxes(dtick=1)
    fig.write_html(f"{output_dir}/hourly_average_total_amount.html")
