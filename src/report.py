from pathlib import Path


def df_to_md_or_text(df):
    try:
        return df.round(3).to_markdown()
    except Exception:
        return df.round(3).to_string()


def generate_report(
    output_dir,
    descriptive_stats,
    correlation_df,
    ttest_info,
    performance_df,
    ml_metrics,
    model_path,
):
    out = Path(output_dir)
    report_path = out / "report.md"

    parts = []
    parts.append("# NYC Taxi Analysis Report")
    parts.append("")
    parts.append("## Summary")
    parts.append("")
    parts.append(f"- Rows after preprocessing: {int(descriptive_stats.loc['trip_distance','count']):,}")
    parts.append("")
    parts.append("## Technical Statistics")
    parts.append("")
    parts.append(df_to_md_or_text(descriptive_stats))
    parts.append("")
    parts.append("## Correlation Matrix")
    parts.append("")
    parts.append(df_to_md_or_text(correlation_df))
    parts.append("")
    parts.append("![Correlation Heatmap](correlation_heatmap.png)")
    parts.append("")
    parts.append("## T-test: Rush vs Non-rush Total Amount")
    parts.append("")
    parts.append(f"- Rush sample size: {ttest_info['rush_n']:,}")
    parts.append(f"- Non-rush sample size: {ttest_info['non_rush_n']:,}")
    parts.append(f"- Rush mean: ${ttest_info['rush_mean']:.3f}")
    parts.append(f"- Non-rush mean: ${ttest_info['non_rush_mean']:.3f}")
    parts.append(f"- t-statistic: {ttest_info['t_statistic']:.4f}")
    parts.append(f"- p-value: {ttest_info['p_value']:.6g}")
    parts.append("")
    parts.append("## Visualizations")
    parts.append("")
    parts.append("- Static distribution: total_amount_distribution.png")
    parts.append("- Boxplot: total_amount_boxplot.png")
    parts.append("- Interactive hourly average: hourly_average_total_amount.html")
    parts.append("")
    parts.append("## Performance Comparison")
    parts.append("")
    try:
        parts.append(performance_df.to_markdown())
    except Exception:
        parts.append(performance_df.to_string())

    parts.append("")
    parts.append("## Machine Learning Results")
    parts.append("")
    parts.append(f"- MAE: {ml_metrics['MAE']:.3f}")
    parts.append(f"- MSE: {ml_metrics['MSE']:.3f}")
    parts.append(f"- R²: {ml_metrics['R2']:.3f}")
    parts.append(f"- Saved model: {Path(model_path).name}")

    parts.append("")
    parts.append("---")
    parts.append("")
    parts.append("Report generated automatically.")

    report_path.write_text("\n".join(parts), encoding="utf-8")
    return report_path
