import pandas as pd
import pytest

from src.visualize import plot_benchmark_bar, plot_hourly_fare_line, setup_korean_font


@pytest.fixture
def timings_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "operation": ["로딩", "로딩", "필터링", "필터링"],
            "tool": ["pandas", "polars", "pandas", "polars"],
            "seconds": [0.3, 0.1, 0.2, 0.15],
        }
    )


@pytest.fixture
def hourly_df() -> pd.DataFrame:
    return pd.DataFrame({"pickup_hour": list(range(24)), "avg_total_amount": [20.0 + i for i in range(24)]})


def test_setup_korean_font_does_not_raise():
    setup_korean_font()


def test_plot_benchmark_bar_saves_png(timings_df, tmp_path):
    save_path = tmp_path / "bench.png"
    fig = plot_benchmark_bar(timings_df, save_path=save_path)
    assert save_path.exists()
    assert save_path.stat().st_size > 0
    assert len(fig.axes) >= 1


def test_plot_benchmark_bar_missing_column_raises():
    with pytest.raises(KeyError):
        plot_benchmark_bar(pd.DataFrame({"a": [1]}))


def test_plot_benchmark_bar_empty_df_raises():
    with pytest.raises(ValueError):
        plot_benchmark_bar(pd.DataFrame(columns=["operation", "tool", "seconds"]))


def test_plot_hourly_fare_line_saves_html(hourly_df, tmp_path):
    save_path = tmp_path / "hourly.html"
    fig = plot_hourly_fare_line(hourly_df, save_path=save_path)
    assert save_path.exists()
    content = save_path.read_text(encoding="utf-8")
    assert "plotly" in content.lower()
    assert len(fig.data) > 0


def test_plot_hourly_fare_line_missing_column_raises():
    with pytest.raises(KeyError):
        plot_hourly_fare_line(pd.DataFrame({"a": [1]}))
