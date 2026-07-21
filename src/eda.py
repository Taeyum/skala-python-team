# 작성자 : 이다은
# 작성일 : 2026-07-21
"""핵심 변수(fare_amount, total_amount 등)의 기술통계 출력."""

import pandas as pd

from src.config import DISTANCE_COL, FARE_COL, TOTAL_COL


def run_eda(df: pd.DataFrame) -> pd.DataFrame:
    """fare_amount·total_amount 등 핵심 변수의 기술통계를 출력한다."""
    print("\n=== 기술통계 (평균/표준편차/분위수) ===")
    eda_cols = [FARE_COL, TOTAL_COL, DISTANCE_COL, "trip_duration_min"]
    summary = df[eda_cols].describe(percentiles=[0.25, 0.5, 0.75])
    print(summary)
    return summary
