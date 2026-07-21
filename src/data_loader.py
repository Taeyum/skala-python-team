# 작성자 : 이다은
# 작성일 : 2026-07-21
"""데이터 준비: 로컬 캐시 확보, Pandas/Polars 로딩·필터링·그룹집계 성능 비교."""

import time
from pathlib import Path

import pandas as pd
import polars as pl
import requests

from src.config import PICKUP_COL, TOTAL_COL


def ensure_local_parquet(url: str, cache_path: Path) -> Path:
    """로컬에 parquet 캐시가 없으면 URL에서 내려받아 저장한 뒤 경로를 반환한다.

    Pandas와 Polars의 로딩 속도를 "공정하게" 비교하려면 두 라이브러리가
    똑같은 로컬 파일을 읽어야 한다. 매번 네트워크에서 새로 받으면 그때그때
    네트워크 상태에 따라 시간이 들쭉날쭉해서 비교가 무의미해지기 때문이다.
    """
    if cache_path.is_file():
        return cache_path

    print(f"로컬 캐시가 없어 다운로드를 시작합니다: {url}")
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    response = requests.get(url, timeout=300, stream=True)
    response.raise_for_status()  # 4xx/5xx 응답이면 즉시 예외로 전환

    with open(cache_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            f.write(chunk)

    print(f"다운로드 완료: {cache_path}")
    return cache_path


def time_task(label: str, func, *args, **kwargs):
    """함수 실행 시간을 측정해 (결과, 소요시간초) 튜플로 반환하는 헬퍼."""
    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - start
    print(f"[{label}] 소요 시간: {elapsed:.4f}초")
    return result, elapsed


def compare_pandas_vs_polars(data_path: Path) -> tuple[pd.DataFrame, list[dict]]:
    """로딩/필터링/그룹집계 3개 작업에 대해 Pandas와 Polars 소요 시간을 비교한다.

    반환값
        pandas_df   : 이후 전처리·시각화·통계·ML에 계속 사용할 Pandas DataFrame
        timing_rows : [{"task": ..., "library": ..., "seconds": ...}, ...]
                      -> visualization.plot_speed_comparison_seaborn()의 입력
    """
    timing_rows: list[dict] = []

    # --- 작업 1: 로딩 ---
    pandas_df, pandas_load_sec = time_task("Pandas 로딩", pd.read_parquet, data_path)
    polars_df, polars_load_sec = time_task("Polars 로딩", pl.read_parquet, data_path)
    timing_rows.append({"task": "로딩", "library": "Pandas", "seconds": pandas_load_sec})
    timing_rows.append({"task": "로딩", "library": "Polars", "seconds": polars_load_sec})

    # --- 작업 2: 필터링 (총요금이 0보다 큰 행만) ---
    _, pandas_filter_sec = time_task(
        "Pandas 필터링", lambda: pandas_df[pandas_df[TOTAL_COL] > 0]
    )
    _, polars_filter_sec = time_task(
        "Polars 필터링", lambda: polars_df.filter(pl.col(TOTAL_COL) > 0)
    )
    timing_rows.append({"task": "필터링", "library": "Pandas", "seconds": pandas_filter_sec})
    timing_rows.append({"task": "필터링", "library": "Polars", "seconds": polars_filter_sec})

    # --- 작업 3: 그룹 집계 (시간대별 평균 총요금) ---
    def pandas_groupby():
        hours = pd.to_datetime(pandas_df[PICKUP_COL]).dt.hour
        return pandas_df.groupby(hours)[TOTAL_COL].mean()

    def polars_groupby():
        return (
            polars_df.with_columns(pl.col(PICKUP_COL).dt.hour().alias("hour"))
            .group_by("hour")
            .agg(pl.col(TOTAL_COL).mean())
        )

    _, pandas_group_sec = time_task("Pandas 그룹집계", pandas_groupby)
    _, polars_group_sec = time_task("Polars 그룹집계", polars_groupby)
    timing_rows.append({"task": "그룹집계", "library": "Pandas", "seconds": pandas_group_sec})
    timing_rows.append({"task": "그룹집계", "library": "Polars", "seconds": polars_group_sec})

    return pandas_df, timing_rows
