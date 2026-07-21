from pathlib import Path
from time import perf_counter
from urllib.request import urlretrieve

import pandas as pd
import polars as pl


def ensure_data(path: Path, url: str) -> Path:
    if not path.exists():
        print(f"데이터 파일을 다운로드합니다: {path}")
        urlretrieve(url, path)
        print(f"다운로드 완료: {path.resolve()}")
    else:
        print(f"기존 데이터 파일을 사용합니다: {path.resolve()}")
    return path


def load_pandas(path: Path, columns: list):
    start = perf_counter()
    df = pd.read_parquet(path, columns=columns)
    return df, perf_counter() - start


def load_polars(path: Path, columns: list):
    start = perf_counter()
    df = pl.read_parquet(path, columns=columns)
    return df, perf_counter() - start
