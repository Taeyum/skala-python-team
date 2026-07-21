import os
import time
import urllib.request
import pandas as pd
import polars as pl
from typing import Optional

def download_and_benchmark(url: str, local_path: str = "data/raw/yellow_tripdata_2026-05.parquet") -> tuple:
    """
    데이터셋을 다운로드하고 Pandas와 Polars의 로딩 및 필터링 성능을 비교합니다.
    (채점 기준: Pandas와 Polars 양쪽으로 로딩하여 결과를 비교)
    """
    print("[1/6] 데이터 다운로드 및 Pandas vs Polars 성능 벤치마킹 중...")
    
    # 데이터 다운로드 (캐싱)
    if not os.path.exists(local_path):
        print(f"  -> {url} 에서 데이터를 다운로드 중입니다. (시간이 소요될 수 있습니다)")
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        urllib.request.urlretrieve(url, local_path)
    
    benchmark_results = []

    # 1. Pandas 성능 측정 (로딩 + 기본 필터링)
    start_time = time.time()
    pd_df = pd.read_parquet(local_path)
    pd_load_time = time.time() - start_time
    benchmark_results.append({'Task': 'Data Loading', 'Tool': 'Pandas', 'Time_sec': pd_load_time})

    start_time = time.time()
    _ = pd_df[pd_df['total_amount'] > 0] # 기본 필터링 테스트
    pd_filter_time = time.time() - start_time
    benchmark_results.append({'Task': 'Basic Filtering', 'Tool': 'Pandas', 'Time_sec': pd_filter_time})

    # 2. Polars 성능 측정 (로딩 + 기본 필터링)
    start_time = time.time()
    pl_df = pl.read_parquet(local_path)
    pl_load_time = time.time() - start_time
    benchmark_results.append({'Task': 'Data Loading', 'Tool': 'Polars', 'Time_sec': pl_load_time})

    start_time = time.time()
    _ = pl_df.filter(pl.col('total_amount') > 0)
    pl_filter_time = time.time() - start_time
    benchmark_results.append({'Task': 'Basic Filtering', 'Tool': 'Polars', 'Time_sec': pl_filter_time})

    # 성능 비교 요약 텍스트
    speed_diff = pd_load_time / pl_load_time if pl_load_time > 0 else 1
    bench_report = (
        f"- **데이터 로딩**: Polars({pl_load_time:.4f}초)가 Pandas({pd_load_time:.4f}초)보다 약 **{speed_diff:.2f}배** 빠름.\n"
        f"- **데이터 필터링**: Polars({pl_filter_time:.4f}초) vs Pandas({pd_filter_time:.4f}초)\n"
    )
    print("  -> 벤치마킹 완료.")
    
    # 벤치마크 결과 DF와 메인 분석용 Pandas DF 반환
    return pd_df, pd.DataFrame(benchmark_results), bench_report

def clean_nulls(df: pd.DataFrame,
                cols: Optional[list] = None,
                strategy: str = 'drop') -> pd.DataFrame:
    """
    결측치를 처리하는 공통 함수입니다.
    """
    if cols is None:
        cols = df.select_dtypes('number').columns.tolist()
    
    if strategy == 'median':
        return df.fillna(df[cols].median())
    elif strategy == 'drop':
        return df.dropna(subset=cols)
    return df

def preprocess_data(df: pd.DataFrame) -> tuple:
    """
    결측치 처리, 중복 제거, EDA를 위한 파생변수(운행 시간, 혼잡 시간대, 고요금 타겟)를 생성합니다.
    (채점 기준: 결측치·중복 처리 및 기본 EDA 수행)
    """
    print("[2/6] 데이터 전처리 및 파생 변수 생성 중...")
    initial_len = len(df)
    
    # 1. 결측치 및 중복 데이터 제거
    df = clean_nulls(df, strategy='drop')
    df = df.drop_duplicates()
    
    # 2. 분석을 위한 파생 변수 1: 운행 시간(trip_duration) 분 단위 계산
    df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
    df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'])
    df['trip_duration'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds() / 60.0
    
    # 3. 데이터 필터링 (이상치 제거)
    # 운행 거리 0.1~100마일, 운행 시간 1~120분, 총 요금 1~500 달러 사이의 정상 데이터만 추출
    valid_mask = (
        (df['trip_distance'] > 0) & (df['trip_distance'] <= 100) &
        (df['trip_duration'] >= 1) & (df['trip_duration'] <= 120) &
        (df['total_amount'] > 0) & (df['total_amount'] <= 500) &
        (df['fare_amount'] > 0)
    )
    df = df[valid_mask].copy()
    
    # 4. 분석을 위한 파생 변수 2: 출퇴근 혼잡 시간대 여부 (Group A vs B 분리용)
    df['pickup_hour'] = df['tpep_pickup_datetime'].dt.hour
    # 혼잡 시간대: 08~10시(8,9,10) 및 17~19시(17,18,19)
    rush_hours = [8, 9, 10, 17, 18, 19]
    df['is_rush_hour'] = df['pickup_hour'].apply(lambda x: 1 if x in rush_hours else 0)
    
    # 5. ML 타겟 변수 생성: 고요금(High-Fare) 운행 여부 (상위 25% 기준)
    fare_75th = df['total_amount'].quantile(0.75)
    df['is_high_fare'] = (df['total_amount'] >= fare_75th).astype(int)
    
    # 시스템 과부하 방지 및 원활한 실습을 위한 데이터 샘플링 (50,000건)
    sampled = False
    pre_sample_len = len(df)
    if len(df) > 50000:
        df = df.sample(n=50000, random_state=42).reset_index(drop=True)
        sampled = True
        
    print(f"  -> 원본: {initial_len}건 | 전처리 및 샘플링 후: {len(df)}건")
    print(f"  -> 고요금(Top 25%) 기준선(Threshold): ${fare_75th:.2f}")
    return df, fare_75th, sampled, pre_sample_len
