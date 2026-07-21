# 작성자 : 이다은
# 작성일 : 2026-07-21
# 본 파일은 SKALA Day2 종합 실습 파일입니다.
# 변경사항 내역
# 2026-07-21, 최초 작성 (단일 파일, Pandas vs Polars 성능 비교 + 고요금 운행 예측)
# 2026-07-21, 기능별 모듈로 분리 (config/data_loader/preprocessing/eda/
#             visualization/stats_analysis/ml_pipeline/report) 후 main.py에서 조립

"""Pandas vs Polars 처리 성능 비교 & 고요금(High-Fare) 운행 예측 - 엔트리 포인트

프로그램 설명
    NYC Yellow Taxi 2026-05 데이터를 Pandas와 Polars 양쪽으로 로딩/처리하여
    소요 시간을 비교하고, 고요금 운행 여부(is_high_fare)를 예측하는
    분류 모델을 sklearn Pipeline으로 학습한다.
    분석 결과는 report.md로 자동 생성된다.

모듈 구성 (src/)
    config.py         : 경로/상수/컬럼명 공통 설정
    data_loader.py     : 로컬 캐시 확보, Pandas/Polars 로딩·필터링·그룹집계 성능 비교
    preprocessing.py   : 결측치/중복 제거, 파생 컬럼(운행시간/혼잡여부/고요금타겟) 생성
    eda.py             : 기술통계 출력
    visualization.py   : Seaborn 정적 차트 + Plotly 인터랙티브 차트
    stats_analysis.py  : 상관분석(가설 1), t-test(가설 2)
    ml_pipeline.py     : sklearn Pipeline 학습/평가/joblib 저장
    report.py          : report.md 자동 생성
    main.py (본 파일)  : 위 모듈들을 순서대로 호출하는 엔트리 포인트

가설
    H1. 운행 시간(trip_duration)과 이동 거리(trip_distance)는
        총요금(total_amount)과 강한 양의 상관관계를 가질 것이다.
    H2. 출퇴근 혼잡 시간대(08~10시, 17~19시) 운행은 비혼잡 시간대보다
        평균 총요금이 높을 것이다.

실행 방법 (프로젝트 루트에서)
    python -m src.main
"""

import pandas as pd
import requests

from src.config import (
    DATA_PATH,
    DATA_URL,
    HOURLY_FARE_HTML_PATH,
    MODEL_PATH,
    OUTPUT_DIR,
    REPORT_PATH,
    SPEED_FIG_PATH,
)
from src.data_loader import compare_pandas_vs_polars, ensure_local_parquet
from src.eda import run_eda
from src.ml_pipeline import build_ml_pipeline
from src.preprocessing import clean_and_engineer_features
from src.report import generate_report_md
from src.stats_analysis import compute_correlations, run_ttest
from src.visualization import plot_hourly_fare_plotly, plot_speed_comparison_seaborn


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. 데이터 준비 (로컬 캐시 없으면 다운로드)
    local_path = ensure_local_parquet(DATA_URL, DATA_PATH)

    # 2. Pandas vs Polars 성능 비교 (로딩/필터링/그룹집계)
    pandas_df, timing_rows = compare_pandas_vs_polars(local_path)

    # 3. 전처리 및 파생변수 생성 (이후 단계는 Pandas 기준으로 진행)
    filtered_df = clean_and_engineer_features(pandas_df)

    # 4. EDA
    eda_summary = run_eda(filtered_df)

    # 5. 시각화
    plot_speed_comparison_seaborn(timing_rows, SPEED_FIG_PATH)
    plot_hourly_fare_plotly(filtered_df, HOURLY_FARE_HTML_PATH)

    # 6. 상관분석 (가설 1) / t-test (가설 2)
    corr = compute_correlations(filtered_df)
    ttest_result = run_ttest(filtered_df)

    # 7. ML Pipeline
    ml_metrics = build_ml_pipeline(filtered_df, MODEL_PATH)

    # 8. report.md 자동 생성
    generate_report_md(
        timing_rows, eda_summary, corr, ttest_result, ml_metrics, REPORT_PATH
    )


if __name__ == "__main__":
    try:
        main()

    except FileNotFoundError as error:
        print(f"파일 오류: {error}")

    except requests.exceptions.RequestException as error:
        print(f"네트워크 요청 실패: {error}")

    except pd.errors.EmptyDataError:
        print("데이터 파일이 비어 있습니다.")

    except KeyError as error:
        print(f"컬럼 오류: {error}")

    except TypeError as error:
        print(f"자료형 오류: {error}")

    except ValueError as error:
        print(f"데이터 오류: {error}")

    except ImportError as error:
        print(f"필수 라이브러리가 설치되어 있지 않습니다: {error}")

    except Exception as error:
        print(f"예상하지 못한 오류: {error}")
