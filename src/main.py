from src.clean import download_and_benchmark, preprocess_data
from src.viz import generate_visualizations
from src.stats import run_statistical_analysis
from src.models import build_ml_pipeline
from src.report import generate_markdown_report

def main():
    target_url = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2026-05.parquet"
    
    # 각 단계별 결과를 저장할 변수 초기화
    raw_df = bench_df = bench_report = None
    clean_df = fare_threshold = None
    stats_report = ml_report = None
    sampled = False
    pre_sample_len = 0
    
    # 1. 다운로드 및 벤치마킹 수행
    try:
        raw_df, bench_df, bench_report = download_and_benchmark(target_url)
    except Exception as e:
        print(f"\n[ERROR] [1/6] 데이터 다운로드/벤치마킹 단계에서 오류 발생: {str(e)}")
        return
    
    # 2. 전처리 및 파생변수 생성
    try:
        clean_df, fare_threshold, sampled, pre_sample_len = preprocess_data(raw_df)
    except Exception as e:
        print(f"\n[ERROR] [2/6] 데이터 전처리 단계에서 오류 발생: {str(e)}")
        return
    
    # 3. 데이터 시각화 (이미지 및 HTML 저장)
    try:
        generate_visualizations(bench_df, clean_df)
    except Exception as e:
        print(f"\n[ERROR] [3/6] 시각화 생성 단계에서 오류 발생: {str(e)}")
    
    # 4. 통계 및 가설 검정
    try:
        stats_report = run_statistical_analysis(clean_df)
    except Exception as e:
        print(f"\n[ERROR] [4/6] 통계 분석 단계에서 오류 발생: {str(e)}")
        stats_report = "(통계 분석 수행 중 오류가 발생하여 결과를 생성하지 못했습니다.)\n"
    
    # 5. ML 분류 파이프라인
    try:
        ml_report = build_ml_pipeline(clean_df)
    except Exception as e:
        print(f"\n[ERROR] [5/6] ML 파이프라인 단계에서 오류 발생: {str(e)}")
        ml_report = "(ML 파이프라인 수행 중 오류가 발생하여 결과를 생성하지 못했습니다.)\n"
    
    # 6. 자동화 리포트 생성
    try:
        generate_markdown_report(bench_report, stats_report, ml_report,
                                fare_threshold, sampled, pre_sample_len)
    except Exception as e:
        print(f"\n[ERROR] [6/6] 리포트 생성 단계에서 오류 발생: {str(e)}")

if __name__ == "__main__":
    main()
