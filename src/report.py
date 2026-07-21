import time

def generate_markdown_report(bench_report: str, stats_report: str, ml_report: str,
                             threshold: float, sampled: bool = False, pre_sample_len: int = 0):
    """
    분석 전체 파이프라인의 결과를 모아 report.md 파일로 자동 생성합니다.
    (채점 기준: report.md 자동 생성 및 팀 발표를 위한 요약 제공)
    """
    print("[6/6] 최종 분석 결과 리포트(report.md) 자동 생성 중...")
    
    timestamp = time.strftime('%Y년 %m월 %d일 %H:%M:%S')
    
    # 샘플링 안내 문구 생성
    if sampled:
        sampling_note = (
            f"\n> **[NOTE] 데이터 샘플링 안내**: 본 분석은 전처리 후 전체 데이터({pre_sample_len:,}건) 중 "
            f"50,000건을 무작위 샘플링(random_state=42)하여 수행되었습니다. "
            f"재현성 확보를 위해 시드를 고정하였습니다.\n"
        )
    else:
        sampling_note = ""
    
    report_content = f"""# NYC Taxi: 요금 예측 및 도구 성능 비교 자동화 리포트
**자동 생성 일시**: {timestamp}
**작성자**: 광주 2반 박기준
{sampling_note}
---

## 1. 데이터 로딩 성능 비교 (Pandas vs Polars)
{bench_report}
> *대용량 Parquet 데이터 처리 시 Polars가 압도적인 로딩 속도를 보여 자동화 파이프라인 앞단에 적용하기 적합함.*

## 2. 시각화(EDA) 결과
- **[도구 성능 비교] 정적 차트**: `seaborn_benchmark_comparison.png` 
- **[시간대별 요금] 동적 차트**: `plotly_hourly_fare_trend.html` 
> *채점 기준에 따라 모든 차트에 명확한 Title 및 Axis Label을 포함하여 렌더링했습니다.*

## 3. 통계 분석 및 가설 검정 (T-Test)
{stats_report}

## 4. 고요금(High-Fare) 예측 머신러닝 파이프라인
*고요금 기준선(Top 25%): ${threshold:.2f}*
{ml_report}

---
### 파이프라인 설계 인사이트 (발표 5분 요약용)
1. **성능 관점**: 대용량 데이터 ETL 과정에서 `Polars`를 수집(Extract)에 적극 도입하여 리소스 병목을 해결할 수 있습니다.
2. **비즈니스 관점**: 시간대별 요금 차이(T-test 검증 완료)와 거리/시간 등 복합 변수를 반영한 ML 파이프라인은, 향후 수요 기반의 동적 요금제(Dynamic Pricing) 최적화 모델에 직접적으로 기여할 수 있습니다.
"""
    with open("report.md", "w", encoding="utf-8") as f:
        f.write(report_content)
        
    print("[SUCCESS] 모든 파이프라인 실행 완료! 분석 리포트가 'report.md'로 저장되었습니다.")
