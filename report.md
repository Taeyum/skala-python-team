# NYC Yellow Taxi 고요금(High-Fare) 예측 분석 리포트

- 생성 시각: 2026-07-21 16:44:56
- 데이터: data/raw/yellow_tripdata_2026-05.parquet (원본 4,090,836행 → 정제 후 3,434,526행)

## 1. 데이터 준비

### Pandas vs Polars 처리 속도 비교
| 작업 | Pandas(초) | Polars(초) |
|---|---|---|
| 로딩 | 0.1420 | 0.1125 |
| 필터링 | 0.1734 | 0.1619 |
| 시간대별집계 | 0.0577 | 0.0159 |


### 결측치 (상위 컬럼)
| 컬럼 | 결측 비율(%) |
|---|---|
| passenger_count | 23.35 |
| RatecodeID | 23.35 |
| store_and_fwd_flag | 23.35 |
| congestion_surcharge | 23.35 |
| Airport_fee | 23.35 |


- 중복행 제거: 0건

### fare_amount / total_amount 기술통계
| 통계량 | fare_amount | total_amount |
|---|---|---|
| mean | 21.51 | 30.49 |
| std | 19.01 | 22.97 |
| min | -950.00 | -951.00 |
| 25% | 10.00 | 17.64 |
| 50% | 16.30 | 23.94 |
| 75% | 26.80 | 34.95 |
| max | 5525.99 | 5530.74 |


## 2. 시각화
- Seaborn 처리속도 비교 바차트: `output/benchmark_speed.png`
- Plotly 시간대별 평균 총요금 라인차트: `output/hourly_fare_line.html`

## 3. 통계 분석

### 상관계수 (가설1: 운행시간·거리는 총요금과 강한 양의 상관관계를 가질 것이다)
| | trip_distance | trip_duration | total_amount |
|---|---|---|---|
| trip_distance | 1.0000 | 0.6834 | 0.6722 |
| trip_duration | 0.6834 | 1.0000 | 0.7048 |
| total_amount | 0.6722 | 0.7048 | 1.0000 |


**결과**: [상관분석] trip_distance-total_amount r=0.6722, trip_duration-total_amount r=0.7048 -> 가설1 채택(강한 양의 상관 확인) (기준: r>=0.5)

### 혼잡시간대 t-test (가설2: 혼잡시간대(08-10,17-19시)는 비혼잡시간대보다 평균 총요금이 높을 것이다)
- 혼잡시간대 평균: 24.71 (n=1,089,260)
- 비혼잡시간대 평균: 24.88 (n=2,345,266)
- t=-12.2058, p=2.9033e-34

**결과**: [t-test] 혼잡시간대(n=1089260, mean=24.71) vs 비혼잡시간대(n=2345266, mean=24.88): t=-12.2058, p=2.9033e-34 -> p<0.05, 통계적으로 유의미한 차이 있음 | 가설2 기각(혼잡시간대 평균요금이 더 높다고 보기 어려움)

## 4. ML Pipeline — is_high_fare(고요금) 분류

- 타겟: total_amount 상위 25% (임계값 $29.82)
- 학습 2,747,620건 / 평가 686,906건
- 모델: ColumnTransformer(수치 스케일링+범주 원핫) + LogisticRegression(class_weight=balanced)

| 지표 | 값 |
|---|---|
| Accuracy | 0.8961 |
| Precision | 0.7380 |
| Recall | 0.9084 |
| F1 | 0.8144 |

### Confusion Matrix
| | 예측: 일반 | 예측: 고요금 |
|---|---|---|
| 실제: 일반 | 458867 | 55596 |
| 실제: 고요금 | 15804 | 156639 |

- 모델 저장: `output/model.joblib` (재로딩 예측 일치: True)

## 5. 결론

- 가설1: 채택
- 가설2: 기각