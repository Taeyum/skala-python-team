# skala-python-team

## 프로젝트 개요
SKALA 파이썬 데이터 분석 팀 종합실습(Day 2 End2End 프로젝트) 저장소입니다.

**주제**: Pandas vs Polars 처리 성능 비교 기반 고요금(High-Fare) 운행 예측
**데이터**: [NYC TLC Yellow Taxi Trip Records](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) 2026년 5월분 (약 409만 건의 택시 운행 기록)

뉴욕 옐로우캡의 한 달치 운행 데이터를 가지고,
1. 같은 작업을 Pandas와 Polars 양쪽으로 구현해 처리 속도를 비교하고
2. 정제된 데이터로 요금 관련 패턴을 시각화·통계 검정하고
3. 한 운행이 "고요금(상위 25%)"에 해당하는지 예측하는 분류 모델을 만들고
4. 전체 분석 결과를 `report.md`로 자동 생성

하는 것까지 하나의 파이프라인(`python -m src.main`)으로 실행되도록 구성했습니다.

## 분석 파이프라인

| 단계 | 내용 | 담당 모듈 |
|---|---|---|
| 1. 데이터 준비 | Pandas·Polars 로딩/필터링/집계 속도 비교, 결측치·중복 처리, 이상치(비즈니스 규칙+IQR) 정제 | `src/io_compare.py`, `src/eda.py` |
| 2. 시각화 | Seaborn 처리속도 비교 바차트, Plotly 시간대별 평균요금 인터랙티브 라인차트 | `src/visualize.py` |
| 3. 통계 분석 | 상관계수(거리·시간↔요금) + 혼잡시간대 t-test | `src/stats_analysis.py` |
| 4. ML Pipeline | `is_high_fare`(고요금 여부) 분류, ColumnTransformer+LogisticRegression, joblib 저장/재로딩 | `src/ml_pipeline.py` |
| 5. 리포트 자동화 | 위 결과를 Jinja2 템플릿으로 `report.md` 생성 | `src/report.py`, `templates/report_template.md.j2` |

가공 없는 원본 그대로 쓰면 안 되는 이유(음수 요금, 0거리 트립, 307,491마일짜리 이상치 등)를 확인하고 정제 규칙을 만들었고, `is_high_fare`는 `total_amount`에서 파생된 값이라 요금 구성 컬럼(fare_amount, tip_amount 등)은 예측 피처에서 제외해 데이터 누수를 방지했습니다.

## 주요 결과

### Pandas vs Polars 처리 속도
| 작업 | Pandas | Polars |
|---|---|---|
| 로딩 | ~0.22초 | ~0.14초 (더 빠름) |
| 필터링 | ~0.19초 | ~0.20초 (비슷하거나 Pandas가 근소 우위) |
| 시간대별 집계 | ~0.07초 | ~0.02초 (3~4배 빠름) |

→ 대용량(409만행) 단순 로딩·집계에서는 Polars가 확실히 유리하지만, 조건 필터링처럼 작업에 따라서는 차이가 미미하거나 역전되기도 함.

### 데이터 정제
- 원본 4,090,836행 → 비즈니스 규칙(음수요금·0거리·날짜범위·운행시간>0) + IQR 적용 후 **3,434,526행** (16% 제거)
- 결측치: `passenger_count`/`RatecodeID`/`store_and_fwd_flag`/`congestion_surcharge`/`Airport_fee` 각 23.35%

### 통계 분석
- **가설1 (운행거리·시간과 총요금은 강한 양의 상관관계)**: **채택** — `trip_distance`-`total_amount` r=0.672, `trip_duration`-`total_amount` r=0.705
- **가설2 (혼잡시간대(08-10,17-19시)가 비혼잡시간대보다 평균요금이 높다)**: **기각** — 혼잡시간대 평균 $24.71 vs 비혼잡시간대 $24.88로 오히려 낮음(p=2.9e-34로 유의미하지만 방향이 반대). 단위 거리/시간당 단가는 혼잡시간대가 더 비싸지만(정체로 인한 시간요금 가산), 혼잡시간대엔 짧은 시내 통근 트립이 많아 총 이동거리가 짧고, 그 결과 총액 기준으로는 역전됨.

### ML Pipeline — is_high_fare 분류
- 타겟: `total_amount` 상위 25% (임계값 $29.82)
- LogisticRegression(class_weight=balanced), 피처: 거리·운행시간·승객수·시간대·요일·결제수단 (요금 관련 컬럼 제외)
- **Accuracy 0.896 / Precision 0.738 / Recall 0.908 / F1 0.814**

전체 결과와 상세 표는 파이프라인 실행 후 생성되는 [`report.md`](./report.md)에서 확인할 수 있습니다.

## 폴더 구조
```
skala-python-team/
├── data/
│   ├── raw/          # 원본 데이터 (절대 수정 금지, git 추적 안 함)
│   ├── processed/    # 전처리 완료 데이터 (git 추적 안 함)
│   └── external/     # 외부 참조 데이터 (git 추적 안 함)
├── notebooks/
│   └── end2end_analysis.ipynb   # 파이프라인을 셀 단위로 대화형 실행
├── src/                          # 재사용 가능한 Python 모듈
│   ├── io_compare.py             # Pandas/Polars 벤치마크
│   ├── eda.py                    # 결측치·중복·이상치 정제, 기술통계
│   ├── visualize.py              # Seaborn/Plotly 차트
│   ├── stats_analysis.py         # 상관계수·t-test
│   ├── ml_pipeline.py            # is_high_fare 분류 Pipeline
│   ├── report.py                 # report.md 자동 생성
│   └── main.py                   # 전체 파이프라인 진입점
├── templates/
│   └── report_template.md.j2     # report.md Jinja2 템플릿
├── tests/                        # pytest 테스트 코드 (34개)
├── output/                       # 산출물(차트 PNG/HTML, 모델 joblib)
├── report.md                     # 자동 생성되는 분석 리포트
├── requirements.txt
└── README.md
```

## 개발 환경 설정
```bash
git clone https://github.com/Taeyum/skala-python-team.git
cd skala-python-team
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 실행 방법
```bash
# 전체 파이프라인 한 번에 실행 (데이터준비→시각화→통계→ML→report.md)
python -m src.main

# 노트북으로 단계별 대화형 실행
jupyter notebook notebooks/end2end_analysis.ipynb

# 테스트 실행
pytest tests/
```

## 데이터 출처
- `data/raw/`에는 원본 데이터를 직접 커밋하지 않습니다. 데이터 출처(URL·수집 스크립트)는 이 섹션에 기록합니다.

| 파일 | 출처 | 다운로드 |
|---|---|---|
| `yellow_tripdata_2026-05.parquet` | NYC TLC Trip Record Data (Yellow Taxi, 2026-05) | `curl -L -o data/raw/yellow_tripdata_2026-05.parquet https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2026-05.parquet` |

## 브랜치 전략
- `main`: 프로젝트 공통 구조 및 리뷰 완료된 코드
- 팀원별 개인 브랜치에서 작업 후 PR로 병합
