# skala-python-team

## 프로젝트 개요
SKALA 파이썬 데이터 분석 팀 실습 프로젝트

## 폴더 구조
```
skala-python-team/
├── data/
│   ├── raw/          # 원본 데이터 (절대 수정 금지, git 추적 안 함)
│   ├── processed/    # 전처리 완료 데이터 (git 추적 안 함)
│   └── external/     # 외부 참조 데이터 (git 추적 안 함)
├── notebooks/         # EDA·실험용 Jupyter 노트북
├── src/                # 재사용 가능한 Python 모듈
│   └── __init__.py
├── tests/              # pytest 테스트 코드
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
