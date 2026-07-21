"""분석 결과를 report.md로 자동 생성"""

from jinja2 import Template

TEMPLATE = """# 요금 예측 & 도구 성능 비교 분석 리포트

- 생성 일시: {{ generated_at }}
- 데이터: NYC Yellow Taxi Trip Data (2026-05), 전처리 후 {{ n_rows }}건

## 1. Pandas vs Polars 처리 속도

| 단계 | Pandas(초) | Polars(초) |
|---|---|---|
| 로딩 | {{ "%.4f"|format(timings.load.Pandas) }} | {{ "%.4f"|format(timings.load.Polars) }} |
| 전처리 | {{ "%.4f"|format(timings.preprocess.Pandas) }} | {{ "%.4f"|format(timings.preprocess.Polars) }} |

![처리 속도 비교](speed_comparison.png)

## 2. 상관분석

가설 1: 운행 시간과 이동 거리는 총요금과 강한 양의 상관관계를 가질 것이다.

```
{{ correlation_table }}
```

## 3. t-test: 혼잡 시간대 vs 비혼잡 시간대 평균 총요금

가설 2: 출퇴근 혼잡 시간대(08~10시, 17~19시) 운행은 비혼잡 시간대보다 평균 총요금이 높을 것이다.

- 혼잡시간대 평균: {{ "%.2f"|format(ttest.group_a_mean) }}달러 (n={{ ttest.group_a_n }})
- 비혼잡시간대 평균: {{ "%.2f"|format(ttest.group_b_mean) }}달러 (n={{ ttest.group_b_n }})
- t통계량: {{ "%.4f"|format(ttest.t_stat) }}, p-value: {{ "%.6f"|format(ttest.p_value) }}
- 결론: {{ ttest.verdict }}

## 4. 시간대별 평균 총요금

[인터랙티브 차트 보기](hourly_fare_trend.html)

## 5. ML Pipeline: is_high_fare 분류 (총요금 상위 25%)

| 지표 | 값 |
|---|---|
{% for name, value in metrics.items() -%}
| {{ name }} | {{ "%.4f"|format(value) }} |
{% endfor %}

모델 파일: `{{ model_filename }}`
"""


def generate_report(context, save_path):
    """context 딕셔너리를 Jinja2 템플릿에 렌더링해 report.md로 저장"""
    report_text = Template(TEMPLATE).render(**context)
    save_path.write_text(report_text, encoding="utf-8")
    print(f"[OK] 리포트 생성: {save_path}")
