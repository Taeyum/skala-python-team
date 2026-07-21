# 작성자 : 이다은
# 작성일 : 2026-07-21
"""전처리+분류모델을 Pipeline으로 구성해 고요금(is_high_fare) 여부를 예측하고 저장."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET


def build_ml_pipeline(df: pd.DataFrame, model_path: Path) -> dict:
    """전처리 + 분류 모델을 Pipeline으로 구성해 is_high_fare를 예측하고 저장한다."""
    feature_df = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    target = df[TARGET]

    x_train, x_test, y_train, y_test = train_test_split(
        feature_df, target, test_size=0.2, random_state=42, stratify=target
    )

    # 수치형/범주형 컬럼은 전처리 방식이 다르므로 각각 파이프라인으로 구성 후 결합
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )

    # class_weight="balanced": 고요금(1)이 전체의 25%뿐인 불균형 데이터이므로
    # 소수 클래스를 무시하지 않도록 가중치를 보정한다.
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(max_iter=1000, class_weight="balanced"),
            ),
        ]
    )

    print("\n=== sklearn Pipeline 학습 (고요금 운행 분류) ===")
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)

    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    print(f"정확도(Accuracy): {accuracy:.4f}")
    print(f"F1-score: {f1:.4f}")

    joblib.dump(pipeline, model_path)
    print(f"모델 저장 완료: {model_path}")

    return {"accuracy": accuracy, "f1_score": f1}
