"""전처리 + 분류 모델 Pipeline (고요금(is_high_fare) 운행 예측)"""

import joblib
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config


def build_pipeline():
    """수치형 StandardScaler + 범주형 OneHotEncoder 전처리와 LogisticRegression을
    하나의 Pipeline 객체로 구성해 반환"""
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), config.NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), config.CATEGORICAL_FEATURES),
    ])
    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(max_iter=1000)),
    ])


def train_evaluate_save(df, model_path):
    """is_high_fare를 예측하는 Pipeline을 학습·평가(정확도·정밀도·재현율·F1)하고
    joblib으로 저장한 뒤 재로딩 검증까지 수행"""
    X = df[config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES]
    y = df[config.TARGET_COL]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
    }
    print("[ML Pipeline] is_high_fare 분류 평가 지표")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")

    joblib.dump(pipeline, model_path)
    reloaded = joblib.load(model_path)
    reloaded_acc = accuracy_score(y_test, reloaded.predict(X_test))
    assert abs(reloaded_acc - metrics["accuracy"]) < 1e-9, "재로딩한 모델의 정확도가 원본과 다릅니다"
    print(f"[OK] 모델 저장({model_path}) 및 재로딩 검증 통과")

    return pipeline, metrics
