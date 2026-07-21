from pathlib import Path

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def train_fare_model(df: pd.DataFrame, output_dir: Path):
    # 모델에 사용할 입력값
    feature_columns = [
        "trip_distance",
        "trip_duration_minutes",
        "pickup_hour",
    ]

    # 예측할 값
    target_column = "total_amount"

    X = df[feature_columns]
    y = df[target_column]

    # 학습용 데이터와 평가용 데이터 분리
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    # 전처리와 모델을 하나의 Pipeline으로 구성
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]
    )

    # 모델 학습
    pipeline.fit(X_train, y_train)

    # 평가용 데이터 예측
    y_pred = pipeline.predict(X_test)

    # 평가 지표 계산
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    metrics = {
        "MAE": mae,
        "MSE": mse,
        "R2": r2,
    }

    # 학습된 Pipeline 전체 저장
    model_path = output_dir / "fare_prediction_pipeline.joblib"
    joblib.dump(pipeline, model_path)

    return metrics, model_path