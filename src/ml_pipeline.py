"""[Day2] ML Pipeline — is_high_fare(고요금) 분류.

src.eda.clean_trip_data로 정제된 DataFrame(trip_duration 포함)을 입력으로
total_amount 상위 25%를 고요금(1)/그 외(0)로 라벨링하고, ColumnTransformer
+ LogisticRegression Pipeline을 학습·평가·joblib 저장·재로딩까지 수행한다.

데이터 누수 방지: is_high_fare는 total_amount에서 파생되므로 total_amount·
fare_amount·tip_amount 등 요금 구성 컬럼은 피처에서 제외하고, 요금과 직접
무관한 거리/시간/승객수/시간대/요일/결제수단만 사용한다.
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_QUANTILE = 0.75
NUMERIC_FEATURES = ["trip_distance", "trip_duration", "passenger_count"]
CATEGORICAL_FEATURES = ["pickup_hour", "pickup_weekday", "payment_type"]


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """pickup_hour/pickup_weekday 파생 컬럼을 추가한다."""
    if "tpep_pickup_datetime" not in df.columns:
        raise KeyError("'tpep_pickup_datetime' 컬럼이 없습니다.")
    data = df.copy()
    data["pickup_hour"] = data["tpep_pickup_datetime"].dt.hour
    data["pickup_weekday"] = data["tpep_pickup_datetime"].dt.weekday
    return data


def make_target(
    df: pd.DataFrame, column: str = "total_amount", quantile: float = TARGET_QUANTILE
) -> tuple[pd.DataFrame, float]:
    """column 상위 (1-quantile)를 is_high_fare=1로 라벨링한다."""
    if column not in df.columns:
        raise KeyError(f"'{column}' 컬럼이 없습니다.")
    threshold = df[column].quantile(quantile)
    data = df.copy()
    data["is_high_fare"] = (data[column] >= threshold).astype(int)
    return data, threshold


def build_pipeline(
    num_cols: list[str] = NUMERIC_FEATURES, cat_cols: list[str] = CATEGORICAL_FEATURES
) -> Pipeline:
    """전처리(ColumnTransformer) + LogisticRegression을 하나의 Pipeline으로 구성한다."""
    preprocessor = ColumnTransformer(
        [
            (
                "num",
                Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]),
                num_cols,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cat_cols,
            ),
        ]
    )
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    return Pipeline([("prep", preprocessor), ("model", model)])


def train_evaluate_save(
    df: pd.DataFrame, model_path: str | Path = "output/model.joblib", random_state: int = 42
) -> dict:
    """is_high_fare 타겟으로 학습·평가한 뒤 joblib으로 저장·재로딩까지 검증한다."""
    required = {"total_amount", *NUMERIC_FEATURES, "tpep_pickup_datetime"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"학습에 필요한 컬럼이 없습니다: {sorted(missing)}")

    data = add_temporal_features(df)
    data, threshold = make_target(data)
    if len(data) < 10:
        raise ValueError(f"학습 가능한 행이 너무 적습니다: {len(data)}건")

    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X = data[feature_cols]
    y = data["is_high_fare"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    metrics = {
        "threshold": threshold,
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions),
        "recall": recall_score(y_test, predictions),
        "f1": f1_score(y_test, predictions),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
    }

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)

    reloaded = joblib.load(model_path)
    reload_matches = bool((reloaded.predict(X_test) == predictions).all())

    print(
        f"[Pipeline] is_high_fare 임계값(total_amount>={threshold:.2f}) | "
        f"학습 {len(X_train)}건 / 평가 {len(X_test)}건\n"
        f"accuracy={metrics['accuracy']:.4f} precision={metrics['precision']:.4f} "
        f"recall={metrics['recall']:.4f} f1={metrics['f1']:.4f}\n"
        f"confusion_matrix={metrics['confusion_matrix']}\n"
        f"저장={model_path}, 재로딩 예측 일치={reload_matches}"
    )

    return {
        **metrics,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "model_path": str(model_path),
        "reload_matches": reload_matches,
    }
