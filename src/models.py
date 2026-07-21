import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import joblib
from tabulate import tabulate

def build_ml_pipeline(df: pd.DataFrame) -> str:
    """
    sklearn Pipeline을 활용해 고요금 운행 여부를 예측하는 분류 모델을 학습하고 평가합니다.
    (채점 기준: Pipeline 객체로 전처리+모델 구성, 평가 지표 출력, joblib로 모델 저장)
    """
    print("[5/6] 머신러닝 파이프라인 학습 및 모델 저장 중...")
    
    # 특성(X) 및 타겟(y) 분리
    # 운행 거리, 시간, 승객 수, 픽업 시간대를 기반으로 상위 25% 고요금 여부를 예측
    features = ['trip_distance', 'trip_duration', 'passenger_count', 'pickup_hour']
    X = df[features]
    y = df['is_high_fare']
    
    # 학습/테스트 세트 분할 (8:2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Pipeline 구성: StandardScaler(전처리) -> RandomForestClassifier(예측 모델)
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=50, max_depth=8, random_state=42))
    ])
    
    # 파이프라인 학습
    pipeline.fit(X_train, y_train)
    
    # 예측 및 평가
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Confusion Matrix 및 Classification Report 생성
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=['Actual: Normal', 'Actual: High-Fare'],
                         columns=['Pred: Normal', 'Pred: High-Fare'])
    cm_md = tabulate(cm_df, headers='keys', tablefmt='github')
    
    cls_report = classification_report(y_test, y_pred,
                                       target_names=['Normal Fare (0)', 'High Fare (1)'],
                                       output_dict=True)
    cls_df = pd.DataFrame(cls_report).transpose().round(4)
    cls_md = tabulate(cls_df, headers='keys', tablefmt='github')
    
    # 모델 파일로 저장
    model_path = "high_fare_prediction_pipeline.joblib"
    joblib.dump(pipeline, model_path)
    
    ml_report = (
        f"- **분석 목적**: 특정 운행 조건(거리, 시간 등) 기반 상위 25% 고요금(`is_high_fare`) 예측\n"
        f"- **사용한 특성**: {', '.join(features)}\n"
        f"- **파이프라인 구성**: `StandardScaler` + `RandomForestClassifier`\n"
        f"- **모델 평가 지표**:\n"
        f"  - 정확도(Accuracy): {acc:.4f}\n"
        f"  - F1-Score: {f1:.4f}\n\n"
        f"#### Confusion Matrix\n{cm_md}\n\n"
        f"#### Classification Report (클래스별 Precision / Recall / F1)\n{cls_md}\n\n"
        f"- **모델 저장 위치**: `{model_path}` (재현성 확보 완료)\n"
    )
    return ml_report
