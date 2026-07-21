import numpy as np
import pandas as pd
import pytest

from src.ml_pipeline import add_temporal_features, build_pipeline, make_target, train_evaluate_save


@pytest.fixture
def ml_sample_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 200
    pickup = pd.to_datetime("2026-05-01") + pd.to_timedelta(rng.integers(0, 28 * 24 * 60, n), unit="m")
    distance = rng.uniform(0.5, 10, n)
    duration = rng.uniform(3, 40, n)
    return pd.DataFrame(
        {
            "tpep_pickup_datetime": pickup,
            "trip_distance": distance,
            "trip_duration": duration,
            "passenger_count": rng.integers(1, 5, n),
            "payment_type": rng.choice([1, 2], n),
            "total_amount": distance * 8 + duration * 1.5 + rng.normal(0, 2, n),
        }
    )


def test_add_temporal_features_adds_hour_and_weekday(ml_sample_df):
    result = add_temporal_features(ml_sample_df)
    assert "pickup_hour" in result.columns
    assert "pickup_weekday" in result.columns
    assert result["pickup_hour"].between(0, 23).all()
    assert result["pickup_weekday"].between(0, 6).all()


def test_make_target_labels_top_quantile(ml_sample_df):
    result, threshold = make_target(ml_sample_df, quantile=0.75)
    assert result["is_high_fare"].sum() == pytest.approx(len(ml_sample_df) * 0.25, abs=2)
    assert (result.loc[result["is_high_fare"] == 1, "total_amount"] >= threshold).all()


def test_build_pipeline_has_prep_and_model_steps():
    pipeline = build_pipeline()
    assert [name for name, _ in pipeline.steps] == ["prep", "model"]


def test_train_evaluate_save_returns_metrics_and_saves_model(ml_sample_df, tmp_path):
    model_path = tmp_path / "model.joblib"
    result = train_evaluate_save(ml_sample_df, model_path=model_path)

    for key in ["accuracy", "precision", "recall", "f1"]:
        assert 0.0 <= result[key] <= 1.0
    assert model_path.exists()
    assert result["reload_matches"] is True
    assert result["n_train"] + result["n_test"] == len(ml_sample_df)


def test_train_evaluate_save_missing_column_raises(ml_sample_df):
    df = ml_sample_df.drop(columns=["trip_distance"])
    with pytest.raises(KeyError):
        train_evaluate_save(df)


def test_train_evaluate_save_too_few_rows_raises(ml_sample_df):
    df = ml_sample_df.head(3)
    with pytest.raises(ValueError):
        train_evaluate_save(df)
