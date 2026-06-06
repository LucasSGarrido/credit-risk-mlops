# tests/test_models.py
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from xgboost import XGBClassifier

from src.models.evaluate import compute_metrics, ks_statistic


@pytest.fixture
def trained_model():
    """Modelo XGBoost treinado num dataset sintético para testes."""
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        random_state=42,
        weights=[0.9, 0.1],
    )
    X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(20)])
    y_series = pd.Series(y)
    model = XGBClassifier(
        n_estimators=10,
        random_state=42,
        eval_metric="auc",
    )
    model.fit(X_df, y_series)
    return model, X_df, y_series


def test_compute_metrics_returns_all_keys(trained_model):
    model, X_val, y_val = trained_model
    metrics = compute_metrics(model, X_val, y_val)
    assert "val_auc_roc" in metrics
    assert "val_avg_precision" in metrics
    assert "val_f1" in metrics
    assert "val_ks" in metrics


def test_compute_metrics_auc_in_range(trained_model):
    model, X_val, y_val = trained_model
    metrics = compute_metrics(model, X_val, y_val)
    assert 0.0 <= metrics["val_auc_roc"] <= 1.0


def test_compute_metrics_f1_in_range(trained_model):
    model, X_val, y_val = trained_model
    metrics = compute_metrics(model, X_val, y_val)
    assert 0.0 <= metrics["val_f1"] <= 1.0


def test_ks_statistic_in_range(trained_model):
    model, X_val, y_val = trained_model
    y_prob = model.predict_proba(X_val)[:, 1]
    ks = ks_statistic(y_val.values, y_prob)
    assert 0.0 <= ks <= 1.0


def test_ks_statistic_perfect_separation():
    """KS deve ser 1.0 para separação perfeita."""
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_prob = np.array([0.1, 0.1, 0.1, 0.9, 0.9, 0.9])
    ks = ks_statistic(y_true, y_prob)
    assert ks == pytest.approx(1.0, abs=0.01)


def test_compute_metrics_all_numeric(trained_model):
    model, X_val, y_val = trained_model
    metrics = compute_metrics(model, X_val, y_val)
    for key, value in metrics.items():
        assert isinstance(value, float), f"{key} não é float"
