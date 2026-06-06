# tests/test_api.py
import pytest
from pydantic import ValidationError

from src.api.schemas import HealthResponse, MetricsResponse, PredictionRequest, PredictionResponse

VALID_INPUT = {
    "AMT_CREDIT": 500000.0,
    "AMT_INCOME_TOTAL": 150000.0,
    "AMT_ANNUITY": 25000.0,
    "DAYS_BIRTH": -14000,
    "DAYS_EMPLOYED": -2000,
}


def test_prediction_request_valid():
    req = PredictionRequest(**VALID_INPUT)
    assert req.AMT_CREDIT == 500000.0


def test_prediction_request_missing_required_field():
    with pytest.raises(ValidationError):
        PredictionRequest(AMT_CREDIT=500000.0)  # missing required fields


def test_prediction_request_negative_credit_fails():
    bad_input = VALID_INPUT.copy()
    bad_input["AMT_CREDIT"] = -100.0
    with pytest.raises(ValidationError):
        PredictionRequest(**bad_input)


def test_prediction_request_zero_income_fails():
    bad_input = VALID_INPUT.copy()
    bad_input["AMT_INCOME_TOTAL"] = 0.0
    with pytest.raises(ValidationError):
        PredictionRequest(**bad_input)


def test_prediction_response_schema():
    resp = PredictionResponse(
        default_probability=0.2,
        prediction=0,
        threshold=0.5,
        model_version="1",
    )
    assert resp.prediction == 0


def test_health_response_schema():
    h = HealthResponse(
        status="healthy",
        model_loaded=True,
        model_name="credit-risk-xgboost",
        model_version="1",
    )
    assert h.status == "healthy"


def test_metrics_response_schema():
    m = MetricsResponse(
        total_predictions=100,
        avg_default_probability=0.15,
        predictions_above_threshold=10,
    )
    assert m.total_predictions == 100


# ---------------------------------------------------------------------------
# Task 10 — predict module tests
# ---------------------------------------------------------------------------
from unittest.mock import MagicMock

import numpy as np

MOCK_FEATURE_NAMES = [
    "AMT_CREDIT",
    "AMT_INCOME_TOTAL",
    "AMT_ANNUITY",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "CODE_GENDER_M",
    "FLAG_OWN_CAR_Y",
    "FLAG_OWN_REALTY_Y",
    "CNT_CHILDREN",
    "CNT_FAM_MEMBERS",
    "CREDIT_INCOME_RATIO",
    "ANNUITY_INCOME_RATIO",
    "CREDIT_TERM",
    "AGE_YEARS",
    "EMPLOYMENT_YEARS",
]


def make_mock_model(prob: float = 0.2):
    mock = MagicMock()
    mock.predict_proba.return_value = np.array([[1 - prob, prob]])
    mock.get_booster.return_value.feature_names = MOCK_FEATURE_NAMES
    return mock


def test_predict_returns_dict():
    import src.api.predict as pred

    pred._model = make_mock_model(0.2)
    pred._model_version = "1"

    result = pred.predict(VALID_INPUT)
    assert isinstance(result, dict)
    assert "default_probability" in result
    assert "prediction" in result
    assert "threshold" in result
    assert "model_version" in result


def test_predict_probability_low_means_class_zero():
    import src.api.predict as pred

    pred._model = make_mock_model(0.1)
    pred._model_version = "1"

    result = pred.predict(VALID_INPUT)
    assert result["prediction"] == 0
    assert result["default_probability"] == pytest.approx(0.1, abs=0.01)


def test_predict_probability_high_means_class_one():
    import src.api.predict as pred

    pred._model = make_mock_model(0.9)
    pred._model_version = "1"

    result = pred.predict(VALID_INPUT)
    assert result["prediction"] == 1


def test_predict_missing_features_filled_with_zero():
    import src.api.predict as pred

    pred._model = make_mock_model(0.3)
    pred._model_version = "1"

    minimal_input = {k: v for k, v in VALID_INPUT.items()}
    result = pred.predict(minimal_input)
    assert "default_probability" in result


# ---------------------------------------------------------------------------
# Task 11 — FastAPI endpoint tests
# ---------------------------------------------------------------------------
from fastapi.testclient import TestClient


@pytest.fixture
def api_client():
    """TestClient com modelo mockado para testes de endpoint."""
    import src.api.predict as pred

    pred._model = make_mock_model(0.25)
    pred._model_version = "2"

    from src.api.main import app

    return TestClient(app)


def test_health_endpoint_200(api_client):
    resp = api_client.get("/health")
    assert resp.status_code == 200


def test_health_model_loaded(api_client):
    data = api_client.get("/health").json()
    assert data["model_loaded"] is True
    assert data["status"] == "healthy"
    assert data["model_version"] == "2"


def test_predict_endpoint_200(api_client):
    resp = api_client.post("/predict", json=VALID_INPUT)
    assert resp.status_code == 200


def test_predict_response_fields(api_client):
    data = api_client.post("/predict", json=VALID_INPUT).json()
    assert "default_probability" in data
    assert "prediction" in data
    assert "threshold" in data
    assert "model_version" in data


def test_predict_probability_between_0_and_1(api_client):
    data = api_client.post("/predict", json=VALID_INPUT).json()
    assert 0.0 <= data["default_probability"] <= 1.0


def test_predict_invalid_payload_422(api_client):
    resp = api_client.post("/predict", json={"campo_invalido": "valor"})
    assert resp.status_code == 422


def test_predict_negative_credit_422(api_client):
    bad = VALID_INPUT.copy()
    bad["AMT_CREDIT"] = -100.0
    resp = api_client.post("/predict", json=bad)
    assert resp.status_code == 422


def test_metrics_endpoint_200(api_client):
    resp = api_client.get("/metrics")
    assert resp.status_code == 200


def test_metrics_response_fields(api_client):
    data = api_client.get("/metrics").json()
    assert "total_predictions" in data
    assert "avg_default_probability" in data
    assert "predictions_above_threshold" in data


def test_metrics_increments_after_predict(api_client):
    initial = api_client.get("/metrics").json()["total_predictions"]
    api_client.post("/predict", json=VALID_INPUT)
    after = api_client.get("/metrics").json()["total_predictions"]
    assert after == initial + 1
