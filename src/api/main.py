# src/api/main.py
"""
FastAPI app — Credit Risk MLOps API.
Endpoints: POST /predict | GET /health | GET /metrics
"""

from collections import deque
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from src.api import predict as pred_module
from src.api.predict import MODEL_NAME, THRESHOLD, predict
from src.api.schemas import HealthResponse, MetricsResponse, PredictionRequest, PredictionResponse

# Log in-memory de predições (máx 10k, FIFO)
_prediction_log: deque = deque(maxlen=10_000)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: tentar carregar modelo antes de aceitar requests
    try:
        pred_module.get_model()
    except Exception as e:
        print(f"Aviso: modelo não carregado no startup: {e}")
    yield
    # Shutdown: nada a limpar


app = FastAPI(
    title="Credit Risk MLOps API",
    description=(
        "Predição de risco de inadimplência de crédito. " "Modelo XGBoost + MLflow + FastAPI."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    """Retorna status da API e do modelo carregado."""
    model_loaded = pred_module._model is not None
    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        model_loaded=model_loaded,
        model_name=MODEL_NAME,
        model_version=str(pred_module._model_version) if pred_module._model_version else None,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["ml"])
def predict_endpoint(request: PredictionRequest) -> PredictionResponse:
    """Recebe features financeiras e retorna probabilidade de inadimplência."""
    try:
        result = predict(request.model_dump())
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    _prediction_log.append(result["default_probability"])
    return PredictionResponse(**result)


@app.get("/metrics", response_model=MetricsResponse, tags=["ops"])
def metrics() -> MetricsResponse:
    """Métricas operacionais da API (in-memory, reseta no restart)."""
    probs = list(_prediction_log)
    return MetricsResponse(
        total_predictions=len(probs),
        avg_default_probability=float(sum(probs) / len(probs)) if probs else 0.0,
        predictions_above_threshold=sum(1 for p in probs if p >= THRESHOLD),
    )
