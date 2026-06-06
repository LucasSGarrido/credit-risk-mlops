# src/api/predict.py
"""
Carrega o modelo do MLflow Registry e faz inferência.
O modelo é carregado uma vez no startup e cacheado em memória.
"""

import os
from typing import Optional

import mlflow
import mlflow.xgboost
import pandas as pd

from src.data.features import engineer_features

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME = os.getenv("MODEL_NAME", "credit-risk-xgboost")
THRESHOLD = float(os.getenv("PREDICTION_THRESHOLD", "0.5"))

# Cache em memória — carregado no startup da API
_model = None
_model_version: Optional[str] = None


def load_model() -> None:
    """Carrega o modelo em Production do MLflow Registry para memória."""
    global _model, _model_version
    mlflow.set_tracking_uri(MLFLOW_URI)
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])
    if not versions:
        raise RuntimeError(
            f"Nenhuma versão em Production para o modelo '{MODEL_NAME}'. "
            "Execute register.py primeiro."
        )
    _model_version = versions[0].version
    model_uri = f"models:/{MODEL_NAME}/Production"
    _model = mlflow.xgboost.load_model(model_uri)
    print(f"Modelo '{MODEL_NAME}' v{_model_version} carregado com sucesso.")


def get_model():
    """Retorna o modelo cacheado, carregando se necessário."""
    global _model
    if _model is None:
        load_model()
    return _model


def predict(features: dict) -> dict:
    """
    Recebe dicionário de features brutas, computa features engineered
    (mesmo pipeline do treinamento) e retorna predição do modelo.
    Colunas restantes ausentes são preenchidas com 0.
    """
    model = get_model()
    df = pd.DataFrame([features])

    # Reproduzir o mesmo pipeline de feature engineering do treinamento
    df = engineer_features(df)

    # Alinhar colunas com as do modelo (preencher ausentes com 0)
    model_features = model.get_booster().feature_names
    for col in model_features:
        if col not in df.columns:
            df[col] = 0.0
    df = df[model_features]

    prob = float(model.predict_proba(df)[0, 1])
    pred = int(prob >= THRESHOLD)

    return {
        "default_probability": prob,
        "prediction": pred,
        "threshold": THRESHOLD,
        "model_version": str(_model_version),
    }
