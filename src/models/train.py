# src/models/train.py
"""
Treina XGBoost no dataset processado e loga experimento no MLflow.
Uso: python -m src.models.train
"""

import os
from pathlib import Path

import mlflow
import mlflow.xgboost
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from src.data.features import engineer_features
from src.models.evaluate import compute_metrics

PROCESSED = Path(__file__).resolve().parents[2] / "data" / "processed" / "train.parquet"
TARGET_COL = "TARGET"
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = "credit-risk"

DEFAULT_PARAMS = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "scale_pos_weight": 10,
    "eval_metric": "auc",
    "random_state": 42,
    "n_jobs": -1,
}


def load_data(path: Path = PROCESSED):
    """Carrega parquet processado e separa features do target."""
    if not path.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {path}\n"
            "Execute primeiro: python -m src.data.download && python -m src.data.preprocess"
        )
    df = pd.read_parquet(path)
    df = engineer_features(df)  # mesmo pipeline de predict.py
    X = df.drop(columns=[TARGET_COL, "SK_ID_CURR"], errors="ignore")
    y = df[TARGET_COL]
    return X, y


def train(params: dict = None, data_path: Path = PROCESSED) -> str:
    """
    Treina XGBoost, loga no MLflow (params + métricas + artefato do modelo).
    Retorna o run_id do experimento criado.
    """
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    params = params or DEFAULT_PARAMS
    X, y = load_data(data_path)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    with mlflow.start_run() as run:
        model = XGBClassifier(**params)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=50,
        )

        mlflow.log_params(params)

        metrics = compute_metrics(model, X_val, y_val)
        mlflow.log_metrics(metrics)

        mlflow.xgboost.log_model(model, artifact_path="model")

        run_id = run.info.run_id
        print(f"\nRun ID: {run_id}")
        print(f"Métricas: {metrics}")
        return run_id


if __name__ == "__main__":
    train()
