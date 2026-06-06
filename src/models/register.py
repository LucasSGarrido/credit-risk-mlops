# src/models/register.py
"""
Registra o melhor run do MLflow no Model Registry e o promove para Production.
Uso: python -m src.models.register
"""

import os

import mlflow
from mlflow.tracking import MlflowClient

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME = os.getenv("MODEL_NAME", "credit-risk-xgboost")
EXPERIMENT_NAME = "credit-risk"


def get_best_run(metric: str = "val_auc_roc") -> str:
    """Retorna o run_id com melhor métrica no experimento."""
    mlflow.set_tracking_uri(MLFLOW_URI)
    client = MlflowClient()
    exp = client.get_experiment_by_name(EXPERIMENT_NAME)
    if exp is None:
        raise ValueError(f"Experimento '{EXPERIMENT_NAME}' não encontrado no MLflow.")
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=[f"metrics.{metric} DESC"],
        max_results=1,
    )
    if not runs:
        raise ValueError("Nenhum run encontrado. Execute train.py primeiro.")
    best = runs[0]
    print(f"Melhor run: {best.info.run_id} — {metric}={best.data.metrics.get(metric, 'N/A'):.4f}")
    return best.info.run_id


def register_model(run_id: str = None) -> str:
    """
    Registra o modelo do run_id no Model Registry e promove para Production.
    Se run_id=None, usa o melhor run disponível.
    Retorna a versão do modelo registrada.
    """
    mlflow.set_tracking_uri(MLFLOW_URI)
    client = MlflowClient()

    if run_id is None:
        run_id = get_best_run()

    model_uri = f"runs:/{run_id}/model"
    result = mlflow.register_model(model_uri=model_uri, name=MODEL_NAME)

    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=result.version,
        stage="Production",
        archive_existing_versions=True,
    )
    print(f"Modelo '{MODEL_NAME}' v{result.version} promovido para Production.")
    return str(result.version)


if __name__ == "__main__":
    register_model()
