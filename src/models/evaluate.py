# src/models/evaluate.py
"""
Métricas de avaliação para o modelo de risco de crédito.
Foco em métricas de ranking (AUC, KS) mais que acurácia bruta.
"""

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    roc_auc_score,
)


def ks_statistic(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    KS Statistic: separação máxima entre distribuições de scores
    de bons e maus pagadores. Métrica padrão do setor financeiro.
    Retorna valor entre 0 (pior) e 1 (perfeito).
    """
    pos_probs = y_prob[y_true == 1]
    neg_probs = y_prob[y_true == 0]
    ks, _ = ks_2samp(pos_probs, neg_probs)
    return float(ks)


def compute_metrics(model, X_val: pd.DataFrame, y_val: pd.Series) -> dict:
    """
    Calcula o conjunto completo de métricas para um modelo treinado.
    Retorna dicionário com chaves padronizadas para logging no MLflow.
    """
    y_prob = model.predict_proba(X_val)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    return {
        "val_auc_roc": float(roc_auc_score(y_val, y_prob)),
        "val_avg_precision": float(average_precision_score(y_val, y_prob)),
        "val_f1": float(f1_score(y_val, y_pred, zero_division=0)),
        "val_ks": ks_statistic(y_val.values, y_prob),
    }
