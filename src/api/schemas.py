# src/api/schemas.py
"""Modelos Pydantic para request/response da API."""

from typing import Optional

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Features de entrada para predição de risco de crédito."""

    AMT_CREDIT: float = Field(..., gt=0, description="Valor total do crédito solicitado (R$)")
    AMT_INCOME_TOTAL: float = Field(..., gt=0, description="Renda anual do solicitante (R$)")
    AMT_ANNUITY: float = Field(..., gt=0, description="Valor da parcela mensal (R$)")
    DAYS_BIRTH: int = Field(..., lt=0, description="Dias desde o nascimento (valor negativo)")
    DAYS_EMPLOYED: int = Field(
        ..., description="Dias de emprego (negativo) ou 365243 se desempregado"
    )

    CODE_GENDER_M: int = Field(default=0, ge=0, le=1, description="1 se masculino")
    FLAG_OWN_CAR_Y: int = Field(default=0, ge=0, le=1, description="1 se possui carro")
    FLAG_OWN_REALTY_Y: int = Field(default=0, ge=0, le=1, description="1 se possui imóvel")
    CNT_CHILDREN: int = Field(default=0, ge=0, description="Número de filhos")
    CNT_FAM_MEMBERS: float = Field(default=2.0, gt=0, description="Membros na família")

    model_config = {
        "json_schema_extra": {
            "example": {
                "AMT_CREDIT": 500000.0,
                "AMT_INCOME_TOTAL": 150000.0,
                "AMT_ANNUITY": 25000.0,
                "DAYS_BIRTH": -14000,
                "DAYS_EMPLOYED": -2000,
            }
        }
    }


class PredictionResponse(BaseModel):
    """Resposta da predição com probabilidade e classificação."""

    default_probability: float = Field(..., ge=0.0, le=1.0)
    prediction: int = Field(..., ge=0, le=1, description="0=adimplente, 1=inadimplente")
    threshold: float = Field(default=0.5)
    model_version: str


class HealthResponse(BaseModel):
    """Status de saúde da API e do modelo carregado."""

    status: str = Field(..., description="'healthy' ou 'degraded'")
    model_loaded: bool
    model_name: str
    model_version: Optional[str] = None


class MetricsResponse(BaseModel):
    """Métricas operacionais da API (in-memory, resetadas no restart)."""

    total_predictions: int
    avg_default_probability: float
    predictions_above_threshold: int
