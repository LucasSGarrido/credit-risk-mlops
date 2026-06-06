---
title: Arquitetura — Credit Risk MLOps Platform
updated: 2026-06-06
---

# Arquitetura — Credit Risk MLOps Platform

## Visão Geral das 4 Camadas

```
┌─────────────────────────────────────────────────┐
│  CAMADA 1 — DATA                                │
│  src/data/download.py  → Kaggle API             │
│  src/data/preprocess.py → limpeza + encoding    │
│  src/data/features.py  → feature engineering   │
│  Saída: data/processed/train.parquet            │
└──────────────────┬──────────────────────────────┘
                   ↓ parquet
┌─────────────────────────────────────────────────┐
│  CAMADA 2 — TRAINING                            │
│  src/models/train.py   → XGBoost + MLflow       │
│  src/models/evaluate.py → métricas              │
│  src/models/register.py → Model Registry       │
│  MLflow: tracking + registry + PostgreSQL       │
└──────────────────┬──────────────────────────────┘
                   ↓ MLflow Registry (HTTP)
┌─────────────────────────────────────────────────┐
│  CAMADA 3 — SERVING                             │
│  src/api/predict.py → carrega de Production     │
│  src/api/main.py    → FastAPI                   │
│  POST /predict | GET /health | GET /metrics     │
└──────────────────┬──────────────────────────────┘
                   ↓ HTTP + Docker network
┌─────────────────────────────────────────────────┐
│  CAMADA 4 — INFRA & MONITORING                  │
│  docker-compose: api + mlflow + postgres        │
│  GitHub Actions: lint → test → build → push     │
│  src/monitoring/dashboard.py → Streamlit        │
└─────────────────────────────────────────────────┘
```

## Decisões de Design

### Por que XGBoost e não LightGBM/CatBoost?
XGBoost é o padrão da indústria financeira, tem suporte nativo no MLflow e é defensável em entrevista.

### Por que PostgreSQL como backend do MLflow?
Persistência real dos experimentos — reiniciar os containers não perde o histórico.

### Por que FastAPI e não Flask?
Geração automática de OpenAPI/Swagger, validação nativa com Pydantic, async-ready.

### Por que docker-compose e não Kubernetes?
Portfólio local → simplicidade. O objetivo é demonstrar containerização e orquestração sem overhead de k8s.
