# Credit Risk MLOps Platform

> Plataforma de Machine Learning end-to-end para predição de risco de inadimplência de crédito.
> Do dado bruto ao deploy: pipeline reproduzível, API REST, experiment tracking, CI/CD automatizado.

[![CI/CD](https://github.com/LucasSGarrido/credit-risk-mlops/actions/workflows/ci.yml/badge.svg)](https://github.com/LucasSGarrido/credit-risk-mlops/actions)
[![Python](https://img.shields.io/badge/python-3.12-blue)](https://www.python.org)
[![MLflow](https://img.shields.io/badge/MLflow-3.x-orange)](https://mlflow.org)
[![AUC-ROC](https://img.shields.io/badge/AUC--ROC-0.768-success)](docs/model_card.md)

---

## O Problema

Instituições financeiras precisam decidir em segundos se aprovam um crédito. Errar para um lado significa perder receita; errar para o outro significa calote.

**Este projeto demonstra como um modelo de ML vai do dado bruto à produção de forma confiável**: com rastreamento de experimentos, versionamento de modelos, API documentada, testes automatizados e CI/CD.

---

## Arquitetura

```
CSV Kaggle → Preprocess → Feature Eng → XGBoost → MLflow Registry
                                                         ↓
                                                   FastAPI /predict
                                                         ↓
                                              docker-compose + GitHub Actions
                                                         ↓
                                               Streamlit Dashboard
```

Cada camada tem uma responsabilidade única e pode ser testada isoladamente.

---

## Stack

| Categoria | Tecnologia |
|-----------|-----------|
| Modelo | XGBoost · Scikit-learn |
| Tracking | MLflow (tracking + model registry) |
| API | FastAPI · Pydantic · Uvicorn |
| Dados | Pandas · PyArrow · NumPy |
| Infra | Docker · docker-compose · PostgreSQL |
| CI/CD | GitHub Actions |
| Dashboard | Streamlit · Plotly |
| Qualidade | pytest · ruff · black |

---

## Dataset

**Home Credit Default Risk** (Kaggle)
- 307.511 solicitantes · 122 features originais
- 8,1% de inadimplência (dados desbalanceados)
- [Acessar dataset](https://www.kaggle.com/c/home-credit-default-risk)

---

## Métricas do Modelo

> Validation set (20% holdout, stratificado) — XGBoost v1

| Métrica | Valor | Contexto |
|---------|-------|---------|
| **AUC-ROC** | **0.768** | Separa bem adimplentes de inadimplentes |
| **KS Statistic** | **0.402** | Separação máxima entre distribuições de score |
| **Avg Precision** | **0.259** | Benchmark adequado para dados 8% positivos |
| **F1 Score** | **0.296** | Threshold = 0.5 (ajustável por custo de negócio) |

O modelo discrimina bem: solicitante de baixo risco (crédito/renda = 0.25) → prob. 5%; solicitante de alto risco (crédito/renda = 3.3) → prob. 83%.

---

## Como Executar

### Pré-requisitos
- Docker + docker-compose
- Python 3.12
- Conta Kaggle com API key

### 1. Clonar e configurar

```bash
git clone https://github.com/LucasSGarrido/credit-risk-mlops.git
cd credit-risk-mlops
cp .env.example .env
# Editar .env com KAGGLE_USERNAME e KAGGLE_KEY
```

### 2. Subir a infraestrutura

```bash
docker-compose up -d
```

Serviços disponíveis:
- MLflow UI: http://localhost:5000
- API docs: http://localhost:8000/docs
- Dashboard: `streamlit run src/monitoring/dashboard.py`

### 3. Baixar e processar dados

```bash
pip install -r requirements.txt
python -m src.data.download
python -c "
from src.data.preprocess import preprocess, save_processed
from src.data.features import engineer_features
df = preprocess()
df = engineer_features(df)
save_processed(df)
print('Dados processados com sucesso.')
"
```

### 4. Treinar e registrar modelo

```bash
python -m src.models.train
python -m src.models.register
```

### 5. Testar a API

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "AMT_CREDIT": 500000,
    "AMT_INCOME_TOTAL": 150000,
    "AMT_ANNUITY": 25000,
    "DAYS_BIRTH": -14000,
    "DAYS_EMPLOYED": -2000
  }'
```

### 6. Rodar testes

```bash
pytest tests/ -v
```

---

## Estrutura do Projeto

```
credit-risk-mlops/
├── src/
│   ├── data/          # Download, preprocess, features
│   ├── models/        # Train, evaluate, register
│   ├── api/           # FastAPI + schemas + predict
│   └── monitoring/    # Streamlit dashboard
├── tests/             # pytest (TDD)
├── docker/            # Dockerfiles
├── notebooks/         # EDA
├── docs/              # data_card, model_card, arquitetura
└── .github/workflows/ # CI/CD
```

---

## Documentação

- [Arquitetura detalhada](docs/arquitetura.md)
- [Data Card](docs/data_card.md)
- [Model Card](docs/model_card.md)
- [API Docs](http://localhost:8000/docs) (com servidor rodando)

---

## Autor

**Lucas Santos Garrido** — [LinkedIn](https://linkedin.com/in/lucas-garrido-8a119236a) · [GitHub](https://github.com/LucasSGarrido)
