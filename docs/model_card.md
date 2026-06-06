---
title: Model Card — credit-risk-xgboost
updated: 2026-06-06
---

# Model Card — credit-risk-xgboost

## Visão Geral

| Campo | Valor |
|-------|-------|
| **Algoritmo** | XGBoost (Gradient Boosting) |
| **Versão** | v1 |
| **Tarefa** | Classificação binária (risco de crédito) |
| **Dataset** | Home Credit Default Risk |
| **Data de treino** | 2026-06-06 |

## Métricas de Avaliação (validation set, 20%)

| Métrica | Valor | Notas |
|---------|-------|-------|
| AUC-ROC | **0.7683** | Acima do alvo de 0.74 |
| KS Statistic | **0.4024** | Boa separação de classes |
| Avg Precision | **0.2586** | Relevante para dados desbalanceados (8% positivos) |
| F1 Score | **0.2958** | Threshold = 0.5 |

## Hiperparâmetros

| Parâmetro | Valor |
|-----------|-------|
| n_estimators | 500 |
| max_depth | 6 |
| learning_rate | 0.05 |
| scale_pos_weight | 10 |
| subsample | 0.8 |
| colsample_bytree | 0.8 |

## Uso Pretendido

- ✅ Ranqueamento de risco de solicitantes de crédito
- ✅ Triagem inicial em processos de concessão de crédito
- ✅ Demonstração de MLOps e pipeline de produção (portfólio)

## Uso Não Pretendido

- ❌ Decisão automatizada final de concessão sem revisão humana
- ❌ Aplicação em mercados sem calibração local do threshold
- ❌ Uso com dados de distribuição muito diferente do dataset de treino

## Limitações

- Modelo treinado em dados históricos — deriva se o perfil de solicitantes mudar
- Threshold de 0.5 é arbitrário — ajustar com base no custo relativo de FP vs FN
- Explainability individual (SHAP) planejada para v2
- 8% de inadimplência no treino — recalibrar em mercados com taxa diferente
