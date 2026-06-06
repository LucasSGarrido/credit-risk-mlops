---
title: Data Card — Home Credit Default Risk
updated: 2026-06-06
---

# Data Card — Home Credit Default Risk

## Origem

| Campo | Valor |
|-------|-------|
| **Nome** | Home Credit Default Risk |
| **Fonte** | Kaggle Competition |
| **URL** | https://www.kaggle.com/c/home-credit-default-risk |
| **Organização** | Home Credit Group |
| **Licença** | Kaggle Competition rules — uso educacional/portfólio permitido |
| **Data de acesso** | Junho 2026 |

## Estatísticas

| Campo | Valor |
|-------|-------|
| Tabela principal | `application_train.csv` |
| Linhas | 307.511 solicitantes |
| Colunas brutas | 122 features |
| Target | `TARGET` (0=adimplente, 1=inadimplente) |
| Desbalanceamento | ~8,1% de inadimplência |
| Período | Não especificado (dados históricos) |
| País | Múltiplos países (Europa Central/Ásia) |

## Features Usadas na v1

Apenas `application_train.csv`. Features com >60% de missing foram removidas.

### Numéricas (exemplos)
- `AMT_CREDIT` — Valor total do crédito
- `AMT_INCOME_TOTAL` — Renda anual declarada
- `AMT_ANNUITY` — Valor da parcela
- `DAYS_BIRTH` — Dias desde o nascimento (negativo)
- `DAYS_EMPLOYED` — Dias no emprego atual

### Derivadas (feature engineering)
- `CREDIT_INCOME_RATIO` — AMT_CREDIT / AMT_INCOME_TOTAL
- `ANNUITY_INCOME_RATIO` — AMT_ANNUITY / AMT_INCOME_TOTAL
- `CREDIT_TERM` — AMT_CREDIT / AMT_ANNUITY
- `AGE_YEARS` — Idade em anos
- `EMPLOYMENT_YEARS` — Tempo de emprego em anos

## Decisões de Tratamento

| Decisão | Justificativa |
|---------|---------------|
| Remover colunas >60% missing | Alta ausência indica feature estruturalmente incompleta |
| Imputação por mediana | Mediana é robusta a outliers em dados financeiros |
| One-hot encoding | Compatível com XGBoost e interpretável |
| scale_pos_weight=10 | Penaliza erro na classe minoritária (inadimplentes) |
| Sem SMOTE | Oversampling sintético piora calibração de probabilidades |

## Limitações

- Dados históricos — distribuição pode ter derivado desde a coleta
- Contexto geográfico diferente do Brasil — calibrar threshold para o mercado local
- Tabelas auxiliares (bureau, previous applications) não usadas na v1 — potencial para melhoria
- Renda declarada pode ser imprecisa — campo não verificado
