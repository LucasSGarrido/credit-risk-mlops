# src/monitoring/dashboard.py
"""
Dashboard Streamlit — Monitoramento do modelo em produção.
Conecta ao MLflow para histórico de experimentos e à API para métricas operacionais.
Uso: streamlit run src/monitoring/dashboard.py
"""

import os

import mlflow
import pandas as pd
import plotly.express as px
import requests
import streamlit as st

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
API_URL = os.getenv("API_URL", "http://localhost:8000")
MODEL_NAME = os.getenv("MODEL_NAME", "credit-risk-xgboost")

st.set_page_config(
    page_title="Credit Risk MLOps — Dashboard",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Credit Risk MLOps — Dashboard de Monitoramento")
st.caption(f"MLflow: `{MLFLOW_URI}` | API: `{API_URL}`")
st.divider()

# ─── Modelo em Produção ──────────────────────────────────────────────────────
client = None

st.header("🤖 Modelo em Produção")

try:
    mlflow.set_tracking_uri(MLFLOW_URI)
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])

    if versions:
        v = versions[0]
        st.success(f"**{MODEL_NAME}** · v{v.version} · Status: **Production**")

        run = client.get_run(v.run_id)
        m = run.data.metrics

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("AUC-ROC", f"{m.get('val_auc_roc', 0):.4f}")
        col2.metric("Avg Precision", f"{m.get('val_avg_precision', 0):.4f}")
        col3.metric("F1 Score", f"{m.get('val_f1', 0):.4f}")
        col4.metric("KS Statistic", f"{m.get('val_ks', 0):.4f}")
    else:
        st.warning(f"Nenhum modelo em Production encontrado para '{MODEL_NAME}'.")
except Exception as e:
    st.error(f"❌ Erro ao conectar ao MLflow: {e}")

st.divider()

# ─── Métricas Operacionais da API ───────────────────────────────────────────
st.header("⚡ Métricas Operacionais da API")

try:
    r = requests.get(f"{API_URL}/metrics", timeout=3)
    if r.status_code == 200:
        op = r.json()
        col1, col2, col3 = st.columns(3)
        col1.metric("Total de Predições", op["total_predictions"])
        col2.metric("Prob. Média de Default", f"{op['avg_default_probability']:.3f}")
        col3.metric("Acima do Threshold (0.5)", op["predictions_above_threshold"])
    else:
        st.warning(f"API retornou status {r.status_code}.")
except Exception as e:
    st.warning(f"⚠️ API não disponível: {e}")

st.divider()

# ─── Histórico de Experimentos ──────────────────────────────────────────────
st.header("🔬 Histórico de Experimentos")

try:
    if client:
        exp = client.get_experiment_by_name("credit-risk")
        if exp:
            runs = client.search_runs(
                experiment_ids=[exp.experiment_id],
                order_by=["start_time DESC"],
                max_results=20,
            )
            if runs:
                data = []
                for r in runs:
                    data.append(
                        {
                            "Run ID": r.info.run_id[:8] + "...",
                            "AUC-ROC": r.data.metrics.get("val_auc_roc"),
                            "KS": r.data.metrics.get("val_ks"),
                            "F1": r.data.metrics.get("val_f1"),
                            "Data": pd.Timestamp(r.info.start_time, unit="ms").strftime(
                                "%Y-%m-%d %H:%M"
                            ),
                            "Status": r.info.status,
                        }
                    )
                df = pd.DataFrame(data).dropna(subset=["AUC-ROC"])
                st.dataframe(df, use_container_width=True, hide_index=True)

                fig = px.line(
                    df,
                    x="Data",
                    y="AUC-ROC",
                    title="AUC-ROC ao longo dos treinos",
                    markers=True,
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Nenhum experimento encontrado.")
        else:
            st.info("Experimento 'credit-risk' não encontrado no MLflow.")
    else:
        st.info("Conecte ao MLflow para ver o histórico de experimentos.")
except Exception as e:
    st.error(f"❌ Erro ao carregar histórico: {e}")

# ─── Teste Rápido de Predição ─────────────────────────────────────────────
st.divider()
st.header("🧪 Teste de Predição")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    with col1:
        amt_credit = st.number_input("Valor do Crédito (R$)", value=500000.0, min_value=1.0)
        amt_income = st.number_input("Renda Anual (R$)", value=150000.0, min_value=1.0)
        amt_annuity = st.number_input("Parcela Mensal (R$)", value=25000.0, min_value=1.0)
    with col2:
        days_birth = st.number_input("Dias desde nascimento (negativo)", value=-14000)
        days_employed = st.number_input("Dias de emprego (negativo)", value=-2000)
    submitted = st.form_submit_button("Predizer")

if submitted:
    payload = {
        "AMT_CREDIT": amt_credit,
        "AMT_INCOME_TOTAL": amt_income,
        "AMT_ANNUITY": amt_annuity,
        "DAYS_BIRTH": days_birth,
        "DAYS_EMPLOYED": days_employed,
    }
    try:
        r = requests.post(f"{API_URL}/predict", json=payload, timeout=5)
        if r.status_code == 200:
            result = r.json()
            prob = result["default_probability"]
            pred = result["prediction"]
            color = "🔴" if pred == 1 else "🟢"
            st.metric(
                f"{color} Probabilidade de Default",
                f"{prob:.1%}",
                help="Acima de 50%: Alto risco",
            )
            st.json(result)
        else:
            st.error(f"Erro da API: {r.text}")
    except Exception as e:
        st.error(f"Não foi possível conectar à API: {e}")
