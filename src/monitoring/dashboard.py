# src/monitoring/dashboard.py
"""
Dashboard Streamlit — Monitoramento do modelo em produção.
Uso: streamlit run src/monitoring/dashboard.py
"""

import os

import mlflow
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

# ── Configuração ──────────────────────────────────────────────────────────────
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
API_URL = os.getenv("API_URL", "http://localhost:8000")
MODEL_NAME = os.getenv("MODEL_NAME", "credit-risk-xgboost")

st.set_page_config(
    page_title="Credit Risk MLOps",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── MLflow client (compartilhado entre tabs) ──────────────────────────────────
_client = None
_prod_version = None
_prod_metrics = {}

try:
    mlflow.set_tracking_uri(MLFLOW_URI)
    from mlflow.tracking import MlflowClient

    _client = MlflowClient()
    versions = _client.get_latest_versions(MODEL_NAME, stages=["Production"])
    if versions:
        v = versions[0]
        _prod_version = v
        _prod_metrics = _client.get_run(v.run_id).data.metrics
except Exception:
    pass  # Tratado em cada seção

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 💳 Credit Risk MLOps")
    st.divider()

    # Status da API
    try:
        _h = requests.get(f"{API_URL}/health", timeout=2).json()
        if _h.get("model_loaded"):
            st.success(f"✅ API healthy\nModelo v{_h.get('model_version', '?')}")
        else:
            st.error("❌ Modelo não carregado")
    except Exception:
        st.error("❌ API offline")

    st.divider()
    st.caption(f"MLflow: `{MLFLOW_URI}`")
    st.caption(f"API: `{API_URL}`")
    st.divider()
    st.caption("Portfólio · Lucas Santos Garrido")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(
    ["📊 Modelo em Produção", "📈 Histórico de Experimentos", "🧪 Teste de Predição"]
)

# ── Tab 1 — Modelo em Produção ────────────────────────────────────────────────
with tab1:
    if _prod_version is None:
        st.warning(f"Nenhum modelo em Production encontrado para '{MODEL_NAME}'.")
        st.stop()

    v = _prod_version
    m = _prod_metrics

    # Cabeçalho
    col_title, col_badge = st.columns([4, 1])
    with col_title:
        try:
            run_info = _client.get_run(v.run_id).info
            trained_at = pd.Timestamp(run_info.end_time, unit="ms").strftime("%d/%m/%Y %H:%M")
        except Exception:
            trained_at = "—"
        st.markdown(f"## 💳 {MODEL_NAME}")
        st.caption(f"Versão **{v.version}** · Treinado em {trained_at}")
    with col_badge:
        st.markdown("<br>", unsafe_allow_html=True)
        st.success("🟢 Production")

    st.divider()

    # 4 métricas de validação
    auc = m.get("val_auc_roc", 0.0)
    ks = m.get("val_ks", 0.0)
    f1 = m.get("val_f1", 0.0)
    ap = m.get("val_avg_precision", 0.0)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "AUC-ROC",
        f"{auc:.4f}",
        delta="✓ Bom" if auc >= 0.75 else ("⚠ Médio" if auc >= 0.65 else "✗ Baixo"),
        delta_color="normal" if auc >= 0.75 else "inverse",
        help="Área sob a curva ROC. >0.75 = bom poder discriminatório.",
    )
    col2.metric(
        "KS Statistic",
        f"{ks:.4f}",
        delta="✓ Bom" if ks >= 0.40 else "⚠ Médio",
        delta_color="normal" if ks >= 0.40 else "off",
        help="Separação máxima entre bons e maus pagadores. >0.40 é padrão no setor financeiro.",
    )
    col3.metric(
        "F1 Score",
        f"{f1:.4f}",
        help="Média harmônica de Precision e Recall (threshold=0.5). Baixo em datasets desbalanceados — normal para 8% de inadimplência.",
    )
    col4.metric(
        "Avg Precision",
        f"{ap:.4f}",
        help="Área sob a curva Precision-Recall. Mais informativo que AUC-ROC para classes desbalanceadas.",
    )

    st.divider()

    # Métricas operacionais
    st.subheader("⚡ Métricas Operacionais da API")
    try:
        _r = requests.get(f"{API_URL}/metrics", timeout=3)
        if _r.status_code == 200:
            op = _r.json()
            total = op["total_predictions"]
            avg_prob = op["avg_default_probability"]
            above = op["predictions_above_threshold"]

            c1, c2, c3 = st.columns(3)
            c1.metric("Total de Predições", f"{total:,}")
            c2.metric("Prob. Média de Default", f"{avg_prob:.1%}")

            if total > 0:
                above_pct = above / total
                c3.metric(
                    "Alto Risco (≥50%)",
                    f"{above:,}",
                    delta=f"{above_pct:.1%} das predições",
                    delta_color="inverse" if above_pct > 0.20 else "normal",
                    help="Predições com probabilidade de default >= 50%.",
                )
                st.caption(f"**{above_pct:.1%}** das predições classificadas como alto risco")
                st.progress(min(above_pct, 1.0))
            else:
                c3.metric("Alto Risco (≥50%)", "0")
                st.caption("Nenhuma predição registrada nesta sessão.")
        else:
            st.warning(f"API retornou status {_r.status_code}.")
    except Exception as _e:
        st.warning(f"⚠️ API não disponível para métricas operacionais: {_e}")

# ── Tab 2 — Histórico de Experimentos ────────────────────────────────────────
with tab2:
    if _client is None:
        st.error("❌ Não foi possível conectar ao MLflow.")
        st.stop()

    try:
        exp = _client.get_experiment_by_name("credit-risk")
        if not exp:
            st.info("Experimento 'credit-risk' não encontrado no MLflow.")
            st.stop()

        runs = _client.search_runs(
            experiment_ids=[exp.experiment_id],
            order_by=["start_time DESC"],
            max_results=20,
        )

        # Apenas runs FINISHED (ignora RUNNING deixadas pelo crash de encoding)
        finished = [r for r in runs if r.info.status == "FINISHED"]

        if not finished:
            st.info("Nenhuma run finalizada encontrada.")
            st.stop()

        data = []
        for r in finished:
            data.append(
                {
                    "Run ID": r.info.run_id[:8] + "...",
                    "AUC-ROC": r.data.metrics.get("val_auc_roc"),
                    "KS": r.data.metrics.get("val_ks"),
                    "F1": r.data.metrics.get("val_f1"),
                    "Avg Precision": r.data.metrics.get("val_avg_precision"),
                    "Data": pd.Timestamp(r.info.start_time, unit="ms").strftime("%d/%m/%Y %H:%M"),
                    "Status": r.info.status,
                }
            )

        df = pd.DataFrame(data).dropna(subset=["AUC-ROC"])

        # Tabela
        st.subheader("📋 Runs Finalizadas")
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "AUC-ROC": st.column_config.NumberColumn(format="%.4f"),
                "KS": st.column_config.NumberColumn(format="%.4f"),
                "F1": st.column_config.NumberColumn(format="%.4f"),
                "Avg Precision": st.column_config.NumberColumn(format="%.4f"),
            },
        )

        st.divider()

        # Gráfico multi-métrica (linha)
        st.subheader("📈 Evolução das Métricas ao Longo dos Treinos")
        df_melt = df[["Data", "AUC-ROC", "KS", "F1", "Avg Precision"]].melt(
            id_vars="Data", var_name="Métrica", value_name="Valor"
        )
        fig_lines = px.line(
            df_melt,
            x="Data",
            y="Valor",
            color="Métrica",
            markers=True,
            title="Todas as métricas — evolução temporal",
            labels={"Valor": "Score (0–1)", "Data": "Data do Treino"},
            template="plotly_dark",
        )
        fig_lines.update_layout(
            yaxis=dict(range=[0, 1]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        # Linha de referência AUC-ROC = 0.75
        fig_lines.add_hline(
            y=0.75,
            line_dash="dot",
            line_color="rgba(255,255,255,0.3)",
            annotation_text="AUC ≥ 0.75 (bom)",
            annotation_position="bottom right",
        )
        st.plotly_chart(fig_lines, use_container_width=True)

        # Gráfico de barras comparando runs (apenas se houver >= 2)
        if len(df) >= 2:
            st.divider()
            st.subheader("📊 Comparação das Últimas Runs")
            df_bar = df.head(5)[["Run ID", "AUC-ROC", "KS", "F1", "Avg Precision"]].melt(
                id_vars="Run ID", var_name="Métrica", value_name="Valor"
            )
            fig_bar = px.bar(
                df_bar,
                x="Run ID",
                y="Valor",
                color="Métrica",
                barmode="group",
                title="Comparação das 5 últimas runs finalizadas",
                labels={"Valor": "Score", "Run ID": "Run"},
                template="plotly_dark",
            )
            fig_bar.update_layout(yaxis=dict(range=[0, 1]))
            st.plotly_chart(fig_bar, use_container_width=True)

    except Exception as _e:
        st.error(f"❌ Erro ao carregar histórico: {_e}")

# ── Tab 3 — Teste de Predição ─────────────────────────────────────────────────
with tab3:
    st.subheader("🧪 Simulador de Análise de Crédito")
    st.caption(
        "Preencha os dados do solicitante. "
        "O modelo retorna a probabilidade de inadimplência com base em padrões do dataset Home Credit."
    )

    with st.form("prediction_form"):
        st.markdown("**💰 Dados Financeiros**")
        col1, col2 = st.columns(2)

        with col1:
            amt_credit = st.number_input(
                "Valor do Crédito (R$)",
                value=500_000.0,
                min_value=1.0,
                step=10_000.0,
                help="Valor total do crédito solicitado.",
            )
            amt_income = st.number_input(
                "Renda Anual (R$)",
                value=150_000.0,
                min_value=1.0,
                step=5_000.0,
            )
            amt_annuity = st.number_input(
                "Parcela Mensal (R$)",
                value=25_000.0,
                min_value=1.0,
                step=500.0,
            )

        with col2:
            age_years = st.slider(
                "🎂 Idade (anos)",
                min_value=18,
                max_value=70,
                value=38,
                help="Convertido internamente para DAYS_BIRTH.",
            )
            employed_years = st.slider(
                "👔 Anos de emprego atual",
                min_value=0,
                max_value=40,
                value=5,
                help="0 = desempregado. Convertido internamente para DAYS_EMPLOYED.",
            )

        st.markdown("**👤 Dados Pessoais** (opcionais)")
        col3, col4, col5 = st.columns(3)
        with col3:
            gender_m = st.checkbox("Masculino")
            own_car = st.checkbox("Possui carro próprio")
        with col4:
            own_realty = st.checkbox("Possui imóvel próprio")
            cnt_children = st.number_input("Nº de filhos", min_value=0, max_value=10, value=0)
        with col5:
            cnt_fam = st.number_input(
                "Membros da família",
                min_value=1,
                max_value=15,
                value=2,
                step=1,
            )

        submitted = st.form_submit_button("🔍 Analisar Crédito", use_container_width=True)

    if submitted:
        payload = {
            "AMT_CREDIT": amt_credit,
            "AMT_INCOME_TOTAL": amt_income,
            "AMT_ANNUITY": amt_annuity,
            "DAYS_BIRTH": -(age_years * 365),
            "DAYS_EMPLOYED": -(employed_years * 365) if employed_years > 0 else 365243,
            "CODE_GENDER_M": int(gender_m),
            "FLAG_OWN_CAR_Y": int(own_car),
            "FLAG_OWN_REALTY_Y": int(own_realty),
            "CNT_CHILDREN": cnt_children,
            "CNT_FAM_MEMBERS": float(cnt_fam),
        }

        try:
            _resp = requests.post(f"{API_URL}/predict", json=payload, timeout=5)

            if _resp.status_code == 200:
                result = _resp.json()
                prob = result["default_probability"]
                pred = result["prediction"]
                threshold = result["threshold"]

                col_gauge, col_interpret = st.columns([1, 1])

                # Gauge
                with col_gauge:
                    gauge_color = (
                        "red" if prob >= 0.5 else ("orange" if prob >= 0.3 else "green")
                    )
                    fig_gauge = go.Figure(
                        go.Indicator(
                            mode="gauge+number",
                            value=round(prob * 100, 1),
                            number={"suffix": "%", "font": {"size": 48}},
                            title={
                                "text": "Probabilidade de Default",
                                "font": {"size": 16},
                            },
                            gauge={
                                "axis": {"range": [0, 100], "ticksuffix": "%"},
                                "bar": {"color": gauge_color, "thickness": 0.25},
                                "steps": [
                                    {"range": [0, 30], "color": "rgba(0,200,80,0.15)"},
                                    {"range": [30, 50], "color": "rgba(255,165,0,0.15)"},
                                    {"range": [50, 100], "color": "rgba(255,50,50,0.15)"},
                                ],
                                "threshold": {
                                    "line": {"color": "white", "width": 3},
                                    "thickness": 0.75,
                                    "value": threshold * 100,
                                },
                            },
                        )
                    )
                    fig_gauge.update_layout(
                        height=320,
                        template="plotly_dark",
                        margin=dict(t=60, b=10, l=20, r=20),
                    )
                    st.plotly_chart(fig_gauge, use_container_width=True)

                # Interpretação
                with col_interpret:
                    if pred == 1:
                        st.error("### ⚠️ Alto Risco de Inadimplência")
                        st.markdown(f"**Probabilidade estimada:** `{prob:.1%}`")
                        st.markdown(
                            "O modelo classifica este perfil como **provável inadimplente**. "
                            "Recomenda-se revisar condições, exigir garantias adicionais "
                            "ou reduzir o valor aprovado."
                        )
                    else:
                        st.success("### ✅ Baixo Risco de Inadimplência")
                        st.markdown(f"**Probabilidade estimada:** `{prob:.1%}`")
                        st.markdown(
                            "O modelo classifica este perfil como **bom pagador**. "
                            "Crédito pode ser aprovado nas condições solicitadas."
                        )

                    st.divider()

                    # Indicadores calculados
                    credit_income = amt_credit / amt_income
                    annuity_monthly_income = amt_annuity / (amt_income / 12)

                    st.markdown("**📐 Indicadores do Perfil:**")
                    ci_flag = " ⚠️ Elevado" if credit_income > 3 else ""
                    ai_flag = " ⚠️ Elevado" if annuity_monthly_income > 0.40 else ""
                    st.markdown(f"- 💳 Crédito / Renda anual: `{credit_income:.1f}x`{ci_flag}")
                    st.markdown(
                        f"- 📅 Parcela / Renda mensal: `{annuity_monthly_income:.1%}`{ai_flag}"
                    )
                    st.markdown(f"- 🎂 Idade: `{age_years} anos`")
                    st.markdown(
                        f"- 👔 Emprego: "
                        f"`{'Desempregado' if employed_years == 0 else f'{employed_years} anos'}`"
                    )
                    st.caption(
                        f"Modelo v{result['model_version']} · threshold `{threshold}` · "
                        f"classificação: `{'inadimplente' if pred == 1 else 'adimplente'}`"
                    )

            else:
                st.error(f"Erro da API ({_resp.status_code}): {_resp.text}")

        except Exception as _e:
            st.error(f"Não foi possível conectar à API: {_e}")
