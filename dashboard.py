"""
dashboard.py — Dashboard opérationnel : état du corpus + usage du chatbot.
Esthétique cohérente avec le frontend React (palette institutionnelle).
Lancer avec : streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
from app.repositories.vector_repository import connect_to_db

st.set_page_config(page_title="Dashboard - RAG Province", layout="wide", page_icon="📋")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:wght@600;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
    --ink: #1C2321;
    --paper: #FAF6EE;
    --paper-raised: #FFFFFF;
    --green-deep: #0B4F3C;
    --green-soft: #E7EFE9;
    --gold: #B8863B;
    --line: #DDD6C8;
    --stamp-red: #9B3B3B;
}

.stApp {
    background: var(--paper);
    font-family: 'IBM Plex Sans', sans-serif;
    color: var(--ink);
}

.letterhead {
    background: var(--green-deep);
    color: var(--paper);
    padding: 24px 28px;
    border-radius: 6px;
    border-bottom: 3px solid var(--gold);
    margin-bottom: 24px;
}

.letterhead h1 {
    font-family: 'Fraunces', serif;
    font-size: 22px;
    font-weight: 700;
    margin: 0;
}

.letterhead p {
    margin: 4px 0 0;
    font-size: 12.5px;
    opacity: 0.8;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}

div[data-testid="stMetric"] {
    background: var(--paper-raised);
    border: 1px solid var(--line);
    border-left: 3px solid var(--green-deep);
    border-radius: 4px;
    padding: 14px 16px;
}

div[data-testid="stMetricLabel"] {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px !important;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--gold) !important;
}

.stTabs [data-baseweb="tab"] {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 13px;
}

div[data-testid="stDataFrame"] {
    border: 1px solid var(--line);
    border-radius: 4px;
}

.section-title {
    font-family: 'Fraunces', serif;
    font-size: 17px;
    font-weight: 600;
    color: var(--green-deep);
    margin: 20px 0 12px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="letterhead">
    <h1>Dashboard opérationnel</h1>
    <p>Province - Assistant documentaire administratif</p>
</div>
""", unsafe_allow_html=True)

conn = connect_to_db()


def run_query(sql: str) -> pd.DataFrame:
    with conn.cursor() as cur:
        cur.execute(sql)
        columns = [desc[0] for desc in cur.description]
        rows = cur.fetchall()
    return pd.DataFrame(rows, columns=columns)


tab_corpus, tab_usage = st.tabs(["📚  CORPUS", "📊  USAGE"])

with tab_corpus:
    st.markdown('<div class="section-title">Composition du corpus</div>', unsafe_allow_html=True)

    total_chunks = run_query("SELECT count(*) as total FROM dev.chunks;").iloc[0]["total"]
    total_tables = run_query("SELECT count(*) as total FROM dev.extracted_tables;").iloc[0]["total"]
    total_docs = run_query("SELECT count(DISTINCT pdf_name) as total FROM dev.chunks;").iloc[0]["total"]

    m1, m2, m3 = st.columns(3)
    m1.metric("Documents indexés", total_docs)
    m2.metric("Chunks total", total_chunks)
    m3.metric("Tableaux extraits", total_tables)

    st.markdown('<div class="section-title">Répartition par méthode et langue</div>', unsafe_allow_html=True)

    df_source = run_query(
        "SELECT source_method, lang, count(*) as nb FROM dev.chunks GROUP BY source_method, lang ORDER BY source_method;"
    )
    col1, col2 = st.columns([2, 1])
    with col1:
        st.bar_chart(df_source.pivot(index="source_method", columns="lang", values="nb").fillna(0), color=["#0B4F3C", "#B8863B"])
    with col2:
        st.dataframe(df_source, width="stretch", hide_index=True)

with tab_usage:
    st.markdown('<div class="section-title">Usage du chatbot</div>', unsafe_allow_html=True)

    try:
        df_logs = run_query("SELECT * FROM dev.query_logs ORDER BY created_at DESC;")
    except Exception:
        df_logs = pd.DataFrame()

    if df_logs.empty:
        st.info("Aucune requête enregistrée pour l'instant — pose des questions via le frontend pour peupler ce dashboard.")
    else:
        m1, m2, m3 = st.columns(3)
        m1.metric("Requêtes totales", len(df_logs))
        m2.metric("Taux d'abstention", f"{df_logs['abstained'].mean() * 100:.0f}%")
        m3.metric("Latence moyenne", f"{df_logs['latency_ms'].mean():.0f} ms")

        st.markdown('<div class="section-title">Répartition par langue</div>', unsafe_allow_html=True)
        st.bar_chart(df_logs["lang"].value_counts(), color="#0B4F3C")

        st.markdown('<div class="section-title">Historique des requêtes</div>', unsafe_allow_html=True)
        st.dataframe(
            df_logs[["created_at", "question", "lang", "abstained", "sources_count", "latency_ms"]],
            width="stretch",
            hide_index=True,
        )