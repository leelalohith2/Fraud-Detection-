"""
AI Fraud Intelligence Command Center — Streamlit Frontend
============================================================
Run with:  streamlit run frontend/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu

# Make backend importable when launched as `streamlit run frontend/app.py`
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from config import get_logger, settings
from backend.data_pipeline import run_full_pipeline
from backend.database import init_db
from backend.hybrid_ai import HybridFraudEngine 
from backend.streaming import get_simulated_stream

from frontend import dashboard

logger = get_logger("frontend.app")

st.set_page_config(
    page_title="AI Fraud Intelligence Command Center",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_css():
    css_path = ROOT_DIR / "frontend" / "assets" / "styles.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_raw_data():
    customers = pd.read_csv(ROOT_DIR / "datasets" / "customers.csv")
    transactions = pd.read_csv(ROOT_DIR / "datasets" / "transactions.csv")
    return transactions, customers


@st.cache_resource(show_spinner="Training Hybrid AI Engine (XGBoost + LightGBM + Isolation Forest)...")
def build_and_train_engine(pipeline_df_hash: str, pipeline_df: pd.DataFrame):
    engine = HybridFraudEngine()
    engine.train(pipeline_df)
    return engine


@st.cache_data(show_spinner="Running Hybrid AI risk scoring across historical transactions...")
def get_scored_dataset(_engine, pipeline_df: pd.DataFrame, cache_key: str):
    return _engine.score(pipeline_df)


def init_session_state():
    if "initialized" not in st.session_state:
        init_db()

        raw_txns, customers = load_raw_data()
        pipeline_df = run_full_pipeline(raw_txns, customers)

        engine = build_and_train_engine("v1", pipeline_df)
        scored_df = get_scored_dataset(engine, pipeline_df, "v1")

        st.session_state.customers_df = customers
        st.session_state.pipeline_df = pipeline_df
        st.session_state.engine = engine
        st.session_state.scored_df = scored_df

        stream = get_simulated_stream(reference_df=customers)
        stream.start(interval_seconds=1.8)
        st.session_state.stream = stream
        st.session_state.live_scored = []
        st.session_state.stream_cursor = 0

        st.session_state.copilot_history = []
        st.session_state.initialized = True


def render_topbar():
    st.markdown(
        f"""
        <div class="cmd-topbar">
            <div class="cmd-brand">
                <div class="cmd-brand-mark">λ</div>
                <div>
                    <div class="cmd-brand-title">AI Fraud Intelligence Command Center</div>
                    <div class="cmd-brand-subtitle">Real-Time Transaction Risk Monitoring · Hybrid AI</div>
                </div>
            </div>
            <div class="cmd-status">
                <span class="pulse-dot"></span> LIVE MONITORING ACTIVE
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> str:
    with st.sidebar:
        st.markdown(
            """
            <div style="text-align:center; padding: 10px 0 20px 0;">
                <div style="font-family:'Space Grotesk',sans-serif; font-size:16px; font-weight:700; color:#E9EEF5;">
                    🛡️ FRAUD SOC
                </div>
                <div style="font-family:'IBM Plex Mono',monospace; font-size:10px; color:#5B6579; letter-spacing:0.08em;">
                    HYBRID AI ENGINE v1.0
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        selected = option_menu(
            menu_title=None,
            options=[
                "Executive Dashboard",
                "Live Monitoring",
                "Fraud Analytics",
                "Fraud Heatmap",
                "Fraud Network",
                "Investigation Queue",
                "AI Fraud Copilot",
                "Fraud Forecasting",
                "Reports Center",
            ],
            icons=[
                "speedometer2", "broadcast", "graph-up-arrow", "geo-alt",
                "diagram-3", "search", "chat-dots", "cloud-lightning", "file-earmark-bar-graph",
            ],
            default_index=0,
            styles={
                "container": {"padding": "0", "background-color": "transparent"},
                "icon": {"color": "#00D9FF", "font-size": "15px"},
                "nav-link": {
                    "font-family": "Inter, sans-serif", "font-size": "13.5px",
                    "text-align": "left", "margin": "3px 0", "border-radius": "8px",
                    "color": "#8E9BAD", "padding": "10px 14px",
                },
                "nav-link-selected": {
                    "background-color": "rgba(0, 217, 255, 0.10)",
                    "color": "#00D9FF", "font-weight": "600",
                    "border": "1px solid rgba(0, 217, 255, 0.25)",
                },
            },
        )

        st.markdown("---")
        st.markdown(
            f"""
            <div style="font-family:'IBM Plex Mono',monospace; font-size:10.5px; color:#5B6579; line-height:1.8;">
            DATASET: {len(st.session_state.scored_df):,} txns<br/>
            CUSTOMERS: {len(st.session_state.customers_df):,}<br/>
            MODELS: XGBoost · LightGBM<br/>
            ANOMALY: Isolation Forest<br/>
            XAI: SHAP
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button("🔄 Reset live stream", use_container_width=True):
            st.session_state.live_scored = []
            st.session_state.stream_cursor = 0
            st.rerun()

    return selected


def main():
    load_css()
    init_session_state()
    render_topbar()
    selected = render_sidebar()

    page_map = {
        "Executive Dashboard": dashboard.render_executive_dashboard,
        "Live Monitoring": dashboard.render_live_monitoring,
        "Fraud Analytics": dashboard.render_fraud_analytics,
        "Fraud Heatmap": dashboard.render_fraud_heatmap,
        "Fraud Network": dashboard.render_fraud_network,
        "Investigation Queue": dashboard.render_investigation_queue,
        "AI Fraud Copilot": dashboard.render_copilot,
        "Fraud Forecasting": dashboard.render_forecasting,
        "Reports Center": dashboard.render_reports_center,
    }
    page_map[selected]()


if __name__ == "__main__":
    main()
