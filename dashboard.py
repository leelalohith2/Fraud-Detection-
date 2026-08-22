"""
Dashboard Page Renderers
=========================
Each function renders one page of the AI Fraud Intelligence Command
Center inside the Streamlit app. Kept separate from app.py so page
logic stays modular and reusable (e.g. for the FastAPI-backed variant).
"""
from __future__ import annotations

import json
import textwrap
from datetime import datetime

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from backend.config import settings
from backend.data_pipeline import engineer_features
from backend.services import (
    copilot_respond,
    forecast_fraud_volume,
    generate_excel_report,
    generate_pdf_report,
    raise_alerts,
)

RISK_COLORS = {
    "Low Risk": "#10E0A0",
    "Medium Risk": "#F5B342",
    "High Risk": "#F97316",
    "Fraud": "#FF3B5C",
}
BADGE_CLASS = {
    "Low Risk": "safe",
    "Medium Risk": "medium",
    "High Risk": "high",
    "Fraud": "fraud",
}

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#E9EEF5", size=12),
    margin=dict(l=10, r=10, t=40, b=10),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
)


# ------------------------------------------------------------------
# Shared UI helpers
# ------------------------------------------------------------------

def html_block(content: str):
    """Safely render a multi-line HTML string.

    Streamlit's markdown renderer parses full CommonMark, which means a
    string with 4+ leading spaces on a line gets treated as an indented
    *code block* and dumped as literal text instead of being rendered as
    HTML — even with unsafe_allow_html=True. Dedenting + stripping every
    line here avoids that gotcha for any multi-line HTML we build.
    """
    lines = [line.strip() for line in textwrap.dedent(content).strip("\n").splitlines()]
    st.markdown("\n".join(lines), unsafe_allow_html=True)


def eyebrow(text: str):
    st.markdown(f'<div class="section-eyebrow">{text}</div>', unsafe_allow_html=True)


KPI_ICONS = {
    "transaction": "◈", "fraud": "⛔", "money": "◆", "protected": "◆",
    "prevention": "◒", "risk": "◉", "case": "▣", "open": "▣",
    "review": "◔", "monitoring": "◔", "unassigned": "○",
    "live": "◎", "stream": "◎", "report": "▤", "period": "▤",
}


def _icon_for(label: str) -> str:
    low = label.lower()
    for key, icon in KPI_ICONS.items():
        if key in low:
            return icon
    return "▹"


def kpi_grid(cards: list[dict]):
    """Render a row of glass KPI cards. Every fragment is built flush-left
    (no leading indentation) and joined with no stray whitespace so the
    Streamlit markdown parser never mistakes it for a code block."""
    parts = ['<div class="kpi-grid">']
    for c in cards:
        risk_cls = c.get("risk_class", "")
        icon = c.get("icon") or _icon_for(c["label"])
        delta_html = ""
        if c.get("delta"):
            delta_cls = c.get("delta_dir", "neutral")
            delta_html = f'<div class="kpi-delta {delta_cls}"><span class="kpi-delta-arrow">{"▲" if delta_cls=="up" else ("▼" if delta_cls=="down" else "•")}</span> {c["delta"]}</div>'
        parts.append(
            f'<div class="kpi-card {risk_cls}"><div class="kpi-icon">{icon}</div>'
            f'<div class="kpi-label">{c["label"]}</div>'
            f'<div class="kpi-value">{c["value"]}</div>'
            f'{delta_html}</div>'
        )
    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


def badge(risk_category: str) -> str:
    cls = BADGE_CLASS.get(risk_category, "safe")
    return f'<span class="badge {cls}"><span class="badge-dot"></span>{risk_category}</span>'


def apply_fig_theme(fig: go.Figure) -> go.Figure:
    fig.update_layout(**PLOTLY_LAYOUT)
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.06)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.06)")
    return fig


# ------------------------------------------------------------------
# 1. Executive Dashboard
# ------------------------------------------------------------------

def render_executive_dashboard():
    df = st.session_state.scored_df

    total_txns = len(df)
    fraud_count = int((df["risk_category"] == "Fraud").sum())
    high_risk_count = int((df["risk_category"] == "High Risk").sum())
    money_protected = df.loc[df["risk_category"].isin(["Fraud", "High Risk"]), "transaction_amount"].sum()
    prevention_rate = (fraud_count + high_risk_count) / total_txns if total_txns else 0
    avg_risk = df["final_risk_score"].mean()

    eyebrow("EXECUTIVE OVERVIEW")
    kpi_grid([
        {"label": "Total Transactions", "value": f"{total_txns:,}", "delta": "Last 30 days", "delta_dir": "neutral"},
        {"label": "Fraud Detected", "value": f"{fraud_count:,}", "risk_class": "risk-fraud",
         "delta": f"{fraud_count/total_txns:.2%} of volume" if total_txns else "", "delta_dir": "up"},
        {"label": "Money Protected", "value": f"₹{money_protected/1e6:.2f}M", "risk_class": "risk-safe",
         "delta": "Blocked fraud + high-risk value", "delta_dir": "down"},
        {"label": "Fraud Prevention Rate", "value": f"{prevention_rate:.1%}", "risk_class": "risk-medium",
         "delta": "Flagged before settlement", "delta_dir": "neutral"},
        {"label": "Average Risk Score", "value": f"{avg_risk:.1%}", "risk_class": "risk-high",
         "delta": "Hybrid AI composite", "delta_dir": "neutral"},
    ])

    col1, col2 = st.columns([1.4, 1])

    with col1:
        st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">📈 Fraud Trend — Last 30 Days</div>', unsafe_allow_html=True)
        st.markdown('<div class="panel-subtitle">Flagged transactions (High Risk + Fraud) by day</div>', unsafe_allow_html=True)

        tmp = df.copy()
        tmp["timestamp"] = pd.to_datetime(tmp["timestamp"], errors="coerce")
        daily = tmp.groupby([tmp["timestamp"].dt.date, "risk_category"]).size().unstack(fill_value=0)
        for cat in ["Low Risk", "Medium Risk", "High Risk", "Fraud"]:
            if cat not in daily.columns:
                daily[cat] = 0

        fig = go.Figure()
        for cat in ["Fraud", "High Risk", "Medium Risk"]:
            fig.add_trace(go.Scatter(
                x=daily.index.astype(str), y=daily[cat], name=cat, mode="lines",
                stackgroup="one", line=dict(width=0.5, color=RISK_COLORS[cat]),
                fillcolor=RISK_COLORS[cat],
            ))
        fig.update_layout(height=320, hovermode="x unified")
        st.plotly_chart(apply_fig_theme(fig), use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">🎯 Risk Distribution</div>', unsafe_allow_html=True)
        st.markdown('<div class="panel-subtitle">Current transaction portfolio</div>', unsafe_allow_html=True)
        dist = df["risk_category"].value_counts().reindex(
            ["Low Risk", "Medium Risk", "High Risk", "Fraud"]).fillna(0)
        fig = go.Figure(go.Pie(
            labels=dist.index, values=dist.values, hole=0.62,
            marker=dict(colors=[RISK_COLORS[c] for c in dist.index]),
            textinfo="percent", textfont=dict(size=11),
        ))
        fig.update_layout(height=320, showlegend=True,
                           legend=dict(orientation="h", y=-0.1, bgcolor="rgba(0,0,0,0)"))
        st.plotly_chart(apply_fig_theme(fig), use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">🏪 Top Risky Merchant Categories</div>', unsafe_allow_html=True)
        top_merch = (df[df["risk_category"].isin(["Fraud", "High Risk"])]
                     .groupby("merchant").size().sort_values(ascending=False).head(8))
        fig = go.Figure(go.Bar(
            x=top_merch.values, y=top_merch.index, orientation="h",
            marker=dict(color=top_merch.values, colorscale=[[0, "#F5B342"], [1, "#FF3B5C"]]),
        ))
        fig.update_layout(height=300, yaxis=dict(autorange="reversed"))
        st.plotly_chart(apply_fig_theme(fig), use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    with col4:
        st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">📱 Fraud by Device Type</div>', unsafe_allow_html=True)
        dev = (df[df["risk_category"].isin(["Fraud", "High Risk"])]
               .groupby("device_type").size().sort_values(ascending=False))
        fig = go.Figure(go.Bar(
            x=dev.index, y=dev.values,
            marker=dict(color="#00D9FF", line=dict(width=0)),
        ))
        fig.update_layout(height=300)
        st.plotly_chart(apply_fig_theme(fig), use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)


# ------------------------------------------------------------------
# 2. Live Transaction Monitoring
# ------------------------------------------------------------------

def _score_live_batch():
    """Pull newly streamed transactions, score them through the trained
    hybrid engine, and append to session-state live feed + investigation
    queue for high-risk hits."""
    stream = st.session_state.stream
    new_items, new_cursor = stream.drain_new(st.session_state.stream_cursor)
    st.session_state.stream_cursor = new_cursor

    if not new_items:
        return

    raw_df = pd.DataFrame(new_items)
    context_df = pd.concat(
        [st.session_state.pipeline_df.tail(500)[
            ["customer_id", "timestamp", "device_type", "transaction_amount"]
         ], raw_df[["customer_id", "timestamp", "device_type", "transaction_amount"]]],
        ignore_index=True,
    )
    engineered = engineer_features(raw_df, customers=st.session_state.customers_df)
    scored = st.session_state.engine.score(engineered)

    scored_records = scored.to_dict("records")
    st.session_state.live_scored = (scored_records + st.session_state.live_scored)[:300]

    alertable = scored[scored["risk_category"].isin(["High Risk", "Fraud"])]
    if not alertable.empty:
        raise_alerts(alertable, persist=False)


def render_live_monitoring():
    eyebrow("LIVE TRANSACTION MONITORING")
    html_block("""
        <div class="glass-panel" style="padding:14px 22px;">
        <span style="font-family:'IBM Plex Mono',monospace; font-size:12px; color:#8E9BAD;">
        Streaming simulated Kafka-equivalent transaction feed. Auto-refreshes every few seconds —
        toggle below to pause.
        </span>
        </div>
        """)

    col_a, col_b, col_c = st.columns([1, 1, 4])
    with col_a:
        auto_refresh = st.toggle("Auto-refresh", value=True)
    with col_b:
        if st.button("⟳ Refresh now"):
            _score_live_batch()

    if auto_refresh:
        _score_live_batch()

    live = st.session_state.live_scored

    fraud_live = sum(1 for t in live if t["risk_category"] == "Fraud")
    high_live = sum(1 for t in live if t["risk_category"] == "High Risk")
    kpi_grid([
        {"label": "Live Transactions Seen", "value": f"{len(live):,}"},
        {"label": "Fraud Flagged (Live)", "value": f"{fraud_live:,}", "risk_class": "risk-fraud"},
        {"label": "High Risk (Live)", "value": f"{high_live:,}", "risk_class": "risk-high"},
        {"label": "Stream Status", "value": "ACTIVE" if auto_refresh else "PAUSED", "risk_class": "risk-safe"},
    ])

    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">📡 Live Feed</div>', unsafe_allow_html=True)

    header = '<div class="feed-row header"><div>Transaction ID</div><div>Customer</div><div>Amount</div><div>Merchant</div><div>Location</div><div>Risk</div></div>'
    st.markdown(header, unsafe_allow_html=True)

    if not live:
        st.info("Waiting for the first streamed transactions… click **Refresh now** or wait a moment.")
    else:
        rows_html = ""
        for txn in live[:25]:
            rows_html += (
                f'<div class="feed-row">'
                f'<div>{txn["transaction_id"]}</div>'
                f'<div>{txn["customer_id"]}</div>'
                f'<div class="feed-amount">₹{txn["transaction_amount"]:,.0f}</div>'
                f'<div>{txn["merchant"]}</div>'
                f'<div>{txn["location"]}</div>'
                f'<div>{badge(txn["risk_category"])}</div>'
                f'</div>'
            )
        st.markdown(rows_html, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if auto_refresh:
        st.caption("Auto-refreshing… (this page re-runs to poll the stream; disable auto-refresh to freeze the feed)")
        import time
        time.sleep(2)
        st.rerun()


# ------------------------------------------------------------------
# 3. Fraud Analytics
# ------------------------------------------------------------------

def render_fraud_analytics():
    df = st.session_state.scored_df
    eyebrow("FRAUD ANALYTICS")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">💳 Payment Method Risk Profile</div>', unsafe_allow_html=True)
        pm = df.groupby("payment_method").agg(
            avg_risk=("final_risk_score", "mean"), count=("transaction_id", "count")
        ).sort_values("avg_risk", ascending=False)
        fig = go.Figure(go.Bar(
            x=pm.index, y=pm["avg_risk"],
            marker=dict(color=pm["avg_risk"], colorscale=[[0, "#10E0A0"], [0.5, "#F5B342"], [1, "#FF3B5C"]]),
            text=[f"{v:.1%}" for v in pm["avg_risk"]], textposition="outside",
        ))
        fig.update_layout(height=340, yaxis_tickformat=".0%")
        st.plotly_chart(apply_fig_theme(fig), use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">⏰ Fraud by Hour of Day</div>', unsafe_allow_html=True)
        hourly = (df[df["risk_category"].isin(["Fraud", "High Risk"])]
                  .groupby("hour_of_day").size().reindex(range(24), fill_value=0))
        fig = go.Figure(go.Bar(
            x=list(range(24)), y=hourly.values,
            marker=dict(color=["#FF3B5C" if (h < 6 or h >= 23) else "#00D9FF" for h in range(24)]),
        ))
        fig.update_layout(height=340, xaxis_title="Hour", xaxis=dict(dtick=2))
        st.plotly_chart(apply_fig_theme(fig), use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">💰 Transaction Amount Distribution</div>', unsafe_allow_html=True)
        fig = go.Figure()
        for cat in ["Low Risk", "Medium Risk", "High Risk", "Fraud"]:
            subset = df[df["risk_category"] == cat]["transaction_amount"]
            if len(subset):
                fig.add_trace(go.Box(y=subset, name=cat, marker_color=RISK_COLORS[cat], boxpoints=False))
        fig.update_layout(height=340, yaxis_title="Amount (₹)")
        st.plotly_chart(apply_fig_theme(fig), use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    with col4:
        st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">🧭 Risk Score Composition (Sample Avg)</div>', unsafe_allow_html=True)
        comp = df[["ml_score", "anomaly_score", "rule_score", "behavioral_score"]].mean()
        labels = ["ML Model (40%)", "Anomaly Detection (30%)", "Rule Engine (20%)", "Behavioral (10%)"]
        fig = go.Figure(go.Barpolar(
            r=comp.values, theta=labels,
            marker=dict(color=["#00D9FF", "#8B7CF6", "#F5B342", "#10E0A0"]),
        ))
        fig.update_layout(height=340, polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(gridcolor="rgba(255,255,255,0.08)", tickfont=dict(size=9)),
            angularaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
        ))
        st.plotly_chart(apply_fig_theme(fig), use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">🔍 Explainable AI — Top Global Fraud Indicators</div>', unsafe_allow_html=True)
    st.markdown('<div class="panel-subtitle">Feature importance from the trained XGBoost model</div>', unsafe_allow_html=True)
    engine = st.session_state.engine
    if engine.is_trained and engine.xgb_model is not None:
        importances = engine.xgb_model.feature_importances_
        feat_imp = pd.Series(importances, index=engine.feature_columns).sort_values(ascending=False).head(10)
        fig = go.Figure(go.Bar(
            x=feat_imp.values, y=feat_imp.index, orientation="h",
            marker=dict(color="#00D9FF"),
        ))
        fig.update_layout(height=380, yaxis=dict(autorange="reversed"))
        st.plotly_chart(apply_fig_theme(fig), use_container_width=True, config={"displayModeBar": False})
    else:
        st.info("Model not yet trained.")
    st.markdown('</div>', unsafe_allow_html=True)


# ------------------------------------------------------------------
# 4. Fraud Heatmap
# ------------------------------------------------------------------

def render_fraud_heatmap():
    df = st.session_state.scored_df
    eyebrow("FRAUD HEATMAP — INDIA")

    agg = df.groupby("location").agg(
        transaction_count=("transaction_id", "count"),
        fraud_count=("risk_category", lambda s: (s == "Fraud").sum()),
        avg_risk=("final_risk_score", "mean"),
        lat=("latitude", "first"),
        lon=("longitude", "first"),
    ).reset_index().dropna(subset=["lat", "lon"])

    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">🗺️ Fraud Hotspots</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="panel-subtitle">Highlighted metros: {", ".join(settings.hotspot_cities)}</div>',
        unsafe_allow_html=True,
    )

    fig = go.Figure()
    fig.add_trace(go.Scattergeo(
        lon=agg["lon"], lat=agg["lat"],
        text=agg.apply(lambda r: f"{r['location']}<br>Transactions: {r['transaction_count']}<br>"
                                  f"Fraud: {r['fraud_count']}<br>Avg Risk: {r['avg_risk']:.1%}", axis=1),
        mode="markers",
        marker=dict(
            size=np.clip(agg["fraud_count"] * 3 + 8, 8, 55),
            color=agg["avg_risk"], colorscale=[[0, "#10E0A0"], [0.5, "#F5B342"], [1, "#FF3B5C"]],
            showscale=True, colorbar=dict(title="Avg Risk", tickformat=".0%"),
            line=dict(width=1, color="rgba(255,255,255,0.3)"), opacity=0.85,
        ),
        hoverinfo="text",
    ))
    fig.update_geos(
        scope="asia", center=dict(lat=22, lon=79), projection_scale=4.2,
        showland=True, landcolor="#0E1420", showocean=True, oceancolor="#05070C",
        showcountries=True, countrycolor="rgba(255,255,255,0.15)",
        showsubunits=True, subunitcolor="rgba(255,255,255,0.08)",
        bgcolor="rgba(0,0,0,0)",
    )
    fig.update_layout(height=560, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(apply_fig_theme(fig), use_container_width=True, config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">📍 City-Level Breakdown</div>', unsafe_allow_html=True)
    display_df = agg.sort_values("fraud_count", ascending=False)[
        ["location", "transaction_count", "fraud_count", "avg_risk"]
    ].rename(columns={"location": "City", "transaction_count": "Transactions",
                       "fraud_count": "Fraud Count", "avg_risk": "Avg Risk Score"})
    display_df["Avg Risk Score"] = display_df["Avg Risk Score"].apply(lambda x: f"{x:.1%}")
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ------------------------------------------------------------------
# 5. Fraud Network Graph
# ------------------------------------------------------------------

def render_fraud_network():
    df = st.session_state.scored_df
    eyebrow("FRAUD RING NETWORK")

    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">🕸️ Customer → Device → Merchant → Transaction Graph</div>', unsafe_allow_html=True)
    st.markdown('<div class="panel-subtitle">Built from the highest-risk transactions to surface potential fraud rings</div>', unsafe_allow_html=True)

    n_nodes = st.slider("Number of high-risk transactions to visualize", 10, 100, 40, step=10)
    subset = df[df["risk_category"].isin(["Fraud", "High Risk"])].sort_values(
        "final_risk_score", ascending=False).head(n_nodes)

    if subset.empty:
        st.info("No high-risk transactions available to build a network yet.")
        st.markdown('</div>', unsafe_allow_html=True)
        return

    G = nx.Graph()
    for _, row in subset.iterrows():
        cust_node = f"CUST::{row['customer_id']}"
        dev_node = f"DEV::{row['device_type']}"
        merch_node = f"MER::{row['merchant']}"
        txn_node = f"TXN::{row['transaction_id']}"

        G.add_node(cust_node, kind="customer", label=row["customer_id"])
        G.add_node(dev_node, kind="device", label=row["device_type"])
        G.add_node(merch_node, kind="merchant", label=row["merchant"])
        G.add_node(txn_node, kind="transaction", label=row["transaction_id"],
                   risk=row["final_risk_score"], category=row["risk_category"])

        G.add_edge(cust_node, txn_node)
        G.add_edge(txn_node, dev_node)
        G.add_edge(txn_node, merch_node)

    pos = nx.spring_layout(G, seed=42, k=0.6)

    edge_x, edge_y = [], []
    for u, v in G.edges():
        edge_x += [pos[u][0], pos[v][0], None]
        edge_y += [pos[u][1], pos[v][1], None]

    edge_trace = go.Scatter(x=edge_x, y=edge_y, mode="lines",
                             line=dict(width=0.6, color="rgba(0, 217, 255, 0.25)"), hoverinfo="none")

    node_colors_map = {"customer": "#8B7CF6", "device": "#F5B342", "merchant": "#00D9FF", "transaction": "#FF3B5C"}
    node_size_map = {"customer": 14, "device": 10, "merchant": 12, "transaction": 8}

    node_traces = []
    for kind in ["customer", "device", "merchant", "transaction"]:
        nodes = [n for n, d in G.nodes(data=True) if d["kind"] == kind]
        node_traces.append(go.Scatter(
            x=[pos[n][0] for n in nodes], y=[pos[n][1] for n in nodes],
            mode="markers", name=kind.title(),
            marker=dict(size=node_size_map[kind], color=node_colors_map[kind],
                        line=dict(width=1, color="rgba(255,255,255,0.4)")),
            text=[G.nodes[n]["label"] for n in nodes], hoverinfo="text",
        ))

    fig = go.Figure(data=[edge_trace] + node_traces)
    fig.update_layout(height=580, showlegend=True, xaxis=dict(visible=False), yaxis=dict(visible=False))
    st.plotly_chart(apply_fig_theme(fig), use_container_width=True, config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">🔗 Shared-Resource Clusters</div>', unsafe_allow_html=True)
    st.markdown('<div class="panel-subtitle">Devices or merchants linked to multiple distinct customers (possible fraud rings)</div>', unsafe_allow_html=True)
    shared = []
    for n, d in G.nodes(data=True):
        if d["kind"] in ("device", "merchant"):
            txn_neighbors = [t for t in G.neighbors(n) if G.nodes[t]["kind"] == "transaction"]
            cust_set = set()
            for t in txn_neighbors:
                cust_set.update(c for c in G.neighbors(t) if G.nodes[c]["kind"] == "customer")
            if len(cust_set) > 1:
                shared.append({"Node": d["label"], "Type": d["kind"].title(),
                                "Linked Customers": len(cust_set), "Linked Transactions": len(txn_neighbors)})
    if shared:
        st.dataframe(pd.DataFrame(shared).sort_values("Linked Customers", ascending=False),
                     use_container_width=True, hide_index=True)
    else:
        st.info("No shared-resource clusters detected in this sample.")
    st.markdown('</div>', unsafe_allow_html=True)


# ------------------------------------------------------------------
# 6. Investigation Queue
# ------------------------------------------------------------------

def render_investigation_queue():
    df = st.session_state.scored_df
    eyebrow("INVESTIGATION QUEUE")

    analysts = ["A. Rao", "S. Menon", "K. Fernandes", "P. Sharma", "Unassigned"]
    queue = df[df["risk_category"].isin(["Fraud", "High Risk", "Medium Risk"])].sort_values(
        "final_risk_score", ascending=False).head(300).copy()
    queue["status"] = np.where(queue["risk_category"] == "Fraud", "Under Review",
                        np.where(queue["risk_category"] == "High Risk", "Open", "Monitoring"))
    rng = np.random.default_rng(7)
    queue["assigned_analyst"] = rng.choice(analysts, size=len(queue), p=[0.22, 0.22, 0.22, 0.22, 0.12])

    kpi_grid([
        {"label": "Open Cases", "value": f"{(queue['status']=='Open').sum():,}", "risk_class": "risk-high"},
        {"label": "Under Review", "value": f"{(queue['status']=='Under Review').sum():,}", "risk_class": "risk-fraud"},
        {"label": "Monitoring", "value": f"{(queue['status']=='Monitoring').sum():,}", "risk_class": "risk-medium"},
        {"label": "Unassigned", "value": f"{(queue['assigned_analyst']=='Unassigned').sum():,}"},
    ])

    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">🗂️ Case Queue</div>', unsafe_allow_html=True)

    fcol1, fcol2, fcol3 = st.columns(3)
    with fcol1:
        risk_filter = st.multiselect("Risk category", ["Fraud", "High Risk", "Medium Risk"],
                                      default=["Fraud", "High Risk", "Medium Risk"])
    with fcol2:
        analyst_filter = st.multiselect("Analyst", analysts, default=analysts)
    with fcol3:
        status_filter = st.multiselect("Status", ["Open", "Under Review", "Monitoring"],
                                        default=["Open", "Under Review", "Monitoring"])

    filtered = queue[
        queue["risk_category"].isin(risk_filter) &
        queue["assigned_analyst"].isin(analyst_filter) &
        queue["status"].isin(status_filter)
    ]

    display_cols = ["transaction_id", "customer_id", "final_risk_score", "risk_category",
                     "status", "assigned_analyst", "merchant", "transaction_amount"]
    display_df = filtered[display_cols].rename(columns={
        "transaction_id": "Transaction ID", "customer_id": "Customer", "final_risk_score": "Risk Score",
        "risk_category": "Risk Category", "status": "Status", "assigned_analyst": "Analyst",
        "merchant": "Merchant", "transaction_amount": "Amount (₹)",
    })
    display_df["Risk Score"] = display_df["Risk Score"].apply(lambda x: f"{x:.1%}")

    try:
        from st_aggrid import AgGrid, GridOptionsBuilder
        gb = GridOptionsBuilder.from_dataframe(display_df)
        gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=15)
        gb.configure_default_column(sortable=True, filter=True, resizable=True)
        AgGrid(display_df, gridOptions=gb.build(), theme="alpine-dark", height=420, fit_columns_on_grid_load=True)
    except Exception:
        st.dataframe(display_df, use_container_width=True, hide_index=True, height=420)

    st.markdown('</div>', unsafe_allow_html=True)


# ------------------------------------------------------------------
# 7. AI Fraud Copilot
# ------------------------------------------------------------------

def render_copilot():
    eyebrow("AI FRAUD COPILOT")
    html_block("""
        <div class="glass-panel" style="padding:14px 22px;">
        <span style="font-family:'IBM Plex Mono',monospace; font-size:12px; color:#8E9BAD;">
        Ask about specific transactions, high-risk users, fraud trends, city hotspots, or forecasts.
        Runs on OpenAI if configured, otherwise a built-in analytical copilot answers from live data.
        </span>
        </div>
        """)

    suggestions = [
        "Show high-risk users",
        "Show fraud trends",
        "Which city has highest fraud activity?",
        "Explain risk score",
        "Predict tomorrow's fraud volume",
    ]
    cols = st.columns(len(suggestions))
    for i, s in enumerate(suggestions):
        if cols[i].button(s, use_container_width=True, key=f"sugg_{i}"):
            st.session_state.copilot_history.append(("user", s))
            answer = copilot_respond(s, st.session_state.scored_df, engine=st.session_state.engine)
            st.session_state.copilot_history.append(("ai", answer))

    st.markdown('<div class="glass-panel" style="min-height:340px;">', unsafe_allow_html=True)
    for role, msg in st.session_state.copilot_history[-16:]:
        cls = "copilot-bubble-user" if role == "user" else "copilot-bubble-ai"
        st.markdown(f'<div class="{cls}">{msg}</div>', unsafe_allow_html=True)
    if not st.session_state.copilot_history:
        st.markdown(
            '<div class="copilot-bubble-ai">👋 Hi, I\'m your Fraud Copilot. Ask me why a transaction was '
            'flagged (include its TXN ID), or try one of the suggestions above.</div>',
            unsafe_allow_html=True,
        )
    st.markdown('</div>', unsafe_allow_html=True)

    question = st.chat_input("Ask the Fraud Copilot… e.g. 'Why was TXN... flagged?'")
    if question:
        st.session_state.copilot_history.append(("user", question))
        answer = copilot_respond(question, st.session_state.scored_df, engine=st.session_state.engine)
        st.session_state.copilot_history.append(("ai", answer))
        st.rerun()


# ------------------------------------------------------------------
# 8. Fraud Forecasting
# ------------------------------------------------------------------

def render_forecasting():
    df = st.session_state.scored_df
    eyebrow("FRAUD FORECASTING")

    forecast = forecast_fraud_volume(df)

    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">🕐 Next 24 Hours — Hourly Forecast</div>', unsafe_allow_html=True)
    hourly = pd.DataFrame(forecast["hourly"])
    fig = go.Figure(go.Scatter(
        x=hourly["hour"], y=hourly["predicted_fraud_count"], mode="lines+markers",
        line=dict(color="#00D9FF", width=2), fill="tozeroy",
        fillcolor="rgba(0, 217, 255, 0.08)", marker=dict(size=5),
    ))
    fig.update_layout(height=320, yaxis_title="Predicted flagged txns")
    st.plotly_chart(apply_fig_theme(fig), use_container_width=True, config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">📅 Next 7 Days — Daily Forecast</div>', unsafe_allow_html=True)
        daily = pd.DataFrame(forecast["daily"])
        fig = go.Figure(go.Bar(
            x=daily["date"], y=daily["predicted_fraud_count"],
            marker=dict(color="#8B7CF6"),
        ))
        fig.update_layout(height=300)
        st.plotly_chart(apply_fig_theme(fig), use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">🗓️ Next 4 Weeks — Weekly Forecast</div>', unsafe_allow_html=True)
        weekly = pd.DataFrame(forecast["weekly"])
        fig = go.Figure(go.Bar(
            x=weekly["week"], y=weekly["predicted_fraud_count"],
            marker=dict(color="#F5B342"),
        ))
        fig.update_layout(height=300)
        st.plotly_chart(apply_fig_theme(fig), use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    st.caption(f"Forecast generated at {forecast['generated_at']} using weighted seasonality + trend "
               "extrapolation from historical flagged-transaction volume.")


# ------------------------------------------------------------------
# 9. Reports Center
# ------------------------------------------------------------------

def render_reports_center():
    df = st.session_state.scored_df
    eyebrow("REPORTS CENTER")

    fraud_count = int((df["risk_category"] == "Fraud").sum())
    high_risk_count = int((df["risk_category"] == "High Risk").sum())
    money_protected = df.loc[df["risk_category"].isin(["Fraud", "High Risk"]), "transaction_amount"].sum()

    kpi_grid([
        {"label": "Report Period", "value": "Last 30 Days"},
        {"label": "Fraud Cases", "value": f"{fraud_count:,}", "risk_class": "risk-fraud"},
        {"label": "High Risk Cases", "value": f"{high_risk_count:,}", "risk_class": "risk-high"},
        {"label": "Money Protected", "value": f"₹{money_protected/1e6:.2f}M", "risk_class": "risk-safe"},
    ])

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">📄 PDF Report</div>', unsafe_allow_html=True)
        st.markdown('<div class="panel-subtitle">Executive summary, top risky merchants, high-risk customers</div>', unsafe_allow_html=True)
        if st.button("Generate PDF Report", use_container_width=True):
            with st.spinner("Generating PDF report..."):
                pdf_bytes = generate_pdf_report(df, title="Daily Fraud Intelligence Summary")
            st.download_button("⬇️ Download PDF", data=pdf_bytes,
                                file_name=f"fraud_report_{datetime.now().strftime('%Y%m%d')}.pdf",
                                mime="application/pdf", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">📊 Excel Report</div>', unsafe_allow_html=True)
        st.markdown('<div class="panel-subtitle">Summary sheet, risk distribution chart, full transaction export</div>', unsafe_allow_html=True)
        if st.button("Generate Excel Report", use_container_width=True):
            with st.spinner("Generating Excel report..."):
                xlsx_bytes = generate_excel_report(df)
            st.download_button("⬇️ Download Excel", data=xlsx_bytes,
                                file_name=f"fraud_report_{datetime.now().strftime('%Y%m%d')}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">🚨 Recent Alerts</div>', unsafe_allow_html=True)
    alerts = raise_alerts(df[df["risk_category"].isin(["Fraud", "High Risk"])].head(500), persist=False)
    if alerts:
        alerts_df = pd.DataFrame(alerts[:50])
        st.dataframe(alerts_df, use_container_width=True, hide_index=True, height=280)
    else:
        st.info("No alerts in the current dataset.")
    st.markdown('</div>', unsafe_allow_html=True)
