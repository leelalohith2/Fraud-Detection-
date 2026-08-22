"""
FastAPI Backend
================
REST API for the AI Fraud Intelligence Command Center. The Streamlit
frontend can either call these HTTP endpoints (when run as a separate
service: `uvicorn backend.api:app --reload`) or import backend modules
directly in-process (the default in app.py, for a zero-dependency
single-process demo). Both paths share the exact same business logic.

Run standalone:
    uvicorn backend.api:app --reload --port 8000
"""
from __future__ import annotations

import io
from typing import Optional

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from backend.config import get_logger
from backend.data_pipeline import run_full_pipeline
from backend.database import fetch_customers, fetch_transactions, init_db
from backend.hybrid_ai import get_engine
from backend.services import (
    copilot_respond,
    forecast_fraud_volume,
    generate_excel_report,
    generate_pdf_report,
    raise_alerts,
)

logger = get_logger(__name__)

app = FastAPI(
    title="AI Fraud Intelligence Command Center API",
    description="Real-time financial fraud detection & transaction risk monitoring backend.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory cache of the last scored dataset (kept simple for demo purposes;
# a production deployment would always read/write through `database.py`).
_state = {"scored_df": None, "customers_df": None}


class CopilotRequest(BaseModel):
    question: str


@app.on_event("startup")
def _startup():
    init_db()
    logger.info("FastAPI backend started")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/upload-transactions")
async def upload_transactions(file: UploadFile = File(...)):
    """Upload a transactions CSV, run it through the pipeline + hybrid AI
    engine, and cache the scored result for subsequent endpoint calls."""
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        customers = _state["customers_df"] if _state["customers_df"] is not None else pd.DataFrame()
        pipeline_df = run_full_pipeline(df, customers)

        engine = get_engine()
        if not engine.is_trained and "fraud_label" in pipeline_df.columns:
            engine.train(pipeline_df)
        scored = engine.score(pipeline_df)
        _state["scored_df"] = scored
        return {"rows_ingested": len(scored), "fraud_detected": int((scored["risk_category"] == "Fraud").sum())}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/dashboard-metrics")
def dashboard_metrics():
    df = _state["scored_df"]
    if df is None:
        raise HTTPException(status_code=404, detail="No scored data available yet. Upload transactions first.")
    return {
        "total_transactions": len(df),
        "fraud_detected": int((df["risk_category"] == "Fraud").sum()),
        "high_risk": int((df["risk_category"] == "High Risk").sum()),
        "money_protected": float(df.loc[df["risk_category"].isin(["Fraud", "High Risk"]), "transaction_amount"].sum()),
        "avg_risk_score": float(df["final_risk_score"].mean()),
        "fraud_prevention_rate": float((df["risk_category"].isin(["Fraud", "High Risk"])).mean()),
    }


@app.get("/api/fraud-predictions")
def fraud_predictions(limit: int = 100):
    df = _state["scored_df"]
    if df is None:
        raise HTTPException(status_code=404, detail="No scored data available yet.")
    cols = ["transaction_id", "customer_id", "transaction_amount", "merchant",
            "final_risk_score", "risk_category", "triggered_rules"]
    return df[cols].head(limit).to_dict("records")


@app.get("/api/risk-scores/{transaction_id}")
def risk_score(transaction_id: str):
    df = _state["scored_df"]
    if df is None:
        raise HTTPException(status_code=404, detail="No scored data available yet.")
    row = df[df["transaction_id"] == transaction_id]
    if row.empty:
        raise HTTPException(status_code=404, detail="Transaction not found")
    row = row.iloc[0]
    engine = get_engine()
    explanation = engine.explain_transaction(row)
    return {
        "transaction_id": transaction_id,
        "ml_score": float(row["ml_score"]),
        "anomaly_score": float(row["anomaly_score"]),
        "rule_score": float(row["rule_score"]),
        "behavioral_score": float(row["behavioral_score"]),
        "final_risk_score": float(row["final_risk_score"]),
        "risk_category": row["risk_category"],
        "explanation": explanation,
    }


@app.get("/api/heatmap-data")
def heatmap_data():
    df = _state["scored_df"]
    if df is None:
        raise HTTPException(status_code=404, detail="No scored data available yet.")
    agg = df.groupby("location").agg(
        transaction_count=("transaction_id", "count"),
        fraud_count=("risk_category", lambda s: (s == "Fraud").sum()),
        avg_risk=("final_risk_score", "mean"),
        lat=("latitude", "first"),
        lon=("longitude", "first"),
    ).reset_index()
    return agg.to_dict("records")


@app.get("/api/forecast")
def forecast():
    df = _state["scored_df"]
    if df is None:
        raise HTTPException(status_code=404, detail="No scored data available yet.")
    return forecast_fraud_volume(df)


@app.post("/api/reports/pdf")
def report_pdf():
    df = _state["scored_df"]
    if df is None:
        raise HTTPException(status_code=404, detail="No scored data available yet.")
    pdf_bytes = generate_pdf_report(df)
    return StreamingResponse(io.BytesIO(pdf_bytes), media_type="application/pdf",
                              headers={"Content-Disposition": "attachment; filename=fraud_report.pdf"})


@app.post("/api/reports/excel")
def report_excel():
    df = _state["scored_df"]
    if df is None:
        raise HTTPException(status_code=404, detail="No scored data available yet.")
    xlsx_bytes = generate_excel_report(df)
    return StreamingResponse(
        io.BytesIO(xlsx_bytes),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": "attachment; filename=fraud_report.xlsx"},
    )


@app.post("/api/copilot")
def copilot(request: CopilotRequest):
    df = _state["scored_df"]
    if df is None:
        raise HTTPException(status_code=404, detail="No scored data available yet.")
    engine = get_engine()
    answer = copilot_respond(request.question, df, engine=engine)
    return {"question": request.question, "answer": answer}


@app.get("/api/alerts")
def alerts():
    df = _state["scored_df"]
    if df is None:
        raise HTTPException(status_code=404, detail="No scored data available yet.")
    return raise_alerts(df, persist=False)
