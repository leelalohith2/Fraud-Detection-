"""
Database layer for the AI Fraud Intelligence Command Center.

Uses SQLAlchemy Core so the exact same code works against SQLite
(zero-setup default, used for demos/hackathons) or a production
PostgreSQL instance — just point DATABASE_URL at Postgres in .env
and every table/query below works unchanged.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

import pandas as pd
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    create_engine,
    insert,
    select,
    text,
)

from backend.config import get_logger, settings

logger = get_logger(__name__)

engine = create_engine(settings.database_url, future=True)
metadata = MetaData()

transactions_table = Table(
    "transactions",
    metadata,
    Column("transaction_id", String, primary_key=True),
    Column("customer_id", String, index=True),
    Column("transaction_amount", Float),
    Column("timestamp", String),
    Column("merchant", String),
    Column("device_type", String),
    Column("location", String),
    Column("latitude", Float),
    Column("longitude", Float),
    Column("payment_method", String),
    Column("is_foreign", Boolean),
    Column("fraud_label", Integer, nullable=True),
    Column("ml_score", Float, nullable=True),
    Column("anomaly_score", Float, nullable=True),
    Column("rule_score", Float, nullable=True),
    Column("behavioral_score", Float, nullable=True),
    Column("final_risk_score", Float, nullable=True),
    Column("risk_category", String, nullable=True),
    Column("triggered_rules", String, nullable=True),
    Column("ingested_at", String, default=lambda: datetime.utcnow().isoformat()),
)

customers_table = Table(
    "customers",
    metadata,
    Column("customer_id", String, primary_key=True),
    Column("name", String),
    Column("home_city", String),
    Column("home_lat", Float),
    Column("home_lon", Float),
    Column("account_age_days", Integer),
    Column("avg_monthly_spend", Float),
    Column("risk_segment", String),
)

alerts_table = Table(
    "alerts",
    metadata,
    Column("alert_id", Integer, primary_key=True, autoincrement=True),
    Column("transaction_id", String, index=True),
    Column("customer_id", String, index=True),
    Column("alert_type", String),
    Column("severity", String),
    Column("message", String),
    Column("created_at", String, default=lambda: datetime.utcnow().isoformat()),
    Column("channel_sent", String, nullable=True),
)

investigations_table = Table(
    "investigations",
    metadata,
    Column("case_id", Integer, primary_key=True, autoincrement=True),
    Column("transaction_id", String, index=True),
    Column("customer_id", String, index=True),
    Column("risk_score", Float),
    Column("status", String, default="Open"),
    Column("assigned_analyst", String, nullable=True),
    Column("notes", String, nullable=True),
    Column("created_at", String, default=lambda: datetime.utcnow().isoformat()),
)


def init_db(reset: bool = False) -> None:
    """Create all tables. If reset=True, drop existing tables first."""
    if reset:
        metadata.drop_all(engine)
        logger.info("Dropped all existing tables (reset=True)")
    metadata.create_all(engine)
    logger.info("Database initialized at %s", settings.database_url)


def bulk_upsert_transactions(df: pd.DataFrame) -> int:
    """Insert transactions, skipping duplicates by transaction_id."""
    if df.empty:
        return 0
    with engine.begin() as conn:
        existing_ids = {row[0] for row in conn.execute(select(transactions_table.c.transaction_id))}
        new_rows = df[~df["transaction_id"].isin(existing_ids)].to_dict("records")
        if new_rows:
            conn.execute(insert(transactions_table), new_rows)
    logger.info("Inserted %d new transactions", len(new_rows) if 'new_rows' in locals() else 0)
    return len(new_rows) if 'new_rows' in locals() else 0


def bulk_upsert_customers(df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    with engine.begin() as conn:
        existing_ids = {row[0] for row in conn.execute(select(customers_table.c.customer_id))}
        new_rows = df[~df["customer_id"].isin(existing_ids)].to_dict("records")
        if new_rows:
            conn.execute(insert(customers_table), new_rows)
    return len(new_rows) if 'new_rows' in locals() else 0


def update_transaction_scores(scored_df: pd.DataFrame) -> None:
    """Write ML/anomaly/rule/behavioral/final scores back to the transactions table."""
    if scored_df.empty:
        return
    cols = [
        "transaction_id", "ml_score", "anomaly_score", "rule_score",
        "behavioral_score", "final_risk_score", "risk_category", "triggered_rules",
    ]
    with engine.begin() as conn:
        for row in scored_df[cols].to_dict("records"):
            tid = row.pop("transaction_id")
            conn.execute(
                transactions_table.update()
                .where(transactions_table.c.transaction_id == tid)
                .values(**row)
            )


def insert_alert(transaction_id: str, customer_id: str, alert_type: str,
                  severity: str, message: str, channel_sent: Optional[str] = None) -> None:
    with engine.begin() as conn:
        conn.execute(insert(alerts_table).values(
            transaction_id=transaction_id, customer_id=customer_id,
            alert_type=alert_type, severity=severity, message=message,
            channel_sent=channel_sent,
        ))


def insert_investigation(transaction_id: str, customer_id: str, risk_score: float,
                          assigned_analyst: Optional[str] = None) -> None:
    with engine.begin() as conn:
        conn.execute(insert(investigations_table).values(
            transaction_id=transaction_id, customer_id=customer_id,
            risk_score=risk_score, assigned_analyst=assigned_analyst,
        ))


def fetch_transactions(limit: Optional[int] = None) -> pd.DataFrame:
    query = "SELECT * FROM transactions ORDER BY timestamp DESC"
    if limit:
        query += f" LIMIT {int(limit)}"
    with engine.connect() as conn:
        return pd.read_sql(text(query), conn)


def fetch_customers() -> pd.DataFrame:
    with engine.connect() as conn:
        return pd.read_sql(text("SELECT * FROM customers"), conn)


def fetch_alerts(limit: int = 200) -> pd.DataFrame:
    with engine.connect() as conn:
        return pd.read_sql(
            text(f"SELECT * FROM alerts ORDER BY created_at DESC LIMIT {int(limit)}"), conn
        )


def fetch_investigations() -> pd.DataFrame:
    with engine.connect() as conn:
        return pd.read_sql(text("SELECT * FROM investigations ORDER BY created_at DESC"), conn)


def row_count(table_name: str) -> int:
    with engine.connect() as conn:
        return conn.execute(text(f"SELECT COUNT(*) FROM {table_name}")).scalar_one()
