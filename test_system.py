"""
System tests for the AI Fraud Intelligence Command Center.

Run with:  pytest tests/test_system.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from backend.data_pipeline import (
    clean_pipeline,
    engineer_features,
    handle_missing_values,
    remove_duplicates,
    run_full_pipeline,
    validate_schema,
)
from backend.hybrid_ai import DEFAULT_RULES, HybridFraudEngine, apply_rule_engine, compute_behavioral_score
from backend.services import (
    copilot_respond,
    forecast_fraud_volume,
    generate_excel_report,
    generate_pdf_report,
    raise_alerts,
)


@pytest.fixture(scope="module")
def raw_data():
    customers = pd.read_csv(ROOT_DIR / "datasets" / "customers.csv")
    transactions = pd.read_csv(ROOT_DIR / "datasets" / "transactions.csv").head(3000)
    return transactions, customers


@pytest.fixture(scope="module")
def pipeline_df(raw_data):
    transactions, customers = raw_data
    return run_full_pipeline(transactions, customers)


@pytest.fixture(scope="module")
def trained_engine(pipeline_df):
    engine = HybridFraudEngine()
    engine.train(pipeline_df)
    return engine


@pytest.fixture(scope="module")
def scored_df(trained_engine, pipeline_df):
    return trained_engine.score(pipeline_df)


# ------------------------------------------------------------------
# Data pipeline
# ------------------------------------------------------------------

class TestDataPipeline:
    def test_schema_validation_passes(self, raw_data):
        transactions, _ = raw_data
        is_valid, missing = validate_schema(transactions)
        assert is_valid
        assert missing == []

    def test_schema_validation_fails_on_missing_fields(self):
        df = pd.DataFrame({"transaction_id": ["T1"]})
        is_valid, missing = validate_schema(df)
        assert not is_valid
        assert "customer_id" in missing

    def test_remove_duplicates(self):
        df = pd.DataFrame({"transaction_id": ["T1", "T1", "T2"], "value": [1, 1, 2]})
        deduped = remove_duplicates(df)
        assert len(deduped) == 2

    def test_handle_missing_values_numeric(self):
        df = pd.DataFrame({"transaction_amount": [10.0, None, 30.0]})
        cleaned = handle_missing_values(df)
        assert cleaned["transaction_amount"].isna().sum() == 0

    def test_engineer_features_adds_expected_columns(self, raw_data):
        transactions, customers = raw_data
        cleaned = clean_pipeline(transactions)
        engineered = engineer_features(cleaned, customers=customers)
        for col in ["hour_of_day", "is_night_transaction", "is_rapid_transaction",
                    "device_switch", "amount_zscore", "is_new_device_location"]:
            assert col in engineered.columns

    def test_full_pipeline_runs_end_to_end(self, pipeline_df):
        assert len(pipeline_df) > 0
        assert "final" not in pipeline_df.columns  # scoring happens later, not in pipeline


# ------------------------------------------------------------------
# Hybrid AI Engine
# ------------------------------------------------------------------

class TestRuleEngine:
    def test_rule_engine_flags_high_amount(self, pipeline_df):
        scored = apply_rule_engine(pipeline_df.head(200))
        assert "rule_score" in scored.columns
        assert scored["rule_score"].between(0, 1).all()

    def test_default_rules_have_valid_weights(self):
        for rule in DEFAULT_RULES:
            assert 0 < rule.weight <= 1

    def test_behavioral_score_bounded(self, pipeline_df):
        scored = compute_behavioral_score(pipeline_df.head(200))
        assert scored["behavioral_score"].between(0, 1).all()


class TestHybridFraudEngine:
    def test_engine_trains_successfully(self, trained_engine):
        assert trained_engine.is_trained
        assert trained_engine.xgb_model is not None
        assert trained_engine.lgbm_model is not None
        assert trained_engine.iso_forest is not None

    def test_scoring_produces_valid_output(self, scored_df):
        required_cols = [
            "ml_score", "anomaly_score", "rule_score", "behavioral_score",
            "final_risk_score", "risk_category",
        ]
        for col in required_cols:
            assert col in scored_df.columns
        assert scored_df["final_risk_score"].between(0, 1).all()
        assert set(scored_df["risk_category"].unique()).issubset(
            {"Low Risk", "Medium Risk", "High Risk", "Fraud"}
        )

    def test_hybrid_weights_sum_to_one(self):
        from backend.config import settings
        total = (settings.weight_ml + settings.weight_anomaly +
                  settings.weight_rule + settings.weight_behavioral)
        assert abs(total - 1.0) < 1e-9

    def test_explain_transaction_returns_plain_english(self, trained_engine, scored_df):
        row = scored_df.iloc[0]
        explanation = trained_engine.explain_transaction(row)
        assert "plain_english" in explanation
        assert isinstance(explanation["plain_english"], str)
        assert len(explanation["top_features"]) > 0

    def test_untrained_engine_falls_back_gracefully(self, pipeline_df):
        fresh_engine = HybridFraudEngine()
        scored = fresh_engine.score(pipeline_df.head(50))
        assert "final_risk_score" in scored.columns
        assert scored["final_risk_score"].between(0, 1).all()


# ------------------------------------------------------------------
# Services
# ------------------------------------------------------------------

class TestForecasting:
    def test_forecast_structure(self, scored_df):
        forecast = forecast_fraud_volume(scored_df)
        assert set(forecast.keys()) == {"hourly", "daily", "weekly", "generated_at"}
        assert len(forecast["hourly"]) == 24
        assert len(forecast["daily"]) == 7
        assert len(forecast["weekly"]) == 4
        for item in forecast["hourly"]:
            assert item["predicted_fraud_count"] >= 0


class TestAlerts:
    def test_alerts_only_for_high_risk_and_fraud(self, scored_df):
        alerts = raise_alerts(scored_df, persist=False)
        flagged_count = scored_df["risk_category"].isin(["High Risk", "Fraud"]).sum()
        assert len(alerts) == flagged_count
        for alert in alerts:
            assert alert["severity"] in {"HIGH", "CRITICAL"}


class TestReports:
    def test_pdf_report_generates_bytes(self, scored_df):
        pdf_bytes = generate_pdf_report(scored_df)
        assert isinstance(pdf_bytes, bytes)
        assert pdf_bytes[:4] == b"%PDF"

    def test_excel_report_generates_bytes(self, scored_df):
        xlsx_bytes = generate_excel_report(scored_df)
        assert isinstance(xlsx_bytes, bytes)
        assert len(xlsx_bytes) > 1000


class TestCopilot:
    def test_copilot_answers_high_risk_users(self, scored_df):
        answer = copilot_respond("Show high-risk users", scored_df)
        assert isinstance(answer, str)
        assert len(answer) > 0

    def test_copilot_answers_risk_score_explanation(self, scored_df):
        answer = copilot_respond("Explain risk score", scored_df)
        assert "40%" in answer or "ML" in answer

    def test_copilot_handles_unknown_question_gracefully(self, scored_df):
        answer = copilot_respond("asdkjaksjdaksjd random gibberish", scored_df)
        assert isinstance(answer, str)
        assert len(answer) > 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
