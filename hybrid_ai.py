"""
Hybrid AI Engine
================
Combines five detection layers into a single fraud risk score:

  1. ML Fraud Detection      (XGBoost + LightGBM ensemble)
  2. Anomaly Detection       (Isolation Forest)
  3. Rule Engine             (configurable business rules)
  4. Behavioral Analytics    (per-customer deviation scoring)
  5. Explainable AI          (SHAP feature attribution)

Final Risk Score = 40% ML + 30% Anomaly + 20% Rule + 10% Behavioral
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from backend.config import get_logger, settings

logger = get_logger(__name__)

FEATURE_COLUMNS = [
    "transaction_amount", "hour_of_day", "day_of_week", "is_night_transaction",
    "is_weekend", "is_foreign", "minutes_since_last_txn", "is_rapid_transaction",
    "device_switch", "txn_count_last_24h", "amount_zscore", "is_new_device_location",
    "account_age_days", "is_high_amount", "device_type_enc", "merchant_enc",
    "payment_method_enc",
]

CATEGORICAL_COLUMNS = ["device_type", "merchant", "payment_method"]


# ------------------------------------------------------------------
# Rule Engine
# ------------------------------------------------------------------

@dataclass
class FraudRule:
    name: str
    description: str
    weight: float
    condition: callable  # takes a pd.Series (row) -> bool


DEFAULT_RULES: list[FraudRule] = [
    FraudRule("high_amount", "Transaction amount exceeds 95th percentile", 0.25,
               lambda r: bool(r.get("is_high_amount", 0))),
    FraudRule("foreign_transaction", "Transaction made from a foreign location", 0.20,
               lambda r: bool(r.get("is_foreign", 0))),
    FraudRule("new_device_location", "Transaction location differs from home city", 0.15,
               lambda r: bool(r.get("is_new_device_location", 0))),
    FraudRule("rapid_transaction", "Multiple transactions within 5 minutes (velocity attack)", 0.20,
               lambda r: bool(r.get("is_rapid_transaction", 0))),
    FraudRule("device_switch", "Customer switched device since last transaction", 0.10,
               lambda r: bool(r.get("device_switch", 0))),
    FraudRule("suspicious_merchant", "Merchant flagged as high-risk category", 0.20,
               lambda r: str(r.get("merchant", "")) in {
                   "QuickCash Traders", "GlobalGiftCards", "CryptoFastX",
                   "InstantLoanHub", "Crypto Exchange",
               }),
    FraudRule("velocity_burst", "More than 5 transactions by this customer in 24h", 0.25,
               lambda r: r.get("txn_count_last_24h", 0) > 5),
    FraudRule("night_high_value", "Large transaction made late at night", 0.15,
               lambda r: bool(r.get("is_night_transaction", 0)) and bool(r.get("is_high_amount", 0))),
]


def apply_rule_engine(df: pd.DataFrame, rules: list[FraudRule] = None) -> pd.DataFrame:
    """Evaluate every rule against every row. Adds `rule_score` (0-1, capped)
    and `triggered_rules` (json list of rule names) columns."""
    rules = rules or DEFAULT_RULES
    df = df.copy()

    scores = np.zeros(len(df))
    triggered_lists: list[list[str]] = [[] for _ in range(len(df))]

    for rule in rules:
        try:
            mask = df.apply(rule.condition, axis=1).astype(bool).values
        except Exception as e:
            logger.warning("Rule %s failed to evaluate: %s", rule.name, e)
            continue
        scores[mask] += rule.weight
        for idx in np.where(mask)[0]:
            triggered_lists[idx].append(rule.name)

    df["rule_score"] = np.clip(scores, 0, 1)
    df["triggered_rules"] = [json.dumps(t) for t in triggered_lists]
    return df


# ------------------------------------------------------------------
# Behavioral Analytics
# ------------------------------------------------------------------

def compute_behavioral_score(df: pd.DataFrame) -> pd.DataFrame:
    """Score how much a transaction deviates from the customer's own
    normal behavior (spend, frequency, device, geography)."""
    df = df.copy()

    amount_component = np.tanh(np.abs(df.get("amount_zscore", 0)) / 3.0)
    frequency_component = np.clip(df.get("txn_count_last_24h", 0) / 10.0, 0, 1)
    device_component = df.get("device_switch", 0).astype(float) * 0.5
    geo_component = df.get("is_new_device_location", 0).astype(float) * 0.5

    behavioral = (
        0.40 * amount_component +
        0.25 * frequency_component +
        0.20 * device_component +
        0.15 * geo_component
    )
    df["behavioral_score"] = np.clip(behavioral, 0, 1)
    return df


# ------------------------------------------------------------------
# Hybrid AI Engine (ML + Anomaly + Rules + Behavioral -> Final Score)
# ------------------------------------------------------------------

class HybridFraudEngine:
    """Encapsulates the trained ML models, anomaly detector, and encoders
    so they can be trained once and reused across scoring calls."""

    def __init__(self):
        self.xgb_model = None
        self.lgbm_model = None
        self.iso_forest: Optional[IsolationForest] = None
        self.encoders: dict[str, LabelEncoder] = {}
        self.is_trained = False
        self.shap_explainer = None
        self.feature_columns = FEATURE_COLUMNS

    # -- Encoding -----------------------------------------------------
    def _encode_categoricals(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        df = df.copy()
        for col in CATEGORICAL_COLUMNS:
            enc_col = f"{col}_enc"
            if col not in df.columns:
                df[enc_col] = 0
                continue
            if fit:
                le = LabelEncoder()
                df[enc_col] = le.fit_transform(df[col].astype(str))
                self.encoders[col] = le
            else:
                le = self.encoders.get(col)
                if le is None:
                    df[enc_col] = 0
                else:
                    known = set(le.classes_)
                    safe_vals = df[col].astype(str).apply(lambda v: v if v in known else le.classes_[0])
                    df[enc_col] = le.transform(safe_vals)
        return df

    def _prepare_matrix(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        df = self._encode_categoricals(df, fit=fit)
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0
        X = df[self.feature_columns].fillna(0)
        return X

    # -- Training -------------------------------------------------------
    def train(self, df: pd.DataFrame) -> dict:
        """Train XGBoost, LightGBM, and Isolation Forest on labeled data."""
        import lightgbm as lgb
        import xgboost as xgb
        from sklearn.metrics import accuracy_score, roc_auc_score

        if "fraud_label" not in df.columns:
            raise ValueError("Training data must include a `fraud_label` column")

        X = self._prepare_matrix(df, fit=True)
        y = df["fraud_label"].astype(int)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y if y.nunique() > 1 else None
        )

        scale_pos_weight = max(1.0, (y_train == 0).sum() / max(1, (y_train == 1).sum()))

        self.xgb_model = xgb.XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.08,
            subsample=0.9, colsample_bytree=0.9, eval_metric="logloss",
            scale_pos_weight=scale_pos_weight, random_state=42, n_jobs=-1,
        )
        self.xgb_model.fit(X_train, y_train)

        self.lgbm_model = lgb.LGBMClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.08,
            subsample=0.9, colsample_bytree=0.9, class_weight="balanced",
            random_state=42, n_jobs=-1, verbose=-1,
        )
        self.lgbm_model.fit(X_train, y_train)

        self.iso_forest = IsolationForest(
            n_estimators=200, contamination=max(0.01, min(0.2, y.mean() or 0.05)),
            random_state=42, n_jobs=-1,
        )
        self.iso_forest.fit(X_train)

        xgb_pred = self.xgb_model.predict_proba(X_test)[:, 1]
        lgb_pred = self.lgbm_model.predict_proba(X_test)[:, 1]
        ensemble_pred = (xgb_pred + lgb_pred) / 2

        metrics = {
            "xgb_auc": float(roc_auc_score(y_test, xgb_pred)) if y_test.nunique() > 1 else None,
            "lgbm_auc": float(roc_auc_score(y_test, lgb_pred)) if y_test.nunique() > 1 else None,
            "ensemble_auc": float(roc_auc_score(y_test, ensemble_pred)) if y_test.nunique() > 1 else None,
            "ensemble_accuracy": float(accuracy_score(y_test, (ensemble_pred > 0.5).astype(int))),
            "train_size": len(X_train),
            "test_size": len(X_test),
            "fraud_rate": float(y.mean()),
        }

        try:
            import shap
            self.shap_explainer = shap.TreeExplainer(self.xgb_model)
        except Exception as e:
            logger.warning("Could not build SHAP explainer: %s", e)
            self.shap_explainer = None

        self.is_trained = True
        logger.info("Hybrid AI Engine trained. Metrics: %s", metrics)
        return metrics

    # -- Scoring ----------------------------------------------------------
    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run the full hybrid pipeline: ML + anomaly + rules + behavioral
        -> weighted final risk score + categorical label."""
        df = df.copy()

        # 1. Rule engine + behavioral (don't require training)
        df = apply_rule_engine(df)
        df = compute_behavioral_score(df)

        # 2. ML + anomaly (require a trained model; fall back gracefully)
        if self.is_trained:
            X = self._prepare_matrix(df, fit=False)
            xgb_p = self.xgb_model.predict_proba(X)[:, 1]
            lgb_p = self.lgbm_model.predict_proba(X)[:, 1]
            df["ml_score"] = (xgb_p + lgb_p) / 2

            raw_anomaly = self.iso_forest.decision_function(X)
            # Normalize: more negative = more anomalous -> flip and scale to 0-1
            df["anomaly_score"] = np.clip(
                (raw_anomaly.max() - raw_anomaly) / (raw_anomaly.max() - raw_anomaly.min() + 1e-9), 0, 1
            )
        else:
            logger.warning("Model not trained — using rule/behavioral-only fallback scores")
            df["ml_score"] = df["rule_score"] * 0.6
            df["anomaly_score"] = df["behavioral_score"] * 0.6

        # 3. Weighted hybrid final score
        df["final_risk_score"] = np.clip(
            settings.weight_ml * df["ml_score"] +
            settings.weight_anomaly * df["anomaly_score"] +
            settings.weight_rule * df["rule_score"] +
            settings.weight_behavioral * df["behavioral_score"],
            0, 1,
        )

        df["risk_category"] = df["final_risk_score"].apply(self._categorize)
        return df

    @staticmethod
    def _categorize(score: float) -> str:
        if score >= settings.fraud_threshold:
            return "Fraud"
        if score >= settings.high_risk_threshold:
            return "High Risk"
        if score >= settings.medium_risk_threshold:
            return "Medium Risk"
        return "Low Risk"

    # -- Explainability --------------------------------------------------
    def explain_transaction(self, row: pd.Series) -> dict:
        """Return top SHAP feature contributions + a plain-English explanation
        for a single transaction."""
        if not self.is_trained or self.shap_explainer is None:
            return self._fallback_explanation(row)

        X = self._prepare_matrix(pd.DataFrame([row]), fit=False)
        shap_values = self.shap_explainer.shap_values(X)
        if isinstance(shap_values, list):
            shap_values = shap_values[-1]
        contributions = dict(zip(self.feature_columns, shap_values[0]))
        top = sorted(contributions.items(), key=lambda kv: abs(kv[1]), reverse=True)[:5]

        readable_names = {
            "transaction_amount": "Transaction amount",
            "hour_of_day": "Time of day",
            "is_night_transaction": "Late-night timing",
            "is_foreign": "Foreign transaction location",
            "minutes_since_last_txn": "Time since last transaction",
            "is_rapid_transaction": "Rapid repeat transactions",
            "device_switch": "Device change",
            "txn_count_last_24h": "Transaction frequency (24h)",
            "amount_zscore": "Deviation from usual spend",
            "is_new_device_location": "Unusual transaction location",
            "account_age_days": "Account age",
            "is_high_amount": "Unusually high amount",
            "device_type_enc": "Device type",
            "merchant_enc": "Merchant category",
            "payment_method_enc": "Payment method",
        }

        explanation_lines = []
        for feat, val in top:
            label = readable_names.get(feat, feat)
            direction = "increased" if val > 0 else "decreased"
            explanation_lines.append(f"{label} {direction} the fraud risk")

        plain_english = (
            f"This transaction was flagged mainly because: {', '.join(explanation_lines[:3])}."
            if explanation_lines else "No dominant risk driver identified."
        )

        return {
            "top_features": [{"feature": readable_names.get(f, f), "impact": float(v)} for f, v in top],
            "plain_english": plain_english,
        }

    def _fallback_explanation(self, row: pd.Series) -> dict:
        triggered = json.loads(row.get("triggered_rules", "[]")) if isinstance(row.get("triggered_rules"), str) else []
        rule_lookup = {r.name: r.description for r in DEFAULT_RULES}
        reasons = [rule_lookup.get(r, r) for r in triggered][:5]
        plain_english = (
            f"This transaction was flagged by rule engine + behavioral analytics: {'; '.join(reasons)}."
            if reasons else "No specific rule triggered; flagged by anomaly/behavioral deviation."
        )
        return {
            "top_features": [{"feature": r, "impact": 1.0} for r in reasons],
            "plain_english": plain_english,
        }


# Module-level singleton so the trained engine is shared across the app
_engine_singleton: Optional[HybridFraudEngine] = None


def get_engine() -> HybridFraudEngine:
    global _engine_singleton
    if _engine_singleton is None:
        _engine_singleton = HybridFraudEngine()
    return _engine_singleton
