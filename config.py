"""
Centralized configuration management for the AI Fraud Intelligence
Command Center. Reads from environment variables / .env with sane
defaults so the platform runs out of the box with zero setup.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


def _get_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


@dataclass
class Settings:
    # Paths
    base_dir: Path = BASE_DIR
    dataset_dir: Path = BASE_DIR / "datasets"
    reports_dir: Path = BASE_DIR / "reports"

    # Database
    database_url: str = os.getenv(
        "DATABASE_URL", f"sqlite:///{(BASE_DIR / 'datasets' / 'fraud_command_center.db').as_posix()}"
    )

    # Kafka
    kafka_bootstrap_servers: str = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    kafka_topic: str = os.getenv("KAFKA_TOPIC", "transactions")

    # GenAI
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # Alerting
    smtp_host: str = os.getenv("SMTP_HOST", "")
    smtp_port: int = int(os.getenv("SMTP_PORT", "587"))
    smtp_user: str = os.getenv("SMTP_USER", "")
    smtp_password: str = os.getenv("SMTP_PASSWORD", "")
    alert_email_to: str = os.getenv("ALERT_EMAIL_TO", "")
    telegram_bot_token: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    telegram_chat_id: str = os.getenv("TELEGRAM_CHAT_ID", "")

    # Risk thresholds
    medium_risk_threshold: float = _get_float("MEDIUM_RISK_THRESHOLD", 0.45)
    high_risk_threshold: float = _get_float("HIGH_RISK_THRESHOLD", 0.75)
    fraud_threshold: float = _get_float("FRAUD_THRESHOLD", 0.90)

    # Hybrid risk engine weights (must sum to 1.0)
    weight_ml: float = 0.40
    weight_anomaly: float = 0.30
    weight_rule: float = 0.20
    weight_behavioral: float = 0.10

    hotspot_cities: list = field(default_factory=lambda: [
        "Hyderabad", "Bangalore", "Chennai", "Mumbai", "Delhi",
    ])


settings = Settings()

# Ensure required directories exist
settings.dataset_dir.mkdir(parents=True, exist_ok=True)
settings.reports_dir.mkdir(parents=True, exist_ok=True)


def get_logger(name: str) -> logging.Logger:
    """Return a configured module-level logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger
