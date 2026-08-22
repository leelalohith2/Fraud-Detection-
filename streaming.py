"""
Streaming Module
=================
Wraps Apache Kafka ingestion for real deployments. Because a hackathon /
portfolio environment rarely has a Kafka broker running, this module
auto-detects broker availability and transparently falls back to an
in-process simulated stream (same interface, same behaviour) that
replays/generates transactions in real time — so the "Live Transaction
Monitoring" dashboard always works.
"""
from __future__ import annotations

import json
import random
import threading
import time
import uuid
from collections import deque
from datetime import datetime
from typing import Callable, Optional

import pandas as pd

from backend.config import get_logger, settings

logger = get_logger(__name__)

_KAFKA_AVAILABLE = False
try:
    from kafka import KafkaProducer, KafkaConsumer  # type: ignore
    _KAFKA_AVAILABLE = True
except Exception:
    _KAFKA_AVAILABLE = False


class SimulatedTransactionStream:
    """In-process producer/consumer that mimics a Kafka topic using a
    thread-safe deque. Generates realistic synthetic transactions drawn
    from the same distributions as datasets/generate_data.py so the live
    feed looks and behaves like real Kafka traffic."""

    def __init__(self, buffer_size: int = 500, reference_df: Optional[pd.DataFrame] = None):
        self.buffer: deque = deque(maxlen=buffer_size)
        self.lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self.reference_df = reference_df
        self._subscribers: list[Callable[[dict], None]] = []

    def subscribe(self, callback: Callable[[dict], None]) -> None:
        self._subscribers.append(callback)

    def _generate_transaction(self) -> dict:
        devices = ["Mobile", "Desktop", "POS Terminal", "ATM", "Tablet"]
        merchants = ["Electronics", "Grocery", "Travel", "Jewellery", "Fuel",
                     "Online Marketplace", "Gaming", "Crypto Exchange", "Utilities",
                     "Dining", "QuickCash Traders", "CryptoFastX"]
        cities = [("Hyderabad", 17.3850, 78.4867), ("Bangalore", 12.9716, 77.5946),
                  ("Chennai", 13.0827, 80.2707), ("Mumbai", 19.0760, 72.8777),
                  ("Delhi", 28.7041, 77.1025), ("Dubai", 25.2048, 55.2708)]
        payments = ["Credit Card", "Debit Card", "UPI", "Net Banking", "Wallet"]

        is_fraud_like = random.random() < 0.08
        city, lat, lon = random.choice(cities)
        amount = round(random.gauss(45000, 20000), 2) if is_fraud_like else round(random.gauss(2500, 1800), 2)
        amount = max(50, amount)

        if self.reference_df is not None and len(self.reference_df) > 0:
            customer_id = random.choice(self.reference_df["customer_id"].unique().tolist())
        else:
            customer_id = f"CUST{random.randint(100000, 100999)}"

        return {
            "transaction_id": f"TXN{uuid.uuid4().hex[:12].upper()}",
            "customer_id": customer_id,
            "transaction_amount": amount,
            "timestamp": datetime.now().isoformat(sep=" ", timespec="seconds"),
            "merchant": random.choice(merchants),
            "device_type": random.choice(devices),
            "location": city,
            "latitude": lat,
            "longitude": lon,
            "payment_method": random.choice(payments),
            "is_foreign": city == "Dubai",
        }

    def _run(self, interval_seconds: float):
        while self._running:
            txn = self._generate_transaction()
            with self.lock:
                self.buffer.append(txn)
            for cb in self._subscribers:
                try:
                    cb(txn)
                except Exception as e:
                    logger.warning("Subscriber callback failed: %s", e)
            time.sleep(interval_seconds)

    def start(self, interval_seconds: float = 1.5) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, args=(interval_seconds,), daemon=True)
        self._thread.start()
        logger.info("Simulated Kafka stream started (interval=%.1fs)", interval_seconds)

    def stop(self) -> None:
        self._running = False
        logger.info("Simulated Kafka stream stopped")

    def latest(self, n: int = 20) -> list[dict]:
        with self.lock:
            return list(self.buffer)[-n:][::-1]

    def drain_new(self, since_index: int) -> tuple[list[dict], int]:
        """Return items appended since `since_index` plus the new index cursor."""
        with self.lock:
            items = list(self.buffer)
        new_items = items[since_index:]
        return new_items, len(items)


class KafkaTransactionStream:
    """Thin wrapper over kafka-python for real deployments."""

    def __init__(self, bootstrap_servers: str, topic: str):
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        )

    def produce(self, transaction: dict) -> None:
        self.producer.send(self.topic, transaction)

    def consume(self, callback: Callable[[dict], None], timeout_ms: int = 1000) -> None:
        consumer = KafkaConsumer(
            self.topic, bootstrap_servers=self.bootstrap_servers,
            value_deserializer=lambda v: json.loads(v.decode("utf-8")),
            consumer_timeout_ms=timeout_ms,
        )
        for message in consumer:
            callback(message.value)


def check_kafka_available(bootstrap_servers: str, timeout: float = 1.0) -> bool:
    """Quick TCP-level probe to see if a real Kafka broker is reachable."""
    if not _KAFKA_AVAILABLE:
        return False
    import socket
    try:
        host, port = bootstrap_servers.split(":")
        with socket.create_connection((host, int(port)), timeout=timeout):
            return True
    except Exception:
        return False


def get_stream(reference_df: Optional[pd.DataFrame] = None):
    """Factory: returns a real KafkaTransactionStream if a broker is
    reachable, otherwise a SimulatedTransactionStream with an identical
    `latest()` / `drain_new()` interface for the dashboard to consume."""
    if check_kafka_available(settings.kafka_bootstrap_servers):
        logger.info("Kafka broker detected — using real KafkaTransactionStream")
        return KafkaTransactionStream(settings.kafka_bootstrap_servers, settings.kafka_topic)
    logger.info("No Kafka broker reachable — using SimulatedTransactionStream fallback")
    return SimulatedTransactionStream(reference_df=reference_df)


# Module-level singleton stream used by the Streamlit app
_stream_singleton: Optional[SimulatedTransactionStream] = None


def get_simulated_stream(reference_df: Optional[pd.DataFrame] = None) -> SimulatedTransactionStream:
    global _stream_singleton
    if _stream_singleton is None:
        _stream_singleton = SimulatedTransactionStream(reference_df=reference_df)
    return _stream_singleton
