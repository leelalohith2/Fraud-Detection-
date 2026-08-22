"""
Synthetic data generator for the AI Fraud Intelligence Command Center.

Produces two CSVs shaped like a realistic financial-fraud dataset
(similar in spirit to Kaggle's IEEE-CIS / PaySim fraud sets) so the
whole platform runs end-to-end with zero external downloads:

    datasets/customers.csv
    datasets/transactions.csv

Run directly:  python datasets/generate_data.py
"""
from __future__ import annotations

import random
import string
import uuid
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

random.seed(42)
np.random.seed(42)

CITIES = [
    ("Hyderabad", 17.3850, 78.4867),
    ("Bangalore", 12.9716, 77.5946),
    ("Chennai", 13.0827, 80.2707),
    ("Mumbai", 19.0760, 72.8777),
    ("Delhi", 28.7041, 77.1025),
    ("Pune", 18.5204, 73.8567),
    ("Kolkata", 22.5726, 88.3639),
    ("Ahmedabad", 23.0225, 72.5714),
]

FOREIGN_CITIES = [
    ("Dubai", 25.2048, 55.2708),
    ("Singapore", 1.3521, 103.8198),
    ("London", 51.5072, -0.1276),
    ("New York", 40.7128, -74.0060),
]

DEVICE_TYPES = ["Mobile", "Desktop", "POS Terminal", "ATM", "Tablet"]
PAYMENT_METHODS = ["Credit Card", "Debit Card", "UPI", "Net Banking", "Wallet"]
MERCHANT_CATEGORIES = [
    "Electronics", "Grocery", "Travel", "Jewellery", "Fuel",
    "Online Marketplace", "Gaming", "Crypto Exchange", "Utilities", "Dining",
]
SUSPICIOUS_MERCHANTS = ["QuickCash Traders", "GlobalGiftCards", "CryptoFastX", "InstantLoanHub"]


def _random_customer_id(n: int) -> list[str]:
    return [f"CUST{100000 + i}" for i in range(n)]


def generate_customers(n_customers: int = 1200) -> pd.DataFrame:
    ids = _random_customer_id(n_customers)
    rows = []
    for cid in ids:
        home_city = random.choice(CITIES)
        rows.append({
            "customer_id": cid,
            "name": f"Customer {cid[-4:]}",
            "home_city": home_city[0],
            "home_lat": home_city[1],
            "home_lon": home_city[2],
            "account_age_days": np.random.randint(10, 3650),
            "avg_monthly_spend": round(np.random.gamma(3, 4000), 2),
            "risk_segment": random.choices(
                ["Low", "Medium", "High"], weights=[0.75, 0.20, 0.05]
            )[0],
        })
    return pd.DataFrame(rows)


def generate_transactions(customers: pd.DataFrame, n_transactions: int = 15000) -> pd.DataFrame:
    rows = []
    start_time = datetime.now() - timedelta(days=30)
    cust_records = customers.to_dict("records")

    for _ in range(n_transactions):
        cust = random.choice(cust_records)
        is_fraud = np.random.rand() < 0.045  # ~4.5% base fraud rate

        ts = start_time + timedelta(
            seconds=random.randint(0, 30 * 24 * 3600)
        )

        if is_fraud:
            amount = round(np.random.choice([1, -1]) * 0 + np.random.gamma(2, 15000) + 5000, 2)
            device = random.choice(DEVICE_TYPES)
            merchant = random.choice(SUSPICIOUS_MERCHANTS + MERCHANT_CATEGORIES)
            if np.random.rand() < 0.5:
                city, lat, lon = random.choice(FOREIGN_CITIES)
            else:
                city, lat, lon = random.choice(CITIES)
            payment = random.choice(PAYMENT_METHODS)
            hour = random.choice(list(range(0, 5)) + list(range(0, 24)))
        else:
            amount = round(max(50, np.random.gamma(2, cust["avg_monthly_spend"] / 25 + 1)), 2)
            device = random.choices(DEVICE_TYPES, weights=[0.45, 0.25, 0.15, 0.05, 0.10])[0]
            merchant = random.choice(MERCHANT_CATEGORIES)
            city, lat, lon = cust["home_city"], cust["home_lat"], cust["home_lon"]
            payment = random.choices(PAYMENT_METHODS, weights=[0.3, 0.3, 0.25, 0.1, 0.05])[0]
            hour = ts.hour

        ts = ts.replace(hour=hour % 24)

        rows.append({
            "transaction_id": f"TXN{uuid.uuid4().hex[:12].upper()}",
            "customer_id": cust["customer_id"],
            "transaction_amount": amount,
            "timestamp": ts.isoformat(sep=" ", timespec="seconds"),
            "merchant": merchant,
            "device_type": device,
            "location": city,
            "latitude": lat,
            "longitude": lon,
            "payment_method": payment,
            "is_foreign": city in [c[0] for c in FOREIGN_CITIES],
            "fraud_label": int(is_fraud),
        })

    df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    return df


def main():
    customers = generate_customers()
    transactions = generate_transactions(customers)

    customers.to_csv("datasets/customers.csv", index=False)
    transactions.to_csv("datasets/transactions.csv", index=False)

    print(f"Generated {len(customers)} customers -> datasets/customers.csv")
    print(f"Generated {len(transactions)} transactions -> datasets/transactions.csv")
    print(f"Fraud rate: {transactions['fraud_label'].mean():.2%}")


if __name__ == "__main__":
    main()
