# 🛡️ AI Fraud Intelligence Command Center

**Real-Time Financial Fraud Detection & Transaction Risk Monitoring System**

An enterprise-grade, production-shaped fraud intelligence platform that looks and
behaves like a banking Security Operations Center — built with **Streamlit**,
**FastAPI**, and a **Hybrid AI Engine** combining Machine Learning, Anomaly
Detection, Rule-Based Intelligence, Behavioral Analytics, Explainable AI (SHAP),
and a GenAI-powered investigation copilot.

The whole system runs **out of the box with zero external infrastructure** —
SQLite stands in for PostgreSQL and an in-process simulated stream stands in
for Kafka, both auto-upgrading transparently the moment you point them at a
real Postgres DSN / Kafka broker. Synthetic data ships in `datasets/` so you
can demo it immediately.

---

## ✨ What's inside

| Layer | Technology |
|---|---|
| Frontend | Streamlit, Plotly, streamlit-option-menu, streamlit-aggrid |
| Backend API | FastAPI |
| Database | SQLite by default → PostgreSQL-ready (SQLAlchemy) |
| Streaming | In-process simulated stream → Kafka-ready (`kafka-python`) |
| ML | XGBoost + LightGBM ensemble |
| Anomaly Detection | Isolation Forest |
| Explainable AI | SHAP |
| GenAI Copilot | OpenAI API (optional) → built-in analytical copilot fallback |
| Reporting | ReportLab (PDF), OpenPyXL (Excel) |
| Graph | NetworkX (fraud ring visualization) |

## 🧠 Hybrid Risk Engine

```
Final Risk Score = 40% ML Score + 30% Anomaly Score + 20% Rule Score + 10% Behavioral Score
```

Risk categories: **Low Risk → Medium Risk → High Risk → Fraud**, with
configurable thresholds in `.env`.

## 🗂️ Project structure

```
AI-Fraud-Intelligence-Command-Center/
├── README.md
├── requirements.txt
├── .env.example
├── datasets/
│   ├── generate_data.py      # synthetic data generator (Kaggle-style schema)
│   ├── transactions.csv
│   └── customers.csv
├── backend/
│   ├── config.py              # settings, thresholds, logging
│   ├── database.py            # SQLAlchemy models + queries (SQLite/Postgres)
│   ├── data_pipeline.py       # ingestion, cleaning, feature engineering
│   ├── hybrid_ai.py           # ML + anomaly + rules + behavioral + SHAP
│   ├── streaming.py           # Kafka wrapper + simulated stream fallback
│   ├── services.py            # forecasting, alerts, reports, GenAI copilot
│   └── api.py                 # FastAPI endpoints
├── frontend/
│   ├── app.py                 # Streamlit entry point + navigation
│   ├── dashboard.py           # all 9 dashboard pages
│   └── assets/styles.css      # dark-glassmorphism SOC design system
├── reports/                    # generated PDF/Excel reports land here
└── tests/
    └── test_system.py         # 21 automated tests across every module
```

## 🚀 Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) copy and edit environment config
cp .env.example .env

# 3. Generate the synthetic dataset (already included, re-run to refresh)
python datasets/generate_data.py

# 4. Launch the dashboard
streamlit run frontend/app.py
```

The first load trains the Hybrid AI Engine (XGBoost + LightGBM + Isolation
Forest) on the bundled dataset — this is cached, so subsequent interactions
are instant.

### Run the API separately (optional)

```bash
uvicorn backend.api:app --reload --port 8000
```

Swagger docs at `http://localhost:8000/docs`.

### Run the tests

```bash
pytest tests/test_system.py -v
```

## 📊 Dashboard pages

1. **Executive Dashboard** — KPIs, fraud trend, risk distribution, top merchants/devices
2. **Live Monitoring** — real-time simulated transaction feed with risk badges
3. **Fraud Analytics** — payment-method risk, hourly patterns, amount distributions, SHAP feature importance
4. **Fraud Heatmap** — India-focused geographic fraud hotspots
5. **Fraud Network** — Customer → Device → Merchant → Transaction graph for fraud-ring discovery
6. **Investigation Queue** — filterable/sortable case queue with analyst assignment
7. **AI Fraud Copilot** — chat interface for natural-language fraud investigation
8. **Fraud Forecasting** — hourly/daily/weekly predicted fraud volume
9. **Reports Center** — on-demand PDF & Excel report generation

## 🔌 Connecting real infrastructure

- **PostgreSQL**: set `DATABASE_URL=postgresql+psycopg2://user:pass@host:5432/db` in `.env`
- **Kafka**: set `KAFKA_BOOTSTRAP_SERVERS` and install `kafka-python`; the platform
  auto-detects a reachable broker and switches from the simulated stream automatically
- **OpenAI Copilot**: set `OPENAI_API_KEY` in `.env` to upgrade the copilot from
  its built-in analytical fallback to full GenAI responses
- **Email/Telegram alerts**: configure SMTP / bot credentials in `.env`

## 📄 License

Provided as-is for hackathon, academic, and portfolio use.
