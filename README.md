# 🏥 Predictive Hospital Optimization System

> **A production-grade AI platform for hospital admission forecasting, intelligent staff scheduling, and real-time complication alerts — built with FHIR compliance, differential privacy, and full MLOps observability.**

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Architecture Diagram](#-architecture-diagram)
- [Project Structure](#-project-structure)
- [Pipelines](#-pipelines)
- [Data Flow](#-data-flow)
- [Privacy & Compliance](#-privacy--compliance)
- [API Reference](#-api-reference)
- [Dashboard](#-dashboard)
- [Monitoring & Observability](#-monitoring--observability)
- [Getting Started](#-getting-started)
- [Testing](#-testing)
- [Deployment](#-deployment)
- [Business KPIs](#-business-kpis)
- [Tech Stack](#-tech-stack)

---

## 🎯 Overview

| Problem | Impact | AI Solution |
|---|---|---|
| Unpredictable patient admissions | Overcrowding / understaffing | 24h forecasting (Prophet + LSTM) |
| Inefficient shift scheduling | Legal risk + poor coverage | Linear programming optimizer (PuLP) |
| Late complication detection | Increased mortality risk | Real-time ML alerts + SHAP explainability |

All three pipelines unify under a single **Core API Gateway**, output **FHIR R4-compliant** resources, apply **differential privacy**, and are fully observable via Prometheus + Grafana + ELK.

---

## 📊 Architecture Diagram
```mermaid
graph TD
    subgraph Data Sources
        EHR[EHR / Hospital DB] --> |FHIR/HL7| Ingestion[Ingestion Service]
        Staff[Staff Scheduling System] --> Ingestion
    end

    subgraph Data & ETL Layer
        Ingestion --> |Raw Data| RawDB[(Raw Data Storage)]
        Ingestion --> ETL_Job[ETL & Anonymization]
        ETL_Job --> |Cleaned Data| CleanedDB[(Cleaned Data Storage)]
        ETL_Job --> FE[Feature Engineering Pipeline]
    end

    subgraph ML_Platform_Offline
        FE --> |Features| FeatureStore[(Feature Store)]
        FeatureStore --> |Training Data| Train_Forecast[Forecasting Model Training]
        FeatureStore --> |Training Data| Train_Alert[Alert Model Training]
        Train_Forecast --> |Validated Model| ModelRegistry[Model Registry]
        Train_Alert --> |Validated Model| ModelRegistry
        ModelRegistry --> |Model Version| Predict_Job[Batch Prediction Job]
    end

    subgraph Serving & Integration Online
        FeatureStore --> |Real-time Features| Serve_API[Model Serving API]
        ModelRegistry --> |Model Version| Serve_API
        Serve_API --> |Predictions| PredictionDB[(Prediction DB)]
        PredictionDB --> Serve_API
        CleanedDB --> Optimizer[Optimization Engine]
        PredictionDB --> Optimizer
        CoreAPI[Core API Gateway] --> Serve_API
        CoreAPI --> |Optimization Request| Optimizer
        CoreAPI --> PredictionDB
        CoreAPI --> AuditDB[(Audit Log DB)]
    end

    subgraph User Interface
        CoreAPI --> Dashboard[Streamlit Dashboard]
    end

    subgraph Monitoring
        AllServices --> Prom[Prometheus]
        Prom --> Grafana[Grafana Dashboard]
        AllServices --> ELK[ELK / EFK Stack]
    end

    classDef db fill:#f9f,stroke:#333,stroke-width:2px;
    classDef model fill:#ff9,stroke:#333,stroke-width:2px;
    classDef api fill:#9f9,stroke:#333,stroke-width:2px;
    class RawDB,CleanedDB,FeatureStore,PredictionDB,ModelRegistry,AuditDB db;
    class Train_Forecast,Train_Alert,Serve_API,Predict_Job model;
    class Ingestion,ETL_Job,FE,Optimizer,CoreAPI,Dashboard api;
```

If Mermaid is not rendered in your environment, use the static fallback:

<img src="./docs/architecture_fallback.svg" alt="Architecture Diagram (Static Fallback)" width="100%" />

---

## 📁 Project Structure
```
hospital-optimization-system/
│
├── etl/
│   ├── extract.py              # SQL queries against MIMIC / EHR
│   ├── transform.py            # Feature engineering & imputation
│   ├── anonymize.py            # GDPR anonymization + DP injection
│   ├── validate.py             # Schema checks (pandera)
│   ├── load.py                 # Write to Parquet feature store
│   └── pipeline.py             # ETL orchestrator
│
├── pipelines/
│   ├── forecasting/
│   │   ├── train_prophet.py    # Prophet seasonal forecasting
│   │   ├── train_lstm.py       # LSTM (TensorFlow/Keras)
│   │   ├── ensemble.py         # Prophet + LSTM weighted ensemble
│   │   └── evaluate.py         # MAE, MAPE, backtesting
│   ├── staffing/
│   │   ├── optimizer.py        # PuLP LP formulation
│   │   ├── constraints.py      # Legal + operational constraints
│   │   ├── demand_converter.py # Forecast → staff demand mapping
│   │   └── schedule_export.py  # Output to DataFrame / FHIR
│   └── alerts/
│       ├── train_xgboost.py    # XGBoost classifier
│       ├── train_rf.py         # Random Forest baseline
│       ├── explain.py          # SHAP explainability
│       ├── bias_audit.py       # Fairness evaluation per subgroup
│       └── evaluate.py         # Precision, recall, ROC-AUC
│
├── privacy/
│   ├── dp_trainer.py           # DP-SGD via tensorflow-privacy
│   ├── laplace.py              # Laplace mechanism for outputs
│   └── epsilon_sweep.py        # Privacy-utility tradeoff analysis
│
├── api/
│   ├── main.py                 # FastAPI app init + middleware
│   ├── routers/                # forecast / staffing / alerts / health
│   ├── schemas/                # Pydantic + FHIR R4 models
│   ├── services/
│   │   └── model_service.py    # Model loader + cache
│   └── middleware/             # JWT auth + structured JSON logging
│
├── dashboard/
│   ├── app.py                  # Streamlit multi-page app
│   └── pages/
│       ├── doctor.py           # Patient risk alerts view
│       ├── manager.py          # Staffing schedule view
│       └── executive.py        # KPI & forecast view
│
├── monitoring/
│   ├── prometheus/prometheus.yml
│   ├── grafana/dashboards/
│   └── elk/logstash.conf
│
├── tests/
│   ├── unit/                   # ETL, LP constraints, FHIR mapping
│   ├── integration/            # API endpoints, pipeline end-to-end
│   ├── model/                  # KPI assertions, bias audit
│   └── load/                   # Locust scripts
│
├── deploy/
│   ├── docker-compose.yml
│   ├── Dockerfile.api
│   ├── Dockerfile.dashboard
│   ├── Dockerfile.model-svc
│   └── nginx/nginx.conf
│
├── docs/
│   ├── model_card.md
│   ├── epsilon_report.md
│   ├── bias_audit_report.md
│   └── fhir_conformance.md
│
├── .env.example
├── requirements.txt
└── README.md
```

---

## 🔵 Pipelines

### Pipeline 1 — Admission Forecasting

**Goal:** Predict hospital admissions for the next 24 hours. **Target KPI: MAE < 2h**

| Stage | Model | Expected MAE |
|---|---|---|
| Baseline | 7-day moving average | ~3–5 |
| Statistical | Prophet | ~2–3 |
| Deep Learning | LSTM (TensorFlow) | < 2 (target) |
| Ensemble | Prophet + LSTM weighted avg | Best overall |
```bash
python pipelines/forecasting/train_prophet.py --config configs/prophet.yaml
python pipelines/forecasting/train_lstm.py --config configs/lstm.yaml
python pipelines/forecasting/ensemble.py --evaluate
```

---

### Pipeline 2 — Staffing Optimization

**Goal:** Generate the optimal shift schedule from forecasted demand. **Target KPI: Waiting time −20%**

**Linear Program formulation:**
```
Variable:    X[i,t] ∈ {0,1}  — staff i works at time slot t

Objective:   Minimize  Σ cost[i]·X[i,t]  +  λ · Σ max(0, demand[t] − Σ_i X[i,t])

Constraints:
  Σ_t X[i,t] · hours  ≤  40          ∀i   (max weekly hours)
  Rest gap between consecutive shifts  ≥  11h  (EU Working Time Directive)
  Σ_i X[i,t]           ≥  demand[t]  ∀t   (coverage)
  X[i,t] = 0 if not ICU-certified          (skill constraint)
```
```bash
python pipelines/staffing/optimizer.py --forecast-date 2025-01-15 --horizon 24
```

---

### Pipeline 3 — Complication Alerts

**Goal:** Detect early complications. **Target KPI: Precision > 85%**

Every prediction returns a SHAP explanation:
```json
{
  "patient_id": "P-00471",
  "risk_score": 0.87,
  "risk_level": "HIGH",
  "explanation": [
    { "feature": "creatinine_last", "impact": 0.40 },
    { "feature": "spo2_mean_24h",   "impact": -0.25 },
    { "feature": "age",             "impact": 0.12 }
  ]
}
```

**Bias audit subgroups:** Age >65 / <65, Male / Female, Ethnicity groups. Disparity threshold: < 5%.
```bash
python pipelines/alerts/train_xgboost.py --config configs/xgboost.yaml
python pipelines/alerts/bias_audit.py --model-version latest
```

---

## 🔄 Data Flow
```
EHR (FHIR/HL7)
    ↓
Ingestion Service       ← schema validation, internal ID assignment
    ├──→ Raw DB          ← unmodified archive (audit trail)
    ↓
ETL & Anonymization     ← pseudonymization, DP noise injection
    ↓
Cleaned Data Storage    ← validated Parquet files
    ↓
Feature Engineering     ← rolling aggregations, time decomposition
    ↓
Feature Store           ← versioned, shared by all three pipelines
    ├──→ Offline: Training → Model Registry → Batch Jobs
    └──→ Online:  Serving API → Prediction DB → Core API Gateway
                                                      ↓
                              Optimization Engine ←───┤
                              Audit Log DB        ←───┤
                                                      ↓
                                            Streamlit Dashboard
```

---

## 🔒 Privacy & Compliance

### Differential Privacy

| Level | Mechanism | Applied To |
|---|---|---|
| Training | DP-SGD (`tensorflow-privacy`) | LSTM gradient updates |
| Output | Laplace mechanism | Forecasted admission counts |

Epsilon tested across `{0.1, 0.5, 1.0, 2.0, 5.0, 10.0}`. See [`docs/epsilon_report.md`](docs/epsilon_report.md).

### FHIR R4 Compliance

| Output | FHIR Resource | Key Fields |
|---|---|---|
| Admission forecast | `Schedule` + `Slot` | `start`, `end`, `serviceType` |
| Complication risk | `RiskAssessment` | `prediction.probability`, `basis` (SHAP) |
| Staffing schedule | `Schedule` + `Practitioner` | `actor`, `planningHorizon` |
| Vital signs | `Observation` | `code` (LOINC), `valueQuantity` |

See [`docs/fhir_conformance.md`](docs/fhir_conformance.md).

---

## ⚙️ API Reference

Base URL: `https://your-hospital.domain/api/v1`
Auth: `Authorization: Bearer <token>`

| Endpoint | Method | Description |
|---|---|---|
| `/forecast` | GET | 24h admission predictions with confidence intervals |
| `/staffing` | GET | Optimized shift schedule (`?date=&ward=`) |
| `/alerts` | GET | Active complication risk scores (`?ward=&threshold=`) |
| `/health` | GET | System health check for all downstream services |

---

## 📊 Dashboard

Access at `http://localhost:8501`

| Role | View | Key Widgets |
|---|---|---|
| **Doctor** | Patient risk alerts | Risk score cards, SHAP bar charts, threshold slider |
| **Manager** | Staffing schedule | Gantt chart, coverage heatmap, violation flags |
| **Executive** | KPI overview | MAE trend, waiting time gauge, precision history |

---

## 📡 Monitoring & Observability

| Metric | Description |
|---|---|
| `forecast_mae_rolling_7d` | Rolling 7-day MAE |
| `alert_precision_rolling_7d` | Rolling alert precision |
| `model_prediction_latency_ms` | Inference latency per endpoint |
| `data_drift_score` | Feature drift vs training baseline |

**Alerting rules:**

| Condition | Action |
|---|---|
| MAE > 3h for 3 consecutive days | Trigger retraining pipeline |
| Alert precision < 80% | Page on-call ML engineer |
| API p99 latency > 500ms | Scale `model-svc` container |
| Data drift score > 0.15 | Flag for manual review |

---

## 🚀 Getting Started

### Prerequisites

- Docker Engine >= 24.0, Docker Compose >= 2.20
- Python 3.10+
- PhysioNet account with MIMIC-III/IV access

### Quick Start
```bash
# 1. Clone
git clone https://github.com/your-org/hospital-optimization-system.git
cd hospital-optimization-system

# 2. Configure
cp .env.example .env
# Edit .env with your credentials

# 3. Start all services
docker-compose up -d

# 4. Run ETL
docker-compose exec api python etl/pipeline.py --source mimic

# 5. Train models
docker-compose exec api python pipelines/forecasting/train_prophet.py
docker-compose exec api python pipelines/alerts/train_xgboost.py

# 6. Open dashboard → http://localhost:8501
```

---

## ⚙️ Configuration
```env
POSTGRES_URL=postgresql://user:pass@postgres:5432/hospital
MODEL_REGISTRY_PATH=/models
PRIVACY_EPSILON=1.0
FHIR_VERSION=R4
JWT_SECRET=<your-secret>
LOG_LEVEL=INFO
```

---

## 🧪 Testing
```bash
# Unit tests (target: >80% coverage)
pytest tests/unit/ -v --cov=. --cov-report=term-missing

# Integration tests (requires running stack)
pytest tests/integration/ -v

# Model KPI assertions
pytest tests/model/ -v

# Load tests
locust -f tests/load/locustfile.py --host=http://localhost:8000
```

---

## 🐳 Deployment

| Service | Port | Description |
|---|---|---|
| `nginx` | 443 | Reverse proxy + SSL |
| `api` | 8000 | FastAPI Core API Gateway |
| `model-svc` | 8001 | Model Serving API (internal) |
| `dashboard` | 8501 | Streamlit Dashboard |
| `postgres` | 5432 | Feature store + audit DB |
| `prometheus` | 9090 | Metrics |
| `grafana` | 3000 | Dashboards |
| `kibana` | 5601 | Logs |
```bash
docker-compose build
docker-compose up -d
curl http://localhost:8000/health
```

---

## 📈 Business KPIs

| KPI | Target |
|---|---|
| Admission Forecast MAE | < 2 hours |
| Complication Alert Precision | > 85% |
| Waiting Time Reduction | >= 20% |
| API Availability | 99.9% uptime |
| Inference Latency p99 | < 200ms |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Ingestion | Python, `fhir.resources` (FHIR R4) |
| ETL | Pandas, SQLAlchemy, Pandera, Parquet |
| Forecasting | Prophet, TensorFlow / Keras (LSTM) |
| Optimization | PuLP, CBC / GLPK |
| Classification | XGBoost, Scikit-learn, SHAP, Optuna |
| Privacy | `tensorflow-privacy` (DP-SGD), Laplace |
| API | FastAPI, Pydantic, Uvicorn |
| Dashboard | Streamlit, Plotly, Altair |
| Database | PostgreSQL 15, Parquet |
| Deployment | Docker, Docker Compose, Nginx |
| Monitoring | Prometheus, Grafana, ELK Stack |
| Testing | pytest, pytest-cov, Locust |

---

## 📄 Responsible AI

| Document | Description |
|---|---|
| [`docs/model_card.md`](docs/model_card.md) | Training data, intended use, limitations, fairness results |
| [`docs/epsilon_report.md`](docs/epsilon_report.md) | DP privacy-utility tradeoff curve |
| [`docs/bias_audit_report.md`](docs/bias_audit_report.md) | Precision/recall per demographic subgroup |
| [`docs/fhir_conformance.md`](docs/fhir_conformance.md) | Supported FHIR R4 resources and profiles |

---

## 🤝 Contributing

1. `git checkout -b feature/my-feature`
2. `pytest tests/ --cov=. --cov-fail-under=80`
3. Open a pull request — one approval + passing CI required

---

## 📜 License

MIT License. See [`LICENSE`](LICENSE).

> **Note:** MIMIC data use is subject to the PhysioNet Credentialed Health Data License.

---

*Built for Senior AI Engineering — Healthcare Systems · Applied ML · HealthTech AI*
