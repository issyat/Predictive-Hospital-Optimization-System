<div align="center">

# 🏥 Predictive Hospital Optimization System

### *Production-grade Clinical AI Platform on Databricks Lakehouse*

[![Databricks](https://img.shields.io/badge/Platform-Databricks-FF3621?style=for-the-badge&logo=databricks)](https://databricks.com)
[![Delta Lake](https://img.shields.io/badge/Storage-Delta_Lake-003366?style=for-the-badge)](https://delta.io)
[![MLflow](https://img.shields.io/badge/ML_Ops-MLflow-0194E2?style=for-the-badge)](https://mlflow.org)
[![FHIR R4](https://img.shields.io/badge/Standard-FHIR_R4-orange?style=for-the-badge)](https://hl7.org/fhir/)
[![GDPR](https://img.shields.io/badge/Compliance-GDPR_%2B_DP-green?style=for-the-badge)](https://gdpr.eu)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python)](https://python.org)

---

*Admission Forecasting · Staffing Optimization · Complication Alerts · Differential Privacy · HL7 FHIR R4*

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Why Databricks](#-why-databricks)
- [System Architecture](#-system-architecture)
- [Architecture Diagram](#-architecture-diagram)
- [Medallion Architecture](#-medallion-architecture-bronze--silver--gold)
- [Project Structure](#-project-structure)
- [ML Pipelines](#-ml-pipelines)
  - [Pipeline 1 — Admission Forecasting](#pipeline-1--admission-forecasting)
  - [Pipeline 2 — Staffing Optimization](#pipeline-2--staffing-optimization)
  - [Pipeline 3 — Complication Alerts](#pipeline-3--complication-alerts)
- [MLflow & Model Registry](#-mlflow--model-registry)
- [Databricks Feature Store](#-databricks-feature-store)
- [Privacy & Compliance](#-privacy--compliance)
- [API Gateway Layer](#-api-gateway-layer)
- [Dashboard](#-dashboard)
- [Monitoring & Observability](#-monitoring--observability)
- [Getting Started](#-getting-started)
- [Databricks Setup](#-databricks-setup)
- [Configuration](#-configuration)
- [Testing](#-testing)
- [Deployment](#-deployment)
- [Business KPIs](#-business-kpis)
- [Tech Stack](#-tech-stack)
- [Responsible AI](#-responsible-ai)

---

## 🎯 Overview

The **Predictive Hospital Optimization System** is a full-stack clinical intelligence platform built entirely on the **Databricks Lakehouse**, solving three critical operational problems in modern hospitals:

| Problem | Clinical Impact | AI Solution | KPI Target |
|---|---|---|---|
| Unpredictable patient admissions | Overcrowding, understaffing, ICU overload | 24h forecasting — Prophet + LSTM ensemble | MAE < 2 hours |
| Inefficient shift scheduling | Legal risk, poor coverage, staff burnout | Linear programming optimizer (PuLP) | Waiting time −20% |
| Late complication detection | Increased mortality, delayed intervention | Real-time XGBoost alerts + SHAP explainability | Precision > 85% |

All three pipelines share a **unified Databricks Lakehouse**: one Delta Lake storage layer, one Databricks Feature Store, one MLflow Model Registry, and one set of Databricks Workflows — eliminating data silos and reducing operational overhead.

---

## 🧱 Why Databricks

This project was designed to run natively on Databricks. Here is why each major platform capability maps to a project requirement:

| Project Requirement | Databricks Capability | Benefit |
|---|---|---|
| Scalable ETL from EHR/MIMIC | **Auto Loader + PySpark** | Incremental ingestion, schema evolution, fault tolerance |
| Versioned, shared feature store | **Databricks Feature Store** | Point-in-time lookups, no training/serving skew |
| Experiment tracking + model versioning | **MLflow (native)** | Built-in, no extra infrastructure |
| Hyperparameter tuning at scale | **Hyperopt (distributed)** | Parallel trials across cluster workers |
| Real-time model endpoints | **Databricks Model Serving** | One-click deployment from MLflow registry |
| Data drift + model quality monitoring | **Lakehouse Monitoring** | Automatic drift detection on Delta tables |
| Audit logging + GDPR lineage | **Delta Lake + Unity Catalog** | Full data lineage, column-level access control |
| Scheduled ML pipelines | **Databricks Workflows** | DAG-based job orchestration, no Airflow needed |

> **Architecture decision:** The system uses Databricks as the core data + ML platform, with two thin external components: a lightweight **FastAPI container** (FHIR wrapping + auth) and a **Streamlit container** (clinical dashboard). Everything else runs natively inside Databricks.

---

## 📊 Architecture Diagram

```mermaid
graph TD
    subgraph Sources["Data Sources"]
        EHR["EHR / Hospital DB"]
        Staff["Staff Scheduling System"]
    end

    subgraph Medallion["Delta Lake - Medallion Architecture"]
        AutoLoader["Databricks Auto Loader"]
        Bronze[("Bronze Layer<br/>Raw Delta Tables")]
        ETL["PySpark ETL Job<br/>Anonymization + Validation"]
        Silver[("Silver Layer<br/>Cleaned Delta Tables")]
        FE["Feature Engineering<br/>PySpark Pipeline"]
        Gold[("Gold Layer<br/>Databricks Feature Store")]
    end

    subgraph MLPlatform["ML Platform - Offline"]
        Train_Forecast["Pipeline 1<br/>Prophet + LSTM Training"]
        Train_Alert["Pipeline 3<br/>XGBoost + SHAP Training"]
        Optimizer["Pipeline 2<br/>PuLP Staffing Optimizer"]
        MLflowReg["MLflow Model Registry"]
        ModelServing["Databricks Model Serving<br/>REST Endpoints"]
        BatchJob["Batch Prediction Jobs"]
    end

    subgraph Serving["Serving Layer"]
        PredDB[("Prediction DB<br/>Delta Table")]
        APIGateway["FastAPI Gateway<br/>FHIR R4 + Auth"]
        AuditDB[("Audit Log<br/>Delta Table")]
    end

    subgraph UI["User Interface"]
        Dashboard["Streamlit Dashboard<br/>Doctor / Manager / Executive"]
    end

    subgraph Observability["Observability"]
        LHM["Lakehouse Monitoring"]
        Alerts["Databricks Alerts"]
        Grafana["Grafana Dashboard"]
        ELK["ELK Stack"]
    end

    EHR -->|"FHIR/HL7"| AutoLoader
    Staff --> AutoLoader
    AutoLoader -->|"Raw Data"| Bronze
    Bronze --> ETL
    ETL -->|"Cleaned Data"| Silver
    Silver --> FE
    FE -->|"Feature Tables"| Gold

    Gold -->|"Training Data"| Train_Forecast
    Gold -->|"Training Data"| Train_Alert
    Gold -->|"Demand Data"| Optimizer

    Train_Forecast -->|"Validated Model + Metrics"| MLflowReg
    Train_Alert -->|"Validated Model + Metrics"| MLflowReg
    MLflowReg -->|"Promoted Model"| ModelServing
    MLflowReg -->|"Scheduled"| BatchJob

    Gold -->|"Real-time Features"| ModelServing
    ModelServing -->|"Predictions"| PredDB
    Optimizer -->|"Schedule"| PredDB

    APIGateway --> ModelServing
    APIGateway --> PredDB
    APIGateway --> AuditDB

    APIGateway --> Dashboard

    LHM -->|"Drift Alerts"| Alerts
    MLflowReg -->|"Model Metrics"| Grafana
    APIGateway -->|"Structured Logs"| ELK

    classDef delta fill:#003366,color:#fff,stroke:#003366
    classDef mlflow fill:#0194E2,color:#fff,stroke:#0194E2
    classDef api fill:#2E7D32,color:#fff,stroke:#2E7D32
    classDef source fill:#FF6B35,color:#fff,stroke:#FF6B35
    classDef monitor fill:#6A1B9A,color:#fff,stroke:#6A1B9A

    class Bronze,Silver,Gold,PredDB,AuditDB delta
    class Train_Forecast,Train_Alert,MLflowReg,ModelServing,BatchJob mlflow
    class FE,ETL,AutoLoader,Optimizer,APIGateway,Dashboard api
    class EHR,Staff source
    class LHM,Alerts,Grafana,ELK monitor
```

---

## 🥉🥈🥇 Medallion Architecture: Bronze → Silver → Gold

The entire data lifecycle follows Databricks' Medallion Architecture, implemented as Delta Lake tables in Unity Catalog.

### Bronze Layer — Raw Ingestion

| Table | Source | Ingestion Method |
|---|---|---|
| `bronze.admissions_raw` | EHR FHIR R4 Bundle | Auto Loader (JSON stream) |
| `bronze.chartevents_raw` | MIMIC CHARTEVENTS | Auto Loader (CSV batch) |
| `bronze.labevents_raw` | MIMIC LABEVENTS | Auto Loader (CSV batch) |
| `bronze.staff_roster_raw` | HR Scheduling System | Auto Loader (CSV stream) |

All Bronze tables are **append-only**, retain full history, and have `_ingestion_timestamp` and `_source_file` metadata columns for audit lineage.

### Silver Layer — Cleaned & Anonymized

Databricks Workflows orchestrate the Bronze → Silver ETL job, which applies:
- **Pseudonymization** of patient identifiers (SHA-256 hash + salt stored in Databricks Secret Scope)
- **Schema validation** via Delta constraints and Great Expectations on Databricks
- **Outlier clipping** (heart rate: 20–300 bpm, BP: 40–250 mmHg)
- **Missing value imputation** (forward-fill for vitals, median for labs)
- **FHIR resource parsing** into structured Delta columns

### Gold Layer — Feature Store (Databricks Feature Store)

The Gold layer is managed entirely by **Databricks Feature Store**, enabling:
- **Point-in-time correct joins** — critical for clinical ML to prevent label leakage
- **Shared feature reuse** across all three pipelines
- **Automatic online/offline consistency** — same features at training and serving time
- **Full lineage** back to source Silver tables

| Feature Table | Key Features | Used By |
|---|---|---|
| `gold.admission_time_features` | `admission_hour`, `dow`, `month`, `is_holiday`, `admissions_last_7d` | Pipeline 1 |
| `gold.patient_vitals_features` | `hr_mean_24h`, `bp_mean_24h`, `spo2_mean_24h`, `temp_last` | Pipeline 3 |
| `gold.patient_lab_features` | `creatinine_last`, `glucose_last`, `wbc_last`, `lactate_last` | Pipeline 3 |
| `gold.icu_stay_features` | `los_icu_days`, `num_procedures`, `icu_readmission_flag` | Pipeline 3 |
| `gold.staff_demand_features` | `forecasted_admissions`, `current_census`, `ward_capacity` | Pipeline 2 |

---

## 📁 Project Structure

```
hospital-optimization-system/
│
├── notebooks/                          # Databricks Notebooks (exploratory + dev)
│   ├── 01_bronze_exploration.py        # MIMIC data profiling
│   ├── 02_silver_etl_dev.py            # ETL prototyping
│   ├── 03_feature_engineering_dev.py   # Feature development
│   ├── 04_forecasting_dev.py           # Prophet + LSTM prototyping
│   ├── 05_staffing_dev.py              # PuLP LP development
│   └── 06_alerts_dev.py                # XGBoost + SHAP development
│
├── src/                                # Production Python source (synced to DBFS)
│   │
│   ├── etl/
│   │   ├── auto_loader_config.py       # Auto Loader stream config per source
│   │   ├── bronze_to_silver.py         # PySpark anonymization + validation job
│   │   ├── silver_to_gold.py           # Feature engineering PySpark job
│   │   ├── fhir_parser.py              # FHIR R4 Bundle → Delta columns
│   │   └── schema_registry.py          # Delta schema definitions + constraints
│   │
│   ├── pipelines/
│   │   ├── forecasting/
│   │   │   ├── train_prophet.py        # Prophet training + MLflow logging
│   │   │   ├── train_lstm.py           # LSTM training + MLflow logging (TF)
│   │   │   ├── ensemble.py             # Ensemble logic + final MLflow model
│   │   │   └── evaluate.py             # MAE, MAPE, backtesting utilities
│   │   │
│   │   ├── staffing/
│   │   │   ├── optimizer.py            # PuLP LP formulation
│   │   │   ├── constraints.py          # EU Working Time Directive constraints
│   │   │   ├── demand_converter.py     # Forecast → staff demand mapping
│   │   │   └── fhir_schedule.py        # Output → FHIR Schedule resource
│   │   │
│   │   └── alerts/
│   │       ├── train_xgboost.py        # XGBoost + Hyperopt tuning + MLflow
│   │       ├── train_rf.py             # Random Forest baseline + MLflow
│   │       ├── explain.py              # SHAP explainability (inference-time)
│   │       ├── bias_audit.py           # Fairness evaluation per subgroup
│   │       └── evaluate.py             # Precision, recall, ROC-AUC
│   │
│   ├── privacy/
│   │   ├── dp_sgd_trainer.py           # DP-SGD wrapper (tensorflow-privacy)
│   │   ├── laplace_mechanism.py        # Laplace output perturbation
│   │   └── epsilon_sweep.py            # Privacy-utility tradeoff analysis job
│   │
│   ├── feature_store/
│   │   ├── register_features.py        # Feature table creation + registration
│   │   ├── feature_lookup.py           # FeatureLookup config per pipeline
│   │   └── online_store_sync.py        # Delta → online store sync for serving
│   │
│   └── utils/
│       ├── mlflow_helpers.py           # MLflow run management utilities
│       ├── delta_helpers.py            # Delta table read/write utilities
│       └── fhir_adapter.py             # Core FHIR R4 resource mapping logic
│
├── api/                                # FastAPI Gateway (thin container)
│   ├── main.py                         # FastAPI app init + CORS + middleware
│   ├── routers/
│   │   ├── forecast.py                 # GET /forecast → Databricks serving
│   │   ├── staffing.py                 # GET /staffing → Delta prediction table
│   │   ├── alerts.py                   # GET /alerts → Databricks serving
│   │   └── health.py                   # GET /health — all services check
│   ├── schemas/
│   │   ├── forecast_schema.py          # Pydantic request/response models
│   │   ├── alert_schema.py
│   │   ├── staffing_schema.py
│   │   └── fhir_schema.py              # FHIR R4 Pydantic resource models
│   └── middleware/
│       ├── auth.py                     # JWT / Databricks token auth
│       └── audit_logger.py             # Append predictions to Delta audit table
│
├── dashboard/                          # Streamlit Dashboard
│   ├── app.py                          # Multi-page Streamlit app
│   ├── pages/
│   │   ├── doctor.py                   # Real-time patient risk alerts
│   │   ├── manager.py                  # Staffing schedule + Gantt chart
│   │   └── executive.py                # KPI overview + forecast trends
│   └── components/
│       ├── risk_card.py                # Risk score + SHAP widget
│       ├── gantt_chart.py              # Shift schedule visualization
│       └── databricks_client.py        # Databricks SDK query helpers
│
├── workflows/                          # Databricks Workflows (JSON definitions)
│   ├── etl_pipeline.json               # Bronze → Silver → Gold DAG
│   ├── forecasting_training.json       # Weekly Prophet + LSTM retraining
│   ├── alerts_training.json            # Weekly XGBoost retraining
│   ├── batch_predictions.json          # Nightly batch prediction job
│   └── epsilon_sweep.json              # Monthly DP calibration job
│
├── monitoring/
│   ├── lakehouse_monitor_config.py     # Databricks Lakehouse Monitor setup
│   ├── grafana/dashboards/             # Pre-built Grafana dashboard JSON
│   └── elk/logstash.conf               # API Gateway log pipeline
│
├── tests/
│   ├── unit/                           # ETL functions, LP constraints, FHIR mapping
│   ├── integration/                    # API endpoints, Databricks Connect tests
│   ├── model/                          # KPI assertions, bias audit assertions
│   └── load/                           # Locust load tests for API Gateway
│
├── deploy/
│   ├── docker-compose.yml              # Local dev (API + Dashboard)
│   ├── docker-compose.prod.yml         # Production (API + Dashboard + monitoring)
│   ├── Dockerfile.api
│   ├── Dockerfile.dashboard
│   └── terraform/                      # IaC for Databricks workspace setup
│       ├── main.tf                     # Workspace, clusters, Unity Catalog
│       ├── feature_store.tf            # Feature Store online tables
│       └── model_serving.tf            # Serving endpoints + autoscaling
│
├── docs/
│   ├── model_card.md                   # All three models: data, use, limits, fairness
│   ├── epsilon_report.md               # DP privacy-utility tradeoff analysis
│   ├── bias_audit_report.md            # Precision/recall per demographic subgroup
│   ├── fhir_conformance.md             # Supported FHIR R4 resources + profiles
│   └── databricks_architecture.md      # Cluster sizing, Unity Catalog design
│
├── .env.example                        # Environment variable template
├── databricks.yml                      # Databricks Asset Bundle config (DAB)
├── requirements.txt                    # Python dependencies
├── pyproject.toml
└── README.md
```

---

## 🔵 ML Pipelines

### Pipeline 1 — Admission Forecasting

**Goal:** Predict the number of hospital admissions for the next 24 hours.
**KPI Target:** MAE < 2 hours

#### Model Strategy

| Stage | Model | Databricks Implementation | Expected MAE |
|---|---|---|---|
| Baseline | 7-day moving average | PySpark rolling window | ~3–5 admissions |
| Statistical | Prophet (Meta) | Single-node cluster, MLflow autolog | ~2–3 admissions |
| Deep Learning | LSTM (TensorFlow) | GPU cluster, `tensorflow-privacy` | < 2 (target) |
| Final | Prophet + LSTM ensemble | Weighted average, MLflow pyfunc model | Best overall |

---

### Pipeline 2 — Staffing Optimization

**Goal:** Generate the optimal nurse/doctor schedule from forecasted demand.
**KPI Target:** Waiting time reduction ≥ 20%

#### Mathematical Formulation

```
Decision variable:   X[i,t] ∈ {0,1}   — staff i works at time slot t

Objective:   Minimize  Σ cost[i]·X[i,t]  +  λ · Σ max(0, demand[t] − Σ_i X[i,t])

Constraints:
  Σ_t X[i,t] · slot_hours     ≤  40          ∀i   (EU max weekly hours)
  Rest gap between shifts      ≥  11h              (EU Working Time Directive)
  Σ_i X[i,t]                  ≥  demand[t]   ∀t   (demand coverage)
  X[i,t] = 0 if not ICU-certified                  (skill constraint)
```

---

### Pipeline 3 — Complication Alerts

**Goal:** Detect early patient complications in real time.
**KPI Target:** Precision > 85%

#### Label Definition Strategy

> A senior expert invests significant time here — this is the highest-leverage decision in the pipeline.

| Option | Label | Pros | Cons |
|---|---|---|---|
| **A (recommended start)** | 30-day in-hospital mortality | Clearest signal, widely available in MIMIC | Less clinically specific |
| **B (production target)** | ICD-coded events: sepsis (995.91), AKI (584.x) | Clinically precise | Requires ICD coding quality |
| **C (stretch)** | ICU readmission within 30 days | Actionable, avoidable | Lower prevalence |

**Sample alert response:**
```json
{
  "patient_id": "P-00471",
  "risk_score": 0.87,
  "risk_level": "HIGH",
  "explanation": [
    { "feature": "creatinine_last",  "impact": +0.40, "value": 4.2 },
    { "feature": "spo2_mean_24h",    "impact": -0.25, "value": 91.3 },
    { "feature": "age",              "impact": +0.12, "value": 78 }
  ],
  "fhir_resource": { "resourceType": "RiskAssessment", "..." : "..." }
}
```

---

## 📦 MLflow & Model Registry

All three models are tracked and versioned through **MLflow (native Databricks)**.

### Model Lifecycle

```
Development (notebook)
    → Registered in MLflow Model Registry
    → Staging: automated validation (KPI tests + bias audit)
    → Production: promoted by CI/CD pipeline after all checks pass
    → Archived: previous version retained for rollback
```

### Registered Models

| Model Name | Current Stage | KPI |
|---|---|---|
| `admission_forecasting_ensemble` | Production | MAE: 1.7h |
| `complication_alert_xgboost` | Production | Precision: 87% |
| `complication_alert_rf` | Staging | Precision: 84% |

---

## 🗄️ Databricks Feature Store

The Feature Store is the single source of truth for all model features — at training time and serving time.

---

## 🔒 Privacy & Compliance

### Differential Privacy

Patient data is protected at two levels using the differential privacy framework:

| Level | Mechanism | Tool | Applied To |
|---|---|---|---|
| Training | DP-SGD (gradient perturbation) | `tensorflow-privacy` | LSTM model training |
| Output | Laplace mechanism (output perturbation) | Custom implementation | Forecasted admission counts |

**Epsilon calibration** is run as a scheduled **Databricks Job** (monthly), testing values across `{0.1, 0.5, 1.0, 2.0, 5.0, 10.0}`. Each run logs MAE and precision metrics to MLflow, producing the privacy-utility tradeoff curve that serves as GDPR compliance evidence.

**Selected epsilon:** The value achieving all KPI targets with minimum privacy cost. See [`docs/epsilon_report.md`](docs/epsilon_report.md).

### FHIR R4 Compliance

All API responses are available as HL7 FHIR R4 resources, mapped via the `fhir_adapter.py` layer in the API Gateway:

| Pipeline Output | FHIR Resource | Key Fields |
|---|---|---|
| Admission forecast | `Schedule` + `Slot` | `start`, `end`, `serviceType`, `comment` |
| Complication risk | `RiskAssessment` | `prediction.probability`, `basis` (SHAP values) |
| Staffing schedule | `Schedule` + `Practitioner` | `actor`, `planningHorizon` |
| Vital signs | `Observation` | `code` (LOINC), `valueQuantity`, `subject` |

See [`docs/fhir_conformance.md`](docs/fhir_conformance.md).

### GDPR & Unity Catalog

- **Column-level access control** via Unity Catalog — clinicians only see de-identified data
- **Data lineage** fully tracked from source EHR through to prediction output
- **Audit log** stored in Delta table `audit.prediction_log` — append-only, immutable
- **Data retention** policies enforced by Delta table TTL on Bronze raw tables

---

## ⚙️ API Gateway Layer

The API Gateway is a **lightweight FastAPI container** that sits between the Databricks platform and external clients. It handles:
- Routing requests to **Databricks Model Serving** endpoints
- Wrapping responses in **FHIR R4 format**
- **JWT authentication** (or Databricks PAT token validation)
- Appending every prediction to the **Delta audit log**

### Endpoints

| Endpoint | Method | Description | Backend |
|---|---|---|---|
| `/api/v1/forecast` | `GET` | 24h admission predictions + confidence intervals | Databricks Model Serving |
| `/api/v1/staffing` | `GET` | Optimized shift schedule (`?date=&ward=`) | Delta table `gold.staff_schedules` |
| `/api/v1/alerts` | `GET` | Active complication risk alerts (`?ward=&threshold=`) | Databricks Model Serving |
| `/api/v1/health` | `GET` | System health check — all services | Databricks REST API |

### Sample Forecast Response (FHIR R4)

```json
{
  "generated_at": "2025-01-15T08:00:00Z",
  "horizon_hours": 24,
  "model_version": "admission_forecasting_ensemble/3",
  "predictions": [
    {
      "hour": "2025-01-15T09:00:00Z",
      "predicted_admissions": 12,
      "confidence_interval": { "lower": 9, "upper": 15 }
    }
  ],
  "fhir_resource": {
    "resourceType": "Schedule",
    "status": "active",
    "serviceType": [{ "text": "Admission Forecast" }],
    "comment": "Predicted admissions: 12 (CI: 9–15)"
  }
}
```

---

## 📊 Dashboard

The Streamlit dashboard connects to the API Gateway and directly to Delta Lake via the Databricks SQL Connector.

> **Deployment note:** Use **Databricks Apps** (currently in preview) for native integration, or deploy as a container on Azure Container Apps / AWS Fargate.

Access the dashboard at: `http://localhost:8501` (dev) or your production URL.

| Role | View | Key Widgets |
|---|---|---|
| 🩺 **Doctor** | Real-time patient risk alerts | Risk score cards, SHAP explanation bar chart, alert threshold slider, patient drill-down |
| 📋 **Manager** | Today's staffing schedule | Gantt-style shift chart, coverage heatmap, constraint violation flags, export to Excel |
| 📈 **Executive** | KPI overview & trends | MAE 30-day trend, waiting time delta gauge, precision history chart, cost savings estimate |

---

## 📡 Monitoring & Observability

### Databricks Lakehouse Monitoring

Configured via `monitoring/lakehouse_monitor_config.py`, Lakehouse Monitoring automatically tracks:
- **Data quality drift** on all Silver and Gold Delta tables
- **Feature distribution drift** vs training baseline (per-column statistics)
- **Model prediction drift** — output distribution shift over time
- **Profile metrics** stored in a `_monitoring` Delta table for each monitored table

### MLflow Model Quality Tracking

MLflow logs and tracks the following metrics after every batch prediction run:

| Metric | Alert Threshold |
|---|---|
| `rolling_mae_7d` | > 3h → trigger retraining |
| `rolling_precision_7d` | < 80% → page on-call engineer |
| `max_bias_disparity` | > 5% → automated retraining block |
| `data_drift_score` | > 0.15 → flag for manual review |

### API Gateway Observability

| Layer | Tool | What is Tracked |
|---|---|---|
| Metrics | Prometheus + Grafana | Request rate, latency p99, error rate per endpoint |
| Logs | ELK Stack | Structured JSON prediction audit logs |
| Traces | Databricks REST API | Model serving invocation latency |

---

## 🚀 Getting Started

### Prerequisites

- Databricks workspace (AWS / Azure / GCP) with Unity Catalog enabled
- Python 3.10+
- Databricks CLI >= 0.200
- Docker Engine >= 24.0 (for API Gateway + Dashboard)
- PhysioNet account with MIMIC-III/IV access approved

### Step 1 — Configure Databricks CLI

```bash
databricks configure --token
# Enter your workspace URL: https://adb-xxxx.azuredatabricks.net
# Enter your personal access token: dapixxxx
```

### Step 2 — Deploy Databricks Asset Bundle

```bash
git clone https://github.com/your-org/hospital-optimization-system.git
cd hospital-optimization-system

# Deploy all notebooks, jobs, and cluster configs to Databricks workspace
databricks bundle deploy --target dev
```

### Step 3 — Initialize Unity Catalog & Delta Tables

```bash
databricks bundle run setup_unity_catalog --target dev
```

### Step 4 — Run the ETL Pipeline

```bash
# Trigger the Bronze → Silver → Gold Databricks Workflow
databricks bundle run etl_pipeline --target dev
```

### Step 5 — Train the Models

```bash
databricks bundle run forecasting_training --target dev
databricks bundle run alerts_training --target dev
```

### Step 6 — Start the API Gateway & Dashboard (local dev)

```bash
cp .env.example .env
# Edit .env — add DATABRICKS_HOST, DATABRICKS_TOKEN, serving endpoint URLs

docker-compose up -d
# API Gateway: http://localhost:8000
# Dashboard:   http://localhost:8501
```

---

## ⚙️ Databricks Setup

### Cluster Configuration

| Cluster | Purpose | Recommended Size |
|---|---|---|
| ETL Cluster | PySpark ETL jobs | `Standard_DS3_v2` × 4 workers (autoscaling 2–8) |
| Training Cluster (CPU) | Prophet, XGBoost, PuLP | `Standard_DS4_v2` × 4 workers |
| Training Cluster (GPU) | LSTM (TensorFlow) | `Standard_NC6s_v3` × 2 GPU workers |
| Serving Cluster | Databricks Model Serving | Managed (serverless) |
| SQL Warehouse | Dashboard queries | Serverless SQL Warehouse |

### Databricks Workflows

All pipelines are orchestrated as **Databricks Workflows** (defined in `workflows/`):

| Workflow | Schedule | Steps |
|---|---|---|
| `etl_pipeline` | Every 6 hours | Bronze ingest → Silver clean → Gold features → Feature Store update |
| `forecasting_training` | Weekly (Monday 02:00) | Train Prophet → Train LSTM → Ensemble → MLflow log → Promote if KPI met |
| `alerts_training` | Weekly (Monday 03:00) | Train XGBoost → Bias audit → MLflow log → Promote if KPI met |
| `batch_predictions` | Daily (05:00) | Load models → Score all current patients → Write to `gold.admission_predictions` |
| `epsilon_sweep` | Monthly | DP calibration → Log tradeoff → Update epsilon config |

---

## ⚙️ Configuration

**Critical environment variables (`.env`):**

```env
# Databricks
DATABRICKS_HOST=https://adb-xxxx.azuredatabricks.net
DATABRICKS_TOKEN=dapixxxx
DATABRICKS_WAREHOUSE_ID=xxxx

# Model Serving Endpoints
FORECAST_SERVING_ENDPOINT=https://.../serving-endpoints/admission_forecasting/invocations
ALERTS_SERVING_ENDPOINT=https://.../serving-endpoints/complication_alert/invocations

# Feature Store
FEATURE_STORE_CATALOG=hospital_prod
FEATURE_STORE_SCHEMA=gold

# Privacy
PRIVACY_EPSILON=1.0
PRIVACY_DELTA=1e-5

# API
JWT_SECRET=<your-secret>
LOG_LEVEL=INFO

# FHIR
FHIR_VERSION=R4
FHIR_SERVER_URL=https://your-fhir-server.azurehealthcareapis.com
```

---

## 🧪 Testing

```bash
# Unit tests — ETL functions, LP constraints, FHIR mapping
pytest tests/unit/ -v --cov=src --cov-report=term-missing

# Integration tests — API endpoints + Databricks Connect
pytest tests/integration/ -v

# Model KPI assertion tests — MAE, precision, bias thresholds
pytest tests/model/ -v

# FHIR contract tests — validate against official R4 JSON schema
pytest tests/integration/test_fhir_contracts.py -v

# Load tests
locust -f tests/load/locustfile.py --host=http://localhost:8000 --users=100
```

**Coverage target: > 80%**

CI/CD pipeline (GitHub Actions) runs all unit + integration tests and the model KPI suite on every pull request. Merge to `main` is blocked if any test fails or coverage drops below 80%.

---

## 🐳 Deployment

### Local Development

```bash
docker-compose up -d
# API Gateway → http://localhost:8000
# Dashboard   → http://localhost:8501
```

### Production (Azure example)

```bash
# 1. Provision Databricks workspace + Unity Catalog (Terraform)
cd deploy/terraform && terraform apply

# 2. Deploy Asset Bundle to production workspace
databricks bundle deploy --target prod

# 3. Deploy API Gateway to Azure Container Apps
az containerapp up --name hospital-api --image your-registry/hospital-api:latest

# 4. Deploy Dashboard to Azure Container Apps
az containerapp up --name hospital-dashboard --image your-registry/hospital-dashboard:latest
```

### Production Service Map

| Service | Hosting | Port | Description |
|---|---|---|---|
| Databricks Model Serving | Databricks (managed) | HTTPS | Forecast + Alert REST endpoints |
| API Gateway (FastAPI) | Azure Container Apps / AWS Fargate | 8000 | FHIR wrapping + auth |
| Dashboard (Streamlit) | Databricks Apps / Container | 8501 | Role-based clinical dashboard |
| SQL Warehouse | Databricks (serverless) | — | Dashboard SQL queries |
| Grafana | Container / Azure Monitor | 3000 | Operational metrics |
| Kibana | ELK Stack | 5601 | API Gateway logs |

---

## 📈 Business KPIs

| KPI | Target | Measurement Method |
|---|---|---|
| Admission Forecast MAE | < 2 hours | Rolling 7-day avg on live predictions vs actuals |
| Complication Alert Precision | > 85% | Validated against confirmed clinical outcomes |
| Waiting Time Reduction | ≥ 20% | Pre/post deployment comparison (A/B baseline) |
| API Availability | 99.9% uptime | Databricks serving + container health checks |
| Model Inference Latency | < 200ms p99 | Databricks serving endpoint metrics |
| Feature Freshness | < 6 hours lag | ETL workflow SLA monitoring |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Cloud Platform** | Azure Databricks / AWS Databricks / GCP Databricks |
| **Data Storage** | Delta Lake (Bronze / Silver / Gold), Unity Catalog |
| **Ingestion** | Databricks Auto Loader, `fhir.resources` (FHIR R4) |
| **ETL** | PySpark, Pandas on Spark, Great Expectations on Databricks |
| **Feature Store** | Databricks Feature Store (offline + online) |
| **Forecasting** | Prophet (Meta), TensorFlow / Keras (LSTM) |
| **Optimization** | PuLP (LP solver), CBC / GLPK backend |
| **Classification** | XGBoost, Scikit-learn, SHAP, Optuna / Hyperopt |
| **Privacy** | `tensorflow-privacy` (DP-SGD), Laplace mechanism |
| **ML Tracking** | MLflow (native Databricks), Hyperopt (distributed) |
| **Model Serving** | Databricks Model Serving (serverless REST) |
| **Orchestration** | Databricks Workflows (DAG job scheduler) |
| **API Gateway** | FastAPI, Pydantic, Uvicorn |
| **Dashboard** | Streamlit, Plotly, Altair, Databricks SQL Connector |
| **IaC** | Terraform (Databricks provider), Databricks Asset Bundles |
| **Monitoring** | Databricks Lakehouse Monitoring, Prometheus, Grafana, ELK |
| **Testing** | pytest, pytest-cov, Locust |
| **CI/CD** | GitHub Actions + Databricks Asset Bundle deploy |

---

## 📄 Responsible AI

| Document | Location | Description |
|---|---|---|
| Model Card | [`docs/model_card.md`](docs/model_card.md) | Training data, intended use, limitations, fairness evaluation for all three models |
| Epsilon Report | [`docs/epsilon_report.md`](docs/epsilon_report.md) | DP privacy-utility tradeoff curve with GDPR justification for chosen epsilon |
| Bias Audit Report | [`docs/bias_audit_report.md`](docs/bias_audit_report.md) | Precision/recall per demographic subgroup (age, gender, ethnicity) |
| FHIR Conformance | [`docs/fhir_conformance.md`](docs/fhir_conformance.md) | Supported FHIR R4 resources, profiles, and known limitations |
| Databricks Architecture | [`docs/databricks_architecture.md`](docs/databricks_architecture.md) | Cluster sizing, Unity Catalog design, cost optimization decisions |

---

## 🤝 Contributing

1. Branch: `git checkout -b feature/my-feature`
2. Test: `pytest tests/ --cov=src --cov-fail-under=80`
3. Validate bundle: `databricks bundle validate`
4. Open a pull request — one approval + passing CI required before merge

---

## 📜 License

MIT License. See [`LICENSE`](LICENSE) for details.

> **MIMIC Data:** Use of MIMIC-III/IV is subject to the [PhysioNet Credentialed Health Data License](https://physionet.org/content/mimiciii/view-license/). Ensure your team holds valid credentialed access before using real patient data in this system.

---

<div align="center">

*Built for Senior AI Engineering — Databricks Lakehouse · Healthcare Systems · Applied ML · HealthTech AI*

</div>
