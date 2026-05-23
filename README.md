# CreditPulse — AI-Powered Credit Risk Intelligence Platform

Real-time fraud detection and credit risk scoring platform with SHAP explainability, counterfactual generation, LLM-powered analyst agent, and full model governance — built to the standards of AmEx, PayPal, and Amazon-scale financial ML systems.

---

## Architecture

```
Synthetic Transactions (150K, 12.5% fraud rate)
        │
        ▼
   Kafka Producer ──────────────────────────────────────────────────────────┐
        │                                                                    │
        ▼                                                                    ▼
  PostgreSQL (raw.*)                                                  alerts.fraud topic
        │
        ▼
  dbt staging layer (staging.*)
        │
        ▼
  PySpark Streaming Features ──→ Feast Feature Store ──→ Redis (online store)
        │
        ▼
  ML Inference Pipeline
  ┌─────────────────────────────────────────────────────────────┐
  │  XGBoost Fraud Detector  │  Isolation Forest  │  XGBoost    │
  │  + SHAP top-5 features   │  Anomaly Detector  │  Risk Scorer│
  └─────────────────────────────────────────────────────────────┘
        │                         │
        ▼                         ▼
  Dice-ML Counterfactuals    Fairlearn Gate + PSI Drift Monitor
        │                         │
        └────────────┬────────────┘
                     ▼
              mart.* + audit.*
                     │
                     ▼
         ┌───────────────────────┐
         │   FastAPI (16 routes) │ ← WebSocket live feed
         └───────────────────────┘
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
   React Dashboard        LangChain ReAct Agent
   (5 pages)              + FAISS RAG (8 regulatory docs)
```

---

## Stack

| Layer | Technology |
|---|---|
| Event streaming | Kafka (`transactions.raw`, `alerts.fraud`) |
| Warehouse | PostgreSQL · 4 schemas: `raw / staging / mart / audit` |
| Feature engineering | PySpark + Feast + Redis (velocity features) |
| ML | XGBoost · Isolation Forest · SHAP · Dice-ML counterfactuals |
| Governance | MLflow · Fairlearn · PSI drift monitor |
| LLM / Agent | LangChain ReAct · FAISS RAG · Gemini Flash (free tier) |
| API | FastAPI (async · WebSocket) |
| Frontend | React · Vite · Recharts · Tailwind CSS |
| Infra | Docker Compose (dev) · Kubernetes minikube + KEDA (demo) |

---

## Key Features

### Real-Time Fraud Scoring
- Sub-100ms p99 inference via XGBoost + pre-computed Feast features
- Composite decision: **FRAUD** (score > 75) / **REVIEW** (40–75) / **CLEAR** (< 40)
- Live WebSocket feed — watch scores stream in real time

### Explainability
- **SHAP** — top-5 feature contributions per transaction
- **Dice-ML counterfactuals** — "change X to Y to get a different decision" (FCRA §615(a) adverse action compliance)
- What-If Simulator — adjust transaction attributes and see score change instantly

### Model Governance
- **Fairlearn** — bias audit across 3 protected groups (demographic parity δ < 0.05)
- **PSI drift monitor** — alerts when feature distribution shifts > 0.2
- **MLflow** — experiment tracking, model versioning, artifact lineage
- **OCC SR 11-7** model risk card — [`docs/model_risk_card_v1.0.0.md`](docs/model_risk_card_v1.0.0.md)

### LLM Analyst Agent
- LangChain ReAct agent with 5 tools: `get_risk_summary`, `explain_decision`, `run_what_if`, `query_sql`, `search_regulations`
- FAISS RAG over 8 regulatory documents (FCRA, PCI DSS, OCC SR 11-7, Regulation E)
- Ask in plain English: *"Why was transaction T-1234 flagged?"* or *"What's the current false positive rate for REVIEW decisions?"*

---

## Quick Start

```bash
# 1. Clone and enter
git clone https://github.com/Anshitaa/CreditPulse.git
cd CreditPulse

# 2. Create .env from example — add your Gemini key for the agent
cp .env.example .env

# 3. Start infrastructure
docker-compose -f infra/docker/docker-compose.yml up -d

# 4. Install Python deps (conda env recommended)
pip install -r requirements.txt

# 5. Start API (Terminal 1)
PYTHONPATH=$(pwd) uvicorn api.main:app --port 8000 --reload

# 6. Start frontend (Terminal 2)
cd frontend && npm install && npm run dev -- --port 5174
```

| Service | URL |
|---|---|
| React Dashboard | http://localhost:5174 |
| API docs (Swagger) | http://127.0.0.1:8000/docs |
| Live feed (WebSocket) | ws://127.0.0.1:8000/ws/scores |

---

## Testing

```bash
export PYTHONPATH=$(pwd)

# Unit tests — no Docker needed (15 tests)
pytest tests/unit/test_models.py -v

# Integration tests — requires Docker stack + API running (105 tests)
pytest tests/integration/ \
  --ignore=tests/integration/test_agent.py \
  --ignore=tests/integration/test_kafka.py \
  -v

# Agent tests — requires Gemini API key (~60s, rate-limit guard)
set -a && source .env && set +a
pytest tests/integration/test_agent.py -v
```

---

## Repository Structure

```
data/                              # Synthetic transaction generator (150K rows)
ingestion/kafka/                   # Idempotent Kafka producer
features/                          # PySpark streaming features → Feast → Redis
models/
  fraud_detector.py                # XGBoost + Optuna HPO + SHAP
  anomaly.py                       # Isolation Forest
  credit_risk.py                   # XGBoost regression
  counterfactual.py                # Dice-ML counterfactuals (FCRA compliance)
governance/
  fairness_gate.py                 # Fairlearn — 3 protected groups, δ < 0.05
  drift_monitor.py                 # PSI per feature → audit.drift_reports
agent/
  react_agent.py                   # LangChain ReAct + 5 tools
  rag/indexer.py                   # FAISS index over regulatory docs
api/
  main.py                          # FastAPI + WebSocket + lifespan model load
  routers/                         # score · explain · risk · governance · agent
frontend/                          # React dashboard (5 pages)
tests/
  unit/test_models.py              # 15 unit tests
  integration/                     # 105+ integration tests
docs/model_risk_card_v1.0.0.md    # OCC SR 11-7 model risk card
infra/docker/docker-compose.yml   # Postgres · Kafka · Redis · Zookeeper
infra/k8s/                         # Kubernetes manifests + KEDA ScaledObject
.kiro/
  specs/                           # EARS requirement specs (3 features)
  hooks/                           # fairness-gate · psi-check · auto-test
  mcp/                             # MCP servers: postgres · kafka · mlflow
```

---

## Architecture Decisions

| ADR | Decision | Why |
|---|---|---|
| ADR-001 | Kafka over direct DB writes | Fan-out to multiple consumers; replay on failure |
| ADR-002 | XGBoost over neural nets | SHAP explainability required for FCRA adverse action; < 100ms inference |
| ADR-003 | Dice-ML counterfactuals | FCRA §615(a) adverse action notice — "what would change the decision?" |
| ADR-004 | KEDA on Kafka consumer lag | Lag is the leading indicator of backpressure; CPU-based HPA lags behind bursts |
| ADR-005 | File-based MLflow in dev | No server dependency; swap tracking URI for production MLflow server |

---

## Compliance Coverage

| Regulation | Implementation |
|---|---|
| FCRA §615(a) | Dice-ML counterfactuals stored per decision in `audit.explanations` |
| PCI DSS | No raw card data stored; transaction IDs only; audit trail in `audit.*` |
| OCC SR 11-7 | Model risk card · validation metrics · governance workflow documented |
| Regulation E | Dispute workflow hooks in Kiro spec `fraud-detection.md` |

---

## Author

**Anshita Bhardwaj** — MS Data Science, Arizona State University (May 2026)  
Target roles: AI/ML Platform · Data Engineering · MLE (AmEx · Tesla · Amazon · PayPal)
