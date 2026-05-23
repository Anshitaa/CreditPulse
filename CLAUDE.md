# CLAUDE.md — CreditPulse

Auto-loaded by Claude Code on every session.

**Owner:** Anshita Bhardwaj — MS Data Science ASU May 2026  
**Target roles:** AmEx (AI/ML Platform), Tesla (Data Eng), Amazon (Data Eng CDP), PayPal (MLE)

---

## Build Status

| Phase | Status | Checkpoint |
|---|---|---|
| 1 — Synthetic Data | ✅ Built | 150K transactions, 12.5% fraud rate, loaded to `raw.transactions` |
| 2 — Spark Streaming Features | ✅ Built | `features/spark_streaming_features.py` (needs JDBC JAR + Kafka) |
| 3 — ML Models | ✅ Built | `fraud_detector.pkl`, `anomaly_detector.pkl`, `credit_risk_scorer.pkl` |
| 4 — SHAP + Counterfactuals | ✅ Built | Dice-ML + SHAP top-5 per transaction |
| 5 — Fairness + Drift | ✅ Built | Fairlearn gate + PSI monitor → `audit.*` tables |
| 6 — LLM Agent | ✅ Built | LangChain ReAct + FAISS RAG (16 chunks, 8 regulatory docs) |
| 7 — FastAPI | ✅ Built | 5 routers, 16 routes, WebSocket live feed |
| 8 — React Frontend | ✅ Built | 5 pages: Dashboard, Live Feed, What-If, Governance, Agent Chat |
| 9 — Kiro Specs + Hooks | ✅ Built | 3 EARS specs, 3 hooks, 3 MCP servers |
| 10 — Docker + K8s | ✅ Built | docker-compose + minikube manifests + KEDA ScaledObject |
| 11 — Unit Tests | ✅ Built | 15/15 passing (`tests/unit/test_models.py`) |
| 12 — Integration Tests | ✅ Built | 105/105 passing (`tests/integration/`) |

---

## Stack

| Layer | Technology |
|---|---|
| Event streaming | Kafka (topics: `transactions.raw`, `alerts.fraud`) |
| Warehouse | PostgreSQL (port **5435**) — 4 schemas: raw/staging/mart/audit |
| Batch features | PySpark + JDBC → `mart.feature_store` |
| ML | XGBoost (fraud + credit risk) + Isolation Forest (anomaly) + SHAP + Dice-ML |
| Governance | MLflow (file-based) + Fairlearn + PSI drift monitoring |
| LLM/Agent | LangChain ReAct, FAISS RAG (8 regulatory docs), Gemini Flash (free) |
| API | FastAPI (async, WebSocket) |
| Frontend | React (Vite + Recharts + Tailwind) |
| Infra | Docker Compose (dev), Kubernetes minikube (demo) + KEDA |

---

## Critical Environment Facts

- **PostgreSQL port: 5435** (not 5432/5433 — Mac has 3 Postgres instances)
- **Conda env:** `talentlens` (shared with TalentLens project)
- **Python:** `/opt/anaconda3/envs/talentlens/bin/python`
- **MLFLOW_TRACKING_URI:** `file:///Users/anshita/Desktop/CreditPulse/mlruns` (file-based, no server needed)
- **Frontend port:** 5174 (not 5173 — avoids TalentLens conflict)
- **DATABASE_URL:** `postgresql://creditpulse:creditpulse@localhost:5435/creditpulse`
- **Java for PySpark:** `export JAVA_HOME=/opt/anaconda3`
- **JDBC JAR:** `spark/jars/postgresql-42.7.3.jar` (downloaded)
- **Gemini rate limit:** 15 RPM free tier — agent tests have 5s sleep guard

---

## Quick Start (Full Stack)

```bash
cd /Users/anshita/Desktop/CreditPulse

# ── 1. Start Docker infrastructure ──────────────────────────────────────────
open -a Docker && sleep 15
docker-compose -f infra/docker/docker-compose.yml up -d postgres kafka redis zookeeper

# ── 2. Load .env ─────────────────────────────────────────────────────────────
set -a && source .env && set +a

# ── 3. Start API (Terminal 1) ─────────────────────────────────────────────────
PYTHONPATH=/Users/anshita/Desktop/CreditPulse \
  /opt/anaconda3/envs/talentlens/bin/uvicorn api.main:app --port 8000

# ── 4. Start frontend (Terminal 2) ───────────────────────────────────────────
cd frontend && npm run dev -- --port 5174

# ── 5. Verify ────────────────────────────────────────────────────────────────
curl http://127.0.0.1:8000/health/ready
# → {"status":"ready","checks":{"model":true,"counterfactual_engine":true,"database":true}}
```

**URLs:**
- Dashboard: `http://localhost:5174`
- API docs: `http://127.0.0.1:8000/docs`
- Live feed WebSocket: `ws://127.0.0.1:8000/ws/scores`

---

## Run Tests

```bash
set -a && source .env && set +a
export PYTHONPATH=/Users/anshita/Desktop/CreditPulse

# Unit tests (no Docker needed)
pytest tests/unit/test_models.py -v
# → 15/15 PASS

# Integration tests (requires Docker stack + API running)
pytest tests/integration/ \
  --ignore=tests/integration/test_agent.py \
  --ignore=tests/integration/test_kafka.py \
  -v
# → 105/105 PASS

# Agent tests (slow — Gemini rate limit guard, ~60s)
pytest tests/integration/test_agent.py -v

# Kafka tests (requires Kafka container)
pytest tests/integration/test_kafka.py -v
```

---

## Port Reference

| Service | Port | Notes |
|---|---|---|
| CreditPulse API | 8000 | `http://127.0.0.1:8000/docs` |
| React Frontend | 5174 | Vite dev server |
| PostgreSQL | 5435 | Avoid Mac 5432/5433 conflicts |
| Kafka | 9092 | |
| Redis | 6379 | Feast online store |
| MLflow | file-based | No server — uses `mlruns/` directory |

---

## Repository Structure

```
data/synthetic_transactions.py     # 150K transactions, 12.5% fraud rate
ingestion/kafka/producer.py        # Idempotent Kafka producer (simulate/replay modes)
features/spark_streaming_features.py  # PySpark → Feast → Redis velocity features
models/
  fraud_detector.py                # XGBoost + Optuna HPO + SHAP
  anomaly.py                       # Isolation Forest
  credit_risk.py                   # XGBoost regression
  counterfactual.py                # Dice-ML counterfactuals
governance/
  fairness_gate.py                 # Fairlearn (3 protected groups, δ < 0.05)
  drift_monitor.py                 # PSI per feature → audit.drift_reports
agent/
  react_agent.py                   # LangChain ReAct + 5 tools
  rag/indexer.py                   # FAISS index over 8 regulatory docs
api/
  main.py                          # FastAPI app + WebSocket + lifespan model load
  routers/                         # score, explain, risk, governance, agent
frontend/                          # React 5-page dashboard (Vite + Recharts)
tests/
  unit/test_models.py              # 15 unit tests
  integration/                     # 105+ integration tests (6 files + agent + kafka)
docs/model_risk_card_v1.0.0.md    # OCC SR 11-7 model risk card
.kiro/
  specs/                           # 3 EARS requirement specs
  hooks/                           # fairness-gate.sh, psi-check.sh, auto-test.sh
  mcp/                             # 3 MCP servers (postgres, kafka, mlflow)
  steering/                        # ml-standards, fairness-standards, coding-standards
infra/docker/docker-compose.yml    # Postgres/Kafka/Redis/Zookeeper
infra/k8s/deployments.yaml         # K8s manifests
scripts/minikube-up.sh             # One-shot K8s demo deploy
```

---

## Key Architecture Decisions

- **ADR-001 Kafka over direct DB writes** — fan-out to consumers, replay on failure
- **ADR-002 XGBoost over neural nets** — SHAP explainability required for financial compliance; < 100ms inference
- **ADR-003 Dice-ML counterfactuals** — FCRA §615(a) adverse action notice compliance
- **ADR-004 KEDA on Kafka consumer lag** — lag is the leading indicator; HPA on CPU lags behind burst traffic
- **ADR-005 File-based MLflow** — no server dependency in dev; swap to MLflow server URI for production

---

## Deployment Honesty Rule

Only claim EKS deployment on resume if actually deployed there.  
For portfolio demo: deploy to minikube → state *"architected for AWS EKS; local demo runs on minikube."*  
Amazon interviewers ask detailed EKS/IAM/node-group questions.
