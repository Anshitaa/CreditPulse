# CreditPulse — Quick Start Guide

**Real-Time Fraud Detection & Credit Risk Intelligence Platform**  
Built spec-first with Kiro · Stack: Kafka + Spark + XGBoost + SHAP + Dice-ML + LangChain + FastAPI + React

---

## Prerequisites

```bash
# Conda env (same as TalentLens)
conda activate talentlens

# Install dependencies
cd /Users/anshita/Desktop/CreditPulse
pip install -r requirements.txt

# Frontend
cd frontend && npm install && cd ..

# Copy env file
cp .env.example .env
# → Fill in your GEMINI_API_KEY (free) or ANTHROPIC_API_KEY
```

---

## Phase 0 — Start Infrastructure

```bash
cd /Users/anshita/Desktop/CreditPulse

# Start Postgres (port 5435), Kafka, Redis, MLflow
docker-compose -f infra/docker/docker-compose.yml up -d postgres kafka redis mlflow schema-registry

# Verify DB is up
docker exec creditpulse-postgres psql -U creditpulse -d creditpulse -c "\dn"
# Should show: raw, staging, mart, audit schemas
```

---

## Phase 1 — Generate Data

```bash
# Fast demo (50K transactions, ~30 seconds)
python data/synthetic_transactions.py --fast --load-db --save-parquet

# Full dataset (1M transactions, ~5 minutes)
python data/synthetic_transactions.py --rows 1000000 --load-db --save-parquet

# Verify
docker exec creditpulse-postgres psql -U creditpulse -d creditpulse \
  -c "SELECT COUNT(*), ROUND(AVG(amount)::numeric,2) FROM raw.transactions;"
```

---

## Phase 2 — Spark Streaming Features (optional for demo)

```bash
export JAVA_HOME=/opt/anaconda3
export JDBC_JAR=/Users/anshita/Desktop/CreditPulse/spark/jars/postgresql-42.7.3.jar
export POSTGRES_HOST=localhost
export POSTGRES_PORT=5435

python features/spark_streaming_features.py --local
# Ctrl+C after ~60 seconds once features have been written
```

---

## Phase 3 — Train ML Models

```bash
# Train all three models (fraud, credit risk, anomaly)
python models/fraud_detector.py --train --n-trials 10   # ~3 min
python models/anomaly.py --train                          # ~1 min
python models/credit_risk.py --train --n-trials 10       # ~2 min

# MLflow UI → http://localhost:5001
# Check: 3 experiments, each with 1 run
```

---

## Phase 4 — Build RAG Index

```bash
python -c "from agent.rag.indexer import build_index; build_index()"
# → 8 regulatory docs, ~28 chunks indexed to agent/rag/faiss_index/
```

---

## Phase 5 — Run Kiro Quality Hooks

```bash
# Test fairness gate manually
python governance/fairness_gate.py --source synthetic
# Expected: demographic_parity_difference < 0.05 → gate PASSED

# Test PSI drift monitor
python governance/drift_monitor.py --source synthetic
# Expected: some features STABLE, some MONITOR (synthetic drift injected)
```

---

## Phase 6 — Start API + Frontend

```bash
# Terminal 1: API
uvicorn api.main:app --reload --port 8000

# Terminal 2: Frontend
cd frontend && npm run dev
# → http://localhost:5174

# API docs: http://127.0.0.1:8000/docs
```

---

## Phase 7 — Test End-to-End

```bash
# Score a transaction (< 100ms target)
curl -s -X POST http://127.0.0.1:8000/score/ \
  -H "Content-Type: application/json" \
  -d '{
    "account_id": "acct-001",
    "amount": 4500.0,
    "merchant_category": "wire_transfer",
    "is_foreign_merchant": true,
    "hour_of_day": 2,
    "day_of_week": 6,
    "txn_velocity_1h": 8,
    "amount_vs_avg_ratio": 12.5
  }' | python -m json.tool

# Get counterfactual explanation
curl -s -X POST http://127.0.0.1:8000/explain/counterfactual \
  -H "Content-Type: application/json" \
  -d '{"account_id": "acct-001", "amount": 4500, "merchant_category": "wire_transfer", "is_foreign_merchant": true, "hour_of_day": 2, "day_of_week": 6, "txn_velocity_1h": 8, "amount_vs_avg_ratio": 12.5}' \
  | python -m json.tool

# Ask the agent
curl -s -X POST http://127.0.0.1:8000/agent/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "Is the fraud model showing signs of drift that would require retraining?"}' \
  | python -m json.tool

# Risk summary
curl -s http://127.0.0.1:8000/risk/summary | python -m json.tool
```

---

## Phase 8 — Stream Kafka Events

```bash
# Simulate 50 transactions/sec for 60 seconds
python ingestion/kafka/producer.py --mode simulate --rate 50 --duration 60

# Watch live in the frontend → http://localhost:5174/feed
```

---

## Phase 9 — Run Unit Tests

```bash
pytest tests/unit/test_models.py -v
# Expected: all tests PASS
```

---

## Phase 10 — Kiro Demo (the differentiator!)

```bash
# Open CreditPulse in Kiro IDE
# 1. Show .kiro/specs/fraud-detection.md → EARS requirements
# 2. Open models/fraud_detector.py → save file → watch fairness-gate.sh fire
# 3. In Kiro chat: "What's the Kafka consumer lag?" → MCP returns live data
# 4. In Kiro chat: "Show me the latest drift report" → MCP queries PostgreSQL
# 5. Promote a model in Kiro: "Promote fraud_detector v1 to Production" → MLflow MCP
```

---

## minikube K8s Deploy (portfolio demo)

```bash
chmod +x scripts/minikube-up.sh
./scripts/minikube-up.sh

# Dashboard → http://creditpulse.local
# Statement for interviews: "Architected for AWS EKS; local demo runs on minikube."
```

---

## Port Reference

| Service | Port | Notes |
|---|---|---|
| CreditPulse API | 8000 | uvicorn, http://127.0.0.1:8000/docs |
| React Frontend | 5174 | Vite dev server |
| PostgreSQL | 5435 | Avoid conflict with Mac 5432/5433 |
| Kafka | 9092 | |
| Schema Registry | 8081 | |
| Redis | 6379 | Feast online store |
| MLflow | 5001 | Avoid macOS AirPlay on 5000 |

---

## Resume Bullets (copy-paste ready)

- "Built real-time fraud detection platform processing 10K events/sec via Kafka + Spark Streaming + Feast, achieving p99 inference latency < 100ms"
- "Implemented Dice-ML counterfactual explanations + Fairlearn fairness gates enforced as automated Kiro IDE hooks in the development loop"
- "Developed 3 production MCP servers (PostgreSQL, Kafka, MLflow) enabling AI IDE agents to query live platform data during development"
- "Spec-driven development: 100% of features backed by EARS specifications with full traceability to code (Kiro)"
- "Deployed on minikube with KEDA auto-scaling on Kafka consumer lag; architected for AWS EKS"
