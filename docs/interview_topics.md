# CreditPulse — Interview Topics & Study Guide

**Owner:** Anshita Bhardwaj | MS Data Science, ASU May 2026  
**Target roles:** AmEx (AI/ML Platform) · Tesla (Data Eng) · Amazon (Data Eng CDP) · PayPal (MLE)

---

## What is CreditPulse? (30-second pitch)

CreditPulse is a **real-time fraud detection and credit risk intelligence platform** — the kind of system a fintech or bank runs to automatically score every transaction for fraud, explain *why* it was flagged (legally required under FCRA), monitor for model drift and fairness violations, and let analysts query the system in plain English via an AI agent.

**End-to-end flow:**
```
150K synthetic transactions
  → Kafka (event stream)
  → PostgreSQL medallion warehouse (raw → staging → mart → audit)
  → PySpark (velocity features) → Feast → Redis (online store)
  → XGBoost fraud detector + Isolation Forest anomaly detector
  → SHAP top-5 explanations + Dice-ML counterfactuals
  → Fairlearn fairness gate + PSI drift monitor
  → FastAPI (16 routes, WebSocket) + React dashboard (5 pages)
  → LangChain ReAct agent + FAISS RAG (8 regulatory docs)
  → Docker Compose (dev) + Kubernetes/KEDA (demo)
```

**Why each company cares:**
| Company | What maps directly |
|---|---|
| AmEx | Model governance, fairness gate, SHAP, counterfactuals, OCC SR 11-7 risk card |
| PayPal | Real-time inference SLA, FCRA adverse action, PCI DSS, WebSocket live feed |
| Amazon | Kafka + Spark + Medallion warehouse, KEDA autoscaling, 105-test integration suite |
| Tesla | Kafka pipeline, feature store (Feast), data quality, PSI drift monitoring |

---

## Topic 1 — Data Architecture (Medallion / Lakehouse)

**What we built:** 4-schema PostgreSQL warehouse: `raw` → `staging` → `mart` → `audit`

**What to know:**
- **Raw layer:** immutable, append-only. Never modify. Source of truth for replay.
- **Staging:** cleaned, deduplicated, type-cast. No business logic.
- **Mart:** aggregated, business-ready tables (`feature_store`, `risk_scores`, `account_summaries`). Optimised for reads.
- **Audit:** append-only compliance trail. Rows are never updated. Every score, every governance run is logged here.
- **Why this layering?** Each layer fails independently. Bad staging transform doesn't corrupt raw. Mart rebuild is idempotent.

**Interview question you'll get:** *"Walk me through how you'd design a data warehouse for fraud detection."*

**Your answer anchor:** CreditPulse's 4-schema medallion architecture. Lead with the audit layer — most candidates miss it. Explain immutability as a compliance requirement, not just good practice.

---

## Topic 2 — Kafka & Event-Driven Architecture

**What we built:** Producer publishes to `transactions.raw` topic. Consumer scores each event, publishes fraud alerts to `alerts.fraud`. KEDA scales consumers on lag.

**What to know:**
- **Why Kafka over direct DB writes:** fan-out to multiple consumers, replay on failure, decouples producers from consumers (ADR-001)
- **Consumer groups:** each group gets its own offset pointer — the fraud scorer and the analytics consumer are separate groups on the same topic
- **Consumer lag:** how far behind the consumer is from the latest message — KEDA uses this as its autoscaling signal
- **Idempotent producer:** setting `enable.idempotence=True` prevents duplicate messages on retry
- **Retention:** Kafka is not a database. Default retention is 7 days. For long-term storage, consume to a warehouse.
- **Offset management:** `auto_offset_reset='earliest'` vs `'latest'` — matters when a consumer group is brand new

**Issue we hit:** Background pytest killed the API because of Unix process group signal propagation (SIGINT from Ctrl+C goes to the entire foreground process group).

**Interview question:** *"Why would you use Kafka instead of writing directly to a database?"*

---

## Topic 3 — PySpark & Feature Engineering

**What we built:** `features/spark_streaming_features.py` reads raw transactions via JDBC, computes velocity features (transaction count per account per hour, average amount per merchant), writes to `mart.feature_store` via Feast → Redis.

**What to know:**
- **Spark execution model:** DAG of transformations, lazy evaluation, actions trigger execution
- **Partitioning:** partition by account_id for velocity aggregations — co-locates data for `groupBy`
- **JDBC JAR:** PySpark needs a JDBC driver JAR to connect to PostgreSQL. It's NOT bundled with Spark. In production, store on S3 and reference via `--jars`.
- **Feast feature store:** separates feature computation (offline) from feature serving (online, via Redis). Training/serving skew prevention.
- **Velocity features:** count of transactions in last 1h/24h/7d per account — critical fraud signal. A card doing 50 transactions in 1 hour is suspicious.

**Issue we hit:** `spark/jars/postgresql-42.7.3.jar` was gitignored — fresh clone broke Spark. Fix: manual download step documented in CLAUDE.md.

**Interview question:** *"How do you prevent training/serving skew in a real-time ML system?"*

**Your answer:** Feast. Features computed offline go to the offline store for training; same feature logic computed at inference time goes to Redis via the online store. Same feature definitions, same transformations — single source of truth.

---

## Topic 4 — Machine Learning Models

**What we built:**
- **XGBoost fraud detector** — `models/fraud_detector.py` — AUC 0.681, Optuna HPO, 12.5% fraud rate
- **Isolation Forest** — anomaly detection for novel patterns the supervised model hasn't seen
- **XGBoost credit risk scorer** — regression output (risk score 0–1)

**What to know:**
- **Why XGBoost over neural nets for fraud?** SHAP TreeExplainer requires tree models. Inference is ~3ms vs ~50ms for a small neural net. FCRA requires explainability. (ADR-002)
- **Class imbalance:** 12.5% fraud rate — not severe, but threshold tuning matters. Optimize for business cost: FN (missed fraud) costs more than FP (false alarm). Use precision-recall curve, not ROC, to set threshold.
- **AUC 0.681:** honest number — synthetic data is intentionally hard. In a real interview: "On synthetic data with intentional overlap, 0.681. On clean production data with real fraud signals, we'd expect 0.85+"
- **Optuna HPO:** Bayesian optimisation of `n_estimators`, `max_depth`, `learning_rate`, `subsample`. Faster than grid search.
- **Isolation Forest:** no labels needed — great for anomaly types the fraud labels don't capture (account takeover, new attack patterns)

**Issue we hit:** Fairness gate is non-deterministic at threshold boundaries — `account_age_group` predictive parity was 0.055, barely above the 0.05 gate. XGBoost multi-threading causes tiny float differences that flip the gate. Fix: use hysteresis (fail only if >threshold for 3 consecutive runs).

**Interview question:** *"How do you handle class imbalance in fraud detection?"*

---

## Topic 5 — SHAP & Explainability

**What we built:** `models/fraud_detector.py` computes SHAP values for every scored transaction, stores top-5 features with direction (positive = increases fraud probability).

**What to know:**
- **TreeExplainer vs KernelExplainer:** TreeExplainer is O(n log n) — use for tree models. KernelExplainer is model-agnostic but slow (O(n²) samples).
- **SHAP values:** contribution of each feature to the model's output relative to the base value (expected output). Positive = pushed score up, negative = pushed down.
- **What "top-5 features" means in a fraud context:** tells the investigator *why* the model fired. "high_velocity_1h (↑), unusual_merchant_category (↑), card_present=False (↑)"
- **Legal requirement:** FCRA §615(a) — adverse action notice. If you deny credit or flag a transaction, you must give the customer the top reasons. SHAP outputs are the technical mechanism for this.
- **SHAP for feature selection:** aggregate SHAP values across training set → feature importance that respects interactions (unlike raw feature importance which is biased toward high-cardinality features)

**Interview question:** *"How do you explain a fraud model decision to a compliance officer?"*

**Your answer:** SHAP top-5 features with direction, mapped to plain-English descriptions. "Your transaction was flagged primarily because the velocity in the last hour was 14× your average, and the merchant category (online electronics) is high-risk for this card profile."

---

## Topic 6 — Counterfactual Explanations (Dice-ML)

**What we built:** `models/counterfactual.py` — given a denied/flagged transaction, generate the minimum change that would flip the decision.

**What to know:**
- **Why counterfactuals?** SHAP tells you *why* the model scored the transaction high. Counterfactuals tell the customer *what to do differently*. "Your score would drop below the threshold if your transaction amount were $400 instead of $1,200."
- **FCRA §615(a) adverse action notice** — legally requires actionable reasons, not just "we denied you." Counterfactuals operationalize this.
- **Dice-ML:** generates diverse counterfactuals — multiple paths to flipping the decision, not just the nearest one. Diversity avoids gaming (if only one counterfactual, users game it).
- **Actionability constraint:** some features are immutable (age, account creation date). Dice-ML supports feature constraints to exclude these.
- **What-if page:** the React What-If page calls `/explain/what-if` — lets a compliance analyst adjust feature values and see the score change live.

**Interview question:** *"What's the difference between SHAP and counterfactual explanations? When do you need each?"*

---

## Topic 7 — Model Governance (Fairlearn + PSI)

**What we built:**
- **Fairlearn fairness gate** — `governance/fairness_gate.py` — checks 3 protected groups (gender, age, race proxy), 3 metrics (demographic parity, equal opportunity, predictive parity), gate passes if all deltas < 0.05
- **PSI drift monitor** — `governance/drift_monitor.py` — Population Stability Index per feature, logs to `audit.drift_reports`

**What to know:**
- **PSI thresholds:** 0–0.10 = stable, 0.10–0.20 = monitor, >0.20 = retrain. PSI is a leading indicator — feature distributions shift before model accuracy degrades.
- **PSI vs AUC degradation:** AUC needs labels (you need to wait for fraud outcomes, which can take days). PSI is computed on input features only — you see drift immediately.
- **Fairlearn metrics:**
  - *Demographic parity:* approval rate should be equal across groups. Ignores actual fraud rate differences.
  - *Equal opportunity:* TPR should be equal across groups. Allows different approval rates if fraud rates differ.
  - *Predictive parity:* PPV (precision) equal across groups. Minority groups shouldn't face higher false-positive rates.
- **Which metric matters?** In credit/fraud: equal opportunity + predictive parity are most defensible under disparate impact law. Demographic parity is too strict when base rates differ.
- **OCC SR 11-7:** Fed/OCC guidance requiring banks to have a model risk management framework — model inventory, validation, governance. The model risk card (`docs/model_risk_card_v1.0.0.md`) is the artifact.

**Issue we hit:** Fairness gate returned different results on consecutive runs due to floating-point non-determinism at threshold boundary. Fix: test consistency (if gate passes, no failing metrics), not equality.

**Interview question:** *"How do you monitor a fraud model for fairness in production?"*

---

## Topic 8 — FastAPI & Real-Time Serving

**What we built:** 5 routers, 16 routes, WebSocket live feed for scored transactions. Models loaded at startup via FastAPI lifespan, not per request.

**What to know:**
- **Lifespan context manager:** load models at startup, close DB connections at shutdown. If you load per-request: first request p99 spikes to 2s instead of 5ms.
- **Async vs sync routes:** FastAPI runs sync routes in a thread pool. CPU-bound work (XGBoost inference) in an `async def` route will block the event loop. Use `run_in_executor` for CPU-bound work.
- **WebSocket for live feed:** the fraud score feed uses `ws://localhost:8000/ws/scores`. Each scored transaction is pushed to all connected clients. Backpressure is not handled — in production add a Redis Pub/Sub buffer.
- **Latency SLA (CREDIT-001 NFR-001):** p99 model inference < 100ms. XGBoost is ~3ms. The 100ms budget leaves room for SHAP (~30ms), DB write (~5ms), Kafka publish (~2ms).
- **Health probes:** `/health/live` — is the process up? `/health/ready` — is the model loaded AND DB reachable? K8s sends traffic only when ready probe passes.

**Issue we hit:** `X-Latency-MS` header vs `inference_latency_ms` body field had a 1000ms discrepancy — they measure different things. Header = full request time (model + DB + Kafka). Body field = XGBoost `.predict_proba()` only.

**Interview question:** *"What's the difference between a liveness probe and a readiness probe in Kubernetes?"*

---

## Topic 9 — LangChain ReAct Agent + RAG

**What we built:** `agent/react_agent.py` — LangChain ReAct agent with 5 tools (score transaction, explain transaction, get account history, run fairness check, run drift check). FAISS RAG over 8 regulatory documents (FCRA, PCI DSS, SR 11-7, Reg E, CFPB fair lending guidance).

**What to know:**
- **ReAct pattern:** Reasoning + Acting. The agent thinks step-by-step (Thought), picks a tool (Action), observes the result (Observation), loops until done.
- **FAISS RAG:** documents are chunked, embedded, stored in a FAISS index. At query time, top-k chunks retrieved by cosine similarity are injected into the LLM prompt.
- **MMR retrieval (from TalentLens):** Maximum Marginal Relevance — avoids returning 5 copies of the same chunk. Balances relevance with diversity.
- **Gemini Flash (free tier):** 15 RPM limit. In test suite: 5-second sleep fixture between agent calls keeps RPM under 12.
- **Rate limiting patterns:** token bucket (fixed capacity, refills at rate r), leaky bucket (smooths bursts), exponential backoff with jitter (retry after 2^n + random delay).
- **Why file-based MLflow, not server:** no server dependency in dev. For production: swap `MLFLOW_TRACKING_URI` to `http://mlflow-server:5001`. File-based MLflow is not multi-user safe.

**Issue we hit:** MLflow HTTP URI caused 60-second timeout when `/governance/fairness/run` fired and MLflow server wasn't running. Fix: file-based tracking. **Interview talking point:** observability should not block the critical path — MLflow logging should be async/fire-and-forget.

**Interview question:** *"How does a ReAct agent decide when to call a tool vs answer directly?"*

---

## Topic 10 — Kubernetes + KEDA Autoscaling

**What we built:** `infra/k8s/deployments.yaml` — K8s manifests for all services. `infra/k8s/keda-scaled-object.yaml` — KEDA ScaledObject watches `transactions.raw` consumer lag, scales fraud-scorer pods from 1→10.

**What to know:**
- **KEDA vs HPA:** HPA scales on CPU/memory (lagging indicators for event-driven workloads). KEDA scales on Kafka consumer lag — when lag grows, more pods are spawned. ADR-004.
- **Consumer lag as leading indicator:** if 10,000 messages are queued, you need more consumers *now*, not after CPU spikes. By the time CPU reflects the backlog, latency SLA is already broken.
- **ScaledObject fields:** `pollingInterval`, `cooldownPeriod`, `minReplicaCount`, `maxReplicaCount`, `lagThreshold`
- **minikube vs EKS:** portfolio demo runs on minikube. Never claim EKS deployment unless actually deployed there — Amazon interviewers ask about IAM roles, node groups, VPC setup.
- **Deployment honesty rule (from CLAUDE.md):** "Architected for AWS EKS; local demo runs on minikube."

**Interview question:** *"Why would you use KEDA over a standard Kubernetes HPA for a Kafka consumer?"*

---

## Topic 11 — Testing Strategy

**What we built:** 15 unit tests + 105 integration tests across 8 files. Every router has integration test coverage. Data quality validated at the DB layer.

**What to know:**
- **Unit vs integration tests for pipelines:** unit tests verify function logic in isolation. Integration tests verify system boundaries — the `account_id='unknown'` bug passed unit tests and only failed at the integration boundary.
- **Test pyramid:** unit (fast, many) → integration (medium) → end-to-end (slow, few). The 105 integration tests catch wiring bugs that unit tests miss.
- **Fixtures:** `conftest.py` shared fixtures (`client`, `db`, `scored_txn`) avoid duplicated setup. `autouse` fixtures (like the Gemini rate limit sleep) apply to every test in a class automatically.
- **Idempotent test data:** integration tests create known transactions with fixed IDs (`inttest-explain-001`) so assertions are deterministic. Tests clean up after themselves.
- **SLA test:** `test_score.py` measures actual p99 latency across 20 requests and asserts < 100ms. Not just a happy-path schema check.

**Issue we hit:** `account_id` was always `'unknown'` in `mart.risk_scores` — silent data quality bug only caught by the integration test that checked the DB row, not the API response.

**Interview question:** *"You have a unit test suite. Why do you still need integration tests?"*

---

## Topic 12 — Compliance & Regulations (AmEx / PayPal critical)

**What we built:** Counterfactuals (FCRA), model risk card (OCC SR 11-7), fairness gate (CFPB disparate impact), audit tables (PCI DSS logging).

**What to know — these will come up in AmEx/PayPal interviews:**

| Regulation | What it requires | How CreditPulse addresses it |
|---|---|---|
| **FCRA §615(a)** | Adverse action notice — if you deny credit, give the top reasons in plain language | Dice-ML counterfactuals + SHAP top-5 |
| **PCI DSS** | Cardholder data must be encrypted, access logged, CVV/full PAN never stored | Audit tables log decision metadata, not raw card data |
| **OCC SR 11-7** | Banks must have a model risk management framework — inventory, validation, ongoing monitoring | `docs/model_risk_card_v1.0.0.md` with real AUC, fairness metrics, known gaps |
| **Regulation E** | Error resolution rights for electronic fund transfers within 10 business days | The explainability layer supports the dispute resolution workflow |
| **CFPB Fair Lending** | No disparate impact on protected classes | Fairlearn gate with 3 groups, 3 metrics, δ < 0.05 threshold |

**Interview question:** *"What compliance requirements drive model explainability in financial services?"*

**Your answer:** Lead with FCRA §615(a) — adverse action notice. That's the legal forcing function. Then SR 11-7 for model risk governance. SHAP is the technical mechanism for FCRA; the model risk card is the artifact for SR 11-7.

---

## Topic 13 — Data Quality & Defensive Design

**Issues that map directly to interview questions:**

**Aggregate queries on empty sets:**
- `COUNT(*)` always returns a row (value 0), but `AVG()`, `SUM()` on empty sets return NULL
- Your application must coerce NULL → 0. `COALESCE(AVG(score), 0.0)`
- `GET /risk/summary` returned `{}` when no rows in last 24h — broke tests that did `data["total_scored_today"]`

**Silent data bugs vs loud failures:**
- `account_id='unknown'` was silently wrong for weeks — the API returned 200, the row was written, but with garbage data. Only an integration test checking the actual DB row caught it.
- Design for loud failures: DB constraint `NOT NULL` on `account_id` would have caught this at insert time.

**Derived fields vs stored fields:**
- PSI `status` (`"STABLE"/"MONITOR"/"RETRAIN"`) is derived from `psi_score`. Don't store it — recompute on read. Threshold can change without a DB migration.
- Stored: `psi_score` (numeric, immutable). Derived: `status` (business interpretation, can change).

**API contract precision:**
- `features` in drift response was a dict keyed by feature name, not a list. The consumer assumed a list and iterated over dict keys. Document your API response shapes.

---

## Topic 14 — Infrastructure & DevOps

**Issues that map to interview questions:**

**12-factor app (configuration via env vars):**
- `GEMINI_API_KEY` not loaded because API was started before `.env` was created. Solution: always `source .env` before starting services. In production: AWS Secrets Manager, Vault, or K8s Secrets.
- Never hardcode ports, DB credentials, or API keys. Everything via environment variables.

**Port management:**
- Mac has 3 Postgres instances (Homebrew: 5432, Anaconda: 5433). Docker Postgres on 5435 to avoid collision.
- macOS AirPlay Receiver uses port 5000 — MLflow default. Use 5001.
- Always parameterize ports via env vars, never hardcode.

**Process management:**
- `nohup` detaches a process from the shell's process group — SIGINT from Ctrl+C doesn't reach it.
- Without `nohup`: killing pytest killed the API because both were in the same process group.
- In production: systemd, supervisor, or K8s owns the process lifecycle.

**Docker Desktop resource saver:**
- Docker pauses containers after inactivity. CreditPulse looked broken — actually Docker was asleep.
- IaC and K8s maintain desired state — they restart containers that aren't running.

---

## Study Priority by Company

### AmEx — AI/ML Platform
Focus: Model governance → SHAP → Fairlearn → OCC SR 11-7 → FCRA → PSI drift → FastAPI serving
Key talking point: "I built a fairness gate that checks 3 protected groups across 3 metrics before any model goes to production — and backed it with a model risk card following OCC SR 11-7."

### PayPal — MLE
Focus: Real-time inference latency (p99 SLA) → WebSocket → FCRA counterfactuals → PCI DSS → Kafka → Idempotency
Key talking point: "The fraud scorer hits p99 < 100ms including SHAP computation — XGBoost TreeExplainer is the key, not KernelExplainer."

### Amazon — Data Engineering (CDP)
Focus: Medallion architecture → Kafka → Spark → JDBC → KEDA → Integration testing → Data quality
Key talking point: "105 integration tests including an SLA latency test and a DB-layer data quality test that caught a silent `account_id='unknown'` bug."

### Tesla — Data Engineering (People Analytics)
Focus: Kafka pipelines → Feast feature store → PostgreSQL warehouse → PSI drift → Docker/K8s
Key talking point: "Feast separates offline feature computation from online serving — same feature definitions, no training/serving skew."

---

## Quick-Reference: 15 Issues → Interview Topics

| # | Issue | Interview topic |
|---|---|---|
| 1 | 3 Postgres instances on one Mac → port 5435 | Docker networking, port management, env var discipline |
| 2 | macOS AirPlay blocks port 5000 | Know your runtime environment |
| 3 | MLflow HTTP URI → 60s timeout blocking fairness endpoint | Observability must not block critical path; dependency isolation |
| 4 | `account_id='unknown'` in mart.risk_scores | Integration tests catch bugs unit tests miss; system boundary validation |
| 5 | GEMINI_API_KEY not loaded → agent chat error | 12-factor app; secrets management; process restart for new env vars |
| 6 | Port 8000 in use after restart | Process management; `lsof`; K8s owns process lifecycle |
| 7 | Background pytest killed the API | Unix process groups; SIGINT propagation; `nohup` |
| 8 | Gemini 15 RPM → all agent tests fail | Rate limiting (token bucket, leaky bucket, exponential backoff with jitter) |
| 9 | X-Latency-MS ≠ inference_latency_ms | Latency measurement granularity; p50/p95/p99 SLA definition |
| 10 | `/risk/summary` returns `{}` on empty DB | Defensive API design; SQL NULL aggregates; COALESCE |
| 11 | `features` is dict not list | API contract precision; document response shapes |
| 12 | `psi_score` in DB vs `status` in memory | Derived fields; don't persist what you can recompute |
| 13 | Fairness gate non-deterministic at threshold boundary | Threshold fragility; hysteresis; confidence intervals |
| 14 | Docker Desktop sleeps between sessions | IaC; K8s desired state; stateful infrastructure |
| 15 | Spark JDBC JAR not in repo | Binary dependency management; JAR on S3 for production Spark |

---

## What's Left / Known Gaps

- **Spark streaming** — end-to-end not tested (needs Kafka up + JDBC JAR). Unit-level feature logic tested.
- **K8s on EKS** — manifests written, tested on minikube only. Do not claim EKS deployment.
- **WebSocket load testing** — manual tested only. No load test for concurrent WS clients.
- **Fairness gate on test data** — `gate_passed=False` because `account_age_group` predictive parity is 0.055 > 0.05. Documented in model risk card as a known gap. This is honest.
- **AUC 0.681** — honest. Synthetic data is intentionally hard. Say this proactively in interviews — it shows you understand model evaluation, not just model building.
