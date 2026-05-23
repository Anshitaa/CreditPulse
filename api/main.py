"""
CreditPulse — FastAPI Application
Spec: CREDIT-001, CREDIT-002, CREDIT-003

15 routes across 5 routers:
  /health         — liveness + readiness probes
  /score          — real-time fraud scoring (< 100ms target)
  /explain        — SHAP + counterfactual explanations
  /risk           — risk score queries and account risk history
  /governance     — drift reports, fairness metrics, model registry
  /agent          — LangChain ReAct agent chat endpoint
  /ws/scores      — WebSocket: real-time scored transaction feed

Start:
    uvicorn api.main:app --reload --port 8000
"""

import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime

import structlog
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

from api.routers import agent, explain, governance, risk, score

logger = structlog.get_logger(__name__)

# --- Lifespan: load models once at startup ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("startup", service="creditpulse-api")
    try:
        from models.fraud_detector import load_model
        model, label_encoder, explainer = load_model()
        app.state.fraud_model = model
        app.state.label_encoder = label_encoder
        app.state.shap_explainer = explainer
        logger.info("model_loaded", model="fraud_detector")
    except Exception as e:
        logger.warning("model_load_failed", error=str(e), msg="API will start without pre-loaded model")
        app.state.fraud_model = None
        app.state.label_encoder = None
        app.state.shap_explainer = None

    from models.counterfactual import CounterfactualEngine
    app.state.cf_engine = CounterfactualEngine()

    yield

    logger.info("shutdown", service="creditpulse-api")


app = FastAPI(
    title="CreditPulse API",
    description=(
        "Real-time fraud detection and credit risk scoring API. "
        "Built spec-first with Kiro. "
        "Spec linkage: CREDIT-001 (fraud), CREDIT-002 (credit risk), CREDIT-003 (explainability)."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# --- Middleware ---

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://talentlens.local"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)


@app.middleware("http")
async def request_id_middleware(request, call_next):
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    t0 = time.perf_counter()
    response = await call_next(request)
    latency_ms = round((time.perf_counter() - t0) * 1000, 2)
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Latency-MS"] = str(latency_ms)
    if latency_ms > 100 and request.url.path == "/score":
        logger.warning("slow_score_request", latency_ms=latency_ms, path=request.url.path)
    return response


# --- Routers ---

app.include_router(score.router, prefix="/score", tags=["Fraud Scoring"])
app.include_router(explain.router, prefix="/explain", tags=["Explainability"])
app.include_router(risk.router, prefix="/risk", tags=["Risk Queries"])
app.include_router(governance.router, prefix="/governance", tags=["Model Governance"])
app.include_router(agent.router, prefix="/agent", tags=["AI Agent"])


# --- Health endpoints ---

@app.get("/health/live", tags=["Health"])
async def liveness():
    """Kubernetes liveness probe — is the process alive?"""
    return {"status": "alive", "timestamp": datetime.utcnow().isoformat()}


@app.get("/health/ready", tags=["Health"])
async def readiness():
    """Kubernetes readiness probe — is the API ready to serve traffic?"""
    checks = {
        "model": app.state.fraud_model is not None,
        "counterfactual_engine": app.state.cf_engine is not None,
    }
    db_ok = False
    try:
        import psycopg2
        conn = psycopg2.connect(os.environ.get("DATABASE_URL", "postgresql://creditpulse:creditpulse@localhost:5435/creditpulse"), connect_timeout=2)
        conn.close()
        db_ok = True
    except Exception:
        pass
    checks["database"] = db_ok
    ready = all(checks.values())
    return JSONResponse(
        status_code=200 if ready else 503,
        content={"status": "ready" if ready else "degraded", "checks": checks},
    )


# --- WebSocket: real-time scored transaction feed ---

class ConnectionManager:
    def __init__(self):
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        self.active.remove(ws)

    async def broadcast(self, message: dict):
        import json
        dead = []
        for ws in self.active:
            try:
                await ws.send_text(json.dumps(message, default=str))
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.active.remove(ws)


manager = ConnectionManager()
app.state.ws_manager = manager


@app.websocket("/ws/scores")
async def websocket_scores(websocket: WebSocket):
    """Real-time stream of scored transactions. Clients receive a JSON message
    for every transaction scored by the /score endpoint.
    Spec: CREDIT-001 FR-001 (publish result to scored topic)
    """
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive; scored transactions are pushed via broadcast()
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
