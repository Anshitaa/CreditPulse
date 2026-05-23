# CreditPulse — Load Tests

## WebSocket Load Test

Tests concurrent WebSocket clients + REST /score throughput.

### Prerequisites
```bash
pip install locust websocket-client
# API must be running:
PYTHONPATH=$(pwd) uvicorn api.main:app --port 8000
```

### Run (headless — CI-friendly)
```bash
mkdir -p tests/load/results
locust -f tests/load/locustfile_ws.py \
       --headless -u 50 -r 5 -t 60s \
       --host http://127.0.0.1:8000 \
       --csv tests/load/results/ws_load_test
```

### Run (web UI)
```bash
locust -f tests/load/locustfile_ws.py --host http://127.0.0.1:8000
# Open http://localhost:8089 → set users=50, spawn rate=5 → Start
```

### SLA Targets (CREDIT-001 NFR)
| Metric | Target | How to check |
|---|---|---|
| POST /score p99 latency | < 100ms | Locust stats table |
| WS broadcast p95 latency | < 200ms | `ws/scores broadcast` row |
| 50 concurrent WS clients | No failures | Failure count = 0 |
| 100 idle WS connections | Server stable | `WebSocketOnlyUser` task |

### Reading Results
- `tests/load/results/ws_load_test_stats.csv` — per-endpoint stats
- `tests/load/results/ws_load_test_failures.csv` — any failures
- Look for: RPS, median/p95/p99 response time, failure %
