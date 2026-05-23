# Coding Standards — CreditPulse

## Python
- Python 3.11+, type hints everywhere
- `ruff` for linting, `black` for formatting (line length 100)
- `pytest` for all tests, `pytest-asyncio` for async tests
- Import order: stdlib → third-party → local (enforced by ruff)
- Prefer `pathlib.Path` over `os.path`

## FastAPI
- All route handlers must be async
- Request/response models must be Pydantic v2 (`model_config = ConfigDict(...)`)
- Errors must return structured JSON: `{"error": "...", "detail": "...", "request_id": "..."}`
- All routes must have `tags`, `summary`, and `response_model`
- Rate limiting: 100 req/min per IP on public endpoints, 10K req/min on internal

## React / TypeScript
- TypeScript strict mode enabled
- Components: functional only (no class components)
- State management: React Query for server state, Zustand for client state
- Styling: Tailwind CSS (no inline styles)
- Every component must have a corresponding `.test.tsx` file

## Git
- Commits follow Conventional Commits: `feat:`, `fix:`, `chore:`, `docs:`, `test:`
- Branch naming: `feat/<feature>`, `fix/<bug>`, `chore/<task>`
- Every PR must link to a Kiro spec

## Docker
- Multi-stage builds (build → runtime stage)
- Non-root user in all containers
- Health checks on all services
- Pin all image versions (no `latest` tags)

## Environment
- All secrets via environment variables (never hardcoded)
- `.env.example` must be kept up to date
- Production: secrets via AWS Secrets Manager (K8s: ExternalSecrets operator)
