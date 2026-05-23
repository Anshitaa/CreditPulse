# ML Engineering Standards — CreditPulse

## Code Quality
- All ML code must be type-annotated (Python 3.10+)
- Every public function needs a docstring with Args, Returns, and Raises
- No magic numbers — use named constants in `config.py`
- Use `structlog` for all logging (JSON format, machine-parseable)

## Model Development
- Every model training script MUST log to MLflow: params, metrics, artifacts
- Hyperparameter tuning must use Optuna (never manual grid search)
- All classifiers must export SHAP values alongside predictions
- Models must be versioned and registered in MLflow Model Registry before use in API

## Feature Engineering
- Features must be registered in Feast feature store before use in training
- Rolling window features: always use closed="left" (no data leakage)
- Categorical encodings must be saved as MLflow artifacts (not hardcoded)
- All features must have a `feature_description` in the Feast FeatureView

## Testing Requirements
- Unit tests for every feature transformation function
- Integration test for end-to-end: raw event → Kafka → features → model → API response
- Load test: /score endpoint must handle 1000 concurrent requests at p99 < 100ms
- Fairness test: demographic parity delta must be < 0.05 on holdout test set

## Security
- No PII (names, SSNs, card numbers) in logs — use transaction IDs only
- API keys and DB passwords must come from environment variables only
- Validate and sanitize all API inputs with Pydantic models

## Performance
- Online inference must complete in < 100ms (includes Feast feature retrieval)
- Batch scoring jobs must process 1M transactions in < 30 minutes
- Spark jobs must use broadcast joins for lookup tables < 100MB
