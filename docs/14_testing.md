# 14. Testing & Verification

## Test Architecture
- **Framework**: `pytest` with `httpx` / `Starlette TestClient`.
- **Coverage Areas**:
  - `tests/test_database.py`: Verifies database ORM creation, session handling, and model CRUD operations.
  - `tests/test_auth.py`: Tests password hashing, JWT token issue/verification, and unauthorized endpoint rejection.
  - `tests/test_dataset_pipeline.py`: Tests file parsing, syntax validation, text cleaner, and deduplication logic.
  - `tests/test_training_service.py`: Validates job launch, hyperparameter validation, and metrics streaming.
  - `tests/test_eval_and_registry.py`: Tests PPL, BLEU, ROUGE evaluation algorithms and model registration.
  - `tests/test_api_endpoints.py`: End-to-end REST API & WebSocket endpoint testing.
