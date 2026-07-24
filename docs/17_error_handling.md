# 17. Error Handling Strategy

## Fault Tolerance & Error Taxonomy
1. **HTTP Error Responses**: All API exceptions return a uniform JSON schema:
   ```json
   {
     "status": "error",
     "error_code": "INVALID_HYPERPARAMETERS",
     "message": "Learning rate must be positive.",
     "timestamp": "2026-07-25T00:00:00Z"
   }
   ```
2. **Database Integrity Protection**: All mutations run inside SQLAlchemy session transactions with automatic rollback on unhandled exceptions.
3. **Background Worker Resilience**: Uncaught exceptions during training or dataset processing update job state to `FAILED` and record the stack trace into `error_log` for diagnostic review.
4. **Graceful Hardware Fallback**: Automatic detection of PyTorch CUDA / MPS / CPU availability with automatic CPU/Mock fallback for non-GPU cloud deployments (e.g. standard Render web instances).
