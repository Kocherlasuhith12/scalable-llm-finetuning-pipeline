# 10. FastAPI Backend Architecture

## Backend Stack
- **Framework**: FastAPI (Asynchronous Python Web Framework).
- **ORM**: SQLAlchemy with SQLite/PostgreSQL support.
- **Security**: OAuth2 Bearer with JWT (JSON Web Tokens) & Passlib/Bcrypt password hashing.
- **WebSocket Manager**: Connection manager maintaining active client connections for live progress and telemetry.
- **Background Tasks**: FastAPI `BackgroundTasks` and process pool execution for non-blocking ML operations.

## Middleware & Exception Handlers
- **CORS Middleware**: Allows cross-origin requests for API access.
- **Global Error Handling**: Standardized JSON error response format (`status`, `message`, `error_code`, `timestamp`).
