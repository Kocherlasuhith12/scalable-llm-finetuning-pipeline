# 12. Authentication & Authorization

## Security Implementation
- **JWT Token Authentication**: Tokens signed using HMAC SHA-256 with configurable secret key (`SECRET_KEY`) and expiration time (`ACCESS_TOKEN_EXPIRE_MINUTES`).
- **Password Hashing**: PBKDF2 with SHA-256 / Bcrypt via `passlib` to ensure raw passwords are never stored.
- **API Key Management**: Generated secure API keys (`sk_live_...`) for external inference server client authorization (`/v1/chat/completions`).
- **Protected Endpoints**: FastAPI Dependency `get_current_user` enforces valid bearer token authorization across administrative routes.
