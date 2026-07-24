# 02. System Architecture

## Architecture Diagram

```
+-----------------------------------------------------------------------+
|                       Frontend Dashboard UI                           |
|       (HTML5 / CSS Glassmorphism / Vanilla JS / WebSockets / Chart.js) |
+-----------------------------------+-----------------------------------+
                                    | REST APIs / WebSockets
                                    v
+-----------------------------------------------------------------------+
|                          FastAPI Server                               |
|  +--------------------+  +--------------------+  +-----------------+  |
|  | Auth & Security    |  | REST Controllers   |  | OpenAI API      |  |
|  | (JWT / Password)   |  | (Datasets, Jobs)   |  | (/v1/chat)      |  |
|  +---------+----------+  +---------+----------+  +--------+--------+  |
+------------|-----------------------|----------------------|-----------+
             v                       v                      v
+-----------------------------------------------------------------------+
|                          Service Layer                                |
|  +--------------------+  +--------------------+  +-----------------+  |
|  | Dataset Service    |  | Training Engine    |  | Eval & Serving  |  |
|  +---------+----------+  +---------+----------+  +--------+--------+  |
+------------|-----------------------|----------------------|-----------+
             |                       |                      |
             v                       v                      v
+-----------------------+  +--------------------+  +--------------------+
| SQLite / Postgres DB  |  | Background Worker  |  | Model Registry     |
| (SQLAlchemy ORM)      |  | (Process / Thread) |  | (Artifact Storage) |
+-----------------------+  +--------------------+  +--------------------+
```

## System Subsystems
1. **API Gateway & Routing**: Handles CORS, OAuth2 JWT authentication, request routing, rate limiting, and standard error formats.
2. **Database Persistence**: Maintains relational mapping for all platform entities with ACID guarantees.
3. **Async Task Execution**: Executes data cleaning, model training, adapter merging, and evaluation without blocking the HTTP server.
4. **WebSocket Manager**: Broadcasts real-time training metrics (step, epoch, loss, learning rate, GPU memory) to connected dashboard clients.
5. **Inference Server Engine**: Provides OpenAI spec endpoints with streaming chunk support.
