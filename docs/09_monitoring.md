# 09. Monitoring & Analytics

## System Monitoring & Cost Tracking
- **Hardware Telemetry**: Continuous tracking of CPU utilization percentage, RAM memory allocation, GPU utilization, and VRAM memory usage.
- **Training Loss Monitoring**: Real-time step-by-step stream of training loss, evaluation loss, learning rate decay, and throughput (tokens/sec).
- **Cost Estimation Engine**: Calculates compute cost in USD based on hardware type and training runtime hours.
- **Log Streamer**: Filterable system and job execution logs exposed via REST `/api/v1/monitoring/logs` and WebSockets `/ws/telemetry`.
