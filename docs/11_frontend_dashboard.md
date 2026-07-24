# 11. Frontend Dashboard UI

## Design System & UX
- **Theme**: Premium Dark Glassmorphism aesthetic featuring neon accents (cyan `#00f2fe`, purple `#4facfe`, emerald `#10b981`), smooth gradients, backdrop blur filters, and micro-animations.
- **Single Page Application (SPA)**: Built using clean, standard Vanilla JS with zero external bundler dependencies, served statically by FastAPI.
- **Real-Time Integration**: Direct WebSockets connection updating Chart.js loss curves, step progress bars, log outputs, and server health gauges without page refreshes.

## Dashboard Modules
1. **Overview**: Cluster metrics, quick stat cards, recent jobs table, active deployments.
2. **Dataset Manager**: File drag-and-drop uploader, data table inspector, cleaning & deduplication trigger.
3. **Training Studio**: Model and hyperparameter configuration form, job execution launcher, live loss/learning rate charts.
4. **Evaluation Benchmark**: Evaluator launcher, ROUGE/BLEU cards, side-by-side output viewer.
5. **Model Registry**: Registered versions, adapter merge action, deployment launcher.
6. **Inference Playground**: Interactive chat playground with streaming completion, OpenAI endpoint integration, parameter controls (temperature, top_p), auto-generated code snippets.
7. **System Monitoring**: Real-time CPU, RAM, GPU gauges, cost estimator, log stream console.
