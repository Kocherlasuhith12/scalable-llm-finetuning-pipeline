# 16. UI Workflow Guide

## Step-by-Step User Journey
1. **User Authentication**: User logs in or registers via the Auth Modal or default Admin account.
2. **Data Preparation**:
   - User navigates to **Datasets**, uploads a `.jsonl` or `.csv` dataset.
   - User clicks **Clean & Deduplicate**, monitoring row reductions and syntax validation results.
3. **Training Launch**:
   - User opens **Training Studio**, selects base model (e.g. `meta-llama/Llama-3.2-1B`, `mistralai/Mistral-7B-v0.1`, `Qwen/Qwen2.5-1.5B`), method (LoRA / QLoRA / SFT / DPO), learning rate, epochs, and LoRA rank.
   - User clicks **Launch Training**.
   - Live Chart.js graph streams step loss, learning rate decay, and GPU memory utilization.
4. **Evaluation & Benchmark**:
   - User opens **Evaluation Studio**, selects fine-tuned checkpoint against test dataset.
   - System calculates Perplexity, BLEU, and ROUGE scores, displaying a comparison table.
5. **Model Registration & Deployment**:
   - User registers candidate model in **Model Registry** under version tag `v1.0.0`.
   - User clicks **Merge Adapters** (if LoRA/QLoRA), then clicks **Deploy Endpoint**.
6. **Inference Playground**:
   - User tests deployed endpoint in **Playground**, sends prompts, adjusts temperature/top_p, inspects response latency, and copies pre-formatted `curl` code snippets.
7. **System Monitoring**:
   - User views cluster health, RAM/CPU/GPU utilization, cost breakdown, and live log console in **Monitoring**.
