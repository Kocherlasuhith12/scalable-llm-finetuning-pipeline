// HyperTune AI - Enterprise Platform Client Logic

let ws = null;
let lossChart = null;
const lossData = [];
const labelsData = [];

document.addEventListener("DOMContentLoaded", () => {
    initNavigation();
    initLossChart();
    initDragAndDrop();
    initCommandPalette();
    connectWebSocket();
    refreshAllData();

    // Telemetry polling interval
    setInterval(fetchTelemetry, 4000);
});

// Navigation Tab Switcher & Breadcrumbs
function initNavigation() {
    const navItems = document.querySelectorAll(".nav-item");
    navItems.forEach(item => {
        item.addEventListener("click", () => {
            const tab = item.getAttribute("data-tab");
            switchTab(tab);
        });
    });
}

function switchTab(tab) {
    document.querySelectorAll(".nav-item").forEach(n => n.classList.remove("active"));
    const activeNavItem = document.querySelector(`.nav-item[data-tab="${tab}"]`);
    if (activeNavItem) activeNavItem.classList.add("active");

    document.querySelectorAll(".page-view").forEach(v => v.classList.remove("active"));
    const targetView = document.getElementById(`view-${tab}`);
    if (targetView) targetView.classList.add("active");

    // Update Breadcrumbs
    const breadcrumbLabel = document.getElementById("breadcrumb-current-page");
    const labelsMap = {
        overview: "Overview",
        datasets: "Dataset Manager",
        training: "Training Studio",
        evaluation: "Evaluation Studio",
        registry: "Model Registry",
        playground: "Inference Playground",
        monitoring: "System Monitoring"
    };
    if (breadcrumbLabel) breadcrumbLabel.textContent = labelsMap[tab] || "Overview";
}

// Chart.js Loss Curve Initialization with Purple Accent
function initLossChart() {
    const ctx = document.getElementById("chart-training-loss");
    if (!ctx) return;

    const chartCtx = ctx.getContext("2d");
    const purpleGradient = chartCtx.createLinearGradient(0, 0, 0, 200);
    purpleGradient.addColorStop(0, 'rgba(139, 92, 246, 0.3)');
    purpleGradient.addColorStop(1, 'rgba(139, 92, 246, 0.0)');

    lossChart = new Chart(chartCtx, {
        type: 'line',
        data: {
            labels: labelsData,
            datasets: [{
                label: 'Training Loss',
                data: lossData,
                borderColor: '#8B5CF6',
                backgroundColor: purpleGradient,
                borderWidth: 2,
                fill: true,
                tension: 0.3,
                pointBackgroundColor: '#A78BFA',
                pointRadius: 3
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: { grid: { color: 'rgba(255, 255, 255, 0.05)' }, ticks: { color: '#64748B' } },
                y: { grid: { color: 'rgba(255, 255, 255, 0.05)' }, ticks: { color: '#64748B' } }
            },
            plugins: {
                legend: { labels: { color: '#F8FAFC', font: { family: 'Inter' } } }
            }
        }
    });
}

// WebSockets Telemetry Stream
function connectWebSocket() {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${protocol}//${window.location.host}/ws/telemetry`;

    ws = new WebSocket(wsUrl);

    ws.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            if (data.type === "training_progress") {
                updateTrainingProgressUI(data);
            } else if (data.type === "training_completed") {
                appendLog(`[TRAINING] Job ${data.job_id} completed successfully. Registered as ${data.model_name}`, 'success');
                refreshAllData();
            }
        } catch (e) {
            console.error("WS parse error:", e);
        }
    };

    ws.onclose = () => {
        setTimeout(connectWebSocket, 3000);
    };
}

function updateTrainingProgressUI(data) {
    document.getElementById("training-active-tag").textContent = `Job #${data.job_id} Running`;
    document.getElementById("text-training-step").textContent = `Step: ${data.step} / ${data.total_steps}`;
    document.getElementById("text-training-loss").textContent = `Loss: ${data.loss}`;
    document.getElementById("bar-training-progress").style.width = `${data.progress_pct}%`;

    labelsData.push(`Step ${data.step}`);
    lossData.push(data.loss);
    if (labelsData.length > 30) {
        labelsData.shift();
        lossData.shift();
    }
    if (lossChart) lossChart.update();

    appendLog(`[STEP ${data.step}/${data.total_steps}] Loss: ${data.loss} | LR: ${data.learning_rate} | VRAM: ${data.gpu_memory_mb} MB`, 'info');
}

// Data Fetchers
async function refreshAllData() {
    await Promise.all([
        fetchTelemetry(),
        fetchDatasets(),
        fetchTrainingJobs(),
        fetchModelRegistry(),
        fetchDeployments()
    ]);
}

async function fetchTelemetry() {
    try {
        const res = await fetch("/api/v1/monitoring/metrics");
        const data = await res.json();

        document.getElementById("gauge-cpu-bar").style.width = `${data.cpu_percent}%`;
        document.getElementById("gauge-cpu-text").textContent = `${data.cpu_percent}%`;

        document.getElementById("gauge-ram-bar").style.width = `${data.ram_percent}%`;
        document.getElementById("gauge-ram-text").textContent = `${data.ram_percent}%`;

        const vramPct = Math.min(100, Math.round((data.vram_used_mb / data.vram_total_mb) * 100));
        document.getElementById("gauge-vram-bar").style.width = `${vramPct}%`;
        document.getElementById("gauge-vram-text").textContent = `${(data.vram_used_mb / 1024).toFixed(1)} GB / ${(data.vram_total_mb / 1024).toFixed(1)} GB`;

        document.getElementById("kpi-active-jobs").textContent = data.active_training_jobs;
        document.getElementById("sidebar-active-jobs-badge").textContent = data.active_training_jobs;
    } catch (e) {
        console.error("Telemetry error:", e);
    }
}

async function fetchDatasets() {
    try {
        const res = await fetch("/api/v1/datasets");
        const datasets = await res.json();

        document.getElementById("kpi-datasets-count").textContent = datasets.length;
        document.getElementById("sidebar-dataset-badge").textContent = datasets.length;

        const tableBody = document.querySelector("#table-datasets tbody");
        const trainDatasetSelect = document.getElementById("select-train-dataset");
        trainDatasetSelect.innerHTML = "";

        if (datasets.length === 0) {
            tableBody.innerHTML = `<tr><td colspan="7" class="text-muted text-center">No workspace datasets uploaded.</td></tr>`;
            return;
        }

        tableBody.innerHTML = datasets.map(d => {
            trainDatasetSelect.innerHTML += `<option value="${d.id}">${d.name} (${d.sample_count} samples)</option>`;
            return `
                <tr>
                    <td class="text-mono">#${d.id}</td>
                    <td><strong>${d.name}</strong></td>
                    <td><span class="status-tag processing">${d.file_type}</span></td>
                    <td>${d.sample_count}</td>
                    <td>${(d.size_bytes / 1024).toFixed(1)} KB</td>
                    <td><span class="status-tag ${d.status === 'preprocessed' ? 'success' : 'processing'}">${d.status}</span></td>
                    <td>
                        <button class="btn btn-secondary btn-sm" onclick="triggerPreprocess(${d.id})">Clean & Dedupe</button>
                    </td>
                </tr>
            `;
        }).join("");
    } catch (e) {
        console.error("Datasets fetch error:", e);
    }
}

async function fetchTrainingJobs() {
    try {
        const res = await fetch("/api/v1/training/jobs");
        const jobs = await res.json();

        const tableBody = document.querySelector("#table-recent-jobs tbody");
        if (jobs.length === 0) {
            tableBody.innerHTML = `<tr><td colspan="3" class="text-muted text-center">No recent training workloads.</td></tr>`;
            return;
        }

        tableBody.innerHTML = jobs.slice(0, 5).map(j => `
            <tr>
                <td><strong>${j.name}</strong></td>
                <td><span class="status-tag processing">${j.method.toUpperCase()}</span></td>
                <td><span class="status-tag ${j.status === 'completed' ? 'success' : (j.status === 'running' ? 'processing' : 'warning')}">${j.status}</span></td>
            </tr>
        `).join("");
    } catch (e) {
        console.error("Training jobs fetch error:", e);
    }
}

async function fetchModelRegistry() {
    try {
        const res = await fetch("/api/v1/models");
        const models = await res.json();

        document.getElementById("kpi-models-count").textContent = models.length;
        document.getElementById("sidebar-models-badge").textContent = models.length;

        const tableBody = document.querySelector("#table-models tbody");
        const evalModelSelect = document.getElementById("select-eval-model");
        evalModelSelect.innerHTML = "";

        if (models.length === 0) {
            tableBody.innerHTML = `<tr><td colspan="7" class="text-muted text-center">No models registered yet.</td></tr>`;
            return;
        }

        tableBody.innerHTML = models.map(m => {
            evalModelSelect.innerHTML += `<option value="${m.id}">${m.name} (${m.version})</option>`;
            return `
                <tr>
                    <td class="text-mono">#${m.id}</td>
                    <td><strong>${m.name}</strong></td>
                    <td><span class="status-tag processing">${m.version}</span></td>
                    <td>${m.base_model}</td>
                    <td>${m.quantization}</td>
                    <td><span class="status-tag ${m.status === 'deployed' ? 'success' : 'processing'}">${m.status}</span></td>
                    <td>
                        <button class="btn btn-secondary btn-sm" onclick="handleMergeModel(${m.id})">Merge Adapter</button>
                        <button class="btn btn-primary btn-sm" onclick="handleDeployModel(${m.id})">Deploy Endpoint</button>
                    </td>
                </tr>
            `;
        }).join("");
    } catch (e) {
        console.error("Model registry fetch error:", e);
    }
}

async function fetchDeployments() {
    try {
        const res = await fetch("/api/v1/deployments");
        const deployments = await res.json();

        document.getElementById("kpi-deployments-count").textContent = deployments.length;

        const playgroundSelect = document.getElementById("select-playground-endpoint");
        playgroundSelect.innerHTML = `<option value="meta-llama/Llama-3.2-1B">meta-llama/Llama-3.2-1B (Default)</option>`;

        deployments.forEach(d => {
            playgroundSelect.innerHTML += `<option value="${d.name}">${d.name} (${d.endpoint_url})</option>`;
        });
    } catch (e) {
        console.error("Deployments fetch error:", e);
    }
}

// User Actions
function quickLaunchJob() {
    switchTab('training');
}

async function handleLaunchTrainingForm(event) {
    event.preventDefault();
    const name = document.getElementById("input-job-name").value;
    const base_model = document.getElementById("select-base-model").value;
    const dataset_id = parseInt(document.getElementById("select-train-dataset").value);
    const method = document.getElementById("select-train-method").value;
    const learning_rate = parseFloat(document.getElementById("input-lr").value);
    const epochs = parseInt(document.getElementById("input-epochs").value);
    const lora_r = parseInt(document.getElementById("input-lora-r").value);
    const lora_alpha = parseInt(document.getElementById("input-lora-alpha").value);

    try {
        const res = await fetch("/api/v1/training/launch", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                name, base_model, dataset_id, method, learning_rate, epochs, batch_size: 4, lora_r, lora_alpha
            })
        });

        if (res.ok) {
            const job = await res.json();
            appendLog(`[TRAINING] Launched fine-tuning workload #${job.id}: ${name}`, 'success');
            labelsData.length = 0;
            lossData.length = 0;
            if (lossChart) lossChart.update();
            refreshAllData();
        }
    } catch (e) {
        alert("Training launch error: " + e.message);
    }
}

async function triggerPreprocess(datasetId) {
    try {
        appendLog(`[DATASET] Running text cleaning and MinHash LSH deduplication on dataset #${datasetId}...`, 'info');
        const res = await fetch(`/api/v1/datasets/${datasetId}/preprocess`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ min_length: 5, dedup_threshold: 0.9 })
        });
        if (res.ok) {
            const prepJob = await res.json();
            alert(`Preprocessing completed! Cleaned samples: ${prepJob.processed_count}. Deduplication ratio: ${(prepJob.deduplication_ratio * 100).toFixed(1)}%`);
            refreshAllData();
        }
    } catch (e) {
        alert("Preprocessing error: " + e.message);
    }
}

async function handleRunEvaluation() {
    const modelId = parseInt(document.getElementById("select-eval-model").value);
    if (!modelId) {
        alert("Please select a target registered model checkpoint.");
        return;
    }

    try {
        appendLog(`[EVALUATION] Calculating PPL, BLEU, and ROUGE metrics for model #${modelId}...`, 'info');
        const res = await fetch("/api/v1/evaluations/run", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ model_id: modelId, dataset_id: 1 })
        });

        if (res.ok) {
            const evalJob = await res.json();
            document.getElementById("val-eval-ppl").textContent = evalJob.metrics.perplexity;
            document.getElementById("val-eval-bleu").textContent = evalJob.metrics.bleu;
            document.getElementById("val-eval-rouge").textContent = evalJob.metrics.rouge_l;
            document.getElementById("val-eval-em").textContent = evalJob.metrics.exact_match;

            const container = document.getElementById("eval-samples-container");
            container.innerHTML = evalJob.sample_outputs.map(s => `
                <div class="card mt-4">
                    <p style="font-weight: 600; color: var(--text-primary);">Prompt: ${s.prompt}</p>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-top: 0.75rem;">
                        <div style="background: var(--bg-surface); padding: 0.85rem; border-radius: 8px; border: 1px solid var(--border-subtle);">
                            <span class="status-tag warning">Base Model</span>
                            <p style="font-size: 0.85rem; margin-top: 0.4rem; color: var(--text-secondary);">${s.base_model_response}</p>
                        </div>
                        <div style="background: rgba(139, 92, 246, 0.05); padding: 0.85rem; border-radius: 8px; border: 1px solid rgba(139, 92, 246, 0.3);">
                            <span class="status-tag success">Fine-Tuned Response</span>
                            <p style="font-size: 0.85rem; margin-top: 0.4rem; color: var(--text-primary);">${s.fine_tuned_response}</p>
                        </div>
                    </div>
                </div>
            `).join("");
        }
    } catch (e) {
        alert("Evaluation error: " + e.message);
    }
}

async function handleMergeModel(modelId) {
    try {
        appendLog(`[MODEL] Merging PEFT LoRA adapter into base model for checkpoint #${modelId}...`, 'info');
        const res = await fetch(`/api/v1/models/${modelId}/merge`, { method: "POST" });
        if (res.ok) {
            alert(`PEFT adapter merged successfully into standalone model!`);
            refreshAllData();
        }
    } catch (e) {
        alert("Merge error: " + e.message);
    }
}

async function handleDeployModel(modelId) {
    try {
        appendLog(`[DEPLOYMENT] Deploying model checkpoint #${modelId} as live endpoint...`, 'info');
        const res = await fetch("/api/v1/deployments", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ model_id: modelId })
        });

        if (res.ok) {
            const dep = await res.json();
            alert(`Endpoint deployed successfully! API Key: ${dep.api_key}`);
            refreshAllData();
        }
    } catch (e) {
        alert("Deploy error: " + e.message);
    }
}

// Interactive Chat Playground
async function sendPlaygroundMessage() {
    const inputEl = document.getElementById("input-chat-prompt");
    const prompt = inputEl.value.trim();
    if (!prompt) return;

    const chatBox = document.getElementById("chat-messages-list");
    chatBox.innerHTML += `
        <div class="chat-bubble user">
            <div class="chat-avatar">U</div>
            <div class="chat-content">${prompt}</div>
        </div>
    `;
    inputEl.value = "";
    chatBox.scrollTop = chatBox.scrollHeight;

    const selectedModel = document.getElementById("select-playground-endpoint").value;

    try {
        const res = await fetch("/v1/chat/completions", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                model: selectedModel,
                messages: [{ role: "user", content: prompt }]
            })
        });

        if (res.ok) {
            const data = await res.json();
            const reply = data.choices[0].message.content;
            chatBox.innerHTML += `
                <div class="chat-bubble assistant">
                    <div class="chat-avatar">AI</div>
                    <div class="chat-content">${reply}</div>
                </div>
            `;
            chatBox.scrollTop = chatBox.scrollHeight;
        }
    } catch (e) {
        chatBox.innerHTML += `
            <div class="chat-bubble assistant">
                <div class="chat-avatar">AI</div>
                <div class="chat-content text-muted">Error generating completion response.</div>
            </div>
        `;
    }
}

// Drag & Drop File Upload
function initDragAndDrop() {
    const dropzone = document.getElementById("dataset-dropzone");
    if (!dropzone) return;

    dropzone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropzone.style.borderColor = "var(--accent-purple)";
    });

    dropzone.addEventListener("dragleave", () => {
        dropzone.style.borderColor = "var(--border-subtle)";
    });

    dropzone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropzone.style.borderColor = "var(--border-subtle)";
        if (e.dataTransfer.files.length > 0) {
            uploadSelectedFile(e.dataTransfer.files[0]);
        }
    });
}

function handleFileUpload(event) {
    if (event.target.files.length > 0) {
        uploadSelectedFile(event.target.files[0]);
    }
}

async function uploadSelectedFile(file) {
    const formData = new FormData();
    formData.append("file", file);

    appendLog(`[DATASET] Uploading file ${file.name}...`, 'info');

    try {
        const res = await fetch("/api/v1/datasets/upload", {
            method: "POST",
            body: formData
        });

        if (res.ok) {
            const dataset = await res.json();
            alert(`Dataset ${dataset.name} uploaded successfully! (${dataset.sample_count} samples parsed)`);
            refreshAllData();
        }
    } catch (e) {
        alert("Upload error: " + e.message);
    }
}

// Command Palette (Raycast / Linear style)
function initCommandPalette() {
    document.addEventListener("keydown", (e) => {
        if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
            e.preventDefault();
            openCommandPalette();
        }
        if (e.key === "Escape") {
            closeCommandPalette();
        }
    });
}

function openCommandPalette() {
    const overlay = document.getElementById("cmd-k-overlay");
    if (overlay) {
        overlay.classList.add("active");
        document.getElementById("cmd-k-input").focus();
    }
}

function closeCommandPalette() {
    const overlay = document.getElementById("cmd-k-overlay");
    if (overlay) overlay.classList.remove("active");
}

function executeCommand(tab) {
    switchTab(tab);
    closeCommandPalette();
}

function filterCommandPalette(event) {
    const query = event.target.value.toLowerCase();
    const items = document.querySelectorAll("#cmd-k-results-list .cmd-k-item");
    items.forEach(item => {
        const text = item.textContent.toLowerCase();
        item.style.display = text.includes(query) ? "flex" : "none";
    });
}

function appendLog(msg, type = 'info') {
    const terminal = document.getElementById("log-terminal-output");
    if (terminal) {
        const timeStr = new Date().toLocaleTimeString();
        terminal.innerHTML += `<div class="log-entry ${type}">[${timeStr}] ${msg}</div>`;
        terminal.scrollTop = terminal.scrollHeight;
    }
}
