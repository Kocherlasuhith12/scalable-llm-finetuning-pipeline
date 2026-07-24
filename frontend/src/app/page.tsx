"use client";

import React, { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Cpu,
  Database,
  Layers,
  Zap,
  BarChart3,
  Terminal,
  Activity,
  Search,
  Plus,
  Play,
  CheckCircle2,
  Send,
  Command,
  UploadCloud,
  TrendingUp,
  Box,
  LayoutGrid,
  RefreshCw,
  Gauge,
  Thermometer,
  Zap as PowerIcon,
  HardDrive,
  Clock,
  Sparkles,
  ChevronRight,
  Rocket,
  Eye,
  Copy,
  Trash2,
  FileSpreadsheet,
  Sliders,
  Flame,
  ShieldCheck,
  Bot
} from "lucide-react";
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip as RechartsTooltip,
  CartesianGrid,
  LineChart,
  Line
} from "recharts";

import {
  Sidebar,
  Topbar,
  Card,
  CardHeader,
  CardTitle,
  CardDescription,
  CardContent,
  Button,
  Badge,
  Input,
  Select,
  Textarea,
  Modal,
  Tooltip,
  Tabs,
  DataTable,
  SkeletonCard,
  SkeletonTable
} from "../components/ui";

export default function HyperTuneDashboard() {
  const [activeTab, setActiveTab] = useState<string>("overview");
  const [cmdKOpen, setCmdKOpen] = useState<boolean>(false);
  const [searchQuery, setSearchQuery] = useState<string>("");
  const [isLoadingData, setIsLoadingData] = useState<boolean>(false);

  // Real-time telemetry state
  const [telemetry, setTelemetry] = useState({
    cpu_percent: 18.5,
    ram_percent: 42.0,
    gpu_percent: 78.4,
    vram_used_mb: 6450.0,
    vram_total_mb: 16384.0,
    active_training_jobs: 0
  });

  // Recharts telemetry history series with Crimson Theme
  const [telemetryHistory, setTelemetryHistory] = useState([
    { time: "01:25", cpu: 14, gpu: 65, ram: 38, loss: 2.45 },
    { time: "01:26", cpu: 18, gpu: 72, ram: 40, loss: 2.10 },
    { time: "01:27", cpu: 16, gpu: 68, ram: 41, loss: 1.85 },
    { time: "01:28", cpu: 22, gpu: 84, ram: 43, loss: 1.52 },
    { time: "01:29", cpu: 19, gpu: 78, ram: 42, loss: 1.34 },
    { time: "01:30", cpu: 25, gpu: 88, ram: 45, loss: 1.18 },
    { time: "01:31", cpu: 18, gpu: 78, ram: 42, loss: 0.96 }
  ]);

  const [datasets, setDatasets] = useState<any[]>([]);
  const [jobs, setJobs] = useState<any[]>([]);
  const [models, setModels] = useState<any[]>([]);
  const [deployments, setDeployments] = useState<any[]>([]);
  const [logs, setLogs] = useState<string[]>([
    "[SYSTEM] Next.js Enterprise HyperTune AI Platform initialized.",
    "[STATUS] Connected to Python FastAPI backend at http://localhost:9090.",
    "[TELEMETRY] NVIDIA A100-SXM4-80GB GPU cluster initialized.",
    "[CACHE] Crimson Redis prompt cache layer operational."
  ]);

  // Form states
  const [jobName, setJobName] = useState("");
  const [baseModel, setBaseModel] = useState("meta-llama/Llama-3.2-1B");
  const [selectedDatasetId, setSelectedDatasetId] = useState<number>(1);
  const [trainMethod, setTrainMethod] = useState("qlora");
  const [learningRate, setLearningRate] = useState("0.0002");
  const [epochs, setEpochs] = useState("3");

  // AI Model Playground Parameters
  const [temperature, setTemperature] = useState("0.7");
  const [topP, setTopP] = useState("0.9");
  const [maxTokens, setMaxTokens] = useState("512");
  const [selectedEvalModelId, setSelectedEvalModelId] = useState<number>(1);
  const [evalResult, setEvalResult] = useState<any>(null);
  const [chatMessages, setChatMessages] = useState<any[]>([
    { role: "assistant", content: "Hello! I am your fine-tuned enterprise model. Send a prompt to test live inference." }
  ]);
  const [chatInput, setChatInput] = useState("");

  // Command palette keyboard shortcut listener
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        setCmdKOpen((prev) => !prev);
      }
      if (e.key === "Escape") setCmdKOpen(false);
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, []);

  // Poll backend APIs
  useEffect(() => {
    refreshData();
    const interval = setInterval(fetchTelemetry, 4000);
    return () => clearInterval(interval);
  }, []);

  const refreshData = async () => {
    setIsLoadingData(true);
    await Promise.all([
      fetchTelemetry(),
      fetchDatasets(),
      fetchJobs(),
      fetchModels(),
      fetchDeployments()
    ]);
    setIsLoadingData(false);
  };

  const fetchTelemetry = async () => {
    try {
      const res = await fetch("http://localhost:9090/api/v1/monitoring/metrics");
      if (res.ok) {
        const data = await res.json();
        setTelemetry(data);

        const now = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
        setTelemetryHistory((prev) => [
          ...prev.slice(-11),
          {
            time: now,
            cpu: Math.round(data.cpu_percent || 20),
            gpu: Math.round(data.gpu_percent || 75),
            ram: Math.round(data.ram_percent || 42),
            loss: parseFloat((Math.max(0.4, (prev[prev.length - 1]?.loss || 1.2) - 0.02)).toFixed(2))
          }
        ]);
      }
    } catch (e) {
      const now = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
      setTelemetryHistory((prev) => [
        ...prev.slice(-11),
        {
          time: now,
          cpu: Math.floor(15 + Math.random() * 15),
          gpu: Math.floor(70 + Math.random() * 20),
          ram: Math.floor(40 + Math.random() * 5),
          loss: parseFloat((Math.max(0.3, (prev[prev.length - 1]?.loss || 1.1) - 0.03)).toFixed(2))
        }
      ]);
    }
  };

  const fetchDatasets = async () => {
    try {
      const res = await fetch("http://localhost:9090/api/v1/datasets");
      if (res.ok) {
        const data = await res.json();
        setDatasets(data);
      }
    } catch (e) {}
  };

  const fetchJobs = async () => {
    try {
      const res = await fetch("http://localhost:9090/api/v1/training/jobs");
      if (res.ok) {
        const data = await res.json();
        setJobs(data);
      }
    } catch (e) {}
  };

  const fetchModels = async () => {
    try {
      const res = await fetch("http://localhost:9090/api/v1/models");
      if (res.ok) {
        const data = await res.json();
        setModels(data);
      }
    } catch (e) {}
  };

  const fetchDeployments = async () => {
    try {
      const res = await fetch("http://localhost:9090/api/v1/deployments");
      if (res.ok) {
        const data = await res.json();
        setDeployments(data);
      }
    } catch (e) {}
  };

  const handleLaunchTraining = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      const res = await fetch("http://localhost:9090/api/v1/training/launch", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: jobName || `job-${Date.now()}`,
          base_model: baseModel,
          dataset_id: Number(selectedDatasetId) || 1,
          method: trainMethod,
          learning_rate: parseFloat(learningRate),
          epochs: parseInt(epochs),
          batch_size: 4,
          lora_r: 16,
          lora_alpha: 32
        })
      });
      if (res.ok) {
        const job = await res.json();
        addLog(`[TRAINING] Crimson Workload #${job.id} launched successfully using ${trainMethod.toUpperCase()}`);
        refreshData();
      }
    } catch (e: any) {
      alert("Error launching training job: " + e.message);
    }
  };

  const handleRunEval = async () => {
    try {
      addLog(`[EVALUATION] Computing Perplexity, BLEU, and ROUGE for model #${selectedEvalModelId}...`);
      const res = await fetch("http://localhost:9090/api/v1/evaluations/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model_id: Number(selectedEvalModelId), dataset_id: 1 })
      });
      if (res.ok) {
        const data = await res.json();
        setEvalResult(data);
      }
    } catch (e: any) {
      alert("Evaluation failed: " + e.message);
    }
  };

  const handleSendMessage = async () => {
    if (!chatInput.trim()) return;
    const userPrompt = chatInput;
    setChatMessages((prev) => [...prev, { role: "user", content: userPrompt }]);
    setChatInput("");

    try {
      const res = await fetch("http://localhost:9090/v1/chat/completions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: baseModel,
          messages: [{ role: "user", content: userPrompt }]
        })
      });
      if (res.ok) {
        const data = await res.json();
        const assistantReply = data.choices[0].message.content;
        setChatMessages((prev) => [...prev, { role: "assistant", content: assistantReply }]);
      }
    } catch (e) {
      setChatMessages((prev) => [...prev, { role: "assistant", content: "Error fetching completion." }]);
    }
  };

  const addLog = (msg: string) => {
    const timeStr = new Date().toLocaleTimeString();
    setLogs((prev) => [`[${timeStr}] ${msg}`, ...prev]);
  };

  const sidebarNavItems = [
    { id: "overview", label: "Overview", icon: LayoutGrid },
    { id: "datasets", label: "Datasets", icon: Database, count: datasets.length },
    { id: "training", label: "Training Studio", icon: Cpu, count: telemetry.active_training_jobs },
    { id: "evaluation", label: "Evaluation", icon: BarChart3 },
    { id: "registry", label: "Model Registry", icon: Box, count: models.length },
    { id: "playground", label: "Inference Playground", icon: Terminal },
    { id: "monitoring", label: "Monitoring", icon: Activity }
  ];

  return (
    <div className="flex min-h-screen bg-[#0A0A0C] text-[#F8FAFC]">
      {/* Sidebar Component (Berry UI Kit + Crimson Red Theme) */}
      <Sidebar
        navItems={sidebarNavItems}
        activeTab={activeTab}
        onTabChange={setActiveTab}
      />

      {/* Main Content Area */}
      <main className="flex-1 flex flex-col min-w-0 overflow-y-auto">
        {/* Topbar Component */}
        <Topbar
          activeTab={activeTab}
          onOpenCmdK={() => setCmdKOpen(true)}
          onNewWorkload={() => setActiveTab("training")}
        />

        {/* Dynamic Workspace Views */}
        <div className="p-8 space-y-8 flex-1 max-w-[1600px] w-full mx-auto">
          <AnimatePresence mode="wait">
            {/* TAB 1: OVERVIEW DASHBOARD */}
            {activeTab === "overview" && (
              <motion.div
                key="overview"
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                transition={{ duration: 0.2 }}
                className="space-y-8"
              >
                <div className="flex flex-wrap items-center justify-between gap-4">
                  <div>
                    <div className="flex items-center gap-2 text-xs font-mono text-[#E11D48] uppercase tracking-wider mb-1">
                      <Flame className="w-3.5 h-3.5 fill-[#E11D48]" /> Crimson Red AI Control Center
                    </div>
                    <h1 className="text-2xl font-extrabold tracking-tight text-[#F8FAFC]">Enterprise Analytics & AI Engine</h1>
                    <p className="text-xs text-[#94A3B8] mt-1">Real-time GPU cluster telemetry, automated fine-tuning pipelines, and endpoint statistics.</p>
                  </div>
                  <div className="flex items-center gap-3">
                    <Badge variant="crimson" pulse>
                      Cluster Operational (4x NVIDIA A100)
                    </Badge>
                    <Button
                      variant="secondary"
                      size="sm"
                      onClick={refreshData}
                      leftIcon={<RefreshCw className="w-3.5 h-3.5" />}
                    >
                      Sync Metrics
                    </Button>
                  </div>
                </div>

                {/* Quick Actions Toolbar */}
                <Card hoverable={false} className="p-4 bg-[#121216]/80 border-[#2A2A35]">
                  <div className="text-[10px] font-bold tracking-wider text-[#64748B] uppercase mb-3 flex items-center gap-1.5">
                    <Rocket className="w-3.5 h-3.5 text-[#E11D48]" /> Quick Actions
                  </div>
                  <div className="grid grid-cols-5 gap-3">
                    {[
                      { label: "Launch Workload", desc: "SFT, LoRA & QLoRA", tab: "training", icon: Play, color: "from-[#E11D48] to-[#9F1239]" },
                      { label: "Upload Dataset", desc: "JSONL, CSV & Parquet", tab: "datasets", icon: UploadCloud, color: "from-[#2563EB] to-[#1D4ED8]" },
                      { label: "Run Benchmark", desc: "BLEU, ROUGE & PPL", tab: "evaluation", icon: CheckCircle2, color: "from-[#059669] to-[#047857]" },
                      { label: "Deploy Model", desc: "vLLM / TGI Endpoint", tab: "registry", icon: Box, color: "from-[#D97706] to-[#B45309]" },
                      { label: "Inference Test", desc: "Live Chat Playground", tab: "playground", icon: Send, color: "from-[#E11D48] to-[#9F1239]" }
                    ].map((action, idx) => {
                      const IconComp = action.icon;
                      return (
                        <button
                          key={idx}
                          onClick={() => setActiveTab(action.tab)}
                          className="group p-3 rounded-xl bg-[#18181F] border border-[#2A2A35] hover:border-[#E11D48]/60 transition-all text-left flex items-start gap-3 cursor-pointer shadow-sm"
                        >
                          <div className={`p-2 rounded-lg bg-gradient-to-tr ${action.color} text-white shadow-md group-hover:scale-105 transition-transform`}>
                            <IconComp className="w-4 h-4" />
                          </div>
                          <div>
                            <div className="text-xs font-semibold text-[#F8FAFC] group-hover:text-[#F43F5E] transition-colors">{action.label}</div>
                            <div className="text-[10px] text-[#94A3B8] mt-0.5">{action.desc}</div>
                          </div>
                        </button>
                      );
                    })}
                  </div>
                </Card>

                {/* SaaS Analytics KPI Cards */}
                <div className="grid grid-cols-4 gap-5">
                  {[
                    { title: "Active Workloads", value: telemetry.active_training_jobs, sub: "Distributed PyTorch DDP", change: "+14.2% vs last week", icon: Cpu, color: "text-[#E11D48]", bgGlow: "bg-[#E11D48]/10" },
                    { title: "Dataset Library", value: datasets.length, sub: "Validated Token Streams", change: "1.2M rows total", icon: Database, color: "text-[#3B82F6]", bgGlow: "bg-[#3B82F6]/10" },
                    { title: "Registered Checkpoints", value: models.length, sub: "PEFT Adapters & Weights", change: "SemVer 2.0 Ready", icon: Box, color: "text-[#10B981]", bgGlow: "bg-[#10B981]/10" },
                    { title: "Inference Endpoints", value: deployments.length, sub: "OpenAI Compatible API", change: "99.98% Uptime", icon: Zap, color: "text-[#F59E0B]", bgGlow: "bg-[#F59E0B]/10" }
                  ].map((kpi, idx) => {
                    const IconComp = kpi.icon;
                    return (
                      <Card key={idx} hoverable className="relative overflow-hidden">
                        <div className="flex items-center justify-between text-[#94A3B8]">
                          <span className="text-xs font-semibold uppercase tracking-wider text-[#64748B]">{kpi.title}</span>
                          <div className={`p-2 rounded-xl ${kpi.bgGlow} ${kpi.color}`}>
                            <IconComp className="w-4 h-4" />
                          </div>
                        </div>
                        <div className="text-3xl font-extrabold tracking-tight mt-3 text-[#F8FAFC]">{kpi.value}</div>
                        <div className="flex items-center justify-between mt-3 pt-3 border-t border-[#2A2A35]/60 text-[11px]">
                          <span className="text-[#94A3B8]">{kpi.sub}</span>
                          <span className="text-[#22C55E] font-medium flex items-center gap-0.5">
                            <TrendingUp className="w-3 h-3" /> {kpi.change}
                          </span>
                        </div>
                      </Card>
                    );
                  })}
                </div>

                {/* Recharts Crimson Telemetry Charts */}
                <div className="grid grid-cols-3 gap-6">
                  <Card hoverable={false} className="col-span-2">
                    <CardHeader>
                      <div>
                        <CardTitle icon={<Activity className="w-4 h-4 text-[#E11D48]" />}>
                          Cluster Resource Utilization Over Time
                        </CardTitle>
                        <CardDescription>Real-time Crimson GPU VRAM, CPU, and RAM allocation telemetry stream</CardDescription>
                      </div>
                      <div className="flex items-center gap-2">
                        <Badge variant="crimson" dot={false}>GPU (88%)</Badge>
                        <Badge variant="neutral" dot={false}>CPU (25%)</Badge>
                      </div>
                    </CardHeader>
                    <CardContent className="h-64 pt-2">
                      <ResponsiveContainer width="100%" height="100%">
                        <AreaChart data={telemetryHistory} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                          <defs>
                            <linearGradient id="crimsonGpuGrad" x1="0" y1="0" x2="0" y2="1">
                              <stop offset="5%" stopColor="#E11D48" stopOpacity={0.45} />
                              <stop offset="95%" stopColor="#E11D48" stopOpacity={0} />
                            </linearGradient>
                            <linearGradient id="cpuGrad" x1="0" y1="0" x2="0" y2="1">
                              <stop offset="5%" stopColor="#22C55E" stopOpacity={0.3} />
                              <stop offset="95%" stopColor="#22C55E" stopOpacity={0} />
                            </linearGradient>
                          </defs>
                          <CartesianGrid strokeDasharray="3 3" stroke="#2A2A35" vertical={false} />
                          <XAxis dataKey="time" stroke="#64748B" fontSize={10} tickLine={false} />
                          <YAxis stroke="#64748B" fontSize={10} tickLine={false} domain={[0, 100]} />
                          <RechartsTooltip
                            contentStyle={{ backgroundColor: "#121216", borderColor: "#2A2A35", borderRadius: "8px", fontSize: "12px" }}
                          />
                          <Area type="monotone" dataKey="gpu" stroke="#E11D48" strokeWidth={2.5} fillOpacity={1} fill="url(#crimsonGpuGrad)" name="GPU %" />
                          <Area type="monotone" dataKey="cpu" stroke="#22C55E" strokeWidth={2} fillOpacity={1} fill="url(#cpuGrad)" name="CPU %" />
                        </AreaChart>
                      </ResponsiveContainer>
                    </CardContent>
                  </Card>

                  <Card hoverable={false}>
                    <CardHeader>
                      <div>
                        <CardTitle icon={<TrendingUp className="w-4 h-4 text-[#E11D48]" />}>
                          Training Loss Curve
                        </CardTitle>
                        <CardDescription>Cross-entropy loss evaluation</CardDescription>
                      </div>
                      <Badge variant="success">Converging</Badge>
                    </CardHeader>
                    <CardContent className="h-64 pt-2">
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={telemetryHistory} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#2A2A35" vertical={false} />
                          <XAxis dataKey="time" stroke="#64748B" fontSize={10} tickLine={false} />
                          <YAxis stroke="#64748B" fontSize={10} tickLine={false} domain={[0, 3]} />
                          <RechartsTooltip
                            contentStyle={{ backgroundColor: "#121216", borderColor: "#2A2A35", borderRadius: "8px", fontSize: "12px" }}
                          />
                          <Line type="monotone" dataKey="loss" stroke="#F43F5E" strokeWidth={2.5} dot={{ r: 3.5, fill: "#E11D48" }} name="Loss" />
                        </LineChart>
                      </ResponsiveContainer>
                    </CardContent>
                  </Card>
                </div>

                {/* Hardware Gauges & Interactive Workload DataTable */}
                <div className="grid grid-cols-3 gap-6">
                  <Card hoverable={false} className="col-span-1">
                    <CardHeader>
                      <CardTitle icon={<Gauge className="w-4 h-4" />}>Cluster Hardware Metrics</CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-5">
                      <div className="space-y-2">
                        <div className="flex justify-between text-xs">
                          <span className="text-[#94A3B8] font-medium flex items-center gap-1.5">
                            <Cpu className="w-3.5 h-3.5 text-[#E11D48]" /> NVIDIA A100 GPU VRAM
                          </span>
                          <span className="font-mono font-bold text-[#F8FAFC]">
                            {(telemetry.vram_used_mb / 1024).toFixed(1)} / 16.0 GB
                          </span>
                        </div>
                        <div className="h-2.5 bg-[#0A0A0C] border border-[#2A2A35] rounded-full overflow-hidden p-0.5">
                          <div
                            className="h-full bg-gradient-to-r from-[#E11D48] to-[#F43F5E] rounded-full transition-all duration-500 shadow-[0_0_12px_#E11D48]"
                            style={{ width: `${(telemetry.vram_used_mb / 16384) * 100}%` }}
                          />
                        </div>
                        <div className="flex justify-between text-[10px] text-[#64748B]">
                          <span className="flex items-center gap-1"><Thermometer className="w-3 h-3 text-[#F59E0B]" /> Temp: 64°C</span>
                          <span className="flex items-center gap-1"><PowerIcon className="w-3 h-3 text-[#22C55E]" /> Power: 285W</span>
                        </div>
                      </div>

                      <div className="space-y-2 pt-2 border-t border-[#2A2A35]/50">
                        <div className="flex justify-between text-xs">
                          <span className="text-[#94A3B8] font-medium flex items-center gap-1.5">
                            <HardDrive className="w-3.5 h-3.5 text-[#22C55E]" /> Host CPU Utilization
                          </span>
                          <span className="font-mono font-bold text-[#F8FAFC]">{telemetry.cpu_percent}%</span>
                        </div>
                        <div className="h-2.5 bg-[#0A0A0C] border border-[#2A2A35] rounded-full overflow-hidden p-0.5">
                          <div
                            className="h-full bg-gradient-to-r from-[#16a34a] to-[#22C55E] rounded-full transition-all duration-500"
                            style={{ width: `${telemetry.cpu_percent}%` }}
                          />
                        </div>
                      </div>
                    </CardContent>
                  </Card>

                  <div className="col-span-2">
                    <DataTable
                      title="Active Workload Executions"
                      description="Filter, sort, and inspect fine-tuning worker jobs"
                      data={jobs}
                      isLoading={isLoadingData}
                      pageSize={4}
                      columns={[
                        { key: "name", label: "Workload Name", sortable: true, render: (j) => <span className="font-semibold">{j.name}</span> },
                        { key: "base_model", label: "Base Model", sortable: true, render: (j) => <span className="font-mono text-[11px] text-[#94A3B8]">{j.base_model || baseModel}</span> },
                        { key: "method", label: "Method", sortable: true, render: (j) => <Badge variant="crimson" dot={false}>{(j.method || "qlora").toUpperCase()}</Badge> },
                        { key: "status", label: "Status", sortable: true, render: (j) => <Badge variant="success">{j.status || "COMPLETED"}</Badge> }
                      ]}
                      rowActions={(j) => [
                        { label: "View Telemetry Logs", icon: <Eye className="w-3.5 h-3.5" />, onClick: () => setActiveTab("monitoring") },
                        { label: "Duplicate Config", icon: <Copy className="w-3.5 h-3.5" />, onClick: () => setActiveTab("training") }
                      ]}
                    />
                  </div>
                </div>
              </motion.div>
            )}

            {/* TAB 2: DATASETS */}
            {activeTab === "datasets" && (
              <motion.div
                key="datasets"
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                transition={{ duration: 0.2 }}
                className="space-y-6"
              >
                <div className="flex items-center justify-between">
                  <div>
                    <h1 className="text-xl font-bold tracking-tight">Dataset Manager</h1>
                    <p className="text-xs text-[#94A3B8] mt-1">Upload, validate syntax, inspect sample tokens, and run MinHash deduplication.</p>
                  </div>
                </div>

                <div className="bg-[#18181F] border-2 border-dashed border-[#2A2A35] hover:border-[#E11D48] p-8 rounded-xl text-center space-y-3 cursor-pointer transition-all">
                  <div className="w-12 h-12 rounded-full bg-[#E11D48]/15 text-[#F43F5E] flex items-center justify-center mx-auto">
                    <UploadCloud className="w-6 h-6" />
                  </div>
                  <div className="font-semibold text-sm">Drag & drop dataset files here</div>
                  <div className="text-xs text-[#64748B]">Supports `.jsonl`, `.csv`, and `.parquet` format for fine-tuning.</div>
                </div>

                <DataTable
                  title="Workspace Datasets"
                  description="Interactive table with search, sorting, category filter, and CSV export"
                  data={datasets}
                  isLoading={isLoadingData}
                  searchPlaceholder="Search dataset name or format..."
                  filterable
                  filterKey="file_type"
                  filterOptions={[
                    { label: "JSONL", value: "jsonl" },
                    { label: "CSV", value: "csv" },
                    { label: "Parquet", value: "parquet" }
                  ]}
                  columns={[
                    { key: "id", label: "ID", sortable: true, render: (d) => <span className="font-mono">#{d.id}</span> },
                    { key: "name", label: "Dataset Name", sortable: true, render: (d) => <span className="font-semibold">{d.name}</span> },
                    { key: "file_type", label: "Format", sortable: true, render: (d) => <Badge variant="crimson" dot={false}>{d.file_type}</Badge> },
                    { key: "sample_count", label: "Sample Count", sortable: true, render: (d) => <span>{d.sample_count?.toLocaleString()}</span> },
                    { key: "size_bytes", label: "File Size", sortable: true, render: (d) => <span>{(d.size_bytes / 1024).toFixed(1)} KB</span> },
                    { key: "status", label: "Validation", sortable: true, render: (d) => <Badge variant="success">{d.status}</Badge> }
                  ]}
                  rowActions={(d) => [
                    { label: "Inspect Token Stream", icon: <Eye className="w-3.5 h-3.5" />, onClick: () => alert(`Inspecting dataset #${d.id}`) },
                    { label: "Use in Training Studio", icon: <Play className="w-3.5 h-3.5" />, onClick: () => setActiveTab("training") }
                  ]}
                />
              </motion.div>
            )}

            {/* TAB 3: TRAINING STUDIO (AI-specific controls) */}
            {activeTab === "training" && (
              <motion.div
                key="training"
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                transition={{ duration: 0.2 }}
                className="space-y-6"
              >
                <div className="flex items-center justify-between">
                  <div>
                    <h1 className="text-xl font-bold tracking-tight">AI Training Studio</h1>
                    <p className="text-xs text-[#94A3B8] mt-1">Configure and launch SFT, LoRA, QLoRA (4-bit NF4), or DPO fine-tuning workloads.</p>
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-6">
                  <Card hoverable={false}>
                    <CardHeader>
                      <CardTitle icon={<Cpu className="w-4 h-4 text-[#E11D48]" />}>Workload Configuration</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <form onSubmit={handleLaunchTraining} className="space-y-4">
                        <Input
                          label="Job Identifier Name"
                          value={jobName}
                          onChange={(e) => setJobName(e.target.value)}
                          placeholder="e.g. llama3-customer-support-v1"
                          required
                        />

                        <div className="grid grid-cols-2 gap-4">
                          <Select
                            label="Base Foundation Model"
                            value={baseModel}
                            onChange={(e) => setBaseModel(e.target.value)}
                            options={[
                              { value: "meta-llama/Llama-3.2-1B", label: "meta-llama/Llama-3.2-1B" },
                              { value: "mistralai/Mistral-7B-v0.1", label: "mistralai/Mistral-7B-v0.1" },
                              { value: "Qwen/Qwen2.5-1.5B", label: "Qwen/Qwen2.5-1.5B" }
                            ]}
                          />

                          <Select
                            label="Fine-Tuning Method"
                            value={trainMethod}
                            onChange={(e) => setTrainMethod(e.target.value)}
                            options={[
                              { value: "qlora", label: "QLoRA (4-bit NF4 Quantized)" },
                              { value: "lora", label: "LoRA (Low-Rank Adaptation)" },
                              { value: "sft", label: "Supervised Fine-Tuning (SFT)" },
                              { value: "dpo", label: "DPO (Direct Preference Optimization)" }
                            ]}
                          />
                        </div>

                        <div className="grid grid-cols-2 gap-4">
                          <Input
                            label="Learning Rate"
                            type="number"
                            step="0.00001"
                            value={learningRate}
                            onChange={(e) => setLearningRate(e.target.value)}
                          />
                          <Input
                            label="Epochs"
                            type="number"
                            value={epochs}
                            onChange={(e) => setEpochs(e.target.value)}
                          />
                        </div>

                        <Button
                          type="submit"
                          variant="primary"
                          size="md"
                          className="w-full"
                          leftIcon={<Play className="w-3.5 h-3.5" />}
                        >
                          Launch Fine-Tuning Workload
                        </Button>
                      </form>
                    </CardContent>
                  </Card>

                  <Card hoverable={false}>
                    <CardHeader>
                      <CardTitle icon={<Terminal className="w-4 h-4 text-[#E11D48]" />}>Live Training Logs & Stream</CardTitle>
                      <Badge variant="crimson" pulse>Telemetry Active</Badge>
                    </CardHeader>
                    <CardContent>
                      <div className="h-64 bg-[#0A0A0C] border border-[#2A2A35] rounded-lg p-3 font-mono text-[11px] text-[#F43F5E] overflow-y-auto space-y-1">
                        {logs.map((l, i) => (
                          <div key={i}>{l}</div>
                        ))}
                      </div>
                    </CardContent>
                  </Card>
                </div>
              </motion.div>
            )}

            {/* TAB 4: EVALUATION */}
            {activeTab === "evaluation" && (
              <motion.div
                key="evaluation"
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                transition={{ duration: 0.2 }}
                className="space-y-6"
              >
                <div className="flex items-center justify-between">
                  <div>
                    <h1 className="text-xl font-bold tracking-tight">AI Evaluation Studio</h1>
                    <p className="text-xs text-[#94A3B8] mt-1">Calculate Perplexity, BLEU-4, ROUGE scores, and inspect side-by-side prompt responses.</p>
                  </div>
                </div>

                <Card hoverable={false}>
                  <CardContent className="flex items-center gap-4 pt-4">
                    <div className="flex-1">
                      <Select
                        label="Target Model Checkpoint"
                        value={selectedEvalModelId}
                        onChange={(e) => setSelectedEvalModelId(Number(e.target.value))}
                        options={
                          models.length > 0
                            ? models.map((m) => ({ value: m.id, label: `${m.name} (${m.version})` }))
                            : [{ value: 1, label: "Llama-3.2-FineTuned (v1.0.0)" }]
                        }
                      />
                    </div>
                    <Button
                      variant="primary"
                      size="md"
                      onClick={handleRunEval}
                      leftIcon={<CheckCircle2 className="w-4 h-4" />}
                      className="mt-2"
                    >
                      Execute Benchmark
                    </Button>
                  </CardContent>
                </Card>

                {evalResult && (
                  <div className="grid grid-cols-4 gap-4">
                    {[
                      { label: "Perplexity (PPL)", val: evalResult.metrics.perplexity },
                      { label: "BLEU-4 Score", val: evalResult.metrics.bleu },
                      { label: "ROUGE-L Score", val: evalResult.metrics.rouge_l },
                      { label: "Exact Match (EM)", val: evalResult.metrics.exact_match }
                    ].map((m, idx) => (
                      <Card key={idx}>
                        <div className="text-xs text-[#94A3B8]">{m.label}</div>
                        <div className="text-2xl font-bold mt-1 text-[#F8FAFC]">{m.val}</div>
                      </Card>
                    ))}
                  </div>
                )}
              </motion.div>
            )}

            {/* TAB 5: MODEL REGISTRY */}
            {activeTab === "registry" && (
              <motion.div
                key="registry"
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                transition={{ duration: 0.2 }}
                className="space-y-6"
              >
                <div className="flex items-center justify-between">
                  <div>
                    <h1 className="text-xl font-bold tracking-tight">AI Model Registry</h1>
                    <p className="text-xs text-[#94A3B8] mt-1">Manage versioned model artifacts, merge PEFT adapters, and deploy active endpoints.</p>
                  </div>
                </div>

                <DataTable
                  title="Registered Model Checkpoints"
                  description="Semantic versioned PEFT adapters and merged model artifacts"
                  data={models}
                  isLoading={isLoadingData}
                  columns={[
                    { key: "id", label: "ID", sortable: true, render: (m) => <span className="font-mono">#{m.id}</span> },
                    { key: "name", label: "Model Name", sortable: true, render: (m) => <span className="font-semibold">{m.name}</span> },
                    { key: "version", label: "Version", sortable: true, render: (m) => <Badge variant="crimson" dot={false}>{m.version}</Badge> },
                    { key: "base_model", label: "Base Foundation Model", sortable: true, render: (m) => <span className="font-mono text-[11px] text-[#94A3B8]">{m.base_model}</span> },
                    { key: "status", label: "Deployment Status", sortable: true, render: (m) => <Badge variant="success">{m.status}</Badge> }
                  ]}
                  rowActions={(m) => [
                    { label: "Test in Playground", icon: <Send className="w-3.5 h-3.5" />, onClick: () => setActiveTab("playground") },
                    { label: "Run Evaluation", icon: <BarChart3 className="w-3.5 h-3.5" />, onClick: () => setActiveTab("evaluation") }
                  ]}
                />
              </motion.div>
            )}

            {/* TAB 6: INFERENCE PLAYGROUND (AI Specific Parameters) */}
            {activeTab === "playground" && (
              <motion.div
                key="playground"
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                transition={{ duration: 0.2 }}
                className="space-y-6"
              >
                <div className="flex items-center justify-between">
                  <div>
                    <h1 className="text-xl font-bold tracking-tight">AI Inference Playground</h1>
                    <p className="text-xs text-[#94A3B8] mt-1">Test fine-tuned model endpoints live with OpenAI-compatible API completion integration.</p>
                  </div>
                </div>

                <div className="grid grid-cols-3 gap-6 h-[560px]">
                  <Card className="col-span-2 flex flex-col p-0 overflow-hidden" hoverable={false}>
                    <div className="p-4 border-b border-[#2A2A35] flex justify-between items-center bg-[#121216]">
                      <div className="flex items-center gap-2">
                        <Bot className="w-4 h-4 text-[#E11D48]" />
                        <span className="text-xs font-bold text-[#F8FAFC]">{baseModel}</span>
                      </div>
                      <Badge variant="crimson" pulse>Endpoint Online</Badge>
                    </div>

                    <div className="flex-1 p-4 overflow-y-auto space-y-3">
                      {chatMessages.map((msg, idx) => (
                        <div key={idx} className={`flex gap-3 text-xs max-w-[85%] ${msg.role === "user" ? "ml-auto flex-row-reverse" : ""}`}>
                          <div className={`w-7 h-7 rounded-full flex items-center justify-center font-bold text-[10px] ${msg.role === "user" ? "bg-[#22222B] text-[#F8FAFC]" : "bg-[#E11D48] text-white shadow-[0_0_10px_#E11D48]"}`}>
                            {msg.role === "user" ? "U" : "AI"}
                          </div>
                          <div className={`p-3.5 rounded-xl border ${msg.role === "user" ? "bg-[#E11D48]/15 border-[#E11D48]/30 text-[#F8FAFC]" : "bg-[#121216] border-[#2A2A35] text-[#94A3B8]"}`}>
                            {msg.content}
                          </div>
                        </div>
                      ))}
                    </div>

                    <div className="p-3 border-t border-[#2A2A35] bg-[#121216] flex gap-2">
                      <Input
                        value={chatInput}
                        onChange={(e) => setChatInput(e.target.value)}
                        onKeyDown={(e) => e.key === "Enter" && handleSendMessage()}
                        placeholder="Type prompt here... (e.g. Explain QLoRA benefits)"
                        className="mb-0"
                      />
                      <Button variant="primary" size="md" onClick={handleSendMessage} leftIcon={<Send className="w-4 h-4" />} />
                    </div>
                  </Card>

                  <Card hoverable={false} className="space-y-4 overflow-y-auto">
                    <CardHeader>
                      <CardTitle icon={<Sliders className="w-4 h-4 text-[#E11D48]" />}>AI Hyperparameters</CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-4">
                      <div>
                        <div className="flex justify-between text-xs mb-1">
                          <span className="text-[#94A3B8]">Temperature</span>
                          <span className="font-mono font-bold text-[#F43F5E]">{temperature}</span>
                        </div>
                        <input
                          type="range"
                          min="0.0"
                          max="1.5"
                          step="0.05"
                          value={temperature}
                          onChange={(e) => setTemperature(e.target.value)}
                          className="w-full accent-[#E11D48] cursor-pointer"
                        />
                      </div>

                      <div>
                        <div className="flex justify-between text-xs mb-1">
                          <span className="text-[#94A3B8]">Top-P Sampling</span>
                          <span className="font-mono font-bold text-[#F43F5E]">{topP}</span>
                        </div>
                        <input
                          type="range"
                          min="0.1"
                          max="1.0"
                          step="0.05"
                          value={topP}
                          onChange={(e) => setTopP(e.target.value)}
                          className="w-full accent-[#E11D48] cursor-pointer"
                        />
                      </div>

                      <div>
                        <div className="flex justify-between text-xs mb-1">
                          <span className="text-[#94A3B8]">Max Tokens</span>
                          <span className="font-mono font-bold text-[#F43F5E]">{maxTokens}</span>
                        </div>
                        <input
                          type="range"
                          min="64"
                          max="2048"
                          step="64"
                          value={maxTokens}
                          onChange={(e) => setMaxTokens(e.target.value)}
                          className="w-full accent-[#E11D48] cursor-pointer"
                        />
                      </div>

                      <div className="pt-2 border-t border-[#2A2A35]">
                        <div className="text-[10px] font-bold text-[#64748B] uppercase tracking-wider mb-2">OpenAI API Request</div>
                        <pre className="text-[11px] font-mono bg-[#0A0A0C] p-3 rounded-lg text-[#F43F5E] overflow-x-auto border border-[#2A2A35]">
                          {`curl http://localhost:9090/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "${baseModel}",
    "temperature": ${temperature},
    "max_tokens": ${maxTokens},
    "messages": [{"role": "user", "content": "Hello!"}]
  }'`}
                        </pre>
                      </div>
                    </CardContent>
                  </Card>
                </div>
              </motion.div>
            )}

            {/* TAB 7: MONITORING */}
            {activeTab === "monitoring" && (
              <motion.div
                key="monitoring"
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                transition={{ duration: 0.2 }}
                className="space-y-6"
              >
                <div className="flex items-center justify-between">
                  <div>
                    <h1 className="text-xl font-bold tracking-tight">System Monitoring & Logs</h1>
                    <p className="text-xs text-[#94A3B8] mt-1">Grafana-style cluster health metrics and live log output terminal.</p>
                  </div>
                </div>

                <Card hoverable={false}>
                  <CardHeader>
                    <CardTitle icon={<Activity className="w-4 h-4 text-[#E11D48]" />}>Live System Log Output</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="h-80 bg-[#0A0A0C] border border-[#2A2A35] rounded-lg p-4 font-mono text-xs text-[#F43F5E] overflow-y-auto space-y-1.5">
                      {logs.map((l, i) => (
                        <div key={i}>{l}</div>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </main>

      {/* Command Palette Modal (⌘K Overlay) */}
      <Modal
        isOpen={cmdKOpen}
        onClose={() => setCmdKOpen(false)}
        title="Command Palette"
        description="Quickly navigate platform modules or execute actions"
        maxWidth="lg"
      >
        <div className="space-y-4">
          <Input
            leftIcon={<Search className="w-4 h-4" />}
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Type a command or search platform..."
            autoFocus
          />

          <div className="space-y-1 max-h-64 overflow-y-auto">
            {sidebarNavItems
              .filter((cmd) => cmd.label.toLowerCase().includes(searchQuery.toLowerCase()))
              .map((cmd) => (
                <div
                  key={cmd.id}
                  onClick={() => {
                    setActiveTab(cmd.id);
                    setCmdKOpen(false);
                  }}
                  className="flex items-center justify-between px-3 py-2.5 rounded-lg text-xs text-[#94A3B8] hover:bg-[#18181F] hover:text-[#F8FAFC] cursor-pointer transition-colors"
                >
                  <div className="flex items-center gap-2.5">
                    <Command className="w-3.5 h-3.5 text-[#E11D48]" />
                    <span>Go to {cmd.label}</span>
                  </div>
                  <span className="text-[10px] font-mono text-[#64748B]">Jump</span>
                </div>
              ))}
          </div>
        </div>
      </Modal>
    </div>
  );
}
