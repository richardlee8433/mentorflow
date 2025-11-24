import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "./components/ui/tabs";
import { Button } from "./components/ui/button";
import {
  Card,
  CardHeader,
  CardTitle,
  CardDescription,
  CardContent,
} from "./components/ui/card";
import { Input } from "./components/ui/input";
import { Textarea } from "./components/ui/textarea";
import { Badge } from "./components/ui/badge";
import { Switch } from "./components/ui/switch";
import { cn } from "./lib/utils";

const API_BASE =
  import.meta.env.VITE_API_BASE || "http://localhost:8000";

type Role = "user" | "assistant";

interface ChatMessage {
  id: string;
  role: Role;
  content: string;
}

interface RagSource {
  text: string;
  score: number;
  metadata?: Record<string, any>;
}

interface ChatResponse {
  reply: string;
  tts_base64?: string | null;
  rag_used?: boolean;
  sources?: RagSource[];
}

interface RagDocument {
  doc_id: string;
  filename: string;
  num_chunks: number;
}

interface RagReport {
  rag_enabled: boolean;
  documents?: RagDocument[];
}

interface RegionMetric {
  region: string;
  user_count: number;
  total_requests: number;
}

function usePersistentUserId(key: string): string {
  const [id] = useState(() => {
    const existing = window.localStorage.getItem(key);
    if (existing) return existing;
    const generated = "user-" + Math.random().toString(36).slice(2, 10);
    window.localStorage.setItem(key, generated);
    return generated;
  });
  return id;
}

function detectRegion(): string {
  let region = "IE";
  try {
    const tz = Intl.DateTimeFormat().resolvedOptions().timeZone || "";
    if (tz.includes("Europe/") && !tz.includes("Dublin")) {
      region = "EU";
    } else {
      region = "IE";
    }
  } catch {
    region = "IE";
  }
  return region;
}

function App() {
  const userId = usePersistentUserId("mentorflow_user_id");
  const region = useMemo(() => detectRegion(), []);

  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);

  const [ttsEnabled, setTtsEnabled] = useState<boolean>(() => {
    const saved = window.localStorage.getItem("mentorflow_tts_enabled");
    if (saved === null) return true;
    try {
      return JSON.parse(saved);
    } catch {
      return true;
    }
  });

  const [activeTab, setActiveTab] = useState<"learner" | "admin">("learner");

  const [ragEnabled, setRagEnabled] = useState<boolean>(true);
  const [documents, setDocuments] = useState<RagDocument[]>([]);
  const [uploading, setUploading] = useState(false);
  const [scratch, setScratch] = useState("");
  const [metricsLoading, setMetricsLoading] = useState(false);
  const [regionStats, setRegionStats] = useState<RegionMetric[]>([]);

  useEffect(() => {
    window.localStorage.setItem(
      "mentorflow_tts_enabled",
      JSON.stringify(ttsEnabled),
    );
  }, [ttsEnabled]);

  async function callChatApi(message: string): Promise<ChatResponse> {
    const resp = await fetch(`${API_BASE}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        user_id: userId,
        message,
        region,
      }),
    });
    if (!resp.ok) {
      throw new Error(`HTTP ${resp.status}`);
    }
    return resp.json();
  }

  async function handleSend() {
    const text = input.trim();
    if (!text || loading) return;

    const userMsg: ChatMessage = {
      id: crypto.randomUUID(),
      role: "user",
      content: text,
    };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setLoading(true);

    try {
      const data = await callChatApi(text);
      let reply = data.reply || "";

      if (data.rag_used && data.sources && data.sources.length > 0) {
        const lines = data.sources.map((src, idx) => {
          const meta = src.metadata || {};
          const label =
            (meta.doc_id as string) ||
            (meta.filename as string) ||
            `Source #${idx + 1}`;
          const score =
            typeof src.score === "number" ? src.score.toFixed(2) : undefined;
          return score ? `- ${label} (score: ${score})` : `- ${label}`;
        });
        reply += `\n\n📚 Sources:\n${lines.join("\n")}`;
      }

      const aiMsg: ChatMessage = {
        id: crypto.randomUUID(),
        role: "assistant",
        content: reply,
      };
      setMessages((prev) => [...prev, aiMsg]);

      if (data.tts_base64 && ttsEnabled) {
        const audio = new Audio(`data:audio/mp3;base64,${data.tts_base64}`);
        audio.play().catch(() => {});
      }
    } catch (err) {
      console.error(err);
      const aiMsg: ChatMessage = {
        id: crypto.randomUUID(),
        role: "assistant",
        content: "⚠️ Error talking to backend. Please try again.",
      };
      setMessages((prev) => [...prev, aiMsg]);
    } finally {
      setLoading(false);
    }
  }

  function handleQuickCommand(cmd: string) {
    setInput(cmd);
  }

  function handleClear() {
    setMessages([]);
    setInput("");
  }

  async function loadRagReport() {
    try {
      const resp = await fetch(`${API_BASE}/report`);
      if (!resp.ok) return;
      const data: RagReport = await resp.json();
      setRagEnabled(data.rag_enabled);
      setDocuments(data.documents || []);
    } catch (err) {
      console.warn("Failed to load RAG report", err);
    }
  }

  async function loadRegionMetrics() {
    setMetricsLoading(true);
    try {
      const resp = await fetch(`${API_BASE}/metrics/regions`);
      if (!resp.ok) return;
      const data = await resp.json();
      setRegionStats(data.regions || []);
    } catch (err) {
      console.warn("Failed to load region metrics", err);
    } finally {
      setMetricsLoading(false);
    }
  }

  useEffect(() => {
    if (activeTab === "admin") {
      loadRagReport();
      loadRegionMetrics();
    }
  }, [activeTab]);

  async function handleToggleRag(next: boolean) {
    setRagEnabled(next);
    try {
      const resp = await fetch(`${API_BASE}/admin/toggle-rag`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ enabled: next }),
      });
      if (!resp.ok) {
        throw new Error(`HTTP ${resp.status}`);
      }
    } catch (err) {
      console.error("Failed to toggle RAG", err);
    }
  }

  async function handleUploadFile(ev: React.ChangeEvent<HTMLInputElement>) {
    const file = ev.target.files?.[0];
    if (!file) return;
    setUploading(true);
    try {
      const form = new FormData();
      form.append("file", file);
      const resp = await fetch(`${API_BASE}/admin/upload`, {
        method: "POST",
        body: form,
      });
      if (!resp.ok) {
        throw new Error(`HTTP ${resp.status}`);
      }
      await resp.json();
      await loadRagReport();
      ev.target.value = "";
    } catch (err) {
      console.error("Upload failed", err);
    } finally {
      setUploading(false);
    }
  }

  function renderMessage(msg: ChatMessage) {
    const isUser = msg.role === "user";
    const bubbleClass = isUser
      ? "bg-emerald-600 text-white rounded-2xl rounded-br-none"
      : "bg-white text-slate-900 border border-slate-200 rounded-2xl rounded-bl-none";

    const label = isUser ? "You" : "MentorFlow";

    return (
      <div
        key={msg.id}
        className={cn(
          "flex flex-col gap-1",
          isUser ? "items-end" : "items-start",
        )}
      >
        <div className="flex items-center gap-2 text-[11px] text-slate-500">
          {!isUser && (
            <span className="inline-flex h-5 w-5 items-center justify-center rounded-full bg-slate-900 text-[10px] font-semibold text-white">
              MF
            </span>
          )}
          <span>{label}</span>
        </div>
        <div
          className={cn(
            "max-w-[90%] px-3 py-2 text-sm shadow-sm whitespace-pre-wrap",
            bubbleClass,
          )}
        >
          {msg.content}
        </div>
      </div>
    );
  }

  const learnerTab = (
    <div className="grid gap-4 md:grid-cols-[minmax(0,2fr)_minmax(0,1fr)]">
      <Card className="h-[520px] flex flex-col">
        <CardHeader className="pb-2">
          <CardTitle className="text-sm font-semibold">
            Lesson &amp; chat
          </CardTitle>
          <CardDescription className="text-xs">
            Type your question, or start a lesson / role-play using quick commands.
          </CardDescription>
        </CardHeader>
        <CardContent className="flex flex-col flex-1 gap-3">
          <div className="flex flex-col flex-1 rounded-lg border border-slate-200 bg-slate-50/80 p-3 overflow-y-auto space-y-3">
            {messages.length === 0 ? (
              <div className="flex h-full items-center justify-center text-xs text-slate-500">
                Start with{" "}
                <code className="mx-1 rounded bg-slate-900/90 px-1.5 py-0.5 text-[10px] text-white">
                  start lesson 1
                </code>{" "}
                or ask anything about your uploaded documents.
              </div>
            ) : (
              messages.map(renderMessage)
            )}
            {loading && (
              <div className="text-xs text-slate-500 animate-pulse">
                MentorFlow is thinking…
              </div>
            )}
          </div>

          <div className="flex flex-wrap gap-2 text-[11px] text-slate-500">
            <span className="mr-1 font-medium">Quick commands:</span>
            <Button
              variant="outline"
              size="xs"
              onClick={() => handleQuickCommand("start lesson 1")}
            >
              start lesson 1
            </Button>
            <Button
              variant="outline"
              size="xs"
              onClick={() => handleQuickCommand("start lesson 2")}
            >
              start lesson 2
            </Button>
            <Button
              variant="outline"
              size="xs"
              onClick={() => handleQuickCommand("start roleplay")}
            >
              start roleplay
            </Button>
            <Button
              variant="ghost"
              size="xs"
              className="ml-auto text-slate-500 hover:text-slate-900"
              onClick={handleClear}
            >
              Clear chat
            </Button>
          </div>

          <div className="flex items-center gap-2">
            <Input
              placeholder="Ask a question or continue the lesson..."
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  handleSend();
                }
              }}
            />
            <Button onClick={handleSend} disabled={loading || !input.trim()}>
              {loading ? "Thinking…" : "Send"}
            </Button>
          </div>

          <div className="flex items-center justify-between text-[11px] text-slate-500 pt-1 border-t border-slate-100">
            <div className="flex items-center gap-2">
              <span className="inline-flex items-center gap-1">
                <span className="h-2 w-2 rounded-full bg-emerald-500" />
                <span>Region: {region}</span>
              </span>
              <span className="hidden sm:inline">
                User ID:{" "}
                <span className="font-mono text-[10px] text-slate-700">
                  {userId.slice(0, 8)}…
                </span>
              </span>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-[11px]">🔊 Auto read</span>
              <Switch
                checked={ttsEnabled}
                onCheckedChange={(val) => setTtsEnabled(Boolean(val))}
              />
            </div>
          </div>
        </CardContent>
      </Card>

      <Card className="h-[520px] flex flex-col">
        <CardHeader className="pb-2">
          <CardTitle className="text-sm font-semibold">
            What can I learn here?
          </CardTitle>
          <CardDescription className="text-xs">
            v0.7 focuses on RAG: document-grounded answers &amp; citations.
          </CardDescription>
        </CardHeader>
        <CardContent className="flex-1 flex flex-col gap-3 text-xs text-slate-600">
          <div className="space-y-1">
            <p className="font-semibold text-slate-800">Tips</p>
            <ul className="list-disc pl-4 space-y-1">
              <li>Upload a TXT/PDF in the Admin tab.</li>
              <li>Ask questions about the uploaded content.</li>
              <li>Look for the 📚 Sources block under AI answers.</li>
            </ul>
          </div>
          <Textarea
            className="mt-2 h-40 text-xs"
            placeholder="Scratchpad – e.g. things you want MentorFlow to explain, PMP notes, exam questions…"
            value={scratch}
            onChange={(e) => setScratch(e.target.value)}
          />
        </CardContent>
      </Card>
    </div>
  );

  const adminTab = (
    <div className="grid gap-4 md:grid-cols-[minmax(0,1.3fr)_minmax(0,1fr)]">
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm font-semibold">RAG control</CardTitle>
          <CardDescription className="text-xs">
            Upload documents and toggle retrieval-augmented generation.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4 text-xs">
          <div className="flex items-center justify-between rounded-lg border border-slate-200 bg-slate-50 px-3 py-2">
            <div>
              <p className="font-medium text-slate-800 text-[13px]">
                RAG mode
              </p>
              <p className="text-[11px] text-slate-500">
                When enabled, answers are grounded in uploaded documents.
              </p>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-[11px] text-slate-500">
                {ragEnabled ? "On" : "Off"}
              </span>
              <Switch checked={ragEnabled} onCheckedChange={handleToggleRag} />
            </div>
          </div>

          <div className="space-y-2">
            <p className="font-medium text-[13px] text-slate-800">
              Upload knowledge base
            </p>
            <p className="text-[11px] text-slate-500">
              Supported: <code>.txt</code>, <code>.pdf</code>. Each upload is
              chunked &amp; embedded into a tiny local vector store.
            </p>
            <div className="flex items-center gap-2">
              <Input
                type="file"
                accept=".txt,.pdf"
                onChange={handleUploadFile}
                className="text-[11px]"
              />
              <Button
                type="button"
                variant="outline"
                size="sm"
                disabled={uploading}
              >
                {uploading ? "Uploading…" : "Upload"}
              </Button>
            </div>
          </div>

          <div className="space-y-2">
            <p className="font-medium text-[13px] text-slate-800">
              Indexed documents
            </p>
            {documents.length === 0 ? (
              <p className="text-[11px] text-slate-500">
                No documents ingested yet. Upload a TXT/PDF to build the KB.
              </p>
            ) : (
              <div className="rounded-md border border-slate-200 bg-white max-h-40 overflow-auto">
                <table className="min-w-full text-[11px]">
                  <thead className="bg-slate-50 text-slate-500">
                    <tr>
                      <th className="px-2 py-1 text-left font-medium">File</th>
                      <th className="px-2 py-1 text-right font-medium">
                        Chunks
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {documents.map((doc) => (
                      <tr
                        key={doc.doc_id}
                        className="border-t border-slate-100"
                      >
                        <td className="px-2 py-1">{doc.filename}</td>
                        <td className="px-2 py-1 text-right">
                          {doc.num_chunks}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm font-semibold">
            Region metrics (demo)
          </CardTitle>
          <CardDescription className="text-xs">
            Simple per-region request counts from the backend.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-3 text-xs">
          {metricsLoading ? (
            <p className="text-slate-500">Loading metrics…</p>
          ) : regionStats.length === 0 ? (
            <p className="text-slate-500">
              No data yet. Send a few chat messages from the learner tab.
            </p>
          ) : (
            <div className="rounded-md border border-slate-200 bg-white max-h-52 overflow-auto">
              <table className="min-w-full text-[11px]">
                <thead className="bg-slate-50 text-slate-500">
                  <tr>
                    <th className="px-2 py-1 text-left font-medium">Region</th>
                    <th className="px-2 py-1 text-right font-medium">
                      Users
                    </th>
                    <th className="px-2 py-1 text-right font-medium">
                      Requests
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {regionStats.map((row) => (
                    <tr
                      key={row.region}
                      className="border-t border-slate-100"
                    >
                      <td className="px-2 py-1">{row.region}</td>
                      <td className="px-2 py-1 text-right">
                        {row.user_count}
                      </td>
                      <td className="px-2 py-1 text-right">
                        {row.total_requests}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );

  return (
    <div className="min-h-screen bg-slate-100/80 text-slate-900">
      <div className="mx-auto flex min-h-screen max-w-5xl flex-col px-4 py-6">
        <header className="mb-4 flex items-center justify-between gap-3">
          <div className="flex flex-col gap-1">
            <div className="flex items-center gap-2">
              <span className="inline-flex h-7 w-7 items-center justify-center rounded-md bg-slate-900 text-[11px] font-semibold text-white">
                MF
              </span>
              <div>
                <h1 className="text-sm font-semibold leading-tight">
                  MentorFlow – Persona / Lesson Engine
                </h1>
                <span className="text-[11px] text-slate-500">
                  Interactive lesson &amp; RAG teaching assistant
                </span>
              </div>
            </div>
            <div className="flex items-center gap-2 text-[11px] text-slate-500">
              <Badge className="bg-emerald-50 text-emerald-700 border border-emerald-200 rounded-full px-2 py-0.5 text-[10px]">
                v0.7 – RAG MVP
              </Badge>
              <span className="hidden sm:inline">
                Backend:{" "}
                <span className="font-mono text-[10px] text-slate-700">
                  {API_BASE}
                </span>
              </span>
            </div>
          </div>
        </header>

        <Tabs
          value={activeTab}
          onValueChange={(v) => setActiveTab(v as "learner" | "admin")}
          className="flex-1 flex flex-col gap-3"
        >
          <TabsList className="w-fit bg-slate-200/70">
            <TabsTrigger value="learner" className="text-xs">
              Learner
            </TabsTrigger>
            <TabsTrigger value="admin" className="text-xs">
              Admin
            </TabsTrigger>
          </TabsList>

          <TabsContent value="learner" className="flex-1 mt-0">
            {learnerTab}
          </TabsContent>
          <TabsContent value="admin" className="flex-1 mt-0">
            {adminTab}
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
}

const container = document.getElementById("root");
if (container) {
  const root = createRoot(container);
  root.render(<App />);
}

export default App;
