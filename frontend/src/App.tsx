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

const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";

type Role = "user" | "assistant";

interface ChatMessage {
  id: string;
  role: Role;
  content: string;
}

interface ChatResponse {
  reply: string;
  tts_base64?: string | null;
}

interface RegionStat {
  region: string;
  user_count: number;
  total_requests: number;
}

interface RegionMetricsResponse {
  regions: RegionStat[];
}

interface RagDocument {
  doc_id?: string;
  filename?: string;
  num_chunks?: number;
}

interface RagReport {
  rag_enabled: boolean;
  documents?: RagDocument[];
}

const USER_ID_KEY = "mentorflow_user_id_v1";
const REGION_KEY = "mentorflow_region_v1";
const TTS_KEY = "mentorflow_tts_enabled_v1";

function getOrCreateUserId(): string {
  if (typeof window === "undefined") return "dev-user";
  const existing = localStorage.getItem(USER_ID_KEY);
  if (existing) return existing;
  const id = crypto.randomUUID();
  localStorage.setItem(USER_ID_KEY, id);
  return id;
}

function detectRegion(): string {
  if (typeof window === "undefined") return "IE";
  const existing = localStorage.getItem(REGION_KEY);
  if (existing) return existing;

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

  localStorage.setItem(REGION_KEY, region);
  return region;
}

export default function App() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState<"learner" | "admin">("learner");

  const [ttsEnabled, setTTSEnabled] = useState<boolean>(() => {
    const v = localStorage.getItem(TTS_KEY);
    if (v === null) return true;
    return v === "1";
  });

  const [regionStats, setRegionStats] = useState<RegionStat[]>([]);
  const [metricsLoading, setMetricsLoading] = useState(false);

  // RAG / knowledge upload state
  const [ragEnabled, setRagEnabled] = useState<boolean | null>(null);
  const [ragLoading, setRagLoading] = useState(false);
  const [ragDocuments, setRagDocuments] = useState<RagDocument[]>([]);
  const [uploading, setUploading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState<string | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);

  const userId = useMemo(() => getOrCreateUserId(), []);
  const region = useMemo(() => detectRegion(), []);

  useEffect(() => {
    localStorage.setItem(TTS_KEY, ttsEnabled ? "1" : "0");
  }, [ttsEnabled]);

  async function callChatApi(message: string): Promise<ChatResponse> {
    const payload = {
      user_id: userId,
      message,
      region,
    };

    const resp = await fetch(`${API_BASE}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
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
      const reply = data.reply || "";

      const aiMsg: ChatMessage = {
        id: crypto.randomUUID(),
        role: "assistant",
        content: reply,
      };
      setMessages((prev) => [...prev, aiMsg]);

      if (data.tts_base64 && ttsEnabled) {
        const audio = new Audio(`data:audio/mp3;base64,${data.tts_base64}`);
        audio.play().catch(() => {
          // ignore autoplay error
        });
      }
    } catch (err) {
      console.error(err);
      const errMsg: ChatMessage = {
        id: crypto.randomUUID(),
        role: "assistant",
        content: "⚠️ Error connecting to server. Please try again.",
      };
      setMessages((prev) => [...prev, errMsg]);
    } finally {
      setLoading(false);
    }
  }

  function handleQuickCommand(cmd: string) {
    setInput(cmd);
  }

  function handleReset() {
    setMessages([]);
    setInput("");
  }

  async function loadRegionMetrics() {
    setMetricsLoading(true);
    try {
      const resp = await fetch(`${API_BASE}/metrics/regions`);
      if (!resp.ok) throw new Error("HTTP error");
      const data: RegionMetricsResponse = await resp.json();
      setRegionStats(data.regions || []);
    } catch (err) {
      console.error(err);
    } finally {
      setMetricsLoading(false);
    }
  }

  async function loadRagReport() {
    setRagLoading(true);
    try {
      const resp = await fetch(`${API_BASE}/report`);
      if (!resp.ok) throw new Error("HTTP error");
      const data: RagReport = await resp.json();
      setRagEnabled(
        typeof data.rag_enabled === "boolean" ? data.rag_enabled : false
      );
      setRagDocuments(data.documents || []);
    } catch (err) {
      console.error(err);
    } finally {
      setRagLoading(false);
    }
  }

  async function handleToggleRag(nextEnabled: boolean) {
    setRagLoading(true);
    try {
      const resp = await fetch(`${API_BASE}/admin/toggle-rag`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ enabled: nextEnabled }),
      });
      if (!resp.ok) throw new Error("HTTP error");
      setRagEnabled(nextEnabled);
    } catch (err) {
      console.error(err);
    } finally {
      setRagLoading(false);
    }
  }

  async function handleUploadKnowledge() {
    if (!selectedFile) return;
    setUploading(true);
    setUploadStatus(null);
    try {
      const formData = new FormData();
      formData.append("file", selectedFile);
      const resp = await fetch(`${API_BASE}/admin/upload`, {
        method: "POST",
        body: formData,
      });
      if (!resp.ok) {
        throw new Error("Upload failed");
      }
      setUploadStatus("✅ Upload successful.");
      setSelectedFile(null);
      // Refresh RAG report so admin sees updated docs/status
      loadRagReport();
    } catch (err) {
      console.error(err);
      setUploadStatus("⚠️ Upload failed. Please try again.");
    } finally {
      setUploading(false);
    }
  }

  useEffect(() => {
    if (activeTab === "admin") {
      if (regionStats.length === 0 && !metricsLoading) {
        loadRegionMetrics();
      }
      if (ragEnabled === null && !ragLoading) {
        loadRagReport();
      }
    }
  }, [activeTab, regionStats.length, metricsLoading, ragEnabled, ragLoading]);

  const learnerMessages = messages;

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 flex flex-col">
      {/* Top Bar */}
      <header className="border-b border-slate-200 bg-white/95 backdrop-blur">
        <div className="mx-auto max-w-6xl px-5 py-3 flex items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="h-10 w-10 rounded-3xl bg-gradient-to-br from-emerald-500 via-emerald-400 to-sky-400 flex items-center justify-center text-slate-950 font-semibold text-sm shadow-sm">
              MF
            </div>
            <div className="flex flex-col">
              <span className="font-semibold leading-tight text-slate-900 text-lg">
                MentorFlow Persona
              </span>
              <span className="text-xs text-slate-500">
                Interactive lesson &amp; coaching demo
              </span>
            </div>
          </div>

          <div className="flex items-center gap-3 text-xs text-slate-500">
            <Badge className="bg-emerald-50 text-emerald-700 border border-emerald-200 rounded-full px-3 py-1 text-[11px]">
              v0.7 – RAG MVP
            </Badge>
            <span className="hidden sm:inline">
              User ID:{" "}
              <span className="font-mono text-[11px] text-slate-700">
                {userId.slice(0, 8)}...
              </span>
            </span>
          </div>
        </div>
      </header>

      {/* Main */}
      <main className="flex-1">
        <div className="mx-auto flex max-w-6xl flex-col gap-5 px-5 py-5 lg:flex-row">
          {/* Left: Chat */}
          <section className="flex-1 space-y-4">
            <Card className="h-[520px] rounded-3xl border border-slate-200 bg-white/80 shadow-sm flex flex-col">
              <CardHeader className="pb-3">
                <div className="flex items-center justify-between gap-3">
                  <div>
                    <CardTitle className="text-base text-slate-900">
                      Lesson &amp; Roleplay
                    </CardTitle>
                    <CardDescription className="text-xs text-slate-500">
                      Ask questions about the lesson, or try role-play commands
                      like{" "}
                      <span className="font-mono text-[11px] bg-slate-100 px-1.5 py-0.5 rounded-full">
                        /lesson 4
                      </span>{" "}
                      or{" "}
                      <span className="font-mono text-[11px] bg-slate-100 px-1.5 py-0.5 rounded-full">
                        /roleplay stakeholder
                      </span>
                      .
                    </CardDescription>
                  </div>

                  <div className="flex flex-col items-end gap-1">
                    <div className="flex items-center gap-2">
                      <span className="text-[11px] text-slate-500">
                        Voice reply
                      </span>
                      <Switch
                        checked={ttsEnabled}
                        onCheckedChange={setTTSEnabled}
                      />
                    </div>
                    <span className="text-[10px] text-slate-400">
                      Region: {region}
                    </span>
                  </div>
                </div>
              </CardHeader>

              <CardContent className="flex-1 flex flex-col gap-3 pt-0">
                <div className="flex-1 rounded-2xl border border-slate-200 bg-slate-50/70 p-3 overflow-y-auto space-y-3">
                  {learnerMessages.length === 0 ? (
                    <div className="flex h-full flex-col items-center justify-center text-center text-xs text-slate-400 gap-2">
                      <p>Start a conversation to see MentorFlow in action.</p>
                      <p>
                        Try:{" "}
                        <button
                          className="font-mono text-[11px] bg-white border border-slate-200 rounded-full px-2.5 py-0.5 shadow-sm hover:bg-slate-50"
                          onClick={() =>
                            handleQuickCommand(
                              "What is a project charter and why is it important?"
                            )
                          }
                        >
                          Ask about project charter
                        </button>
                      </p>
                    </div>
                  ) : (
                    learnerMessages.map((m) => (
                      <ChatMessageBubble key={m.id} message={m} />
                    ))
                  )}
                </div>

                <div className="space-y-2">
                  <div className="flex gap-2">
                    <Textarea
                      value={input}
                      onChange={(e) => setInput(e.target.value)}
                      placeholder="Ask a question or use commands like /lesson 4 or /roleplay stakeholder..."
                      className="min-h-[60px] text-sm resize-none bg-white/80 border-slate-200 focus-visible:ring-emerald-500"
                      onKeyDown={(e) => {
                        if (e.key === "Enter" && !e.shiftKey) {
                          e.preventDefault();
                          handleSend();
                        }
                      }}
                    />
                  </div>

                  <div className="flex items-center justify-between gap-2">
                    <div className="flex flex-wrap gap-1.5">
                      <Button
                        type="button"
                        variant="outline"
                        size="xs"
                        className="rounded-full border-slate-200 bg-white text-[11px] text-slate-700 hover:bg-slate-50"
                        onClick={() =>
                          handleQuickCommand(
                            "Explain why a project is temporary but operations are ongoing."
                          )
                        }
                      >
                        Why project is temporary?
                      </Button>
                      <Button
                        type="button"
                        variant="outline"
                        size="xs"
                        className="rounded-full border-slate-200 bg-white text-[11px] text-slate-700 hover:bg-slate-50"
                        onClick={() => handleQuickCommand("/lesson 4")}
                      >
                        /lesson 4
                      </Button>
                      <Button
                        type="button"
                        variant="outline"
                        size="xs"
                        className="rounded-full border-slate-200 bg-white text-[11px] text-slate-700 hover:bg-slate-50"
                        onClick={() =>
                          handleQuickCommand("/roleplay stakeholder")
                        }
                      >
                        /roleplay stakeholder
                      </Button>
                    </div>

                    <div className="flex items-center gap-1.5">
                      <Button
                        type="button"
                        variant="ghost"
                        size="sm"
                        className="text-xs text-slate-500 hover:text-slate-700"
                        onClick={handleReset}
                      >
                        Reset
                      </Button>
                      <Button
                        type="button"
                        size="sm"
                        onClick={handleSend}
                        disabled={loading || !input.trim()}
                        className="rounded-full bg-emerald-600 hover:bg-emerald-700 text-xs px-4"
                      >
                        {loading ? "Thinking..." : "Send"}
                      </Button>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </section>

          {/* Right: Admin / Debug */}
          <section className="w-full lg:w-[320px] space-y-4">
            <Tabs
              value={activeTab}
              onValueChange={(v) => setActiveTab(v as "learner" | "admin")}
              className="space-y-5"
            >
              <TabsList className="bg-slate-100 border border-slate-200 rounded-full px-1.5 py-1 shadow-sm inline-flex">
                <TabsTrigger
                  value="learner"
                  className="rounded-full px-5 py-1.5 text-sm text-slate-600 data-[state=active]:bg-white data-[state=active]:text-slate-900 data-[state=active]:shadow-sm"
                >
                  Learner View
                </TabsTrigger>
                <TabsTrigger
                  value="admin"
                  className="rounded-full px-5 py-1.5 text-sm text-slate-600 data-[state=active]:bg-white data-[state=active]:text-slate-900 data-[state=active]:shadow-sm"
                >
                  Admin View
                </TabsTrigger>
              </TabsList>

              <TabsContent value="learner" className="space-y-4">
                <Card className="rounded-3xl border border-slate-200 bg-white shadow-sm">
                  <CardHeader className="pb-2">
                    <CardTitle className="text-sm text-slate-900">
                      Lesson context
                    </CardTitle>
                    <CardDescription className="text-xs text-slate-500">
                      This demo focuses on PMP-style project management concepts,
                      with interactive roleplay and lesson snippets.
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-3 text-xs text-slate-600">
                    <ul className="list-disc pl-4 space-y-1">
                      <li>
                        Try asking:{" "}
                        <span className="font-mono text-[11px] bg-slate-100 px-1.5 py-0.5 rounded-full">
                          Why is a project temporary but operations are ongoing?
                        </span>
                      </li>
                      <li>
                        Use{" "}
                        <span className="font-mono text-[11px] bg-slate-100 px-1.5 py-0.5 rounded-full">
                          /lesson 4
                        </span>{" "}
                        to load interactive content from Lesson 4.
                      </li>
                      <li>
                        Use{" "}
                        <span className="font-mono text-[11px] bg-slate-100 px-1.5 py-0.5 rounded-full">
                          /roleplay stakeholder
                        </span>{" "}
                        to simulate real-world conversations.
                      </li>
                    </ul>
                  </CardContent>
                </Card>
              </TabsContent>

              <TabsContent value="admin" className="space-y-4">
                <Card className="rounded-3xl border border-slate-200 bg-white shadow-sm">
                  <CardHeader className="pb-2">
                    <CardTitle className="text-sm text-slate-900">
                      Region metrics
                    </CardTitle>
                    <CardDescription className="text-xs text-slate-500">
                      Aggregated by backend based on user_id and inferred region.
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-2 text-xs text-slate-600">
                    {metricsLoading ? (
                      <p className="text-slate-500">Loading metrics...</p>
                    ) : regionStats.length === 0 ? (
                      <p className="text-slate-400">
                        No metrics yet. Trigger some chat requests first.
                      </p>
                    ) : (
                      <div className="space-y-1.5">
                        {regionStats.map((stat) => (
                          <div
                            key={stat.region}
                            className="flex items-center justify-between rounded-xl bg-slate-50 px-3 py-2"
                          >
                            <div className="flex items-center gap-2">
                              <span className="text-[11px] font-medium text-slate-700">
                                {stat.region}
                              </span>
                            </div>
                            <div className="flex flex-col items-end text-[11px] text-slate-500">
                              <span>{stat.user_count} users</span>
                              <span>{stat.total_requests} requests</span>
                            </div>
                          </div>
                        ))}
                      </div>
                    )}
                  </CardContent>
                </Card>

                <Card className="rounded-3xl border border-slate-200 bg-white shadow-sm">
                  <CardHeader className="pb-3">
                    <CardTitle className="text-sm text-slate-900">
                      Step 1 &amp; 2 – Upload knowledge &amp; enable RAG
                    </CardTitle>
                    <CardDescription className="text-xs text-slate-500">
                      Upload your own PDF / TXT files and tell MentorFlow whether it
                      should use them when answering.
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    <div className="flex flex-col gap-2 sm:flex-row sm:items-center">
                      <Input
                        type="file"
                        accept=".pdf,.txt"
                        onChange={(e) => {
                          const file = e.target.files?.[0] ?? null;
                          setSelectedFile(file);
                          setUploadStatus(null);
                        }}
                        className="text-xs"
                      />
                      <Button
                        type="button"
                        size="sm"
                        className="mt-1 sm:mt-0"
                        disabled={!selectedFile || uploading}
                        onClick={handleUploadKnowledge}
                      >
                        {uploading ? "Uploading..." : "Upload"}
                      </Button>
                    </div>

                    {uploadStatus && (
                      <p className="text-xs text-slate-600">{uploadStatus}</p>
                    )}

                    {ragDocuments.length > 0 && (
                      <div className="mt-2 rounded-xl bg-slate-50 p-3">
                        <p className="mb-1 text-[11px] font-medium uppercase tracking-wide text-slate-500">
                          Uploaded documents
                        </p>
                        <ul className="space-y-1">
                          {ragDocuments.map((doc, idx) => (
                            <li
                              key={doc.doc_id ?? doc.filename ?? idx}
                              className="flex items-center justify-between text-xs text-slate-700"
                            >
                              <span className="truncate pr-2">
                                {doc.filename ?? doc.doc_id ?? "Document"}
                              </span>
                              {typeof doc.num_chunks === "number" && (
                                <span className="text-[11px] text-slate-500">
                                  {doc.num_chunks} chunks
                                </span>
                              )}
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}

                    <div className="mt-3 flex items-center justify-between rounded-xl bg-slate-50 px-3 py-2.5">
                      <div>
                        <p className="text-xs font-medium text-slate-800">
                          Step 2 – Use uploaded docs (RAG)
                        </p>
                        <p className="text-[11px] text-slate-500">
                          When enabled, answers will try to cite your documents first.
                        </p>
                      </div>
                      <div className="flex items-center gap-2">
                        <Switch
                          checked={!!ragEnabled}
                          onCheckedChange={handleToggleRag}
                          disabled={ragLoading}
                        />
                        <span className="text-[11px] font-medium text-slate-600">
                          {ragEnabled === null
                            ? "Unknown"
                            : ragEnabled
                            ? "On"
                            : "Off"}
                        </span>
                      </div>
                    </div>
                    <p className="mt-1 text-[11px] text-slate-400">
                      If a question is not covered by your docs, MentorFlow will fall
                      back to general knowledge.
                    </p>
                  </CardContent>
                </Card>

                <Card className="rounded-3xl border border-slate-200 bg-white shadow-sm">
                  <CardHeader className="pb-3">
                    <CardTitle className="text-sm text-slate-900">
                      Session Debug
                    </CardTitle>
                    <CardDescription className="text-xs text-slate-500">
                      What the backend sees for this browser session.
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-2 text-xs text-slate-600">
                    <div className="grid grid-cols-2 gap-2 rounded-2xl bg-slate-50 p-3">
                      <div className="space-y-0.5">
                        <p className="text-[11px] text-slate-500">User ID</p>
                        <p className="font-mono text-[11px] text-slate-800 break-all">
                          {userId}
                        </p>
                      </div>
                      <div className="space-y-0.5">
                        <p className="text-[11px] text-slate-500">Region</p>
                        <p className="text-[11px] font-medium text-slate-800">
                          {region}
                        </p>
                      </div>
                      <div className="space-y-0.5">
                        <p className="text-[11px] text-slate-500">Messages</p>
                        <p className="text-[11px] font-medium text-slate-800">
                          {messages.length}
                        </p>
                      </div>
                      <div className="space-y-0.5">
                        <p className="text-[11px] text-slate-500">Voice reply</p>
                        <p className="text-[11px] font-medium text-slate-800">
                          {ttsEnabled ? "Enabled" : "Disabled"}
                        </p>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </TabsContent>
            </Tabs>
          </section>
        </div>
      </main>
    </div>
  );
}

function ChatMessageBubble({ message }: { message: ChatMessage }) {
  const isUser = message.role === "user";

  return (
    <div
      className={cn(
        "flex w-full gap-2 text-xs",
        isUser ? "justify-end" : "justify-start"
      )}
    >
      {!isUser && (
        <div className="mt-1 flex h-7 w-7 items-center justify-center rounded-full border border-emerald-200 bg-emerald-50 text-[11px] font-medium text-emerald-700">
          AI
        </div>
      )}
      <div
        className={cn(
          "max-w-[82%] rounded-2xl px-3 py-2 leading-relaxed shadow-sm",
          isUser
            ? "rounded-br-sm bg-slate-900 text-slate-50"
            : "rounded-bl-sm bg-white text-slate-900 border border-slate-200"
        )}
      >
        <p className="whitespace-pre-wrap">{message.content}</p>
      </div>
      {isUser && (
        <div className="mt-1 flex h-7 w-7 items-center justify-center rounded-full border border-slate-200 bg-slate-100 text-[11px] font-medium text-slate-700">
          You
        </div>
      )}
    </div>
  );
}

const container = document.getElementById("root");
if (container) {
  const root = createRoot(container);
  root.render(<App />);
}
