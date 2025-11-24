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

  // Admin state
  const [ragEnabled, setRagEnabled] = useState<boolean>(true);
  const [documents, setDocuments] = useState<RagDocument[]>([]);
  const [uploading, setUploading] = useState(false);
  const [adminNote, setAdminNote] = useState("");
  const [metricsLoading, setMetricsLoading] = useState(false);
  const [regionStats, setRegionStats] = useState<RegionMetric[]>([]);

  useEffect(() => {
    window.localStorage.setItem(
      "mentorflow_tts_enabled",
      JSON.stringify(ttsEnabled)
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

      // 在回答後面加上 citation 區塊
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
        audio.play().catch(() => {
          // ignore autoplay error
        });
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

  function handleReset() {
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
    const lines = msg.content.split("\n");

    return (
      <div
        key={msg.id}
        className={cn("flex flex-col gap-1", isUser ? "items-end" : "items-start")}
      >
        <div className="flex items-center gap-2 text-[11px] text-slate-500">
          {!isUser && (
            <span className="inline-flex h-5 w-5 items-center justify-center rounded-full bg-slate-900 text-[10px] font-semibold text-white">
              MF
            </span>
          )}
          <span>{label}</span>
        </div>
        <div className={cn("max-w-[90%] px-3 py-2 text-sm shadow-sm", bubbleClass)}>
          {lines.map((line, idx) => (
            <p key={idx} className="whitespace-pre-wrap">
              {line}
            </p>
          ))}
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
            Type your question, or start a lesson / role-play using quick c
