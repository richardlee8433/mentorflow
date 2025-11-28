import React, { useEffect, useMemo, useState } from "react";
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

type Role = "user" | "assistant";

type TurnType = "chat" | "lesson" | "lecture" | "roleplay" | "error";

type ChatMessage = {
  id: string;
  role: Role;
  content: string;
  turnType?: TurnType;
  createdAt: number;
};

type ApiResponse = {
  reply: string;
  turn_type: TurnType;
  tts_base64?: string | null;
};

type BackendStatus = "unknown" | "ok" | "error";

function usePersistentUserId(key: string): string {
  const [userId] = useState(
    () => {
      if (typeof window === "undefined") return "anonymous";
      try {
        const existing = window.localStorage.getItem(key);
        if (existing) return existing;
        const generated = `user_${Math.random().toString(36).slice(2, 10)}`;
        window.localStorage.setItem(key, generated);
        return generated;
      } catch {
        return `user_${Math.random().toString(36).slice(2, 10)}`;
      }
    },
  );
  return userId;
}

function useBackendBaseUrl(): string {
  return "https://mentorflow.onrender.com";
}

function App() {
  const userId = usePersistentUserId("mentorflow_user_id");
  const backendBase = useBackendBaseUrl();

  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [ragEnabled, setRagEnabled] = useState(false);
  const [ttsEnabled, setTtsEnabled] = useState(true);
  const [speaking, setSpeaking] = useState(false);
  const [backendStatus, setBackendStatus] = useState<BackendStatus>("unknown");
  const [region, setRegion] = useState("EU-1");
  const [uploading, setUploading] = useState(false);
  const [adminDocsInfo, setAdminDocsInfo] = useState<string | null>(null);

  const backendLabel = useMemo(() => {
    if (backendStatus === "ok") return "backend: online";
    if (backendStatus === "error") return "backend: error";
    return "backend: checking…";
  }, [backendStatus]);

  useEffect(() => {
    let cancelled = false;

    (async () => {
      try {
        const res = await fetch(`${backendBase}/health`);
        if (!res.ok) throw new Error("health not ok");
        const data = await res.json();
        if (!cancelled) {
          setBackendStatus("ok");
          if (data.region && typeof data.region === "string") {
            setRegion(data.region);
          }
        }
      } catch {
        if (!cancelled) {
          setBackendStatus("error");
        }
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [backendBase]);

  useEffect(() => {
    if (!ttsEnabled && speaking) {
      window.speechSynthesis.cancel();
      setSpeaking(false);
    }
  }, [ttsEnabled, speaking]);

  function speakText(text: string) {
    if (!ttsEnabled) return;
    if (!("speechSynthesis" in window)) return;

    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.onend = () => setSpeaking(false);
    utterance.onerror = () => setSpeaking(false);
    setSpeaking(true);
    window.speechSynthesis.speak(utterance);
  }

  async function callChatApi(message: string): Promise<ApiResponse> {
    const res = await fetch(`${backendBase}/chat`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        user_id: userId,
        message,
        rag_enabled: ragEnabled,
        region,
      }),
    });

    if (!res.ok) {
      throw new Error(`Chat API error: ${res.status}`);
    }

    const data = (await res.json()) as ApiResponse;
    return data;
  }

  async function handleSend() {
    const text = input.trim();
    if (!text || loading) return;

    const userMsg: ChatMessage = {
      id: `${Date.now()}-user`,
      role: "user",
      content: text,
      createdAt: Date.now(),
    };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setLoading(true);

    try {
      const data = await callChatApi(text);
      const reply = data.reply ?? "";
      const turnType = data.turn_type ?? "chat";

      const aiMsg: ChatMessage = {
        id: `${Date.now()}-assistant`,
        role: "assistant",
        content: reply,
        turnType,
        createdAt: Date.now(),
      };
      setMessages((prev) => [...prev, aiMsg]);

      // Auto read: prefer backend ElevenLabs audio, fallback to browser TTS
      if (ttsEnabled) {
        if (data.tts_base64) {
          try {
            const audio = new Audio(
              "data:audio/mpeg;base64," + data.tts_base64,
            );
            audio.play().catch((e) => {
              console.error("Audio play error", e);
              speakText(reply);
            });
          } catch (e) {
            console.error("Audio playback failed", e);
            speakText(reply);
          }
        } else {
          speakText(reply);
        }
      }
    } catch (err) {
      console.error(err);
      const aiMsg: ChatMessage = {
        id: `${Date.now()}-assistant-error`,
        role: "assistant",
        content:
          "Something went wrong talking to the backend. Please check if the server is running on 127.0.0.1:8000.",
        turnType: "error",
        createdAt: Date.now(),
      };
      setMessages((prev) => [...prev, aiMsg]);
    } finally {
      setLoading(false);
    }
  }

  function handleQuickCommand(cmd: string) {
    setInput(cmd);
    setTimeout(() => {
      handleSend();
    }, 0);
  }

  function handleClear() {
    setMessages([]);
  }

  async function handleUploadFile(ev: React.ChangeEvent<HTMLInputElement>) {
    const file = ev.target.files?.[0];
    if (!file) return;
    setUploading(true);
    setAdminDocsInfo(null);

    try {
      const form = new FormData();
      form.append("file", file);

      const res = await fetch(`${backendBase}/upload`, {
        method: "POST",
        body: form,
      });

      if (!res.ok) {
        throw new Error(`Upload error: ${res.status}`);
      }

      const data = (await res.json()) as { detail: string; chunks?: number };
      let summary = data.detail;
      if (data.chunks !== undefined) {
        summary += ` (chunks stored: ${data.chunks})`;
      }
      setAdminDocsInfo(summary);
    } catch (err: any) {
      console.error(err);
      setAdminDocsInfo(
        err?.message ?? "Error uploading file (see console for details).",
      );
    } finally {
      setUploading(false);
      ev.target.value = "";
    }
  }

  function renderMessage(msg: ChatMessage) {
    const isUser = msg.role === "user";
    const isError = msg.turnType === "error";

    const bubbleBase =
      "max-w-[90%] rounded-2xl px-3 py-2 text-sm leading-relaxed whitespace-pre-wrap";
    const userBubble = cn(
      bubbleBase,
      "bg-emerald-600 text-white rounded-br-sm shadow-sm",
    );
    const assistantBubble = cn(
      bubbleBase,
      "bg-white text-slate-900 border border-slate-200 rounded-bl-sm shadow-sm",
    );

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
          <span>{isUser ? "You" : "MentorFlow"}</span>
          {msg.turnType && msg.turnType !== "chat" && !isError && (
            <span className="rounded-full bg-slate-100 px-2 py-0.5 text-[10px] font-mono uppercase tracking-wide text-slate-500">
              {msg.turnType}
            </span>
          )}
        </div>
        <div className={isUser ? userBubble : assistantBubble}>
          {msg.content}
        </div>
      </div>
    );
  }

  const learnerTab = (
    <div className="grid gap-4 md:grid-cols-[minmax(0,2fr)_minmax(0,1fr)]">
      {/* Left column: main chat / lesson card */}
      <Card className="flex flex-col">
        <CardHeader className="pb-2">
          <CardTitle className="text-base font-semibold">
            Lesson &amp; chat
          </CardTitle>
          <CardDescription className="text-xs">
            Type your question, or start a lesson / role-play using quick
            commands.
          </CardDescription>
        </CardHeader>
        <CardContent className="flex flex-1 flex-col gap-3 pt-0">
          {/* Message list */}
          <div className="flex-1 space-y-3 rounded-lg border border-slate-100 bg-slate-50/60 px-3 py-3 overflow-y-auto">
            {messages.length === 0 ? (
              <div className="flex h-full items-center justify-center text-xs text-slate-500 text-center">
                <div className="max-w-md space-y-1">
                  <p className="font-medium text-slate-700 mb-1">
                    Try a starting command:
                  </p>
                  <p>
                    <code className="rounded bg-slate-900 px-1.5 py-0.5 text-[11px] text-white">
                      start lecture 1
                    </code>{" "}
                    – podcast-style lecture on Tokens, Embeddings, and Context
                    Windows.
                  </p>
                  <p>
                    <code className="rounded bg-slate-900 px-1.5 py-0.5 text-[11px] text-white">
                      start lesson 1
                    </code>{" "}
                    – guided lesson with Q&amp;A.
                  </p>
                  <p>
                    <code className="rounded bg-slate-900 px-1.5 py-0.5 text-[11px] text-white">
                      start roleplay
                    </code>{" "}
                    – simulate conversations as an AI PM.
                  </p>
                </div>
              </div>
            ) : (
              <>
                {messages.map((m) => renderMessage(m))}
                {loading && (
                  <div className="text-xs text-slate-500 animate-pulse">
                    MentorFlow is thinking…
                  </div>
                )}
              </>
            )}
          </div>

          {/* Quick commands row */}
          <div className="flex flex-wrap items-center gap-2 text-[11px] text-slate-600">
            <span className="font-semibold text-slate-700">Quick commands:</span>
            <Button
              variant="outline"
              size="xs"
              onClick={() => handleQuickCommand("start lecture 1")}
            >
              start lecture 1
            </Button>
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
              onClick={() => handleQuickCommand("start lecture 2")}
            >
              start lecture 2
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
              onClick={() => handleQuickCommand("stop lesson")}
            >
              stop lesson
            </Button>
            <Button
              variant="ghost"
              size="xs"
              className="text-slate-500 hover:text-slate-900"
              onClick={handleClear}
            >
              Clear chat
            </Button>
          </div>

          {/* Input row */}
          <div className="flex items-center gap-2">
            <Input
              placeholder="Ask a question or continue the lesson..."
              value={input}
              onChange={(e) => setInput(e.target.value)}
              className="text-sm text-slate-900 placeholder:text-slate-400"
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  handleSend();
                }
              }}
              disabled={loading}
            />
            <Button onClick={handleSend} disabled={loading || !input.trim()}>
              {loading ? "Thinking…" : "Send"}
            </Button>
          </div>

          {/* Footer row */}
          <div className="flex items-center justify-between border-t border-slate-100 pt-1 text-[11px] text-slate-500">
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
              <span className="inline-flex items-center gap-1">
                <span className="h-2 w-2 rounded-full bg-emerald-500" />
                <span className="capitalize">{backendLabel}</span>
              </span>
              <span className="inline-flex items-center gap-1">
                <span className="h-2 w-2 rounded-full bg-sky-500" />
                <span>
                  Auto read:{" "}
                  <span className="font-mono">
                    {ttsEnabled ? "ElevenLabs" : "off"}
                  </span>
                </span>
              </span>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Right column: RAG / auto read toggles + tips */}
      <Card className="flex flex-col">
        <CardHeader className="pb-2">
          <CardTitle className="text-sm font-semibold">
            What can I learn here?
          </CardTitle>
          <CardDescription className="text-xs">
            v0.8 focuses on lesson flow &amp; podcast-style lectures. v0.7 RAG
            upload remains available in the Admin tab.
          </CardDescription>
        </CardHeader>
        <CardContent className="flex flex-1 flex-col gap-3 pt-0">
          {/* Toggles */}
          <div className="space-y-2 rounded-lg border border-slate-200 bg-white px-3 py-2">
            <div className="flex items-center justify-between gap-3">
              <div className="space-y-0.5">
                <p className="text-xs font-semibold text-slate-700">
                  RAG mode
                </p>
                <p className="text-[11px] text-slate-500">
                  When ON, MentorFlow uses your uploaded documents as a
                  knowledge base and returns citations.
                </p>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-[11px] text-slate-500">
                  {ragEnabled ? "On" : "Off"}
                </span>
                <Switch
                  checked={ragEnabled}
                  onCheckedChange={(val) => setRagEnabled(val)}
                />
              </div>
            </div>
            <div className="h-px bg-slate-100" />
            <div className="flex items-center justify-between gap-3">
              <div className="space-y-0.5">
                <p className="text-xs font-semibold text-slate-700">
                  Auto read (TTS)
                </p>
                <p className="text-[11px] text-slate-500">
                  When enabled, MentorFlow will read answers out loud. With
                  ElevenLabs configured, lectures sound more like a real voice.
                </p>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-[11px] text-slate-500">
                  {ttsEnabled ? "On" : "Off"}
                </span>
                <Switch
                  checked={ttsEnabled}
                  onCheckedChange={(val) => setTtsEnabled(val)}
                />
              </div>
            </div>
          </div>

          {/* Tips */}
          <div className="rounded-lg border border-slate-200 bg-white px-3 py-2">
            <p className="text-[11px] font-semibold text-slate-700 mb-1">
              Tips
            </p>
            <ul className="list-disc pl-4 space-y-1 text-[11px] text-slate-600">
              <li>Use quick commands to start a lesson or role-play.</li>
              <li>
                In lecture mode, type <code>next</code> to hear the next part,
                or <code>stop lesson</code> to exit.
              </li>
              <li>
                Ask for explanations, comparisons, and step-by-step reasoning.
              </li>
              <li>
                In Admin tab, upload documents to power RAG-based answers.
              </li>
            </ul>
          </div>
        </CardContent>
      </Card>
    </div>
  );

  const adminTab = (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-base font-semibold flex items-center gap-2">
          Admin – RAG documents
          <Badge variant="outline" className="text-[10px]">
            v0.8
          </Badge>
        </CardTitle>
        <CardDescription className="text-xs">
          Upload a TXT/PDF file here to use as a knowledge base for RAG-mode
          questions.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-3 pt-0">
        <div className="space-y-1">
          <label className="text-xs font-medium text-slate-700">
            Upload document
          </label>
          <input
            type="file"
            accept=".txt,.pdf"
            onChange={handleUploadFile}
            disabled={uploading}
            className="block w-full text-xs text-slate-700 file:mr-2 file:rounded file:border-0 file:bg-slate-900 file:px-2 file:py-1 file:text-xs file:font-semibold file:text-white hover:file:bg-slate-700"
          />
          <p className="text-[11px] text-slate-500">
            Supported formats: .txt, .pdf. Large files will be chunked before
            indexing.
          </p>
        </div>
        {uploading && (
          <p className="text-xs text-slate-500">Uploading &amp; indexing…</p>
        )}
        {adminDocsInfo && (
          <p className="text-xs text-slate-600 border border-slate-200 rounded-md px-2 py-1 bg-slate-50">
            {adminDocsInfo}
          </p>
        )}
        <div className="space-y-1 border-t border-slate-100 pt-2">
          <p className="text-[11px] font-semibold text-slate-700">Notes</p>
          <ul className="list-disc pl-4 space-y-1 text-[11px] text-slate-600">
            <li>
              RAG mode is controlled from the Learner tab. When ON, answers will
              use your uploaded documents where relevant.
            </li>
            <li>
              For v0.8, the main focus is lesson flow &amp; lecture mode. RAG
              remains a supporting feature.
            </li>
          </ul>
        </div>
      </CardContent>
    </Card>
  );

  return (
    <div className="min-h-screen bg-slate-100 text-slate-900 px-3 py-4 md:px-6 md:py-6">
      <div className="mx-auto flex h-full max-w-6xl flex-col gap-4 rounded-3xl bg-white/90 p-4 shadow-xl ring-1 ring-slate-200 md:p-6">
        {/* Header */}
        <header className="flex flex-col gap-3 border-b border-slate-100 pb-3 md:flex-row md:items-center md:justify-between">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-2xl bg-slate-900 text-sm font-semibold text-white shadow-sm">
              MF
            </div>
            <div>
              <h1 className="text-base font-semibold md:text-lg">
                MentorFlow – Persona / Lesson Engine
              </h1>
              <p className="text-xs text-slate-500">
                Interactive lesson &amp; RAG teaching assistant
              </p>
            </div>
          </div>
          <div className="flex flex-wrap items-center gap-2 text-[11px]">
            <Badge className="bg-emerald-600 text-white hover:bg-emerald-700">
              v0.8 – Lesson Flow
            </Badge>
            <Badge variant="outline" className="font-mono">
              Backend: {backendBase}
            </Badge>
          </div>
        </header>

        {/* Main content */}
        <Tabs defaultValue="learner">
          <TabsList className="mb-3 grid w-full grid-cols-2">
            <TabsTrigger value="learner">Learner</TabsTrigger>
            <TabsTrigger value="admin">Admin</TabsTrigger>
          </TabsList>
          <TabsContent value="learner">{learnerTab}</TabsContent>
          <TabsContent value="admin">{adminTab}</TabsContent>
        </Tabs>
      </div>
    </div>
  );
}

export default App;
