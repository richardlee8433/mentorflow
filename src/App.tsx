import * as React from "react";
import { useEffect, useMemo, useState } from "react";

import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardDescription,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import {
  Tabs,
  TabsList,
  TabsTrigger,
  TabsContent,
} from "@/components/ui/tabs";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import { cn } from "@/lib/utils";

const API_BASE =
  import.meta.env.VITE_API_BASE ??
  "https://persona-lesson-v0-2-backend.onrender.com";

type Role = "user" | "assistant";

interface ChatMessage {
  id: string;
  role: Role;
  content: string;
}

interface RegionStat {
  region: string;
  users: number;
}

interface RegionMetricsResponse {
  regions: RegionStat[];
}

interface ChatResponse {
  reply: string;
  tts_base64?: string | null;
}

const TTS_KEY = "persona_tts_enabled";
const USER_KEY = "persona_user_id";
const REGION_KEY = "persona_region";

function getOrCreateUserId(): string {
  let v = localStorage.getItem(USER_KEY);
  if (!v) {
    v = "u_" + Math.random().toString(36).slice(2);
    localStorage.setItem(USER_KEY, v);
  }
  return v;
}

function detectRegion(): string {
  const stored = localStorage.getItem(REGION_KEY);
  if (stored) return stored;

  let region = "IE"; // default
  try {
    const tz = Intl.DateTimeFormat().resolvedOptions().timeZone || "";
    const lang = (navigator.language || "").toLowerCase();

    if (lang.startsWith("zh")) {
      region = "TW";
    } else if (tz === "Europe/London") {
      region = "UK";
    } else if (tz === "Europe/Dublin") {
      region = "IE";
    } else {
      region = "OTHER";
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

  async function sendMessage(raw: string) {
    const text = raw.trim();
    if (!text || loading) return;

    setLoading(true);
    const userMsg: ChatMessage = {
      id: crypto.randomUUID(),
      role: "user",
      content: text,
    };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");

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
    setInput("");
    sendMessage(cmd);
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

  useEffect(() => {
    if (activeTab === "admin" && regionStats.length === 0 && !metricsLoading) {
      loadRegionMetrics();
    }
  }, [activeTab, regionStats.length, metricsLoading]);

  const learnerMessages = messages;

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 flex flex-col">
      {/* Top Bar */}
      <header className="border-b border-slate-200 bg-white/95 backdrop-blur">
        <div className="mx-auto max-w-6xl px-5 py-3 flex items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="h-10 w-10 rounded-3xl bg-gradient-to-br from-emerald-400 to-cyan-400 flex items-center justify-center text-slate-950 font-semibold text-sm shadow-sm">
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

          <div className="flex items-center gap-4 text-xs">
            <Badge
              variant="outline"
              className="border-emerald-300 text-emerald-800 bg-emerald-50 px-3 py-1 rounded-full"
            >
              MVP v0.6
            </Badge>
            <Separator orientation="vertical" className="h-6 bg-slate-200" />
            <div className="flex flex-col items-end leading-tight text-[11px]">
              <span className="text-slate-700 font-medium">
                User:{" "}
                <span className="font-mono text-slate-900">
                  {userId}
                </span>
              </span>
              <span className="text-slate-500">
                Region:{" "}
                <span className="font-semibold text-slate-900">
                  {region}
                </span>
              </span>
            </div>
          </div>
        </div>
      </header>

      {/* Content */}
      <main className="flex-1">
        <div className="mx-auto max-w-6xl px-5 py-5">
          <Tabs
            value={activeTab}
            onValueChange={(v) => setActiveTab(v as "learner" | "admin")}
            className="space-y-5"
          >
            <TabsList className="bg-slate-100 border border-slate-200 rounded-full px-1.5 py-1 shadow-sm inline-flex">
              <TabsTrigger
                value="learner"
                className="rounded-full px-5 py-1.5 text-sm text-slate-600
                           data-[state=active]:bg-white data-[state=active]:text-slate-900
                           data-[state=active]:shadow-sm"
              >
                Learner View
              </TabsTrigger>
              <TabsTrigger
                value="admin"
                className="rounded-full px-5 py-1.5 text-sm text-slate-600
                           data-[state=active]:bg-white data-[state=active]:text-slate-900
                           data-[state=active]:shadow-sm"
              >
                Admin / Metrics
              </TabsTrigger>
            </TabsList>

            {/* Learner View */}
            <TabsContent value="learner" className="space-y-5">
              <div className="grid gap-5 lg:grid-cols-[2.1fr,1.1fr]">
                {/* Chat / Lesson */}
                <Card className="rounded-3xl border border-slate-200 bg-white shadow-sm">
                  <CardHeader className="pb-4">
                    <CardTitle className="flex items-center justify-between">
                      <span className="text-lg text-slate-900">
                        Lesson &amp; Chat
                      </span>
                      <span className="text-xs font-normal text-slate-500">
                        Type freely or use quick actions.
                      </span>
                    </CardTitle>
                    <CardDescription className="text-slate-500">
                      The assistant can guide you through PMP-style lessons or general
                      Q&amp;A.
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="flex flex-col gap-4">
                    {/* Quick actions */}
                    <div className="flex flex-wrap gap-2">
                      <Button
                        size="sm"
                        variant="outline"
                        className="rounded-full bg-emerald-50 border border-emerald-300 
                                   text-emerald-700 hover:bg-emerald-100 hover:border-emerald-400"
                        onClick={() => handleQuickCommand("start lesson 1")}
                      >
                        Start Lesson 1
                      </Button>
                      <Button
                        size="sm"
                        variant="outline"
                        className="rounded-full bg-cyan-50 border border-cyan-300 
                                   text-cyan-700 hover:bg-cyan-100 hover:border-cyan-400"
                        onClick={() => handleQuickCommand("start lesson 2")}
                      >
                        Start Lesson 2
                      </Button>
                      <Button
                        size="sm"
                        variant="outline"
                        className="rounded-full bg-fuchsia-50 border border-fuchsia-300 
                                   text-fuchsia-700 hover:bg-fuchsia-100 hover:border-fuchsia-400"
                        onClick={() => handleQuickCommand("start roleplay")}
                      >
                        Start Role-play
                      </Button>
                    </div>

                    {/* Chat area */}
                    <div className="mt-1">
                      <Label className="text-xs text-slate-500 mb-1 block">
                        Conversation
                      </Label>
                      <div className="rounded-3xl border border-slate-200 bg-slate-50 overflow-hidden">
                        <ScrollArea className="h-[380px] px-4 py-4">
                          {learnerMessages.length === 0 ? (
                            <div className="text-xs text-slate-500 h-full flex items-center justify-center">
                              No messages yet. Try{" "}
                              <code className="px-1.5 py-0.5 rounded-full bg-white border border-slate-200 text-[11px] text-slate-800">
                                start lesson 1
                              </code>{" "}
                              to begin.
                            </div>
                          ) : (
                            <div className="flex flex-col gap-3">
                              {learnerMessages.map((m) => (
                                <MessageBubble key={m.id} message={m} />
                              ))}
                            </div>
                          )}
                        </ScrollArea>
                      </div>
                    </div>
                  </CardContent>
                </Card>

                {/* Right side: TTS + AI Reflection */}
                <div className="space-y-5">
                  <Card className="rounded-3xl border border-slate-200 bg-white shadow-sm">
                    <CardHeader className="pb-3">
                      <CardTitle className="text-sm text-slate-900">
                        Audio &amp; Status
                      </CardTitle>
                      <CardDescription className="text-xs text-slate-500">
                        Control voice playback and see your current session info.
                      </CardDescription>
                    </CardHeader>
                    <CardContent className="space-y-4">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-3">
                          <div className="h-7 w-7 rounded-full bg-slate-100 flex items-center justify-center text-xs text-emerald-700 border border-slate-200">
                            🔊
                          </div>
                          <div className="flex flex-col">
                            <span className="text-sm font-medium text-slate-900">
                              Voice feedback
                            </span>
                            <span className="text-[11px] text-slate-500">
                              Spoken responses using TTS.
                            </span>
                          </div>
                        </div>
                        <div className="flex items-center gap-3">
                          <Switch
                            id="tts-toggle"
                            checked={ttsEnabled}
                            onCheckedChange={(v) => setTTSEnabled(Boolean(v))}
                          />
                          <Badge
                            variant="outline"
                            className={cn(
                              "text-xs rounded-full px-3 py-1 border-slate-200 bg-slate-50 text-slate-600",
                              loading &&
                                "border-amber-300 text-amber-800 bg-amber-50"
                            )}
                          >
                            {loading ? "Thinking..." : "Idle"}
                          </Badge>
                        </div>
                      </div>

                      <Separator className="bg-slate-200" />

                      <p className="text-[11px] text-slate-500 leading-relaxed">
                        Turn this on when you want the AI coach to read the answers out
                        loud. Great for speaking practice or screen-free listening.
                      </p>
                    </CardContent>
                  </Card>

                  {/* AI Reflection placeholder */}
                  <Card className="rounded-3xl border border-slate-200 bg-white shadow-sm">
                    <CardHeader className="pb-3">
                      <CardTitle className="text-sm text-slate-900">
                        AI Reflection
                      </CardTitle>
                      <CardDescription className="text-xs text-slate-500">
                        Coming soon: let the AI coach summarize the key takeaways for
                        each unit.
                      </CardDescription>
                    </CardHeader>
                    <CardContent>
                      <div className="rounded-2xl border border-slate-200 bg-slate-50 px-3 py-3 text-xs text-slate-700 h-32 flex items-center text-left">
                        Planned feature for v0.7: after you answer a question, click a
                        button to generate a short reflection with key bullet points for
                        what you&apos;ve learned and what to watch out for.
                      </div>
                    </CardContent>
                  </Card>
                </div>
              </div>

              {/* Input row */}
              <Card className="rounded-3xl border border-slate-200 bg-white shadow-sm">
                <CardContent className="pt-4">
                  <div className="flex flex-col gap-3 md:flex-row md:items-center">
                    <Input
                      placeholder="Type your answer or question..."
                      value={input}
                      onChange={(e) => setInput(e.target.value)}
                      onKeyDown={(e) => {
                        if (e.key === "Enter" && !e.shiftKey) {
                          e.preventDefault();
                          sendMessage(input);
                        }
                      }}
                      disabled={loading}
                      className="md:flex-1 bg-slate-50 border-slate-200 text-slate-900 placeholder:text-slate-400 rounded-2xl"
                    />
                    <Button
                      onClick={() => sendMessage(input)}
                      disabled={loading || !input.trim()}
                      className="md:w-32 rounded-2xl bg-emerald-500 hover:bg-emerald-600 text-white font-semibold shadow-sm"
                    >
                      {loading ? "Sending..." : "Send"}
                    </Button>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>

            {/* Admin / Metrics */}
            <TabsContent value="admin">
              <div className="grid gap-5 lg:grid-cols-[1.6fr,1.1fr]">
                <Card className="rounded-3xl border border-slate-200 bg-white shadow-sm">
                  <CardHeader className="pb-3">
                    <CardTitle className="flex items-center justify-between text-slate-900">
                      <span>Region Metrics</span>
                      <Button
                        size="sm"
                        variant="outline"
                        className="border-slate-200 text-slate-700 bg-slate-50 rounded-full px-4 hover:bg-slate-100"
                        onClick={loadRegionMetrics}
                        disabled={metricsLoading}
                      >
                        {metricsLoading ? "Refreshing..." : "Refresh"}
                      </Button>
                    </CardTitle>
                    <CardDescription className="text-xs text-slate-500">
                      Count unique users per region based on their first request.
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    {regionStats.length === 0 ? (
                      <div className="text-xs text-slate-500">
                        No metrics yet. Trigger some chats from the learner view, then
                        refresh.
                      </div>
                    ) : (
                      <div className="grid grid-cols-2 gap-3">
                        {regionStats.map((r) => (
                          <div
                            key={r.region}
                            className="rounded-2xl border border-slate-200 bg-slate-50 px-3 py-3 flex flex-col gap-1 shadow-sm"
                          >
                            <div className="flex items-center justify-between">
                              <span className="text-[11px] text-slate-500">
                                Region
                              </span>
                              <Badge
                                variant="outline"
                                className="text-[10px] border-slate-300 text-slate-700 bg-white rounded-full px-2 py-0.5"
                              >
                                {r.region}
                              </Badge>
                            </div>
                            <div className="flex items-baseline justify-between mt-1.5">
                              <span className="text-2xl font-semibold text-slate-900">
                                {r.users}
                              </span>
                              <span className="text-[11px] text-slate-500">
                                unique users
                              </span>
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
                      Session Debug
                    </CardTitle>
                    <CardDescription className="text-xs text-slate-500">
                      Helpful while iterating on the backend.
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-2 text-xs text-slate-600">
                    <div className="flex justify-between gap-4">
                      <span className="text-slate-500">API Base</span>
                      <span className="ml-2 truncate max-w-[220px] text-right text-slate-900 font-mono text-[11px]">
                        {API_BASE}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-500">User ID</span>
                      <span className="text-slate-900 font-mono text-[11px]">
                        {userId}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-500">Region</span>
                      <span className="text-slate-900 font-semibold">
                        {region}
                      </span>
                    </div>
                    <Separator className="my-2 bg-slate-200" />
                    <p className="text-[11px] text-slate-500 leading-relaxed">
                      This panel is only visible in Admin view and is meant for PM / dev
                      iteration, not end users.
                    </p>
                  </CardContent>
                </Card>
              </div>
            </TabsContent>
          </Tabs>
        </div>
      </main>
    </div>
  );
}

function MessageBubble({ message }: { message: ChatMessage }) {
  const isUser = message.role === "user";

  return (
    <div
      className={cn(
        "flex gap-2 text-sm",
        isUser ? "justify-end" : "justify-start"
      )}
    >
      {!isUser && (
        <div className="mt-1 h-7 w-7 flex items-center justify-center rounded-full bg-emerald-50 text-[11px] text-emerald-700 border border-emerald-200">
          AI
        </div>
      )}

      <div
        className={cn(
          "max-w-[80%] rounded-3xl px-3.5 py-2.5 text-[13px] leading-relaxed border shadow-sm",
          isUser
            ? "bg-emerald-50 border-emerald-200 text-emerald-900"
            : "bg-white border-slate-200 text-slate-900"
        )}
        dangerouslySetInnerHTML={{ __html: message.content }}
      />

      {isUser && (
        <div className="mt-1 h-7 w-7 flex items-center justify-center rounded-full bg-slate-200 text-[11px] text-slate-800 border border-slate-300">
          You
        </div>
      )}
    </div>
  );
}
