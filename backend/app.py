import os
import re
import base64
import hashlib
import time
import requests
from typing import Dict, List, Optional, Any

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from openai import OpenAI

from lesson_content import (
    ROLEPLAY_SCENARIO,
    score_open_answer,
)
from lessons import LESSONS  # contains chapter1~5 loaded from JSON

# ✅ v0.92: Decision-based evaluation
from eval.evaluator_v3 import run_eval

# Lesson progression map for unlock logic
LESSON_PROGRESSION = {
    "chapter1": "chapter2",
    "chapter2": "chapter3",
    "chapter3": "chapter4",
    "chapter4": "chapter5",
}

# =========================
# FastAPI app
# =========================
app = FastAPI(title="MentorFlow v0.92 – Teaching API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # dev: open; prod: tighten
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# OpenAI client
# =========================
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# =========================
# ElevenLabs configuration
# =========================
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
ELEVENLABS_MODEL_ID = os.getenv("ELEVENLABS_MODEL_ID", "eleven_multilingual_v2")

print("[CFG] ELEVENLABS_API_KEY set:", bool(ELEVENLABS_API_KEY))
print("[CFG] ELEVENLABS_VOICE_ID:", ELEVENLABS_VOICE_ID)

# =========================
# In-memory TTS cache
# =========================
TTS_CACHE: Dict[str, bytes] = {}

# =========================
# In-memory sessions
# =========================
SESSIONS: Dict[str, Dict[str, Any]] = {}


def get_session(user_id: str) -> Dict[str, Any]:
    if user_id not in SESSIONS:
        SESSIONS[user_id] = {
            "mode": "chat",
            "history": [],
            # lesson
            "current_lesson": None,
            "current_unit_id": None,
            "correct_in_lesson": 0,
            "unlocked_lessons": ["chapter1"],  # v0.8+ curriculum style
            # role-play
            "roleplay_node": None,
            "roleplay_score": 0,
            "roleplay_done": False,
        }
    return SESSIONS[user_id]


# =========================
# New scorer: semantic, key-point based (reasoning-friendly)
# =========================

SCORER_SYSTEM_PROMPT = """
You are a strict but supportive AI tutor who scores open-ended answers for
AI Product Management lessons.

Rules:
1. Score answers from 0 to 2.
2. Focus on whether the learner hits the key points.
3. If the answer is partially correct, give 1 point.
4. If the answer is clearly incorrect or off-topic, give 0 points.
5. Always explain in simple, constructive language:
   - What the learner got right.
   - What is missing or unclear.
   - Use concise, learner-friendly wording.
6. Your response must be JSON with keys: score, feedback.
""".strip()


def score_answer(
    material: str,
    key_points: List[str],
    learner_answer: str,
    min_points: int,
) -> Dict[str, Any]:
    key_points_text = "\n".join(f"- {kp}" for kp in key_points)
    user_content = f"""
[Lesson Material]
{material}

[Key Points]
{key_points_text}

[Learner Answer]
{learner_answer}

[Scoring Instructions]
You must decide how well the answer reflects the key points above.
Minimum key points for full score: {min_points}.
"""

    messages = [
        {"role": "system", "content": SCORER_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    res = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=messages,
        temperature=0.2,
    )
    reply = res.choices[0].message.content or ""

    match = re.search(r"\{.*\}", reply, re.DOTALL)
    if not match:
        return {
            "score": 0,
            "feedback": "I could not parse the scorer output. Let's refocus on the key ideas.",
            "raw_reply": reply,
        }

    try:
        data = eval(match.group(0), {"__builtins__": {}}, {})
        score = int(data.get("score", 0))
        feedback = str(data.get("feedback", "")).strip() or "Thanks for your answer."
    except Exception:
        score = 0
        feedback = "I could not parse the scorer output. Let's review the key points again."

    score = max(0, min(2, score))
    return {"score": score, "feedback": feedback, "raw_reply": reply}


# =========================
# TTS helper (ElevenLabs + fallback)
# =========================

def synthesize_speech(text: str) -> bytes:
    text = text.strip()
    if not text:
        return b""

    if ELEVENLABS_API_KEY and ELEVENLABS_VOICE_ID:
        provider = "elevenlabs"
        model_id = ELEVENLABS_MODEL_ID
        voice_id = ELEVENLABS_VOICE_ID
    else:
        provider = "openai"
        model_id = "gpt-4o-mini-tts"
        voice_id = "alloy"

    cache_key_src = f"{provider}|{model_id}|{voice_id}|{text}"
    cache_key = hashlib.sha256(cache_key_src.encode("utf-8")).hexdigest()
    cached = TTS_CACHE.get(cache_key)
    if cached:
        return cached

    audio_bytes: bytes = b""

    if provider == "elevenlabs":
        try:
            tts_url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}"
            headers = {
                "Accept": "audio/mpeg",
                "xi-api-key": ELEVENLABS_API_KEY,
                "Content-Type": "application/json",
            }
            payload = {
                "text": text,
                "model_id": ELEVENLABS_MODEL_ID,
                "voice_settings": {
                    "stability": 0.45,
                    "similarity_boost": 0.8,
                    "style": 0.3,
                    "use_speaker_boost": True,
                },
            }
            resp = requests.post(tts_url, headers=headers, json=payload, timeout=45)
            if resp.status_code == 200:
                audio_bytes = resp.content or b""
            else:
                print("[TTS] ElevenLabs error:", resp.status_code, resp.text[:200])
        except Exception as e:
            print("[TTS] ElevenLabs exception:", e)

    # Fallback: OpenAI TTS
    if not audio_bytes:
        try:
            res = client.audio.speech.create(
                model="gpt-4o-mini-tts",
                voice="alloy",
                input=text,
                response_format="mp3",
            )
            audio_bytes = res.content or b""
        except Exception as e:
            print("[TTS] OpenAI fallback error:", e)
            audio_bytes = b""

    if audio_bytes:
        TTS_CACHE[cache_key] = audio_bytes

    return audio_bytes


# =========================
# Lesson helpers (JSON-based)
# =========================

def get_lesson(lesson_key: str) -> Dict[str, Any]:
    lesson = LESSONS.get(lesson_key)
    if not lesson:
        raise KeyError(f"Lesson '{lesson_key}' not found.")
    return lesson


def get_unit(lesson_key: str, unit_id: str) -> Dict[str, Any]:
    lesson = get_lesson(lesson_key)
    for unit in lesson["units"]:
        if unit["id"] == unit_id:
            return unit
    raise KeyError(f"Unit '{unit_id}' not found in lesson '{lesson_key}'.")


def get_first_unit(lesson_key: str) -> Dict[str, Any]:
    lesson = get_lesson(lesson_key)
    return lesson["units"][0]


def start_lesson(user_id: str, lesson_key: str) -> str:
    s = get_session(user_id)

    if lesson_key not in s["unlocked_lessons"]:
        return f"Lesson '{lesson_key}' is locked. Please complete previous lessons first."

    first_unit = get_first_unit(lesson_key)

    s["mode"] = "lesson"
    s["current_lesson"] = lesson_key
    s["current_unit_id"] = first_unit["id"]
    s["correct_in_lesson"] = 0

    lesson = get_lesson(lesson_key)
    return (
        f"📘 Starting {lesson_key}: {lesson['title']}\n\n"
        f"Topic: {first_unit['title']}\n\n"
        f"{first_unit['material']}\n\n"
        f"Question: {first_unit['question']}"
    )


def maybe_unlock_next_lesson(s: Dict[str, Any], completed_lesson_key: str) -> None:
    next_lesson = LESSON_PROGRESSION.get(completed_lesson_key)
    if not next_lesson:
        return
    unlocked = set(s.get("unlocked_lessons", []))
    unlocked.add(next_lesson)
    s["unlocked_lessons"] = list(unlocked)


def continue_lesson(user_id: str, user_message: str) -> str:
    s = get_session(user_id)
    lesson_key = s.get("current_lesson")
    unit_id = s.get("current_unit_id")

    if not lesson_key or not unit_id:
        s["mode"] = "chat"
        return "No lesson is in progress. Type `start chapter1` (or other unlocked chapter)."

    unit = get_unit(lesson_key, unit_id)

    # Score the learner response
    result = score_answer(
        material=unit["material"],
        key_points=unit["key_points"],
        learner_answer=user_message,
        min_points=unit["min_points"],
    )
    s["correct_in_lesson"] = int(s.get("correct_in_lesson", 0)) + result["score"]

    feedback = result["feedback"] or "Thanks for your answer."
    lesson = get_lesson(lesson_key)
    next_unit_id = unit.get("next_unit")

    # HTML-ish formatting (kept from your v0.91 file)
    key_points_formatted = "<ul>" + "".join(f"<li>{kp}</li>" for kp in unit["key_points"]) + "</ul>"
    response_text = (
        "📘 <b>Your Answer Review</b><br>"
        f"{feedback}<br><br>"
        "-----<br><br>"
        "<b>📌 Key Concepts from this Unit</b><br>"
        f"{key_points_formatted}<br>"
    )

    if not next_unit_id:
        response_text += (
            "-----<br><br>"
            f"🎉 <b>You’ve completed {lesson_key}: {lesson['title']}!</b><br>"
        )
        # unlock next
        maybe_unlock_next_lesson(s, lesson_key)

        # reset state
        s["mode"] = "chat"
        s["current_lesson"] = None
        s["current_unit_id"] = None
        return response_text

    s["current_unit_id"] = next_unit_id
    next_unit = get_unit(lesson_key, next_unit_id)

    response_text += (
        "-----<br><br>"
        f"➡️ <b>Next Topic: {next_unit['title']}</b><br><br>"
        f"📖 <b>Concept</b><br>{next_unit['material']}<br><br>"
        f"❓ <b>Question</b><br>{next_unit['question']}"
    )
    return response_text


# =========================
# Role-play engine
# =========================

def start_roleplay(user_id: str) -> str:
    s = get_session(user_id)
    s["mode"] = "roleplay"
    s["roleplay_node"] = "intro"
    s["roleplay_score"] = 0
    s["roleplay_done"] = False

    first = next(n for n in ROLEPLAY_SCENARIO["nodes"] if n["id"] == "intro")
    return f"🎭 Role-play: {ROLEPLAY_SCENARIO['title']}\n\n" + first["npc"]


def handle_roleplay(user_id: str, user_message: str) -> str:
    s = get_session(user_id)
    if s.get("roleplay_done"):
        return "This session is already completed. Type `start roleplay` to begin a new one."

    node_id = s.get("roleplay_node") or "intro"
    node = next(n for n in ROLEPLAY_SCENARIO["nodes"] if n["id"] == node_id)

    if node.get("type") == "terminal":
        if user_message.strip().lower() in ["finish", "done", "end"]:
            s["roleplay_done"] = True
            passed = s["roleplay_score"] >= ROLEPLAY_SCENARIO["passing_score"]
            return (
                f"🎯 Completed! Score {s['roleplay_score']}/{ROLEPLAY_SCENARIO['passing_score']} — "
                + ("✅ Passed!" if passed else "❌ Not passed. Type `start roleplay` to try again.")
            )
        return "Type `finish` to end and calculate your score."

    feedback_lines: List[str] = []

    if node["type"] == "open":
        gained = score_open_answer(user_message, node["key_points"])
        s["roleplay_score"] += gained

        if gained:
            feedback_lines.append("👍 Good reasoning.")
        else:
            feedback_lines.append("📌 You may have missed some key points.")
            if node.get("key_points"):
                bullets = "\n".join([f"- {kp}" for kp in node["key_points"]])
                feedback_lines.append("Here are the key points to cover:\n" + bullets)
            if node.get("model_answer"):
                feedback_lines.append("A concise model answer:\n" + node["model_answer"])

        next_id = node["next"]

    elif node["type"] == "choice":
        chosen = None
        try:
            idx = int(user_message.strip()) - 1
            if 0 <= idx < len(node["choices"]):
                chosen = node["choices"][idx]
        except Exception:
            pass

        if not chosen:
            for c in node["choices"]:
                if c["label"].lower() in user_message.lower():
                    chosen = c
                    break

        if not chosen:
            opts = "\n".join([f"{i+1}) {c['label']}" for i, c in enumerate(node["choices"])])
            return f"Please select one option:\n{opts}"

        s["roleplay_score"] += chosen.get("score", 0)
        if "explain" in node:
            feedback_lines.append(node["explain"])
        next_id = chosen["next"]

    else:
        next_id = node["next"]

    s["roleplay_node"] = next_id
    next_node = next(n for n in ROLEPLAY_SCENARIO["nodes"] if n["id"] == next_id)

    feedback = ("\n".join(feedback_lines) + "\n") if feedback_lines else ""
    return feedback + next_node["npc"]


# =========================
# Core chat router
# =========================

def core_chat(user_id: str, user_message: str) -> str:
    s = get_session(user_id)
    text = user_message.strip().lower()

    # Commands
    if text in ["start roleplay", "start role-play"]:
        return start_roleplay(user_id)

    # v0.8+ chapters
    if text in ["start chapter1", "start chapter 1"]:
        return start_lesson(user_id, "chapter1")
    if text in ["start chapter2", "start chapter 2"]:
        return start_lesson(user_id, "chapter2")
    if text in ["start chapter3", "start chapter 3"]:
        return start_lesson(user_id, "chapter3")
    if text in ["start chapter4", "start chapter 4"]:
        return start_lesson(user_id, "chapter4")
    if text in ["start chapter5", "start chapter 5"]:
        return start_lesson(user_id, "chapter5")

    if text in ["stop lesson", "exit lesson", "end lesson"]:
        s.update(
            {
                "mode": "chat",
                "history": [],
                "current_lesson": None,
                "current_unit_id": None,
                "correct_in_lesson": 0,
            }
        )
        return "🛑 Lesson mode ended. Back to general Q&A."

    if s["mode"] == "roleplay":
        return handle_roleplay(user_id, user_message)

    if s["mode"] == "lesson":
        return continue_lesson(user_id, user_message)

    # Default general chat
    res = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_message},
        ],
    )
    return res.choices[0].message.content or ""


# =========================
# Schemas & Routes
# =========================

class ChatRequest(BaseModel):
    user_id: str
    message: str


class ChatResponse(BaseModel):
    reply: str
    tts_base64: Optional[str] = None


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest):
    reply = core_chat(req.user_id, req.message)

    # strip HTML for speech
    tts_text = re.sub(r"<.*?>", " ", reply)
    tts_text = re.sub(r"\s+", " ", tts_text).strip()

    audio_bytes = synthesize_speech(tts_text) if tts_text else b""
    tts_base64 = base64.b64encode(audio_bytes).decode("utf-8") if audio_bytes else None

    return ChatResponse(reply=reply, tts_base64=tts_base64)


@app.get("/")
def root():
    return {"status": "ok", "message": "MentorFlow v0.92 Teaching API running"}


class ReportRequest(BaseModel):
    user_id: str


class ReportResponse(BaseModel):
    mode: str
    correct_in_lesson: int
    unlocked_lessons: List[str]
    roleplay_score: int
    progress_percent: int


@app.post("/report", response_model=ReportResponse)
def report_endpoint(req: ReportRequest):
    s = get_session(req.user_id)
    passed_roleplay = int(s["roleplay_score"] >= ROLEPLAY_SCENARIO["passing_score"])
    progress = min(100, 20 + s["correct_in_lesson"] * 10 + passed_roleplay * 40)
    return ReportResponse(
        mode=s["mode"],
        correct_in_lesson=s["correct_in_lesson"],
        unlocked_lessons=s["unlocked_lessons"],
        roleplay_score=s["roleplay_score"],
        progress_percent=progress,
    )


class ResetRequest(BaseModel):
    user_id: str


@app.post("/reset")
def reset_endpoint(req: ResetRequest):
    if req.user_id in SESSIONS:
        del SESSIONS[req.user_id]
    return {"ok": True}


# =========================
# v0.92 Evaluation Routes
# =========================

class EvalRunRequest(BaseModel):
    dataset_path: str = Field(default="dataset_v1.json")
    rubric_path: str = Field(default="rubric_v2.json")
    output_path: str = Field(default="output_report_v0_92.json")
    # if true -> return full report including per-case results (large)
    include_results: bool = Field(default=False)


class EvalDecisionSummary(BaseModel):
    decision: str
    risk_level: str
    blocking_reasons: List[str]
    blocking_cases: List[str]
    notes: str


class EvalRunResponse(BaseModel):
    run_id: str
    mentorflow_version: str
    model_under_test: str
    judge_model: str
    started_at: str
    elapsed_seconds: float
    average_score: float
    num_cases: int
    decision_summary: EvalDecisionSummary
    report_path: str
    # optional: full report JSON (can be big)
    report: Optional[Dict[str, Any]] = None


@app.post("/eval/run", response_model=EvalRunResponse)
def eval_run_endpoint(req: EvalRunRequest):
    """
    Runs evaluator_v3 (Decision-based evaluation).
    Advisory output only (not a hard CI gate).
    """
    # Use evaluator_v3 runner
    report = run_eval(
        dataset_path=req.dataset_path,
        rubric_path=req.rubric_path,
        output_path=req.output_path,
    )

    resp = EvalRunResponse(
        run_id=report["run_id"],
        mentorflow_version=report["mentorflow_version"],
        model_under_test=report["model_under_test"],
        judge_model=report["judge_model"],
        started_at=report["started_at"],
        elapsed_seconds=float(report["elapsed_seconds"]),
        average_score=float(report["average_score"]),
        num_cases=int(report["num_cases"]),
        decision_summary=EvalDecisionSummary(**report["decision_summary"]),
        report_path=req.output_path,
        report=(report if req.include_results else None),
    )
    return resp


@app.get("/eval/report")
def eval_get_report(path: str = "output_report_v0_92.json"):
    """
    Convenience endpoint to fetch a saved eval report JSON file.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            return Response(content=f.read(), media_type="application/json")
    except Exception as e:
        return {"ok": False, "error": str(e), "path": path}
