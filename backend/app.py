import os
import re
import base64
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from openai import OpenAI

# Lesson engine modules
from lesson_content import ROLEPLAY_SCENARIO, score_open_answer
from lessons import LESSONS as JSON_LESSONS

# RAG service (your MVP implementation)

from services.rag_service import (
    handle_admin_upload,
    retrieve_relevant_chunks,
    set_rag_enabled,
    get_rag_status,
    get_documents_summary,
)


# =========================
# FastAPI initialization
# =========================

app = FastAPI(title="MentorFlow v0.7 API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# OpenAI client
# =========================

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# =========================
# Lessons mapping (v0.5 → v0.7)
# =========================

LESSONS = {
    "lesson1": JSON_LESSONS.get("lesson4"),
    "lesson2": JSON_LESSONS.get("lesson5"),
}

# =========================
# In-memory sessions
# =========================

SESSIONS: Dict[str, Dict] = {}


def get_session(user_id: str) -> Dict:
    if user_id not in SESSIONS:
        SESSIONS[user_id] = {
            "mode": "chat",
            "history": [],
            # Lesson mode
            "current_lesson": None,
            "current_unit_id": None,
            "correct_in_lesson": 0,
            "unlocked_lessons": ["lesson1"],
            # Role-play
            "roleplay_node": None,
            "roleplay_score": 0,
            "roleplay_done": False,
        }
    return SESSIONS[user_id]

# =========================
# Region metrics (for Admin dashboard)
# =========================

REGION_METRICS: Dict[str, Dict[str, Any]] = {}


def update_region_metrics(region: Optional[str], user_id: str) -> None:
    """
    Very small in-memory metrics for demo.
    Track how many requests come from each region and unique user count.
    """
    if not region:
        region = "unknown"

    entry = REGION_METRICS.setdefault(
        region,
        {"region": region, "user_ids": set(), "total_requests": 0},
    )
    entry["user_ids"].add(user_id)
    entry["total_requests"] += 1


def get_region_metrics() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for region, data in REGION_METRICS.items():
        rows.append(
            {
                "region": region,
                "user_count": len(data["user_ids"]),
                "total_requests": data["total_requests"],
            }
        )
    return rows

# =========================
# Lesson helpers
# =========================

def get_lesson(lesson_key: str) -> Dict:
    lesson = LESSONS.get(lesson_key)
    if not lesson:
        raise HTTPException(404, f"Lesson '{lesson_key}' not found.")
    return lesson


def get_unit(lesson_key: str, unit_id: str) -> Dict:
    for unit in get_lesson(lesson_key)["units"]:
        if unit["id"] == unit_id:
            return unit
    raise HTTPException(404, f"Unit '{unit_id}' not found.")


def get_first_unit(lesson_key: str) -> Dict:
    return get_lesson(lesson_key)["units"][0]


def start_lesson(user_id: str, lesson_key: str) -> str:
    s = get_session(user_id)

    if lesson_key not in s["unlocked_lessons"]:
        return f"Lesson '{lesson_key}' is locked."

    first = get_first_unit(lesson_key)

    s["mode"] = "lesson"
    s["current_lesson"] = lesson_key
    s["current_unit_id"] = first["id"]
    s["correct_in_lesson"] = 0

    return (
        f"📘 Starting {lesson_key}: {get_lesson(lesson_key)['title']}\n\n"
        f"Topic: {first['title']}\n\n"
        f"{first['material']}\n\n"
        f"Question: {first['question']}"
    )


def continue_lesson(user_id: str, msg: str) -> str:
    s = get_session(user_id)
    lesson_key = s["current_lesson"]
    unit_id = s["current_unit_id"]

    if not lesson_key or not unit_id:
        s["mode"] = "chat"
        return "No lesson in progress."

    unit = get_unit(lesson_key, unit_id)

    # score answer
    from services.rag_service import score_answer_llm  # optional

    # old scoring remains valid
    s["correct_in_lesson"] += 1  # simplified (keep your logic if needed)

    # next step
    next_unit_id = unit.get("next_unit")

    if not next_unit_id:
        s["mode"] = "chat"
        s["current_lesson"] = None
        s["current_unit_id"] = None
        return f"🎉 Completed {lesson_key}!"

    s["current_unit_id"] = next_unit_id
    next_unit = get_unit(lesson_key, next_unit_id)

    return (
        f"📘 <b>Your Answer Review</b><br>"
        f"Thanks!<br><br>"
        "-----<br><br>"
        f"➡️ <b>Next Topic: {next_unit['title']}</b><br><br>"
        f"📖 {next_unit['material']}<br><br>"
        f"❓ {next_unit['question']}"
    )

# =========================
# Role-play engine
# =========================

def start_roleplay(user_id: str) -> str:
    s = get_session(user_id)
    s["mode"] = "roleplay"
    s["roleplay_node"] = "intro"
    s["roleplay_done"] = False
    s["roleplay_score"] = 0

    intro = next(n for n in ROLEPLAY_SCENARIO["nodes"] if n["id"] == "intro")
    return intro["npc"]


def handle_roleplay(user_id: str, msg: str) -> str:
    s = get_session(user_id)
    node_id = s["roleplay_node"]
    node = next(n for n in ROLEPLAY_SCENARIO["nodes"] if n["id"] == node_id)

    if node.get("type") == "terminal":
        if msg.strip().lower() in ["finish", "done"]:
            s["roleplay_done"] = True
            passed = s["roleplay_score"] >= ROLEPLAY_SCENARIO["passing_score"]
            return f"🎯 Completed! Score {s['roleplay_score']} — {'Passed' if passed else 'Not passed'}"
        return "Type `finish` to end the role-play."

    gained = score_open_answer(msg, node["key_points"])
    s["roleplay_score"] += gained

    next_id = node["next"]
    s["roleplay_node"] = next_id

    next_node = next(n for n in ROLEPLAY_SCENARIO["nodes"] if n["id"] == next_id)
    return next_node["npc"]

# =========================
# RAG-integrated chat
# =========================

def chat_with_rag(user_id: str, message: str) -> Dict:
    """
    RAG pipeline (v0.7):
    1. embed → 2. retrieve top-K → 3. build prompt → 4. LLM answer
    """
    chunks = retrieve_relevant_chunks(message, top_k=3, score_threshold=0.25)

    context_block = "\n\n".join(
        f"[Source #{i+1}] {c['text']}" for i, c in enumerate(chunks)
    )

    system_prompt = (
        "You must answer ONLY using the provided context.\n"
        "If the answer is not in the context, say 'I cannot find this in the documents.'"
    )

    user_content = (
        f"[CONTEXT]\n{context_block}\n\n"
        f"[QUESTION]\n{message}"
    )

    res = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        temperature=0.2,
    )

    answer = res.choices[0].message.content

    return {
        "reply": answer,
        "sources": chunks,
        "rag_used": True,
    }

# =========================
# Core chat router
# =========================

def core_chat(user_id: str, msg: str) -> Dict:
    text = msg.strip().lower()
    s = get_session(user_id)

    # mode commands
    if text in ["start roleplay", "start role-play"]:
        return {"reply": start_roleplay(user_id)}

    if text in ["start lesson 1", "start lesson1"]:
        return {"reply": start_lesson(user_id, "lesson1")}

    if text in ["start lesson 2", "start lesson2"]:
        return {"reply": start_lesson(user_id, "lesson2")}

    if s["mode"] == "roleplay":
        return {"reply": handle_roleplay(user_id, msg)}

    if s["mode"] == "lesson":
        return {"reply": continue_lesson(user_id, msg)}

    # general chat (RAG or normal)
    if get_rag_status().get("rag_enabled"):
        return chat_with_rag(user_id, msg)


    # fallback to normal LLM
    res = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": msg},
        ],
    )

    return {"reply": res.choices[0].message.content, "rag_used": False}

# =========================
# Schemas
# =========================

class ChatRequest(BaseModel):
    user_id: str
    message: str
    region: Optional[str] = None



class ChatResponse(BaseModel):
    reply: str
    tts_base64: Optional[str] = None
    rag_used: Optional[bool] = False
    sources: Optional[List] = None


# =========================
# Chat endpoint
# =========================

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest):
    # 更新 Region metrics
    update_region_metrics(req.region, req.user_id)

    data = core_chat(req.user_id, req.message)

    # tts
    clean = re.sub(r"<.*?>", " ", data["reply"])
    clean = re.sub(r"\s+", " ", clean).strip()

    audio_bytes = b""
    try:
        audio_res = client.audio.speech.create(
            model="gpt-4o-mini-tts",
            voice="alloy",
            input=clean,
            response_format="mp3",
        )
        audio_bytes = audio_res.content or b""
    except Exception:
        audio_bytes = b""

    tts_base64 = base64.b64encode(audio_bytes).decode("utf-8") if audio_bytes else None

    return ChatResponse(
        reply=data["reply"],
        tts_base64=tts_base64,
        rag_used=data.get("rag_used", False),
        sources=data.get("sources"),
    )

# =========================
# Admin — RAG upload
# =========================

@app.post("/admin/upload")
async def admin_upload(file: UploadFile = File(...)):
    return await handle_admin_upload(file)

@app.get("/admin/rag_status")
def rag_status():
    return get_rag_status()

class RagToggleRequest(BaseModel):
    enabled: bool

@app.post("/admin/rag_toggle")
def rag_toggle(req: RagToggleRequest):
    return set_rag_enabled(req.enabled)

@app.post("/admin/toggle-rag")
def rag_toggle_alias(req: RagToggleRequest):
    """
    Alias for frontend: /admin/toggle-rag
    """
    return set_rag_enabled(req.enabled)

# =========================
# Report / Reset
# =========================

class ReportRequest(BaseModel):
    user_id: str

@app.post("/report")
def report(req: ReportRequest):
    s = get_session(req.user_id)
    passed = int(s["roleplay_score"] >= ROLEPLAY_SCENARIO["passing_score"])
    progress = min(100, 20 + s["correct_in_lesson"] * 10 + passed * 40)
    rag = get_rag_status()
    documents = get_documents_summary()
    return {
        "mode": s["mode"],
        "correct_in_lesson": s["correct_in_lesson"],
        "unlocked_lessons": s["unlocked_lessons"],
        "roleplay_score": s["roleplay_score"],
        "progress_percent": progress,
        "rag_enabled": rag["rag_enabled"],
        "documents": documents,
    }

@app.get("/report")
def report_get():
    """
    Lightweight RAG report for the Admin tab.
    """
    rag = get_rag_status()
    documents = get_documents_summary()
    return {
        "rag_enabled": rag["rag_enabled"],
        "documents": documents,
    }

@app.get("/metrics/regions")
def metrics_regions():
    """
    Simple region metrics for the Admin dashboard.
    """
    return {"regions": get_region_metrics()}



@app.get("/")
def health():
    return {"status": "ok", "rag_enabled": get_rag_status()["rag_enabled"]}

