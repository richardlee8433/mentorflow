import os
import re
import base64
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

from lesson_content import (
    ROLEPLAY_SCENARIO,      # role-play state machine definition
    score_open_answer,      # simple keyword scoring for role-play open questions
)
from lessons import LESSONS as JSON_LESSONS  # JSON-based lesson configs (Lesson 4 & 5)
from services.rag_service import (
    retrieve_relevant_chunks,    # v0.7 RAG retrieval
    build_kb_from_txt_file,      # v0.7 RAG ingestion (TXT)
    build_kb_from_pdf_file,      # v0.7 RAG ingestion (PDF)
)


# Map external lesson keys (lesson1/lesson2) to JSON lesson configs (lesson4/lesson5)
LESSONS = {
    "lesson1": JSON_LESSONS.get("lesson4"),
    "lesson2": JSON_LESSONS.get("lesson5"),
}

# =========================
# FastAPI app
# =========================
app = FastAPI(title="MentorFlow v0.7 Teaching API")
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
# RAG settings (v0.7)
# =========================

# Global RAG enable flag (controlled via /admin/toggle-rag)
RAG_ENABLED: bool = False

# Simple RAG system prompt (can be moved to prompts/ file later)
RAG_SYSTEM_PROMPT = (
    "You are a helpful learning assistant. "
    "Answer only using the provided context. "
    "If the answer is not in the context, clearly say that the documents "
    "do not contain this information."
)

# Directory to store uploaded files for RAG KB building
UPLOAD_DIR = os.getenv("MENTORFLOW_UPLOAD_DIR", "uploaded_docs")

# =========================
# New scorer: semantic, key-point based
# =========================

SCORER_SYSTEM_PROMPT = """
You are a project management coach helping a learner understand PMP concepts.

You will receive:
- A short piece of learning material [Material]
- A list of key points that represent the important ideas for this question [KeyPoints]
- The learner's answer [Answer]
- A minimum number of distinct key points needed for a correct answer [MinPoints]

Your job:
1. Read the material and key points.
2. Compare the learner's answer with the key points using semantic understanding, not exact wording.
   - Treat paraphrases, synonyms, and simple examples as valid matches.
   - If an answer clearly expresses the idea of a key point, count it as covered even if the phrasing is different.
3. Count how many distinct key points are covered by the learner's answer.
4. If the number of key points covered is greater than or equal to [MinPoints], the answer passes (score +1). Otherwise, it does not pass (score +0).
5. Give short, coaching-style feedback:
   - First, briefly affirm what they got right or partially right.
   - Then, if they did not pass, hint at the missing idea(s) in simple language.
6. On the LAST LINE ONLY, output exactly:
   - "#SCORE:+1" for a passing answer
   - "#SCORE:+0" for a non-passing answer

Do NOT put anything after the #SCORE line.
""".strip()


def score_answer(material: str, key_points: List[str], learner_answer: str, min_points: int = 1) -> Dict[str, Any]:
    """
    Call LLM to score a single unit answer using semantic comparison to key points.
    Returns dict: {"score": 0/1, "feedback": "...", "raw_reply": "..."}.
    """
    import re as _re  # local to avoid confusion with global re usage above

    key_points_text = "\n".join(f"- {kp}" for kp in key_points)
    user_content = f"""[Material]
{material}

[KeyPoints]
{key_points_text}

[MinPoints]
{min_points}

[Answer]
{learner_answer}
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
    reply = res.choices[0].message.content.strip()

    lines = reply.splitlines()
    score_line = lines[-1].strip() if lines else ""
    score = 0
    if score_line == "#SCORE:+1":
        score = 1
    elif score_line == "#SCORE:+0":
        score = 0
    else:
        m = _re.search(r"#SCORE:\+([01])", score_line)
        if m:
            score = int(m.group(1))

    feedback = "\n".join(lines[:-1]).strip() if len(lines) > 1 else ""

    return {
        "score": score,
        "feedback": feedback,
        "raw_reply": reply,
    }


# =========================
# TTS helper (v0.5+)
# =========================

def synthesize_speech(text: str, voice: str = "alloy") -> bytes:
    """
    Use OpenAI TTS to synthesize speech from text.
    Returns raw audio bytes (mp3). If text is empty or TTS fails, returns b"".
    """
    text = text.strip()
    if not text:
        return b""

    try:
        # gpt-4o-mini-tts + mp3 output
        res = client.audio.speech.create(
            model="gpt-4o-mini-tts",
            voice=voice,
            input=text,
            response_format="mp3",
        )

        audio_bytes = res.content
        return audio_bytes or b""
    except Exception as e:
        print(f"[TTS ERROR] {e}")
        return b""


# =========================
# RAG helper functions (v0.7)
# =========================

def build_rag_messages(contexts: List[Dict[str, Any]], question: str) -> List[Dict[str, str]]:
    """
    Build chat messages for a RAG-style answer:
    - system: RAG behavior
    - user: context + question
    """
    context_blocks: List[str] = []
    for idx, ctx in enumerate(contexts, start=1):
        text = ctx.get("text", "")
        context_blocks.append(f"[Source {idx}]\n{text}")

    context_text = "\n\n".join(context_blocks)
    user_content = f"Context:\n{context_text}\n\nQuestion:\n{question}"

    return [
        {"role": "system", "content": RAG_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def call_rag_llm(messages: List[Dict[str, str]]) -> str:
    """
    Call the LLM for a RAG answer.
    """
    res = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=messages,
        temperature=0.2,
    )
    return res.choices[0].message.content.strip()


def format_rag_answer(raw_answer: str, contexts: List[Dict[str, Any]]) -> str:
    """
    Format the RAG answer for the front-end:
    - Convert newlines to <br>
    - Append a 📚 Sources block with file / page / chunk info
    """
    answer_html = raw_answer.replace("\n", "<br>")

    if not contexts:
        return answer_html

    source_lines: List[str] = []
    for idx, ctx in enumerate(contexts, start=1):
        metadata = ctx.get("metadata", {}) or {}
        file_name = metadata.get("file_name", "document")
        page = metadata.get("page")
        chunk_id = metadata.get("chunk_id")

        label = f"{idx}) {file_name}"
        if page:
            label += f" (p.{page})"
        if chunk_id:
            label += f" – {chunk_id}"
        source_lines.append(label)

    sources_html = "<br>".join(source_lines)
    return f"{answer_html}<br><br>📚 <b>Sources</b>:<br>{sources_html}"


def get_general_chat_reply(user_message: str) -> str:
    """
    Fallback/general chat without RAG.
    """
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_message},
    ]
    res = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=messages,
    )
    return res.choices[0].message.content


def answer_with_rag(user_id: str, user_message: str) -> str:
    # 1. retrieve chunks
    contexts = retrieve_relevant_chunks(
        question=user_message,
        top_k=3,
        score_threshold=0.20,
    )

    # 2. no matches
    if not contexts:
        return (
            "I could not find any relevant information in the uploaded documents "
            "to answer this question."
        )

    # 3. build messages (corrected argument order)
    messages = build_rag_messages(
        contexts=contexts,
        question=user_message,
    )

    # 4. call LLM
    raw_answer = call_rag_llm(messages)

    # 5. format with citations
    return format_rag_answer(raw_answer, contexts)




# =========================
# In-memory sessions
# =========================
SESSIONS: Dict[str, Dict] = {}


def get_session(user_id: str) -> Dict:
    if user_id not in SESSIONS:
        SESSIONS[user_id] = {
            # global
            "mode": "chat",
            "history": [],
            # lesson
            "current_lesson": None,     # "lesson1" or "lesson2"
            "current_unit_id": None,    # e.g., "lesson4_unit4_1"
            "correct_in_lesson": 0,
            "unlocked_lessons": ["lesson1"],
            # role-play
            "roleplay_node": None,
            "roleplay_score": 0,
            "roleplay_done": False,
        }
    return SESSIONS[user_id]


# =========================
# Lesson helpers (JSON-based)
# =========================

def get_lesson(lesson_key: str) -> Dict:
    lesson = LESSONS.get(lesson_key)
    if not lesson:
        raise KeyError(f"Lesson '{lesson_key}' not found.")
    return lesson


def get_unit(lesson_key: str, unit_id: str) -> Dict:
    lesson = get_lesson(lesson_key)
    for unit in lesson["units"]:
        if unit["id"] == unit_id:
            return unit
    raise KeyError(f"Unit '{unit_id}' not found in lesson '{lesson_key}'.")


def get_first_unit(lesson_key: str) -> Dict:
    lesson = get_lesson(lesson_key)
    return lesson["units"][0]


def start_lesson(user_id: str, lesson_key: str) -> str:
    s = get_session(user_id)

    # check unlock
    if lesson_key not in s["unlocked_lessons"]:
        return f"Lesson '{lesson_key}' is locked. Please complete previous lessons first."

    try:
        first_unit = get_first_unit(lesson_key)
    except KeyError:
        return f"Lesson '{lesson_key}' not found."

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


def continue_lesson(user_id: str, user_message: str) -> str:
    s = get_session(user_id)
    lesson_key = s.get("current_lesson")
    unit_id = s.get("current_unit_id")

    if not lesson_key or not unit_id:
        s["mode"] = "chat"
        return "No lesson is in progress. Type `start lesson 1` or `start lesson 2`."

    try:
        unit = get_unit(lesson_key, unit_id)
    except KeyError:
        s["mode"] = "chat"
        s["current_lesson"] = None
        s["current_unit_id"] = None
        return "Current lesson unit not found. Please restart the lesson."

    # Score this answer with semantic key-point scorer
    result = score_answer(
        material=unit["material"],
        key_points=unit["key_points"],
        learner_answer=user_message,
        min_points=unit["min_points"],
    )
    s["correct_in_lesson"] = int(s.get("correct_in_lesson", 0)) + result["score"]

    feedback = result["feedback"] or "Thanks for your answer."

    lesson = get_lesson(lesson_key)

    # Determine next unit (if any)
    next_unit_id = unit.get("next_unit")

    # ---- Structured HTML formatting ----

    # Format key points of current unit as HTML list
    key_points_formatted = "<ul>" + "".join(
        f"<li>{kp}</li>" for kp in unit["key_points"]
    ) + "</ul>"

    # Base feedback block
    response_text = (
        "📘 <b>Your Answer Review</b><br>"
        f"{feedback}<br><br>"
        "-----<br><br>"
        "<b>📌 Key Concepts from this Unit</b><br>"
        f"{key_points_formatted}<br>"
    )

    # If no next_unit → lesson complete
    if not next_unit_id:
        response_text += (
            "-----<br><br>"
            f"🎉 <b>You’ve completed {lesson_key}: {lesson['title']}!</b><br>"
        )
        # reset lesson state
        s["mode"] = "chat"
        s["current_lesson"] = None
        s["current_unit_id"] = None
        return response_text

    # Not last unit → prepare next
    s["current_unit_id"] = next_unit_id
    next_unit = get_unit(lesson_key, next_unit_id)

    # Next unit block
    response_text += (
        "-----<br><br>"
        f"➡️ <b>Next Topic: {next_unit['title']}</b><br><br>"
        f"📖 <b>Concept</b><br>{next_unit['material']}<br><br>"
        f"❓ <b>Question</b><br>{next_unit['question']}"
    )

    return response_text


def start_lesson1(user_id: str) -> str:
    return start_lesson(user_id, "lesson1")


def start_lesson2(user_id: str) -> str:
    return start_lesson(user_id, "lesson2")


# =========================
# Role-play engine (unchanged, still uses ROLEPLAY_SCENARIO)
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

    # Terminal node
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

    # ===== OPEN QUESTION TYPE =====
    if node["type"] == "open":
        gained = score_open_answer(user_message, node["key_points"])
        s["roleplay_score"] += gained

        if gained:
            feedback_lines.append("👍 Good reasoning.")
        else:
            feedback_lines.append("📌 You may have missed some key points.")
            if "key_points" in node and node["key_points"]:
                bullets = "\n".join([f"- {kp}" for kp in node["key_points"]])
                feedback_lines.append("Here are the key points to cover:\n" + bullets)
            if "model_answer" in node and node["model_answer"]:
                feedback_lines.append("A concise model answer:\n" + node["model_answer"])

        next_id = node["next"]

    # ===== CHOICE QUESTION TYPE =====
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

    # ===== OTHER NODE TYPES =====
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

    if text in ["start lesson 1", "start lesson1"]:
        return start_lesson1(user_id)

    if text in ["start lesson 2", "start lesson2"]:
        return start_lesson2(user_id)

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

    # Mode dispatch
    if s["mode"] == "roleplay":
        return handle_roleplay(user_id, user_message)

    if s["mode"] == "lesson":
        return continue_lesson(user_id, user_message)

    # Default general chat:
    # If RAG is enabled → use RAG pipeline, otherwise general chat
    if RAG_ENABLED:
        return answer_with_rag(user_id, user_message)

    return get_general_chat_reply(user_message)



# =========================
# Schemas & Routes
# =========================

class ChatRequest(BaseModel):
    user_id: str
    message: str


class ChatResponse(BaseModel):
    reply: str
    # v0.5+: AI audio (mp3) returned as base64 string; front-end can optionally play it
    tts_base64: Optional[str] = None


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest) -> ChatResponse:
    # 1) text reply
    reply = core_chat(req.user_id, req.message)

    # 2) build plain text for TTS (strip HTML tags)
    tts_text = re.sub(r"<.*?>", " ", reply)
    tts_text = re.sub(r"\s+", " ", tts_text).strip()

    # 3) call TTS
    audio_bytes = synthesize_speech(tts_text) if tts_text else b""
    tts_base64 = base64.b64encode(audio_bytes).decode("utf-8") if audio_bytes else None

    return ChatResponse(reply=reply, tts_base64=tts_base64)


class RagToggleRequest(BaseModel):
    enabled: bool


class RagToggleResponse(BaseModel):
    rag_enabled: bool

class UploadResponse(BaseModel):
    file_name: str
    file_type: str
    chunks_added: int

@app.post("/admin/upload", response_model=UploadResponse)
async def upload_file_endpoint(file: UploadFile = File(...)) -> UploadResponse:
    """
    Upload a TXT or PDF file and ingest it into the RAG knowledge base.

    - TXT: ingested as a single text blob, chunked by characters.
    - PDF: ingested page by page, each page chunked separately.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="File must have a filename.")

    filename = file.filename
    _, ext = os.path.splitext(filename)
    ext = ext.lower()

    if ext not in [".txt", ".pdf"]:
        raise HTTPException(
            status_code=400,
            detail="Only .txt and .pdf files are supported.",
        )

    # Ensure upload dir exists
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    save_path = os.path.join(UPLOAD_DIR, filename)

    # Save file to disk
    content = await file.read()
    with open(save_path, "wb") as f:
        f.write(content)

    # Build KB depending on file type
    if ext == ".txt":
        chunks_added = build_kb_from_txt_file(save_path)
        file_type = "txt"
    else:
        # ext == ".pdf"
        try:
            chunks_added = build_kb_from_pdf_file(save_path)
        except ImportError as exc:
            # pypdf 未安裝時，給明確錯誤訊息
            raise HTTPException(
                status_code=500,
                detail="PDF ingestion requires 'pypdf' to be installed on the server.",
            ) from exc
        file_type = "pdf"

    return UploadResponse(
        file_name=filename,
        file_type=file_type,
        chunks_added=chunks_added,
    )


@app.post("/admin/toggle-rag", response_model=RagToggleResponse)
def toggle_rag_endpoint(req: RagToggleRequest) -> RagToggleResponse:
    """
    Toggle global RAG mode on/off.

    This is used by the Admin UI to run A/B tests or debugging,
    as described in the v0.7 RAG PRD.
    """
    global RAG_ENABLED
    RAG_ENABLED = bool(req.enabled)
    return RagToggleResponse(rag_enabled=RAG_ENABLED)


@app.get("/")
def root() -> Dict[str, str]:
    return {"status": "ok", "message": "MentorFlow v0.7 Teaching API running"}


class ReportRequest(BaseModel):
    user_id: str


class ReportResponse(BaseModel):
    mode: str
    correct_in_lesson: int
    unlocked_lessons: List[str]
    roleplay_score: int
    progress_percent: int


@app.post("/report", response_model=ReportResponse)
def report_endpoint(req: ReportRequest) -> ReportResponse:
    s = get_session(req.user_id)
    passed_roleplay = int(s["roleplay_score"] >= ROLEPLAY_SCENARIO["passing_score"])
    # Rough progress model: base 20% + 10% per correct lesson answer + 40% if role-play is passed (capped at 100)
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
def reset_endpoint(req: ResetRequest) -> Dict[str, bool]:
    if req.user_id in SESSIONS:
        del SESSIONS[req.user_id]
    return {"ok": True}
