import os
import re
import base64
import hashlib
import requests
from typing import Dict, List, Optional, Tuple

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

from lesson_content import (
    ROLEPLAY_SCENARIO,
    score_open_answer,
)
from lessons import LESSONS  # contains chapter1~5 loaded from JSON

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
app = FastAPI(title="Persona / MentorFlow v0.8 – Teaching API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 開發階段先放寬，之後要可以收緊
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
# 預設 voice：Rachel（通用 podcast 風格，可用環境變數覆蓋）
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
# 建議模型：multilingual v2
ELEVENLABS_MODEL_ID = os.getenv("ELEVENLABS_MODEL_ID", "eleven_multilingual_v2")

print("[CFG] ELEVENLABS_API_KEY set:", bool(ELEVENLABS_API_KEY))
print("[CFG] ELEVENLABS_VOICE_ID:", ELEVENLABS_VOICE_ID)

# =========================
# In-memory TTS cache
# =========================
TTS_CACHE: Dict[str, bytes] = {}

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
"""


def score_answer(
    material: str,
    key_points: List[str],
    learner_answer: str,
    min_points: int,
) -> Dict[str, str]:
    """
    Call OpenAI to semantically score the learner's answer against key points.
    """
    key_points_text = "\n".join(f"- {kp}" for kp in key_points)
    user_content = f"""
[Lesson Material]
{material}

[Key Points]
{key_points_text}

[Question]
What is the most important idea here?

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

    # Attempt to parse JSON-like structure
    match = re.search(r"\{.*\}", reply, re.DOTALL)
    if not match:
        return {
            "score": 0,
            "feedback": "I could not parse the scorer output. Let's try to focus on the key ideas again.",
            "raw_reply": reply,
        }

    try:
        data = eval(match.group(0), {"__builtins__": {}}, {})
        score = int(data.get("score", 0))
        feedback = str(data.get("feedback", "")).strip() or "Thanks for your answer."
    except Exception:
        score = 0
        feedback = "I could not parse your answer correctly, but let's review the key points again."

    # Clamp score to [0, 2]
    if score < 0:
        score = 0
    elif score > 2:
        score = 2

    return {
        "score": score,
        "feedback": feedback,
        "raw_reply": reply,
    }


# =========================
# TTS helper (ElevenLabs + OpenAI fallback)
# =========================

def synthesize_speech(text: str, voice: str = "alloy") -> bytes:
    """
    Synthesize speech from text with a simple in-memory mp3 cache.

    - 若有設定 ELEVENLABS_API_KEY → 優先使用 ElevenLabs TTS。
    - 若 ElevenLabs 失敗或沒設定 → fallback 到 OpenAI TTS (gpt-4o-mini-tts)。
    - 以 (provider + model + voice + text) 產生 cache key，重複文字只會扣一次費用。

    Returns:
        mp3 bytes; 若文字為空或 TTS 全部失敗則回傳 b""。
    """
    text = text.strip()
    if not text:
        return b""

    # ---- 決定 provider 並產生 cache key ----
    if ELEVENLABS_API_KEY and ELEVENLABS_VOICE_ID:
        provider = "elevenlabs"
        cache_key_src = f"{provider}|{ELEVENLABS_MODEL_ID}|{ELEVENLABS_VOICE_ID}|{voice}|{text}"
    else:
        provider = "openai"
        cache_key_src = f"{provider}|gpt-4o-mini-tts|{voice}|{text}"

    cache_key = hashlib.sha256(cache_key_src.encode("utf-8")).hexdigest()

    # ---- 先看 cache 有沒有 ----
    cached = TTS_CACHE.get(cache_key)
    if cached:
        return cached

    audio_bytes: bytes = b""

    # -------------------------
    # Primary: ElevenLabs TTS
    # -------------------------
    if provider == "elevenlabs":
        try:
            print("[TTS] Using ElevenLabs...")
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

            resp = requests.post(
                tts_url,
                headers=headers,
                json=payload,
                timeout=60,
            )

            if resp.ok and resp.content:
                audio_bytes = resp.content
                print("[TTS] ElevenLabs ok, bytes:", len(audio_bytes))
            else:
                print(f"[TTS] ElevenLabs error {resp.status_code}: {resp.text[:200]}")
        except Exception as exc:
            print(f"[TTS] ElevenLabs exception: {exc}")

    # -------------------------
    # Fallback: OpenAI TTS
    # -------------------------
    if not audio_bytes:
        print("[TTS] Falling back to OpenAI TTS")
        try:
            res = client.audio.speech.create(
                model="gpt-4o-mini-tts",
                voice=voice,
                input=text,
                response_format="mp3",
            )
            audio_bytes = res.content or b""
        except Exception as exc:
            print(f"[TTS ERROR] OpenAI fallback failed: {exc}")
            audio_bytes = b""

    # ---- 寫入 cache ----
    if audio_bytes:
        TTS_CACHE[cache_key] = audio_bytes

    return audio_bytes


def generate_spoken_script(title: str, key_points: List[str], concept_text: str) -> str:
    """
    Generate a podcast-style spoken lecture script.
    """
    points_text = "\n".join(key_points) if key_points else "None"

    prompt = f"""
You are a podcast host and an experienced AI product instructor.
Rewrite the following material into a natural, podcast-style spoken script.

Topic:
{title}

Key points to cover:
{points_text}

Concept explanation:
{concept_text}

Tone and Style:
- Warm, confident, slightly energetic — like a real podcast host.
- Short sentences. Natural rhythm.
- Use conversational transitions like:
  "Let’s break this down.",
  "Here’s where it gets interesting.",
  "You might think this is obvious, but…",
  "Now, why does this matter?"
- Allow small pauses using commas, ellipses (...), and occasional em-dash (—).
- Include light rhetorical questions.
- Avoid robotic or academic tone.

Structure:
1. A short narrative opening that sets the scene.
2. A smooth transition into the main idea.
3. Explanation with small examples or analogies.
4. Clarify one or two common misconceptions.
5. End with a clear, strong takeaway.

Constraints:
- No markdown.
- No bullet points.
- No headings.
- No list formatting.
- 230–320 words.
- Output must sound natural when read aloud by TTS.

Your goal:
Make the listener feel like they’re hearing a real human host explain an idea clearly and engagingly.
"""

    try:
        res = client.chat.completions.create(
            model="gpt-4.1",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.9,
        )
        script = res.choices[0].message.content or ""
        return script.strip()
    except Exception as exc:
        print(f"[LECTURE SCRIPT ERROR] {exc}")
        # fallback: use original concept text
        return concept_text.strip()


# =========================
# In-memory sessions
# =========================

SESSIONS: Dict[str, Dict] = {}


def get_session(user_id: str) -> Dict:
    """
    Get or create session for a user.
    """
    if user_id not in SESSIONS:
        SESSIONS[user_id] = {
            "mode": "chat",  # chat / lesson / roleplay / lecture
            "current_lesson": None,
            "current_unit_id": None,
            "correct_in_lesson": 0,
            "unlocked_lessons": ["chapter1"],
            "lecture_lesson": None,
            "lecture_segments": [],
            "lecture_index": 0,
            "roleplay_node": None,
            "roleplay_score": 0,
            "roleplay_done": False,
        }
    return SESSIONS[user_id]


# =========================
# Lesson access helpers
# =========================

def get_lesson(lesson_key: str) -> Dict:
    if lesson_key not in LESSONS:
        raise KeyError(f"Unknown lesson key: {lesson_key}")
    return LESSONS[lesson_key]


def get_first_unit(lesson_key: str) -> Dict:
    lesson = get_lesson(lesson_key)
    units = lesson["units"]
    # list-based structure (chapter JSON)
    if isinstance(units, list):
        return units[0]
    # dict-based fallback
    first_key = sorted(units.keys())[0]
    return units[first_key]


def get_unit(lesson_key: str, unit_id: str) -> Dict:
    lesson = get_lesson(lesson_key)
    units = lesson["units"]
    if isinstance(units, list):
        idx = int(unit_id)
        return units[idx]
    return units[unit_id]


def get_next_unit_id(lesson_key: str, current_unit_id: str) -> Optional[str]:
    lesson = get_lesson(lesson_key)
    units = lesson["units"]

    if isinstance(units, list):
        idx = int(current_unit_id)
        if idx + 1 < len(units):
            return str(idx + 1)
        return None

    keys = sorted(units.keys())
    idx = keys.index(current_unit_id)
    if idx + 1 < len(keys):
        return keys[idx + 1]
    return None


def render_key_points_html(key_points: List[str]) -> str:
    if not key_points:
        return ""
    return "<ul>" + "".join(f"<li>{kp}</li>" for kp in key_points) + "</ul>"


# =========================
# Lesson engine (Flow v2)
# =========================

def start_lesson(user_id: str, lesson_key: str) -> Tuple[str, str]:
    """
    Start a lesson for a given chapter key, resetting progress.
    """
    session = get_session(user_id)

    if lesson_key not in session["unlocked_lessons"]:
        return (
            f"This lesson is currently locked. "
            f"Complete previous chapters before starting {lesson_key}.",
            "locked",
        )

    lesson = get_lesson(lesson_key)
    first_unit = get_first_unit(lesson_key)

    session["mode"] = "lesson"
    session["current_lesson"] = lesson_key
    session["current_unit_id"] = "0"  # works for list-based units
    session["correct_in_lesson"] = 0

    key_points = first_unit.get("key_points") or []
    key_idea = key_points[0] if key_points else ""

    lines = [
        "🎓 Lesson Intro",
        f"You are starting: {lesson['title']}",
        "In this lesson, we will walk through key concepts step by step.",
        "",
        "✨ Key Idea",
        f"- {key_idea}",
        "",
        "📖 Concept",
        first_unit["material"],
        "",
        "❓ Check your understanding",
        first_unit["question"],
    ]

    reply = "\n".join(lines)
    return reply, "intro"


def _chunk_for_lecture(text: str, max_chars: int = 520) -> List[str]:
    """
    Split a long text into smaller chunks suitable for lecture-style TTS.
    Pure text splitting – no extra LLM calls.
    """
    words = text.split()
    segments: List[str] = []
    current: List[str] = []
    length = 0

    for w in words:
        wlen = len(w) + 1  # space
        if length + wlen > max_chars and current:
            segments.append(" ".join(current))
            current = [w]
            length = wlen
        else:
            current.append(w)
            length += wlen

    if current:
        segments.append(" ".join(current))

    return segments


def build_lecture_segments(lesson_key: str) -> List[str]:
    """
    Build a small set of 'spoken' lecture segments for the first unit of a lesson.

    v0.8 Podcast Mode:
    - We call generate_spoken_script(...) once to get a podcast-style script.
    - Then we chunk that script into smaller pieces for "next" navigation.
    - This keeps the flow similar to the old lecture mode, but the content
      is optimized for audio.
    """
    lesson = get_lesson(lesson_key)
    first_unit = get_first_unit(lesson_key)

    title = lesson.get("title", lesson_key)
    key_points = first_unit.get("key_points") or []
    concept_text = first_unit.get("material", "")

    # 1) Generate a spoken-style script for this lecture
    script = generate_spoken_script(title, key_points, concept_text)

    # 2) Split into manageable segments for TTS playback
    chunks = _chunk_for_lecture(script, max_chars=520)
    segments: List[str] = []
    total = len(chunks)

    for idx, chunk in enumerate(chunks, start=1):
        header = f"🎙️ Lecture — part {idx} of {total}" if total > 1 else "🎙️ Lecture"
        segments.append(f"{header}\n{chunk}")

    # 3) Final wrap-up segment
    wrap_lines = [
        "✅ That’s the end of this mini-lecture.",
        "Next, you can switch into interactive practice to check your understanding.",
        "When you are ready, type: start lesson 1, or ask any question about what you just heard.",
    ]
    segments.append("\n".join(wrap_lines))

    return segments


def start_lecture(user_id: str, lesson_key: str) -> Tuple[str, str]:
    """
    Start lecture mode for a given lesson.

    - Uses pre-built segments based on the first unit.
    """
    session = get_session(user_id)

    # unlock check – same as start_lesson
    if lesson_key not in session["unlocked_lessons"]:
        return (
            f"This lecture is locked. Complete previous chapters before starting {lesson_key}.",
            "locked",
        )

    segments = build_lecture_segments(lesson_key)

    session["mode"] = "lecture"
    session["lecture_lesson"] = lesson_key
    session["lecture_segments"] = segments
    session["lecture_index"] = 0

    first_segment = segments[0]
    reply_lines = [
        first_segment,
        "",
        '▶️ Type "next" when you want to continue the lecture, or ask a question anytime.',
        'You can also type "stop lesson" to exit back to general Q&A.',
    ]
    reply = "\n".join(reply_lines)
    return reply, "lecture"


def continue_lecture(user_id: str, user_message: str) -> Tuple[str, str]:
    """
    Continue the current lecture.

    - If the user types 'next' → move to the next segment.
    - If the user asks a question → answer using general chat, but stay in lecture mode.
    """
    session = get_session(user_id)

    lesson_key = session.get("lecture_lesson")
    segments = session.get("lecture_segments") or []
    idx = int(session.get("lecture_index", 0))

    if not lesson_key or not segments:
        # Fallback: nothing to continue
        session["mode"] = "chat"
        session["lecture_lesson"] = None
        session["lecture_segments"] = []
        session["lecture_index"] = 0
        return (
            "There is no active lecture. Type `start lecture 1` to begin the first lecture.",
            "chat",
        )

    text = user_message.strip()
    lowered = text.lower()

    # If the user explicitly asks for the next chunk
    if lowered in ["next", "continue", "n"]:
        idx += 1
        if idx >= len(segments):
            # End of lecture
            session["mode"] = "chat"
            session["lecture_lesson"] = None
            session["lecture_segments"] = []
            session["lecture_index"] = 0
            return (
                "✅ That’s the end of the lecture. You can now type `start lesson 1` "
                "to practice interactively, or ask follow-up questions.",
                "lecture",
            )
        session["lecture_index"] = idx
        reply_lines = [
            segments[idx],
            "",
            '▶️ Type "next" for the following part, or `stop lesson` to exit.',
        ]
        reply = "\n".join(reply_lines)
        return reply, "lecture"

    # Otherwise, treat as a question during lecture
    messages = [
        {
            "role": "system",
            "content": (
                "You are an AI instructor answering questions during a lecture. "
                f"Current lesson: {lesson_key}. Respond concisely and clearly."
            ),
        },
        {"role": "user", "content": user_message},
    ]
    res = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=messages,
    )
    reply = res.choices[0].message.content or ""
    return reply, "lecture"


def continue_lesson(user_id: str, user_message: str) -> Tuple[str, str]:
    """
    Handle learner's answer for the current lesson unit and progress to next unit.
    """
    session = get_session(user_id)
    lesson_key = session.get("current_lesson")
    unit_id = session.get("current_unit_id")

    if not lesson_key or unit_id is None:
        session["mode"] = "chat"
        return (
            "You’re not currently in a lesson. Type `start lesson 1` to begin Chapter 1.",
            "chat",
        )

    try:
        unit = get_unit(lesson_key, unit_id)
    except KeyError:
        session["mode"] = "chat"
        session["current_lesson"] = None
        session["current_unit_id"] = None
        return (
            "Current lesson unit not found. Please restart the lesson.",
            "error",
        )

    result = score_answer(
        material=unit["material"],
        key_points=unit["key_points"],
        learner_answer=user_message,
        min_points=unit["min_points"],
    )
    score = int(result["score"])
    session["correct_in_lesson"] = int(session.get("correct_in_lesson", 0)) + score

    feedback = result["feedback"] or "Thanks for your answer."
    key_points_text = render_key_points_html(unit["key_points"])

    response_parts: List[str] = []

    response_parts.append("📘 Your Answer Review")
    response_parts.append(feedback)
    response_parts.append("")
    response_parts.append("💡 Key Points to Remember")
    response_parts.append(key_points_text)
    response_parts.append("")

    # Determine if there is a next unit
    next_unit_id = get_next_unit_id(lesson_key, unit_id)

    if next_unit_id is None:
        # Lesson completed
        response_parts.append("🎉 You’ve reached the end of this lesson.")

        # Unlock next lesson if defined
        next_lesson = LESSON_PROGRESSION.get(lesson_key)
        if next_lesson and next_lesson not in session["unlocked_lessons"]:
            session["unlocked_lessons"].append(next_lesson)
            response_parts.append(
                f"🔓 Next lesson unlocked: {next_lesson}. "
                f"Type `start lesson {next_lesson[-1]}` to continue."
            )

        session["mode"] = "chat"
        session["current_lesson"] = None
        session["current_unit_id"] = None

        reply = "\n".join(response_parts)
        return reply, "reasoning"

    # Move to the next unit
    session["current_unit_id"] = next_unit_id
    next_unit = get_unit(lesson_key, next_unit_id)

    response_parts.append("-----")
    response_parts.append("📖 Next Concept")
    response_parts.append(next_unit["material"])
    response_parts.append("")
    response_parts.append("❓ Check your understanding")
    response_parts.append(next_unit["question"])

    reply = "\n".join(response_parts)
    return reply, "reasoning"


def start_lesson1(user_id: str) -> Tuple[str, str]:
    """
    Shortcut to start Chapter 1 (AI PM – Tokens, Embeddings, Context Windows).
    """
    return start_lesson(user_id, "chapter1")


def start_lesson2(user_id: str) -> Tuple[str, str]:
    """
    Shortcut to start Chapter 2 (Prompting Fundamentals).
    """
    return start_lesson(user_id, "chapter2")


# =========================
# Role-play engine
# =========================

def start_roleplay(user_id: str) -> Tuple[str, str]:
    """
    Start the role-play scenario.
    """
    session = get_session(user_id)
    session["mode"] = "roleplay"
    session["roleplay_node"] = "intro"
    session["roleplay_score"] = 0
    session["roleplay_done"] = False

    first = next(n for n in ROLEPLAY_SCENARIO["nodes"] if n["id"] == "intro")
    intro_text = f"🎭 Role-play: {ROLEPLAY_SCENARIO['title']}\n\n" + first["npc"]
    return intro_text, "roleplay"


def handle_roleplay(user_id: str, user_message: str) -> Tuple[str, str]:
    """
    Handle one turn of role-play.
    """
    session = get_session(user_id)
    if session.get("roleplay_done"):
        return (
            "This session is already completed. Type `start roleplay` to begin a new one.",
            "roleplay",
        )

    node_id = session.get("roleplay_node") or "intro"
    node = next(n for n in ROLEPLAY_SCENARIO["nodes"] if n["id"] == node_id)

    # Terminal node
    if node.get("type") == "terminal":
        if user_message.strip().lower() in ["finish", "done", "end"]:
            session["roleplay_done"] = True
            passed = session["roleplay_score"] >= ROLEPLAY_SCENARIO["passing_score"]
            text = (
                f"🎯 Completed! Score {session['roleplay_score']}/"
                f"{ROLEPLAY_SCENARIO['passing_score']} — "
            ) + (
                "✅ Passed!"
                if passed
                else "❌ Not passed. Type `start roleplay` to try again."
            )
            return text, "roleplay"
        return ("Type `finish` to end and calculate your score.", "roleplay")

    feedback_lines: List[str] = []

    # Score choices if any
    for rule in node.get("scoring_rules", []):
        pattern = rule["pattern"].lower()
        if pattern in user_message.lower():
            session["roleplay_score"] += rule["score_delta"]
            feedback_lines.append(rule["feedback"])

    # Next node
    next_id = node.get("next_id")
    if not next_id:
        # No next node => mark as done
        session["roleplay_done"] = True
        passed = session["roleplay_score"] >= ROLEPLAY_SCENARIO["passing_score"]
        text = (
            f"🎯 Completed! Score {session['roleplay_score']}/"
            f"{ROLEPLAY_SCENARIO['passing_score']} — "
        ) + (
            "✅ Passed!"
            if passed
            else "❌ Not passed. Type `start roleplay` to try again."
        )
        return text, "roleplay"

    next_node = next(n for n in ROLEPLAY_SCENARIO["nodes"] if n["id"] == next_id)
    session["roleplay_node"] = next_id

    text_parts = []
    if feedback_lines:
        text_parts.append("📝 Feedback on your response:")
        text_parts.extend(feedback_lines)
        text_parts.append("")

    text_parts.append(next_node["npc"])
    text = "\n".join(text_parts)
    return text, "roleplay"


# =========================
# Core chat router
# =========================

def core_chat(user_id: str, user_message: str) -> Tuple[str, Optional[str]]:
    """
    Route user messages to:
    - commands (start lesson / roleplay / lecture / stop lesson)
    - lesson engine (Flow v2 + reasoning)
    - role-play engine
    - lecture engine (podcast-style)
    - general chat
    """
    session = get_session(user_id)
    text = user_message.strip()
    lowered = text.lower()

    # -------- Global stop command --------
    if lowered in ["stop lesson", "exit lesson", "end lesson"]:
        session.update(
            {
                "mode": "chat",
                "current_lesson": None,
                "current_unit_id": None,
                "correct_in_lesson": 0,
                "lecture_lesson": None,
                "lecture_segments": [],
                "lecture_index": 0,
                "roleplay_node": None,
                "roleplay_done": False,
            }
        )
        return "🛑 Lesson / lecture mode ended. Back to general Q&A.", "chat"

    # -------- Explicit commands --------
    # Role-play
    if lowered in ["start roleplay", "start role-play"]:
        return start_roleplay(user_id)

    # Lecture
    if lowered in ["start lecture 1", "start lecture1", "lecture 1"]:
        return start_lecture(user_id, "chapter1")
    if lowered in ["start lecture 2", "start lecture2", "lecture 2"]:
        return start_lecture(user_id, "chapter2")

    # Lessons
    if lowered in ["start lesson 1", "start lesson1"]:
        return start_lesson1(user_id)
    if lowered in ["start lesson 2", "start lesson2"]:
        return start_lesson2(user_id)

    # -------- Mode dispatch (stateful) --------
    mode = session.get("mode", "chat")
    if mode == "roleplay":
        return handle_roleplay(user_id, user_message)
    if mode == "lecture":
        return continue_lecture(user_id, user_message)
    if mode == "lesson":
        return continue_lesson(user_id, user_message)

    # -------- Default general chat --------
    messages = [
        {"role": "system", "content": "You are a helpful teaching assistant."},
        {"role": "user", "content": user_message},
    ]
    res = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=messages,
    )
    reply = res.choices[0].message.content
    return reply, "chat"


# =========================
# Schemas & Routes
# =========================

class ChatRequest(BaseModel):
    user_id: str
    message: str


class ChatResponse(BaseModel):
    reply: str
    tts_base64: Optional[str] = None
    # v0.8: turn_type for flow-aware UI (intro / reasoning / roleplay / chat / lecture / etc.)
    turn_type: Optional[str] = None


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest) -> ChatResponse:
    """
    Main chat endpoint.

    - Routes the message through core_chat().
    - Generates TTS audio for the (HTML-stripped) reply.
    """
    # 1) Route to core engine
    reply, turn_type = core_chat(req.user_id, req.message)

    # 2) Prepare plain text for TTS (strip HTML tags)
    tts_text = re.sub(r"<.*?>", " ", reply)
    tts_text = re.sub(r"\s+", " ", tts_text).strip()

    # 3) Call TTS
    audio_bytes = synthesize_speech(tts_text) if tts_text else b""
    tts_base64 = base64.b64encode(audio_bytes).decode("utf-8") if audio_bytes else None

    return ChatResponse(reply=reply, tts_base64=tts_base64, turn_type=turn_type)


# ---------- 新增：給 PlayCanvas 3D Mentor 用的精簡版 endpoint ----------

class MentorChatRequest(BaseModel):
    message: str
    user_id: Optional[str] = None  # 不傳就用預設 demo user


class MentorChatResponse(BaseModel):
    reply: str


@app.post("/api/mentor-chat", response_model=MentorChatResponse)
def mentor_chat_endpoint(req: MentorChatRequest) -> MentorChatResponse:
    """
    Endpoint for 3D Mentor (PlayCanvas / VIVERSE).

    Request:
      { "message": "Hello from 3D world" }

    Optional:
      { "message": "...", "user_id": "player-123" }

    Response:
      { "reply": "..." }  # 已經去掉 HTML，適合直接顯示在 Text 元件
    """
    user_id = req.user_id or "npc-demo"

    reply, turn_type = core_chat(user_id, req.message)

    # 去掉 HTML，避免 3D UI 出現 <ul><li> 之類標籤
    plain = re.sub(r"<.*?>", " ", reply)
    plain = re.sub(r"\s+", " ", plain).strip()

    if not plain:
        plain = reply  # 萬一整個被清空，就退回原始文字

    return MentorChatResponse(reply=plain)


@app.get("/")
def root() -> Dict[str, str]:
    """
    Health check endpoint.
    """
    return {"status": "ok", "message": "Persona / MentorFlow v0.8 Teaching API running"}


@app.get("/health")
def health_check():
    """
    Simple health endpoint for uptime checks.
    """
    return {
        "status": "ok",
        "service": "mentorflow-backend",
        "version": "v0.8",
        "region": os.getenv("MENTORFLOW_REGION", "local"),
    }


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
    """
    Simple progress report for the front-end status bar.
    """
    session = get_session(req.user_id)
    passed_roleplay = int(
        session["roleplay_score"] >= ROLEPLAY_SCENARIO["passing_score"]
    )

    # Rough heuristic: assume 5 units per lesson for now
    lesson_key = session.get("current_lesson")
    if lesson_key:
        lesson = get_lesson(lesson_key)
        units = lesson["units"]
        total_units = len(units) if isinstance(units, list) else len(units.keys())
        current_idx = int(session.get("current_unit_id") or 0)
        lesson_progress = int((current_idx + 1) / max(total_units, 1) * 100)
    else:
        lesson_progress = 0

    progress = min(
        100,
        lesson_progress
        + passed_roleplay * 20
        + len(session["unlocked_lessons"]) * 10,
    )

    return ReportResponse(
        mode=session["mode"],
        correct_in_lesson=session["correct_in_lesson"],
        unlocked_lessons=session["unlocked_lessons"],
        roleplay_score=session["roleplay_score"],
        progress_percent=progress,
    )


class ResetRequest(BaseModel):
    user_id: str


@app.post("/reset")
def reset_endpoint(req: ResetRequest) -> Dict[str, bool]:
    """
    Reset all progress for a user (lessons + role-play).
    """
    if req.user_id in SESSIONS:
        del SESSIONS[req.user_id]
    return {"ok": True}
