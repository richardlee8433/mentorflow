# lesson_content.py

"""
Lesson content loader for Persona v0.2

- 從外部 .txt 檔載入教材
- LESSONS 字典現在有兩個 lesson：
  lesson1 → Public_Safe_Project_Methodology_Define.txt
  lesson2 → Safe_Public_Project_Charter.txt
"""

from pathlib import Path


def load_lesson(file_path: str) -> str:
    """讀取指定路徑的教材文字檔。"""
    path = Path(file_path)
    if not path.exists():
        # 方便除錯：如果找不到檔案，就回傳錯誤字串（也會被模型看到）
        return f"[ERROR] Lesson file not found: {file_path}"
    return path.read_text(encoding="utf-8")


# 這裡指定每一課要用哪一個 .txt 檔
LESSONS = {
    "lesson1": load_lesson("lesson4_interactive.txt"),
    "lesson2": load_lesson("lesson5_interactive.txt"),
}


# lesson_content.py

TEACHER_SYSTEM_PROMPT = """
You are an English-speaking teaching assistant for role-based lessons.
Always reply in English.
Use a concise, coaching tone.
For every turn:
1) Briefly acknowledge or correct the student's answer.
2) Ask exactly one follow-up question.
3) At the very end, append '#SCORE:+1' if the student's answer covers at least two key ideas from the lesson.
   Otherwise append '#SCORE:+0'.
"""

ROLEPLAY_SCENARIO = {
    "title": "Is this a project or operations?",
    "passing_score": 3,
    "nodes": [
        {
            "id": "intro",
            "type": "open",
            "npc": (
                "🎭 Role-play: Is this a project?\n"
                "Hi, I'm your line manager. We're considering upgrading our fulfillment system "
                "to a new cloud-based platform over the next 6 months.\n\n"
                "Do you think this upgrade is a PROJECT? Why?"
            ),
            "key_points": [
                "Temporary initiative with a defined beginning and end",
                "Creates a unique product/service/result (upgraded system)",
                "Drives change from current state to desired future state",
                "Has constraints such as scope, time, cost, and risk"
            ],
            "model_answer": (
                "Yes, it is a project because it is a temporary initiative with a defined time frame "
                "to deliver a unique change to our fulfillment system. "
                "It moves us from the current state to a desired future state under scope, schedule, "
                "and budget constraints, and requires project management processes to plan, execute, "
                "monitor, and close the work."
            ),
            "next": "business_value"
        },
        {
            "id": "business_value",
            "type": "open",
            "npc": (
                "💰 Good. Now think about *why* the company is doing this.\n"
                "From a business value perspective, what could be the main reason to start this project?"
            ),
            "key_points": [
                "Improving or fixing existing processes or services",
                "Responding to stakeholder/customer requests",
                "Taking advantage of technological advances",
                "Meeting regulatory, legal, or compliance needs"
            ],
            "model_answer": (
                "The main driver is business value. For example, the upgrade might improve efficiency, "
                "reduce errors, and support higher volume, which improves our service and profitability. "
                "It could also be triggered by stakeholder requests, new technology becoming available, "
                "or compliance and security requirements."
            ),
            "next": "project_vs_operations"
        },
        {
            "id": "project_vs_operations",
            "type": "open",
            "npc": (
                "🏭 One more question.\n"
                "Our team also handles daily order processing in the warehouse.\n\n"
                "What is the key difference between this upgrade project and our day-to-day warehouse operations?"
            ),
            "key_points": [
                "The project is temporary, with a defined start and end",
                "Operations are ongoing and repetitive",
                "Project introduces change; operations keep the business running",
                "Both use resources, but their purpose and time horizon differ"
            ],
            "model_answer": (
                "The upgrade is a temporary project with a clear end date and a unique outcome, "
                "while daily order processing is ongoing operations. "
                "The project introduces change and then ends; operations are repetitive activities "
                "that keep the business running continuously."
            ),
            "next": "wrap_up"
        },
        {
            "id": "wrap_up",
            "type": "terminal",
            "npc": (
                "✅ Nice work reaching the end of this scenario.\n"
                "Type `finish` to complete and calculate your score."
            )
        }
    ]
}


# 精簡評分規則：開放題依 key_words 命中數給分
def score_open_answer(text: str, key_points) -> int:
    if not text:
        return 0
    t = text.lower()
    hit = sum(1 for k in key_points if k.lower() in t)
    # 命中≥2 視為答對
    return 1 if hit >= 2 else 0
