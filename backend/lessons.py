import json
from pathlib import Path

def load_json_lesson(path: str):
    """Load a chapter lesson JSON file."""
    return json.loads(Path(path).read_text(encoding="utf-8"))

# v0.8 curriculum lessons
LESSONS = {
    "chapter1": load_json_lesson("lessons/chapter1.json"),
    "chapter2": load_json_lesson("lessons/chapter2.json"),
    "chapter3": load_json_lesson("lessons/chapter3.json"),
    "chapter4": load_json_lesson("lessons/chapter4.json"),
    "chapter5": load_json_lesson("lessons/chapter5.json"),
    "chapter6": load_json_lesson("lessons/chapter6.json"),
    "chapter7": load_json_lesson("lessons/chapter7.json"),
    "chapter8": load_json_lesson("lessons/chapter8.json"),
    "chapter9": load_json_lesson("lessons/chapter9.json"),
    "chapter10": load_json_lesson("lessons/chapter10.json"),
}
