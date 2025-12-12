import json
import os
import time
from datetime import datetime
from typing import Dict, List, Any

from openai import OpenAI

# 使用環境變數 OPENAI_API_KEY（不在這裡硬寫 key）
client = OpenAI()


SCORER_SYSTEM_PROMPT = """
You are an impartial evaluator for an AI teaching assistant.

Your job:
- Compare the model's answer to the ideal reference answer.
- Decide if the answer is ACCEPTABLE (1) or NOT ACCEPTABLE (0).

Rules:
1) Score is binary: 1 or 0.
2) Score 1 if the model answer captures the key ideas of the ideal answer,
   even with different wording.
3) Score 0 if the answer is clearly wrong, off-topic, or misses the core idea.
4) Always return JSON with fields:
   - score: 0 or 1 (integer)
   - reasoning: short natural language explanation (1–3 sentences)

Output format rules (very important):
- You MUST respond with ONLY a valid JSON object.
- Do NOT include any backticks, markdown code fences, or extra explanation.
- Do NOT wrap the JSON in ```json ... ``` or any other formatting.
- If you add anything outside the JSON object, the system will break.
""".strip()


def load_dataset(path: str = "eval/dataset_v1.json") -> List[Dict[str, Any]]:
    """Load the golden dataset from JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def call_model(prompt: str, model: str = "gpt-4.1-mini") -> str:
    """
    Call the model under test to produce an answer.

    This is the "student" model whose behavior we want to evaluate.
    """
    messages = [
        {
            "role": "system",
            "content": "You are MentorFlow, an AI teaching assistant for AI Product Management."
        },
        {"role": "user", "content": prompt},
    ]
    res = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.2,
    )
    return (res.choices[0].message.content or "").strip()


def _extract_json_block(text: str) -> str:
    """
    嘗試從 LLM 回覆中抽出 JSON 區塊：
    - 處理 ```json ... ``` 或 ``` ... ``` 包起來的情況
    - 或是前後有多餘說明文字，只保留第一個 { 到最後一個 } 中間內容
    """
    cleaned = text.strip()

    # 如果是 markdown code block（```json ... ``` / ``` ... ```）
    if cleaned.startswith("```"):
        # 移除開頭和結尾的 ```
        cleaned = cleaned.strip("`").strip()
        # 去掉可能的 'json' 或 'JSON' 語言標籤
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()  # 移除 "json" + 可能的換行

    # 從第一個 { 到最後一個 } 擷取
    if "{" in cleaned and "}" in cleaned:
        start = cleaned.find("{")
        end = cleaned.rfind("}") + 1
        cleaned = cleaned[start:end].strip()

    return cleaned


def judge_answer(
    prompt: str,
    ideal_answer: str,
    model_answer: str,
    model: str = "gpt-4.1-mini",
) -> Dict[str, Any]:
    """
    Use LLM-as-judge to decide 0/1.

    The judge model compares:
    - prompt
    - ideal_answer (reference)
    - model_answer (produced by model_under_test)
    and returns a JSON with "score" and "reasoning".
    """
    user_content = {
        "prompt": prompt,
        "ideal_answer": ideal_answer,
        "model_answer": model_answer,
    }

    res = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SCORER_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(user_content, ensure_ascii=False)},
        ],
        temperature=0,
    )

    raw = (res.choices[0].message.content or "").strip()
    cleaned = _extract_json_block(raw)

    try:
        data = json.loads(cleaned)
    except Exception:
        # 解析失敗時：當作 0 分，但明確標記為 parsing_error，方便之後調試
        return {
            "score": 0,
            "reasoning": "Parsing_error: could not parse judge JSON. Raw output (truncated): "
                         + raw[:200]
        }

    score = int(data.get("score", 0))
    if score not in (0, 1):
        score = 0

    reasoning = str(data.get("reasoning", "")).strip()
    return {"score": score, "reasoning": reasoning}


def run_eval(
    model_under_test: str = "gpt-4.1-mini",
    judge_model: str = "gpt-4.1-mini",
    dataset_path: str = "eval/dataset_v1.json",
    output_path: str = "eval/output_report.json",
    mentorflow_version: str = "v0.9.0-alpha",
) -> Dict[str, Any]:
    """
    Run evaluation on the whole dataset and return a JSON-able report.

    This is the core of v0.90 – Golden Dataset + Basic Evaluator.
    """
    dataset = load_dataset(dataset_path)
    results: List[Dict[str, Any]] = []

    total = len(dataset)
    correct = 0

    started_at = time.time()

    for item in dataset:
        qid = item["id"]
        prompt = item["prompt"]
        ideal = item["ideal_answer"]

        # 1) 由被測模型產生答案
        model_answer = call_model(prompt, model=model_under_test)

        # 2) 由 judge 模型評分
        judge = judge_answer(
            prompt=prompt,
            ideal_answer=ideal,
            model_answer=model_answer,
            model=judge_model,
        )

        score = judge["score"]
        if score == 1:
            correct += 1

        results.append(
            {
                "id": qid,
                "prompt": prompt,
                "ideal_answer": ideal,
                "model_answer": model_answer,
                "score": score,
                "judge_reasoning": judge["reasoning"],
            }
        )

    elapsed = time.time() - started_at
    accuracy = correct / total if total > 0 else 0.0

    # 新增 run_id（UTC, 去掉 microseconds, 加 Z）
    run_id = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

    report = {
        "run_id": run_id,
        "mentorflow_version": mentorflow_version,
        "model_under_test": model_under_test,
        "judge_model": judge_model,
        "dataset_path": dataset_path,
        "total_questions": total,
        "correct": correct,
        "accuracy": accuracy,
        "elapsed_seconds": elapsed,
        "results": results,
    }

    # 寫檔方便之後做 regression / diff / dashboard
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    return report


if __name__ == "__main__":
    # 直接在本機執行：python eval/evaluator_v1.py
    rep = run_eval()
    summary = {
        "run_id": rep["run_id"],
        "mentorflow_version": rep["mentorflow_version"],
        "model_under_test": rep["model_under_test"],
        "total_questions": rep["total_questions"],
        "correct": rep["correct"],
        "accuracy": rep["accuracy"],
        "elapsed_seconds": rep["elapsed_seconds"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
