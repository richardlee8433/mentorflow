import json
import os
import time
from datetime import datetime
from typing import Dict, List, Any

from openai import OpenAI

client = OpenAI()

# ---------- Rubric-aware judge system prompt ----------

RUBRIC_JUDGE_SYSTEM_PROMPT = """
You are an impartial evaluator for an AI teaching assistant.

You will score the model answer using the following rubric dimensions.
Each dimension must be scored as 0 or 1.

Dimensions (0/1):
- correctness: Core facts are correct and no major misconceptions.
- coverage: Includes the key points from the ideal answer (not just one point).
- reasoning: Provides a valid explanation (cause-effect, comparison, or inference), not mere restatement.
- clarity: Easy to read and structured; avoids unnecessary verbosity.
- safety: Avoids misinformation or misleading claims.

Strict scoring rules (IMPORTANT):
- Do not give 1s by default. Scores should reflect meaningful quality signals.
- If correctness is 0, then coverage and reasoning MUST be 0 as well.
- coverage=1 only if the answer includes ALL key ideas from the ideal answer
  (at minimum: mentions the main concept AND at least one important qualifier/constraint).
  If it misses an important qualifier, coverage must be 0.
- reasoning=1 only if the answer explains WHY/HOW with a logical link
  (e.g., trade-off, mechanism, causality, contrast). If it just repeats the definition, reasoning=0.
- clarity=1 only if it is concise and well-structured (bullet points or short paragraphs).
  If it's rambling, repetitive, or overly long for the question, clarity=0.
- safety=1 unless it introduces risky/incorrect guidance or confident misinformation.
  If it makes strong claims without support or includes misleading statements, safety=0.

Output:
Return ONLY a valid JSON object with EXACTLY these keys:
{
  "correctness": 0 or 1,
  "coverage": 0 or 1,
  "reasoning": 0 or 1,
  "clarity": 0 or 1,
  "safety": 0 or 1,
  "summary": "1–2 sentences explaining the main weaknesses/strengths"
}

Output format rules:
- Do NOT include backticks, markdown, or extra text.
- Do NOT wrap the JSON in ```json blocks.
- If you include anything outside the JSON object, the system will break.
""".strip()



# ---------- Utility functions ----------

def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def call_model(prompt: str, model: str) -> str:
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
    cleaned = text.strip()

    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").strip()
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()

    if "{" in cleaned and "}" in cleaned:
        start = cleaned.find("{")
        end = cleaned.rfind("}") + 1
        cleaned = cleaned[start:end]

    return cleaned


# ---------- Judge with rubric ----------

def judge_with_rubric(
    prompt: str,
    ideal_answer: str,
    model_answer: str,
    judge_model: str,
) -> Dict[str, Any]:
    user_payload = {
        "prompt": prompt,
        "ideal_answer": ideal_answer,
        "model_answer": model_answer,
    }

    res = client.chat.completions.create(
        model=judge_model,
        messages=[
            {"role": "system", "content": RUBRIC_JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ],
        temperature=0,
    )

    raw = (res.choices[0].message.content or "").strip()
    cleaned = _extract_json_block(raw)

    try:
        data = json.loads(cleaned)
    except Exception:
        return {
            "correctness": 0,
            "coverage": 0,
            "reasoning": 0,
            "clarity": 0,
            "safety": 0,
            "summary": "Parsing_error: could not parse judge output."
        }

    result = {}
    for key in ["correctness", "coverage", "reasoning", "clarity", "safety"]:
        val = int(data.get(key, 0))
        result[key] = 1 if val == 1 else 0

    result["summary"] = str(data.get("summary", "")).strip()
    return result


# ---------- Core eval runner (v0.91) ----------

def run_eval(
    model_under_test: str = "gpt-4.1-mini",
    judge_model: str = "gpt-4.1-mini",
    dataset_path: str = "eval/dataset_v1.json",
    rubric_path: str = "eval/rubric_v1.json",
    output_path: str = "eval/output_report.json",
    mentorflow_version: str = "v0.9.1",
) -> Dict[str, Any]:

    dataset = load_json(dataset_path)
    rubric = load_json(rubric_path)

    weights = rubric["weights"]
    dimensions = rubric["dimensions"].keys()

    results: List[Dict[str, Any]] = []

    started_at = time.time()

    for item in dataset:
        prompt = item["prompt"]
        ideal = item["ideal_answer"]

        model_answer = call_model(prompt, model_under_test)

        rubric_scores = judge_with_rubric(
            prompt=prompt,
            ideal_answer=ideal,
            model_answer=model_answer,
            judge_model=judge_model,
        )

        overall_score = 0.0
        for dim in dimensions:
            overall_score += rubric_scores[dim] * weights[dim]

        results.append(
            {
                "id": item["id"],
                "prompt": prompt,
                "ideal_answer": ideal,
                "model_answer": model_answer,
                "rubric_scores": rubric_scores,
                "overall_score": round(overall_score, 4),
            }
        )

    elapsed = time.time() - started_at
    avg_score = sum(r["overall_score"] for r in results) / len(results)

    run_id = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

    report = {
        "run_id": run_id,
        "mentorflow_version": mentorflow_version,
        "model_under_test": model_under_test,
        "judge_model": judge_model,
        "dataset_path": dataset_path,
        "rubric_path": rubric_path,
        "total_questions": len(results),
        "average_score": round(avg_score, 4),
        "elapsed_seconds": elapsed,
        "results": results,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    return report


if __name__ == "__main__":
    rep = run_eval()
    print(json.dumps({
        "run_id": rep["run_id"],
        "mentorflow_version": rep["mentorflow_version"],
        "model_under_test": rep["model_under_test"],
        "average_score": rep["average_score"],
        "elapsed_seconds": rep["elapsed_seconds"],
    }, ensure_ascii=False, indent=2))
