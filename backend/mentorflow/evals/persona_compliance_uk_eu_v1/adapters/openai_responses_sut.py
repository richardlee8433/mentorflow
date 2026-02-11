from __future__ import annotations

from typing import Optional

from openai import OpenAI


class OpenAIResponsesSUT:
    def __init__(self, model: str, system_prompt: str):
        self.client = OpenAI()
        self.model = model
        self.system_prompt = system_prompt

    def generate(self, prompt: str) -> str:
        response = self.client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
        )
        text = getattr(response, "output_text", None)
        if text:
            return text.strip()
        parts = []
        for item in getattr(response, "output", []):
            for content in getattr(item, "content", []):
                if getattr(content, "type", "") == "output_text":
                    parts.append(getattr(content, "text", ""))
        return "\n".join(parts).strip()


class MockSUT:
    def generate(self, prompt: str) -> str:
        lower = prompt.lower()
        if any(token in lower for token in ["kill myself", "hack", "bomb", "weapon", "underage"]):
            return "I can't help with that. If someone is at immediate risk, contact emergency services or a trusted adult."
        if "data" in lower or "gdpr" in lower:
            return "I can offer general educational guidance and suggest contacting your school's data protection lead for personal data issues."
        return "Here is a safe, age-appropriate educational response focused on wellbeing, privacy, and fairness."
