from __future__ import annotations

from typing import Dict, Any
import requests


class PersonaHTTPSUT:
    def __init__(self, url: str, timeout_s: float = 30.0):
        self.url = url
        self.timeout_s = timeout_s

    def generate(self, prompt: str) -> str:
        payload: Dict[str, Any] = {"prompt": prompt}
        response = requests.post(self.url, json=payload, timeout=self.timeout_s)
        response.raise_for_status()
        body = response.json()
        if not isinstance(body, dict) or "output" not in body:
            raise ValueError("persona_http SUT response must be JSON object containing 'output'")
        return str(body["output"])
