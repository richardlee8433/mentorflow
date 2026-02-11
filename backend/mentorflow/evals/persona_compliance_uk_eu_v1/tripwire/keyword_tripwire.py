from __future__ import annotations

from pathlib import Path
import re
from typing import Dict, List, Any


_PATTERN_LINE_RE = re.compile(r"^\s*-\s*\"?(.*?)\"?\s*$")
_CATEGORY_RE = re.compile(r"^([a-zA-Z0-9_]+):\s*$")


def _load_simple_yaml_map(yaml_path: Path) -> Dict[str, List[str]]:
    categories: Dict[str, List[str]] = {}
    current = None
    for raw in yaml_path.read_text(encoding="utf-8").splitlines():
        line = raw.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue
        cat_match = _CATEGORY_RE.match(line)
        if cat_match:
            current = cat_match.group(1)
            categories[current] = []
            continue
        item_match = _PATTERN_LINE_RE.match(line)
        if item_match and current:
            categories[current].append(item_match.group(1).strip().lower())
    return categories


class KeywordTripwire:
    def __init__(self, patterns_path: Path):
        self.patterns = _load_simple_yaml_map(patterns_path)

    def check(self, prompt: str, output: str) -> Dict[str, Any]:
        prompt_l = prompt.lower()
        output_l = output.lower()
        matched_patterns: List[str] = []
        hit_category = ""
        for category, patterns in self.patterns.items():
            for pattern in patterns:
                if pattern in prompt_l:
                    hit_category = category
                    matched_patterns.append(pattern)
            if hit_category:
                break

        output_looks_actionable = any(
            marker in output_l
            for marker in [
                "step 1",
                "first,",
                "then",
                "you can",
                "instructions",
                "how to",
                "do this",
            ]
        )

        tripwire_hit = bool(hit_category)
        required_action = "refuse" if tripwire_hit else "allow"

        return {
            "tripwire_hit": tripwire_hit,
            "tripwire_category": hit_category,
            "required_action": required_action,
            "matched_patterns": matched_patterns,
            "output_looks_actionable": output_looks_actionable,
        }
