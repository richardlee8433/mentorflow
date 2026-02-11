from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


DIMENSION_SEED_FILES = {
    "A": "A_privacy.yaml",
    "B": "B_boundaries.yaml",
    "C": "C_online_safety.yaml",
    "D": "D_fairness_bias.yaml",
    "E": "E_transparency.yaml",
}

ONLINE_SAFETY_RISK_TAGS = {"self_harm", "violence", "crime", "sexual_content_minor"}


def _load_seed_file(path: Path) -> List[Dict[str, Any]]:
    # Seed files use JSON-compatible YAML so they can be parsed without external deps.
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid seed file format in {path}: {exc}") from exc
    if not isinstance(data, list):
        raise ValueError(f"Seed file must contain a list: {path}")
    if not all(isinstance(record, dict) for record in data):
        raise ValueError(f"Each seed entry must be an object: {path}")
    return data


def _load_seed_items(seed_dir: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for dimension, filename in DIMENSION_SEED_FILES.items():
        path = seed_dir / filename
        records = _load_seed_file(path)
        for idx, record in enumerate(records, start=1):
            if record.get("dimension") != dimension:
                raise ValueError(f"Dimension mismatch in {path}#{idx}: expected {dimension}, got {record.get('dimension')}")
            expected_id = f"{dimension}-{idx:03d}"
            if record.get("id") != expected_id:
                raise ValueError(f"ID mismatch in {path}#{idx}: expected {expected_id}, got {record.get('id')}")
            risk_tags = set(record.get("risk_tags", []))
            if risk_tags.intersection(ONLINE_SAFETY_RISK_TAGS) and record.get("dimension") != "C":
                raise ValueError(
                    f"Online safety risk tag in non-C dimension for item {record.get('id')} in {path}"
                )
            items.append(record)
    return items


def _items_v1() -> List[Dict[str, Any]]:
    seed_dir = Path(__file__).resolve().parent / "seeds"
    return _load_seed_items(seed_dir)


def build_testbank(version: str) -> List[Dict[str, Any]]:
    if version != "v1":
        raise ValueError(f"Unsupported version: {version}")
    return _items_v1()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Persona UK/EU compliance testbank")
    parser.add_argument("--version", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    items = build_testbank(args.version)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Wrote {len(items)} test items to {out}")


if __name__ == "__main__":
    main()
