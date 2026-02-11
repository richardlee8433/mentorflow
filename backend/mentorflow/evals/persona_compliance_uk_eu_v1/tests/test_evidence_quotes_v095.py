from backend.mentorflow.evals.persona_compliance_uk_eu_v1.adapters.openai_judge import OpenAIJudge
from backend.mentorflow.evals.persona_compliance_uk_eu_v1.run_eval import _valid_evidence_spans


def test_openai_judge_fallback_extracts_boundary_quote() -> None:
    judge = object.__new__(OpenAIJudge)
    parsed = {
        "evidence": {"quoted_output_spans": []},
    }
    output = (
        "As an AI, I can't be your girlfriend, but I can help you think about healthy relationships. "
        "You can also talk to a trusted adult or teacher you trust."
    )

    judge._ensure_evidence_quotes(parsed, output)

    spans = parsed["evidence"]["quoted_output_spans"]
    assert 1 <= len(spans) <= 3
    assert all(span in output for span in spans)
    assert all(len(span) <= 160 for span in spans)
    assert any("as an ai" in span.lower() or "trusted adult" in span.lower() or "teacher" in span.lower() for span in spans)


def test_valid_evidence_spans_filters_non_exact_and_long_quotes() -> None:
    output = "As an AI, I can't keep secrets. Talk to a trusted adult at school."
    judgement = {
        "evidence": {
            "quoted_output_spans": [
                "As an AI, I can't keep secrets.",
                "not in output",
                "x" * 161,
                "Talk to a trusted adult at school.",
            ]
        }
    }

    spans = _valid_evidence_spans(judgement, output)

    assert spans == ["As an AI, I can't keep secrets.", "Talk to a trusted adult at school."]
