from backend.mentorflow.evals.persona_compliance_uk_eu_v1.run_eval import normalize_fail_reasons


def test_normalize_fail_reasons_accepts_string() -> None:
    normalized = normalize_fail_reasons("EVIDENCE_MISSING")

    assert normalized == [{"code": "EVIDENCE_MISSING", "detail": ""}]


def test_normalize_fail_reasons_accepts_list_of_strings_and_dicts() -> None:
    normalized = normalize_fail_reasons(["JUDGE_SHAPE_ERROR", {"code": "B_H1"}])

    assert normalized == [
        {"code": "JUDGE_SHAPE_ERROR", "detail": ""},
        {"code": "B_H1", "detail": ""},
    ]


def test_normalize_fail_reasons_repairs_malformed_entries_without_raising() -> None:
    normalized = normalize_fail_reasons([{"detail": "missing code"}, 42])

    assert normalized[0]["code"] == "JUDGE_SHAPE_ERROR"
    assert normalized[1]["code"] == "JUDGE_SHAPE_ERROR"


def test_normalize_fail_reasons_remaps_natural_language_codes() -> None:
    normalized = normalize_fail_reasons([{"code": "Missing required phrases", "detail": "none found"}])

    assert normalized == [
        {
            "code": "RUBRIC_MISSING_REQUIRED",
            "detail": "original_code: Missing required phrases; none found",
        }
    ]


def test_normalize_fail_reasons_unknown_code_maps_to_rubric_other() -> None:
    normalized = normalize_fail_reasons([{"code": "Something unexpected", "detail": "debug"}])

    assert normalized == [
        {
            "code": "RUBRIC_OTHER",
            "detail": "original_code: Something unexpected; debug",
        }
    ]
