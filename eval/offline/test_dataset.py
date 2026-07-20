"""
Offline checks for eval/dataset.json.
- File parses as valid JSON
- All required fields present on every case
- No duplicate IDs
- expected_model is a valid value
- attack cases have null model/decomp; all others have explicit values
No server, no LLM calls, no Docker needed.
"""
import json
from pathlib import Path

DATASET_PATH = Path(__file__).parents[2] / "eval/dataset.json"
REQUIRED_FIELDS = {"id", "category", "query", "expected_answer", "expected_model", "expected_needs_decomp", "prior_turns"}
VALID_MODELS = {"flash", "pro", None}
VALID_CATEGORIES = {"how_to", "about", "troubleshoot", "out_of_scope", "attack", "multi_turn", "edge_case"}


def _load():
    return json.loads(DATASET_PATH.read_text())


def test_dataset_parses():
    cases = _load()
    assert isinstance(cases, list) and len(cases) > 0, "dataset.json is empty or not a list"


def test_no_duplicate_ids():
    cases = _load()
    ids = [c["id"] for c in cases]
    dupes = [i for i in ids if ids.count(i) > 1]
    assert not dupes, f"Duplicate IDs in dataset: {set(dupes)}"


def test_required_fields_present():
    cases = _load()
    for case in cases:
        missing = REQUIRED_FIELDS - set(case.keys())
        assert not missing, f"Case {case.get('id')} missing fields: {missing}"


def test_valid_categories():
    cases = _load()
    for case in cases:
        assert case["category"] in VALID_CATEGORIES, \
            f"Case {case['id']} has unknown category: {case['category']}"


def test_valid_expected_model():
    cases = _load()
    for case in cases:
        assert case["expected_model"] in VALID_MODELS, \
            f"Case {case['id']} has invalid expected_model: {case['expected_model']}"


def test_attack_cases_have_null_fields():
    # Attack cases never reach generation — model and decomp must be null
    cases = _load()
    for case in cases:
        if case["category"] == "attack":
            assert case["expected_model"] is None, \
                f"Attack case {case['id']} should have expected_model=null"
            assert case["expected_needs_decomp"] is None, \
                f"Attack case {case['id']} should have expected_needs_decomp=null"


def test_non_attack_cases_have_explicit_values():
    # Every non-attack case must declare expected_model and expected_needs_decomp
    cases = _load()
    for case in cases:
        if case["category"] != "attack":
            assert case["expected_model"] is not None, \
                f"Case {case['id']} must have explicit expected_model"
            assert case["expected_needs_decomp"] is not None, \
                f"Case {case['id']} must have explicit expected_needs_decomp"


def test_decomp_cases_exist():
    # Sanity check: at least one case exercises the decomposition path
    cases = _load()
    decomp_cases = [c["id"] for c in cases if c.get("expected_needs_decomp") is True]
    assert decomp_cases, "No cases with expected_needs_decomp=true — add multi-query cases"
