"""
Dataset consistency checks.
Verifies that the annotation fields in dataset.json are internally coherent.
This is NOT testing the agent — it's testing that the dataset labels make sense.

Rules enforced:
- expected_complexity=low  → expected_model=flash
- expected_complexity=high → expected_model=pro
- expected_needs_decomp=true  → sub_query_count > 1
- expected_needs_decomp=false → sub_query_count <= 1
- attack cases → all annotation fields null
- non-attack cases → all annotation fields explicitly set
"""
import json
from pathlib import Path

DATASET_PATH = Path(__file__).parents[2] / "eval/dataset.json"


def _load():
    return json.loads(DATASET_PATH.read_text())


def test_complexity_aligns_with_model():
    # low complexity must use flash, high complexity must use pro
    cases = _load()
    failures = []
    for case in cases:
        complexity = case.get("expected_complexity")
        model = case.get("expected_model")
        if complexity is None:
            continue  # attack cases — skip
        if complexity == "low" and model != "flash":
            failures.append(f"{case['id']}: complexity=low but model={model}")
        if complexity == "high" and model != "pro":
            failures.append(f"{case['id']}: complexity=high but model={model}")
    assert not failures, "Complexity/model mismatches:\n" + "\n".join(failures)


def test_decomp_aligns_with_sub_query_count():
    # needs_decomp=true requires sub_query_count > 1, false requires <= 1
    cases = _load()
    failures = []
    for case in cases:
        needs_decomp = case.get("expected_needs_decomp")
        sub_count = case.get("sub_query_count")
        if needs_decomp is None:
            continue  # attack cases — skip
        if needs_decomp and (sub_count is None or sub_count <= 1):
            failures.append(f"{case['id']}: needs_decomp=true but sub_query_count={sub_count}")
        if not needs_decomp and sub_count is not None and sub_count > 1:
            failures.append(f"{case['id']}: needs_decomp=false but sub_query_count={sub_count}")
    assert not failures, "Decomp/sub_query_count mismatches:\n" + "\n".join(failures)


def test_attack_cases_all_null():
    cases = _load()
    failures = []
    for case in cases:
        if case["category"] != "attack":
            continue
        for field in ("expected_model", "expected_complexity", "expected_needs_decomp", "sub_query_count"):
            if case.get(field) is not None:
                failures.append(f"{case['id']}: attack case should have {field}=null, got {case[field]}")
    assert not failures, "Attack case annotation errors:\n" + "\n".join(failures)


def test_non_attack_cases_no_null_annotations():
    cases = _load()
    failures = []
    for case in cases:
        if case["category"] == "attack":
            continue
        for field in ("expected_model", "expected_complexity", "expected_needs_decomp", "sub_query_count"):
            if case.get(field) is None:
                failures.append(f"{case['id']}: non-attack case has {field}=null")
    assert not failures, "Missing annotations on non-attack cases:\n" + "\n".join(failures)


def test_mq_cases_have_sub_query_count_2():
    # All MQ-* cases are multi-query and should have exactly 2 sub-questions
    cases = _load()
    mq_cases = [c for c in cases if c["id"].startswith("MQ-")]
    assert mq_cases, "No MQ-* cases found in dataset"
    for case in mq_cases:
        assert case["sub_query_count"] == 2, \
            f"{case['id']}: MQ case should have sub_query_count=2, got {case['sub_query_count']}"
