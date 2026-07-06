"""
Offline checks for prompt files.
- All prompt files referenced in code exist and are readable
- Placeholders in each file match what the code actually passes in
No server, no LLM calls, no Docker needed.
"""
import re
from pathlib import Path

PROMPTS_DIR = Path(__file__).parents[2] / "app/prompts/v2"

# Maps each prompt file to the exact placeholder keys the code calls .format() with
EXPECTED_PLACEHOLDERS = {
    "generation.txt":        {"query", "context", "history"},
    "query_intelligence.txt": {"query", "session_history"},
    "faithfulness_judge.txt": {"context", "response"},
    "completeness_judge.txt": {"intent", "sub_queries", "response"},
    "tree_search.txt":        {"tree_json", "query"},
}


def _extract_placeholders(text: str) -> set[str]:
    return set(re.findall(r"\{(\w+)\}", text))


def test_all_prompt_files_exist():
    missing = [f for f in EXPECTED_PLACEHOLDERS if not (PROMPTS_DIR / f).exists()]
    assert not missing, f"Missing prompt files: {missing}"


def test_all_prompt_files_readable():
    for filename in EXPECTED_PLACEHOLDERS:
        content = (PROMPTS_DIR / filename).read_text()
        assert content.strip(), f"{filename} is empty"


def test_generation_placeholders():
    content = (PROMPTS_DIR / "generation.txt").read_text()
    found = _extract_placeholders(content)
    expected = EXPECTED_PLACEHOLDERS["generation.txt"]
    assert expected <= found, f"generation.txt missing placeholders: {expected - found}"


def test_query_intelligence_placeholders():
    content = (PROMPTS_DIR / "query_intelligence.txt").read_text()
    found = _extract_placeholders(content)
    expected = EXPECTED_PLACEHOLDERS["query_intelligence.txt"]
    assert expected <= found, f"query_intelligence.txt missing placeholders: {expected - found}"


def test_faithfulness_placeholders():
    content = (PROMPTS_DIR / "faithfulness_judge.txt").read_text()
    found = _extract_placeholders(content)
    expected = EXPECTED_PLACEHOLDERS["faithfulness_judge.txt"]
    assert expected <= found, f"faithfulness_judge.txt missing placeholders: {expected - found}"


def test_completeness_placeholders():
    content = (PROMPTS_DIR / "completeness_judge.txt").read_text()
    found = _extract_placeholders(content)
    expected = EXPECTED_PLACEHOLDERS["completeness_judge.txt"]
    assert expected <= found, f"completeness_judge.txt missing placeholders: {expected - found}"


def test_tree_search_placeholders():
    content = (PROMPTS_DIR / "tree_search.txt").read_text()
    found = _extract_placeholders(content)
    expected = EXPECTED_PLACEHOLDERS["tree_search.txt"]
    assert expected <= found, f"tree_search.txt missing placeholders: {expected - found}"
