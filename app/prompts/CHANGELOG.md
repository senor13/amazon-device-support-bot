# Prompt Changelog

## v2.1
**Changed:** generation.txt
- Removed hardcoded fixed string from OOS rule — LLM now responds in its own words
- Reordered rules: OOS check first, then greeting, then docs-based, then docs-missing
- Docs-missing rule also uses natural language instead of a fixed string

**Why:** v2 still had a literal string ("I don't have that information...") in the prompt that the LLM copied verbatim. OOS queries also hit the docs-missing rule before reaching the OOS rule because context is always empty for them. Reordering and removing fixed strings lets the LLM respond naturally.

---

## v2
**Changed:** generation.txt
- Restricted answers to docs only — removed "answer from general knowledge" fallback
- Removed Fire TV references (Fire TV docs not yet indexed)
- Added explicit handling for greetings/intros — LLM responds warmly and asks how to help
- Added explicit handling for out-of-scope — LLM deflects naturally instead of hardcoded message
- Removed out_of_scope_node from graph — LLM handles all query types

**Why:** v1 allowed the model to hallucinate answers from general knowledge when docs didn't cover the question. v2 enforces strict doc-grounding and delegates out-of-scope + greeting handling to the LLM for more natural responses.

---

## v1
**Initial version**
- generation.txt: answers from docs + general knowledge fallback
- query_intelligence.txt: intent, complexity, routing, cacheability classification
- completeness_judge.txt: scores whether response covers all sub-queries
