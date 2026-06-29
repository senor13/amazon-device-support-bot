# Prompt Changelog

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
