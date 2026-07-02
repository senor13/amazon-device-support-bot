# Prompt Changelog

## v2.3
**Changed:** generation.txt
- Added "determine PRIMARY INTENT first" instruction — LLM now checks intent before matching rules, not just opening words
- Rule 3 (greeting) now explicitly covers single words or vague messages with no specific question
- Rules labelled by intent type for clearer LLM reasoning

**Why:** EG-002 ("Hi my name is Alex, is Kindle worth buying?") was triggering the greeting rule before the purchase query rule because the message starts with "Hi". EG-001 ("kindle") was getting a generic response but failing faithfulness because context was retrieved. Primary intent check fixes both.

---

## v2.2
**Changed:** generation.txt
- Added rule 0: never repeat PII redaction placeholders (<PERSON>, <EMAIL_ADDRESS>, etc.) in responses
- Added rule 1: purchase/buying queries → redirect to www.amazon.com/help with "post-purchase support only" message
- Renumbered existing rules 1-4 → 2-5

**Why:** Bot was echoing <PERSON> placeholder in responses when user's name was scrubbed. Also needed explicit handling for purchase decision queries separate from general OOS.

---

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
- faithfulness_judge.txt: checks if response claims are supported by context (binary faithful/not faithful)

---

# Prompt Files Reference

| File | Purpose | Used by |
|---|---|---|
| generation.txt | Main response generation | execution.py |
| query_intelligence.txt | Intent, complexity, routing, cacheability | query_intelligence.py |
| faithfulness_judge.txt | LLM-as-judge: are claims grounded in context? | metrics/faithfulness.py |
| completeness_judge.txt | LLM-as-judge: did response cover all sub-queries? | metrics/completeness.py |
