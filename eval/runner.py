"""
Eval runner — fires all dataset cases against the live API, scores them,
and stores results in Postgres.

Usage:
    python -m eval.runner
"""

import asyncio
import json
import os
import subprocess
import time
import uuid
from pathlib import Path

import httpx
import psycopg
from dotenv import load_dotenv
from jose import jwt
from openai import AsyncOpenAI

load_dotenv()

from eval.config import eval_settings

API_BASE = eval_settings.APP_URL
JWT_SECRET = os.environ["JWT_SECRET"]
POSTGRES_DSN = os.environ.get(
    "POSTGRES_DSN", "postgresql://postgres:postgres@localhost:5433/support_bot"
).replace("@postgres:5432", "@localhost:5433")

DATASET_PATH = Path(eval_settings.DATASET_FILE)

FAITHFULNESS_THRESHOLD = eval_settings.FAITHFULNESS_THRESHOLD
COMPLETENESS_THRESHOLD = eval_settings.COMPLETENESS_THRESHOLD
CORRECTNESS_THRESHOLD  = eval_settings.CORRECTNESS_THRESHOLD
EVAL_NOTES             = os.environ.get("EVAL_NOTES", "")

_openai = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])

_PRICING = {
    "gpt-4o-mini": {"input": eval_settings.GPT4O_MINI_INPUT_PRICE, "output": eval_settings.GPT4O_MINI_OUTPUT_PRICE},
    "gpt-4o":      {"input": eval_settings.GPT4O_INPUT_PRICE,      "output": eval_settings.GPT4O_OUTPUT_PRICE},
}


def estimate_cost(model_used: str, response_text: str) -> float:
    """Estimate cost using fixed prompt token assumption + response char count.
    Approximate but consistent — good enough for regression detection."""
    pricing = _PRICING.get(model_used, _PRICING["gpt-4o-mini"])
    completion_tokens = len(response_text) / 4  # chars → tokens approximation
    input_cost  = (eval_settings.AVG_PROMPT_TOKENS / 1_000_000) * pricing["input"]
    output_cost = (completion_tokens               / 1_000_000) * pricing["output"]
    return round(input_cost + output_cost, 6)


async def wait_for_app(client: httpx.AsyncClient) -> None:
    print("Waiting for app to be ready...", end=" ", flush=True)
    for _ in range(30):
        try:
            r = await client.get(f"{API_BASE}/health", timeout=2.0)
            if r.status_code == 200:
                print("ready.")
                return
        except Exception:
            pass
        await asyncio.sleep(2)
    raise RuntimeError("App not ready after 60s — is the container running?")


def make_token(run_id: str) -> str:
    payload = {"sub": f"eval-{run_id[:8]}", "exp": int(time.time()) + 3600}
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")


async def fire_query(client: httpx.AsyncClient, query: str, session_id: str, token: str) -> tuple[dict | None, int, float]:
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {token}"}
    start = time.monotonic()
    try:
        resp = await client.post(
            f"{API_BASE}/query",
            json={"query": query, "session_id": session_id, "skip_cache": True},
            headers=headers,
            timeout=60.0,
        )
        latency_ms = (time.monotonic() - start) * 1000
        body = resp.json() if "application/json" in resp.headers.get("content-type", "") else None
        return body, resp.status_code, latency_ms
    except Exception:
        return None, 0, (time.monotonic() - start) * 1000


_CORRECTNESS_PROMPT = """\
You are evaluating whether an AI support bot correctly answers a question compared to a reference answer.

Question: {query}
Reference Answer: {expected_answer}
AI Response: {response}

Score from 0.0 to 1.0 where 1.0 = covers all key points, 0.0 = wrong or completely misses the reference.
Reply with only a single float and nothing else."""


async def score_correctness(query: str, expected_answer: str, response: str) -> float:
    result = await _openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": _CORRECTNESS_PROMPT.format(
            query=query, expected_answer=expected_answer, response=response
        )}],
        temperature=0,
    )
    try:
        return max(0.0, min(1.0, float(result.choices[0].message.content.strip())))
    except (ValueError, AttributeError):
        return 0.5


async def score_case(case: dict, body: dict | None, status: int) -> dict:
    category        = case["category"]
    expected_answer = case.get("expected_answer")
    failure_reason  = None
    passed          = False

    faithfulness   = body.get("faithfulness_score",  0.0)  if body else 0.0
    completeness   = body.get("completeness_score",  0.0)  if body else 0.0
    rag_precision  = body.get("rag_precision_score", 0.0)  if body else 0.0
    validation_ok  = body.get("validation_passed",   False) if body else False
    model_used     = body.get("model_used", "")             if body else ""
    response_text  = body.get("response",   "")             if body else ""
    request_id     = body.get("request_id", "")             if body else ""
    actual_decomp  = body.get("needs_decomp", False)        if body else False
    actual_queries = body.get("sub_queries", [])            if body else []

    estimated_cost_usd = estimate_cost(model_used, response_text) if status == 200 else 0.0

    correctness = None
    if expected_answer and response_text and status == 200:
        correctness = await score_correctness(case["query"], expected_answer, response_text)

    if category == "attack":
        passed = (status == 403)
        if not passed:
            failure_reason = f"expected 403, got {status}"

    elif category == "out_of_scope":
        # Use LLM-as-judge instead of brittle keyword matching
        if status != 200:
            failure_reason = f"unexpected HTTP {status}"
        elif correctness is not None and correctness < CORRECTNESS_THRESHOLD:
            failure_reason = f"correctness {correctness:.2f} < {CORRECTNESS_THRESHOLD} — bot did not deflect appropriately"
        else:
            passed = True

    else:
        expected_decomp      = case.get("expected_needs_decomp")
        expected_sub_count   = case.get("sub_query_count")

        if status != 200:
            failure_reason = f"unexpected HTTP {status}"
        elif faithfulness < FAITHFULNESS_THRESHOLD:
            failure_reason = f"faithfulness {faithfulness:.2f} < {FAITHFULNESS_THRESHOLD}"
        elif completeness < COMPLETENESS_THRESHOLD:
            failure_reason = f"completeness {completeness:.2f} < {COMPLETENESS_THRESHOLD}"
        elif not validation_ok:
            failure_reason = "validation_passed=False"
        elif correctness is not None and correctness < CORRECTNESS_THRESHOLD:
            failure_reason = f"correctness {correctness:.2f} < {CORRECTNESS_THRESHOLD}"
        elif expected_decomp is True and not actual_decomp:
            failure_reason = "expected needs_decomp=true but bot did not decompose query"
        elif expected_sub_count and len(actual_queries) != expected_sub_count:
            failure_reason = f"expected {expected_sub_count} sub-queries, got {len(actual_queries)}"
        else:
            passed = True

    return {
        "test_case_id":        case["id"],
        "category":            category,
        "query":               case["query"],
        "bot_response":        response_text,
        "expected_answer":     expected_answer,
        "passed":              passed,
        "faithfulness_score":  faithfulness,
        "completeness_score":  completeness,
        "rag_precision_score": rag_precision,
        "correctness_score":   correctness,
        "validation_passed":   validation_ok,
        "model_used":          model_used,
        "http_status":         status,
        "failure_reason":      failure_reason,
        "request_id":          request_id,
        "estimated_cost_usd":  estimated_cost_usd,
        "actual_needs_decomp": actual_decomp,
        "actual_sub_count":    len(actual_queries),
    }


_CREATE_SCHEMA = """
CREATE TABLE IF NOT EXISTS eval_runs (
    run_id                  TEXT PRIMARY KEY,
    created_at              TIMESTAMPTZ DEFAULT NOW(),
    total_cases             INT,
    passed                  INT,
    failed                  INT,
    avg_faithfulness        FLOAT,
    avg_completeness        FLOAT,
    avg_correctness         FLOAT,
    avg_rag_precision       FLOAT,
    avg_latency_ms          FLOAT,
    total_cost_usd          FLOAT,
    git_commit              TEXT,
    notes                   TEXT
);

CREATE TABLE IF NOT EXISTS eval_results (
    id                  SERIAL PRIMARY KEY,
    run_id              TEXT REFERENCES eval_runs(run_id) ON DELETE CASCADE,
    test_case_id        TEXT,
    category            TEXT,
    query               TEXT,
    bot_response        TEXT,
    expected_answer     TEXT,
    passed              BOOLEAN,
    faithfulness_score  FLOAT,
    completeness_score  FLOAT,
    rag_precision_score FLOAT,
    correctness_score   FLOAT,
    validation_passed   BOOLEAN,
    model_used          TEXT,
    http_status         INT,
    latency_ms          FLOAT,
    estimated_cost_usd  FLOAT,
    failure_reason      TEXT,
    request_id          TEXT,
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

-- Add new columns to existing tables if they don't exist yet
ALTER TABLE eval_runs    ADD COLUMN IF NOT EXISTS avg_latency_ms FLOAT;
ALTER TABLE eval_runs    ADD COLUMN IF NOT EXISTS total_cost_usd FLOAT;
ALTER TABLE eval_results ADD COLUMN IF NOT EXISTS estimated_cost_usd FLOAT;
ALTER TABLE eval_results ADD COLUMN IF NOT EXISTS actual_needs_decomp BOOLEAN;
ALTER TABLE eval_results ADD COLUMN IF NOT EXISTS actual_sub_count INT;
"""


async def store_results(run_id: str, results: list[dict], latencies: dict[str, float]) -> None:
    total  = len(results)
    passed = sum(1 for r in results if r["passed"])
    scored = [r for r in results if r["category"] not in ("attack", "out_of_scope")]

    avg_f = sum(r["faithfulness_score"]  for r in scored) / len(scored) if scored else 0.0
    avg_c = sum(r["completeness_score"]  for r in scored) / len(scored) if scored else 0.0
    avg_p = sum(r["rag_precision_score"] for r in scored) / len(scored) if scored else 0.0
    correct_scored = [r for r in scored if r["correctness_score"] is not None]
    avg_k = sum(r["correctness_score"] for r in correct_scored) / len(correct_scored) if correct_scored else 0.0
    avg_latency = sum(latencies.values()) / len(latencies) if latencies else 0.0
    total_cost  = sum(r["estimated_cost_usd"] for r in results)

    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        git_commit = "unknown"

    async with await psycopg.AsyncConnection.connect(POSTGRES_DSN) as conn:
        await conn.execute(_CREATE_SCHEMA)
        await conn.execute(
            """INSERT INTO eval_runs
               (run_id, total_cases, passed, failed, avg_faithfulness, avg_completeness,
                avg_correctness, avg_rag_precision, avg_latency_ms, total_cost_usd, git_commit, notes)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
            (run_id, total, passed, total - passed, avg_f, avg_c, avg_k, avg_p,
             avg_latency, total_cost, git_commit, EVAL_NOTES),
        )
        for r in results:
            await conn.execute(
                """INSERT INTO eval_results
                   (run_id, test_case_id, category, query, bot_response, expected_answer,
                    passed, faithfulness_score, completeness_score, rag_precision_score, correctness_score,
                    validation_passed, model_used, http_status, latency_ms, estimated_cost_usd,
                    failure_reason, request_id, actual_needs_decomp, actual_sub_count)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                (
                    run_id, r["test_case_id"], r["category"], r["query"],
                    r["bot_response"], r["expected_answer"], r["passed"],
                    r["faithfulness_score"], r["completeness_score"], r["rag_precision_score"], r["correctness_score"],
                    r["validation_passed"], r["model_used"], r["http_status"],
                    latencies.get(r["test_case_id"], 0.0), r["estimated_cost_usd"],
                    r["failure_reason"], r["request_id"],
                    r.get("actual_needs_decomp"), r.get("actual_sub_count"),
                ),
            )
        await conn.commit()

    print(f"\nRun {run_id[:8]}: {passed}/{total} passed")
    print(f"Faithfulness {avg_f:.2f} | Completeness {avg_c:.2f} | Correctness {avg_k:.2f} | RAG Precision {avg_p:.2f}")
    print(f"Avg latency {avg_latency:.0f}ms | Total cost ${total_cost:.4f}")
    if EVAL_NOTES:
        print(f"Notes: {EVAL_NOTES}")
    failures = [r for r in results if not r["passed"]]
    for r in failures:
        print(f"  FAIL {r['test_case_id']} ({r['category']}): {r['failure_reason']}")


async def main() -> None:
    dataset = json.loads(DATASET_PATH.read_text())
    run_id  = str(uuid.uuid4())
    token   = make_token(run_id)

    print(f"Eval run {run_id[:8]} — {len(dataset)} cases")

    results:   list[dict]       = []
    latencies: dict[str, float] = {}

    async with httpx.AsyncClient() as client:
        await wait_for_app(client)
        for i, case in enumerate(dataset, 1):
            session_id = f"eval-{run_id[:8]}-{case['id']}"
            print(f"[{i:02d}/{len(dataset)}] {case['id']} ({case['category']})...", end=" ", flush=True)

            for prior in case.get("prior_turns", []):
                await fire_query(client, prior["query"], session_id, token)

            body, status, latency_ms = await fire_query(client, case["query"], session_id, token)
            latencies[case["id"]] = latency_ms

            scored = await score_case(case, body, status)
            results.append(scored)

            print("PASS" if scored["passed"] else "FAIL", f"({latency_ms/1000:.1f}s)")

            await asyncio.sleep(4)

    await store_results(run_id, results, latencies)


if __name__ == "__main__":
    asyncio.run(main())
