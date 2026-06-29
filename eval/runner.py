"""
Eval runner — fires all dataset cases against the live API, scores them,
stores results in Postgres, and prints a summary report.

Usage:
    python -m eval.runner
"""

import asyncio
import json
import os
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import httpx
import psycopg
from dotenv import load_dotenv
from jose import jwt
from openai import AsyncOpenAI

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

API_BASE   = "http://localhost:8000"
JWT_SECRET = os.environ["JWT_SECRET"]
POSTGRES_DSN = os.environ.get(
    "POSTGRES_DSN", "postgresql://postgres:postgres@localhost:5433/support_bot"
).replace("@postgres:5432", "@localhost:5433")

DATASET_PATH = Path(__file__).parent / "dataset.json"

FAITHFULNESS_THRESHOLD  = float(os.environ.get("FAITHFULNESS_THRESHOLD", "0.7"))
COMPLETENESS_THRESHOLD  = float(os.environ.get("COMPLETENESS_THRESHOLD", "0.6"))
CORRECTNESS_THRESHOLD   = 0.6

_openai = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])

# ── ANSI colors ───────────────────────────────────────────────────────────────

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

# ── Auth ──────────────────────────────────────────────────────────────────────

def make_token(run_id: str) -> str:
    payload = {"sub": f"eval-{run_id[:8]}", "exp": int(time.time()) + 3600}
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")

# ── API call ──────────────────────────────────────────────────────────────────

async def fire_query(
    client: httpx.AsyncClient,
    query: str,
    session_id: str,
    token: str,
) -> tuple[dict | None, int, float]:
    """Returns (response_body, http_status, latency_ms)."""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
    }
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
        latency_ms = (time.monotonic() - start) * 1000
        return None, 0, latency_ms

# ── Correctness scorer (LLM-as-judge) ────────────────────────────────────────

_CORRECTNESS_PROMPT = """\
You are evaluating whether an AI support bot correctly answers a question \
compared to a reference answer.

Question: {query}

Reference Answer (key points that must be covered):
{expected_answer}

AI Response:
{response}

Score from 0.0 to 1.0:
- 1.0 = covers all key points from the reference correctly
- 0.7 = covers most key points but misses minor details
- 0.5 = partially correct, misses some important points
- 0.3 = some relevant info but misses key points
- 0.0 = wrong or completely misses the reference

Reply with only a single float and nothing else."""


async def score_correctness(query: str, expected_answer: str, response: str) -> float:
    prompt = _CORRECTNESS_PROMPT.format(
        query=query,
        expected_answer=expected_answer,
        response=response,
    )
    result = await _openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    try:
        return max(0.0, min(1.0, float(result.choices[0].message.content.strip())))
    except (ValueError, AttributeError):
        return 0.5

# ── Scoring ───────────────────────────────────────────────────────────────────

async def score_case(case: dict, body: dict | None, status: int) -> dict:
    category        = case["category"]
    expected_answer = case.get("expected_answer")
    failure_reason  = None
    passed          = False

    faithfulness  = body.get("faithfulness_score", 0.0)  if body else 0.0
    completeness  = body.get("completeness_score", 0.0)  if body else 0.0
    validation_ok = body.get("validation_passed",  False) if body else False
    model_used    = body.get("model_used",          "")   if body else ""
    response_text = body.get("response",            "")   if body else ""

    # Correctness only applies when we have a gold answer and a real response
    correctness = None
    if expected_answer and response_text and status == 200:
        correctness = await score_correctness(case["query"], expected_answer, response_text)

    if category == "attack":
        passed = (status == 403)
        if not passed:
            failure_reason = f"expected 403, got {status}"

    elif category == "out_of_scope":
        deflected = any(
            phrase in response_text.lower()
            for phrase in ["can only help", "amazon.com/help", "outside", "not able to help", "support guide"]
        )
        passed = status == 200 and deflected
        if not passed:
            failure_reason = f"expected deflection, got: {response_text[:80]!r}"

    else:
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
        else:
            passed = True

    return {
        "test_case_id":       case["id"],
        "category":           category,
        "query":              case["query"],
        "bot_response":       response_text,
        "expected_answer":    expected_answer,
        "passed":             passed,
        "faithfulness_score": faithfulness,
        "completeness_score": completeness,
        "correctness_score":  correctness,
        "validation_passed":  validation_ok,
        "model_used":         model_used,
        "http_status":        status,
        "failure_reason":     failure_reason,
    }

# ── Postgres ──────────────────────────────────────────────────────────────────

_CREATE_SCHEMA = """
CREATE TABLE IF NOT EXISTS eval_runs (
    run_id           TEXT PRIMARY KEY,
    created_at       TIMESTAMPTZ DEFAULT NOW(),
    total_cases      INT,
    passed           INT,
    failed           INT,
    avg_faithfulness FLOAT,
    avg_completeness FLOAT,
    avg_correctness  FLOAT,
    git_commit       TEXT
);

CREATE TABLE IF NOT EXISTS eval_results (
    id                 SERIAL PRIMARY KEY,
    run_id             TEXT REFERENCES eval_runs(run_id) ON DELETE CASCADE,
    test_case_id       TEXT,
    category           TEXT,
    query              TEXT,
    bot_response       TEXT,
    expected_answer    TEXT,
    passed             BOOLEAN,
    faithfulness_score FLOAT,
    completeness_score FLOAT,
    correctness_score  FLOAT,
    validation_passed  BOOLEAN,
    model_used         TEXT,
    http_status        INT,
    latency_ms         FLOAT,
    failure_reason     TEXT,
    created_at         TIMESTAMPTZ DEFAULT NOW()
);
"""

async def store_results(run_id: str, results: list[dict], latencies: dict[str, float]) -> None:
    total  = len(results)
    passed = sum(1 for r in results if r["passed"])
    scored = [r for r in results if r["category"] not in ("attack", "out_of_scope")]

    avg_f = sum(r["faithfulness_score"] for r in scored) / len(scored) if scored else 0.0
    avg_c = sum(r["completeness_score"] for r in scored) / len(scored) if scored else 0.0

    correct_scored = [r for r in scored if r["correctness_score"] is not None]
    avg_k = sum(r["correctness_score"] for r in correct_scored) / len(correct_scored) if correct_scored else 0.0

    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        git_commit = "unknown"

    async with await psycopg.AsyncConnection.connect(POSTGRES_DSN) as conn:
        await conn.execute(_CREATE_SCHEMA)
        await conn.execute(
            """INSERT INTO eval_runs
               (run_id, total_cases, passed, failed,
                avg_faithfulness, avg_completeness, avg_correctness, git_commit)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
            (run_id, total, passed, total - passed, avg_f, avg_c, avg_k, git_commit),
        )
        for r in results:
            await conn.execute(
                """INSERT INTO eval_results
                   (run_id, test_case_id, category, query, bot_response, expected_answer,
                    passed, faithfulness_score, completeness_score, correctness_score,
                    validation_passed, model_used, http_status, latency_ms, failure_reason)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                (
                    run_id, r["test_case_id"], r["category"], r["query"],
                    r["bot_response"], r["expected_answer"], r["passed"],
                    r["faithfulness_score"], r["completeness_score"], r["correctness_score"],
                    r["validation_passed"], r["model_used"], r["http_status"],
                    latencies.get(r["test_case_id"], 0.0), r["failure_reason"],
                ),
            )
        await conn.commit()

# ── Report : for developemnt phase────────────────────────────────────────────────────────────────────

def _fmt(val: float | None) -> str:
    return f"{val:.2f}" if val is not None else " n/a"

def print_report(run_id: str, results: list[dict], latencies: dict[str, float]) -> None:
    total  = len(results)
    passed = sum(1 for r in results if r["passed"])
    scored = [r for r in results if r["category"] not in ("attack", "out_of_scope")]

    avg_f = sum(r["faithfulness_score"] for r in scored) / len(scored) if scored else 0.0
    avg_c = sum(r["completeness_score"] for r in scored) / len(scored) if scored else 0.0
    correct_scored = [r for r in scored if r["correctness_score"] is not None]
    avg_k = sum(r["correctness_score"] for r in correct_scored) / len(correct_scored) if correct_scored else None

    pct       = passed / total * 100
    bar_color = GREEN if pct >= 80 else YELLOW if pct >= 60 else RED

    print(f"\n{BOLD}{'='*70}{RESET}")
    print(f"{BOLD}EVAL RUN : {run_id[:8]}  |  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}{RESET}")
    print(f"{'='*70}")
    print(f"\n{BOLD}OVERALL  : {bar_color}{passed}/{total} passed ({pct:.1f}%){RESET}")
    print(f"           Faithfulness {avg_f:.2f}  |  Completeness {avg_c:.2f}  |  Correctness {_fmt(avg_k)}\n")

    categories = sorted(set(r["category"] for r in results))
    print(f"{BOLD}BY CATEGORY{'':>10}pass    faith  comp   correct  lat{RESET}")
    print(f"  {'-'*66}")
    for cat in categories:
        cat_res    = [r for r in results if r["category"] == cat]
        cat_passed = sum(1 for r in cat_res if r["passed"])
        cat_pct    = cat_passed / len(cat_res) * 100
        c_color    = GREEN if cat_pct == 100 else YELLOW if cat_pct >= 60 else RED

        cat_scored = [r for r in cat_res if cat not in ("attack", "out_of_scope")]
        avg_faith = sum(r["faithfulness_score"] for r in cat_scored) / len(cat_scored) if cat_scored else None
        avg_comp  = sum(r["completeness_score"] for r in cat_scored) / len(cat_scored) if cat_scored else None
        cat_corr  = [r for r in cat_scored if r["correctness_score"] is not None]
        avg_corr  = sum(r["correctness_score"] for r in cat_corr) / len(cat_corr) if cat_corr else None
        avg_lat   = sum(latencies.get(r["test_case_id"], 0) for r in cat_res) / len(cat_res)

        print(
            f"  {cat:<16} {c_color}{cat_passed}/{len(cat_res)} ({cat_pct:5.1f}%){RESET}"
            f"  {_fmt(avg_faith)}  {_fmt(avg_comp)}   {_fmt(avg_corr)}    {avg_lat/1000:.1f}s"
        )

    failures = [r for r in results if not r["passed"]]
    if failures:
        print(f"\n{BOLD}{RED}FAILURES{RESET}")
        for r in failures:
            print(f"  {RED}[FAIL]{RESET} {r['test_case_id']:<10} ({r['category']}): {r['failure_reason']}")
    else:
        print(f"\n{GREEN}{BOLD}All {total} cases passed!{RESET}")

    print(f"\n{'='*70}\n")

# ── Main ──────────────────────────────────────────────────────────────────────

async def main() -> None:
    dataset = json.loads(DATASET_PATH.read_text())
    run_id  = str(uuid.uuid4())
    token   = make_token(run_id)

    print(f"\n{BOLD}Eval run {run_id[:8]} — {len(dataset)} cases{RESET}\n")

    results:   list[dict]       = []
    latencies: dict[str, float] = {}

    async with httpx.AsyncClient() as client:
        for i, case in enumerate(dataset, 1):
            session_id = f"eval-{run_id[:8]}-{case['id']}"
            label = f"[{i:02d}/{len(dataset)}] {case['id']:<10} ({case['category']})"
            print(f"  {label}...", end=" ", flush=True)

            # Fire prior turns first for multi-turn cases
            for prior in case.get("prior_turns", []):
                await fire_query(client, prior["query"], session_id, token)

            body, status, latency_ms = await fire_query(client, case["query"], session_id, token)
            latencies[case["id"]] = latency_ms

            scored = await score_case(case, body, status)
            results.append(scored)

            result_str = f"{GREEN}PASS{RESET}" if scored["passed"] else f"{RED}FAIL{RESET}"
            corr_str   = f"  corr={scored['correctness_score']:.2f}" if scored["correctness_score"] is not None else ""
            print(f"{result_str}  ({latency_ms/1000:.1f}s){corr_str}")

            await asyncio.sleep(4)

    print("\nStoring results in Postgres...")
    await store_results(run_id, results, latencies)
    print_report(run_id, results, latencies)


if __name__ == "__main__":
    asyncio.run(main())
