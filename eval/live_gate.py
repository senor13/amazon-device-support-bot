"""
Live eval gate — runs 1 case per category against the live app and compares
against the baseline in eval/baselines/latest.json.

Exits 0 if no regressions, exits 1 if any regression detected.
Prints a markdown metrics table — GitHub Actions captures this for the PR comment.

Usage:
    python -m eval.live_gate
"""
import asyncio
import json
import os
import sys
import time
import uuid
from pathlib import Path

import httpx
from dotenv import load_dotenv
from jose import jwt

load_dotenv()

from eval.config import eval_settings
from eval.runner import fire_query, score_case, estimate_cost, wait_for_app

DATASET_PATH   = Path(eval_settings.DATASET_FILE)
BASELINE_PATH  = Path(eval_settings.BASELINE_FILE)
JWT_SECRET     = os.environ["JWT_SECRET"]


def make_token(run_id: str) -> str:
    payload = {"sub": f"gate-{run_id[:8]}", "exp": int(time.time()) + 3600}
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")


def compare_with_baseline(results: list[dict], latencies: dict[str, float]) -> dict:
    """Compare gate results against baseline. Returns regression report."""
    if not BASELINE_PATH.exists():
        return {"baseline_available": False, "regressions": []}

    baseline      = json.loads(BASELINE_PATH.read_text())
    baseline_sum  = baseline.get("summary", {})
    regressions   = []

    total    = len(results)
    passed   = sum(1 for r in results if r["passed"])
    pass_rate = passed / total if total else 0.0

    scored = [r for r in results if r["category"] not in ("attack", "out_of_scope")]
    avg_faith    = sum(r["faithfulness_score"]  for r in scored) / len(scored) if scored else 0.0
    avg_complete = sum(r["completeness_score"]  for r in scored) / len(scored) if scored else 0.0
    avg_latency  = sum(latencies.values()) / len(latencies) if latencies else 0.0
    total_cost   = sum(estimate_cost(r["model_used"], r["bot_response"]) for r in results)

    # Pass rate — any drop is a regression
    baseline_pass_rate = baseline_sum.get("pass_rate", 1.0)
    if pass_rate < baseline_pass_rate:
        regressions.append({
            "metric": "pass_rate",
            "baseline": f"{baseline_pass_rate:.0%}",
            "current":  f"{pass_rate:.0%}",
            "detail": "Pass rate dropped",
        })

    # Per-case regression — previously passing case now fails
    baseline_cases = {r["id"]: r for r in baseline.get("results", [])}
    for r in results:
        prev = baseline_cases.get(r["test_case_id"])
        if prev and prev.get("passed") and not r["passed"]:
            regressions.append({
                "metric": f"{r['id']} regression",
                "baseline": "PASS",
                "current":  "FAIL",
                "detail": r.get("failure_reason", ""),
            })

    # Latency regression
    baseline_latency = baseline_sum.get("avg_latency_ms", 0)
    if baseline_latency > 0:
        change = (avg_latency - baseline_latency) / baseline_latency
        if change > eval_settings.LATENCY_REGRESSION_THRESHOLD:
            regressions.append({
                "metric": "avg_latency_ms",
                "baseline": f"{baseline_latency:.0f}ms",
                "current":  f"{avg_latency:.0f}ms",
                "detail": f"+{change*100:.1f}% (threshold {eval_settings.LATENCY_REGRESSION_THRESHOLD*100:.0f}%)",
            })

    # Cost regression
    baseline_cost = baseline_sum.get("total_estimated_cost_usd", 0)
    if baseline_cost > 0:
        # Scale baseline cost to gate case count (baseline has 28 cases, gate has 7)
        scale  = total / baseline_sum.get("total_cases", total)
        scaled_baseline_cost = baseline_cost * scale
        change = (total_cost - scaled_baseline_cost) / scaled_baseline_cost if scaled_baseline_cost else 0
        if change > eval_settings.COST_REGRESSION_THRESHOLD:
            regressions.append({
                "metric": "estimated_cost_usd",
                "baseline": f"${scaled_baseline_cost:.4f}",
                "current":  f"${total_cost:.4f}",
                "detail": f"+{change*100:.1f}% (threshold {eval_settings.COST_REGRESSION_THRESHOLD*100:.0f}%)",
            })

    return {
        "baseline_available": True,
        "regressions": regressions,
        "has_regressions": len(regressions) > 0,
        "summary": {
            "pass_rate":    pass_rate,
            "avg_faith":    avg_faith,
            "avg_complete": avg_complete,
            "avg_latency":  avg_latency,
            "total_cost":   total_cost,
        },
        "baseline_summary": baseline_sum,
    }


def print_report(results: list[dict], latencies: dict[str, float], comparison: dict) -> None:
    """Print a markdown-formatted report for the PR comment."""
    print("\n## Live Eval Gate Results\n")

    # Per-case table
    print("| Case | Category | Status | Faithfulness | Completeness | Latency |")
    print("|------|----------|--------|-------------|--------------|---------|")
    for r in results:
        status   = "✅ PASS" if r["passed"] else "❌ FAIL"
        faith    = f"{r['faithfulness_score']:.2f}" if r["faithfulness_score"] else "—"
        complete = f"{r['completeness_score']:.2f}" if r["completeness_score"] else "—"
        latency  = f"{latencies.get(r['test_case_id'], 0):.0f}ms"
        print(f"| {r['test_case_id']} | {r['category']} | {status} | {faith} | {complete} | {latency} |")

    # Summary vs baseline
    if comparison["baseline_available"]:
        s  = comparison["summary"]
        bs = comparison["baseline_summary"]
        print("\n### vs Baseline\n")
        print("| Metric | Baseline | Current |")
        print("|--------|----------|---------|")
        print(f"| Pass rate    | {bs.get('pass_rate', 0):.0%} | {s['pass_rate']:.0%} |")
        print(f"| Faithfulness | {bs.get('avg_faithfulness', 0):.2f} | {s['avg_faith']:.2f} |")
        print(f"| Completeness | {bs.get('avg_completeness', 0):.2f} | {s['avg_complete']:.2f} |")
        print(f"| Avg latency  | {bs.get('avg_latency_ms', 0):.0f}ms | {s['avg_latency']:.0f}ms |")
        print(f"| Est. cost    | ${bs.get('total_estimated_cost_usd', 0) * (len(results) / bs.get('total_cases', len(results))):.4f} | ${s['total_cost']:.4f} |")

    # Regressions
    if comparison["has_regressions"]:
        print("\n### ⚠️ Regressions Detected\n")
        for reg in comparison["regressions"]:
            print(f"- **{reg['metric']}**: {reg['baseline']} → {reg['current']} ({reg['detail']})")
    else:
        print("\n✅ No regressions vs baseline.")


async def main() -> None:
    dataset    = json.loads(DATASET_PATH.read_text())
    gate_cases = [c for c in dataset if c.get("is_gate_case")]

    print(f"Live eval gate — {len(gate_cases)} cases")

    run_id  = str(uuid.uuid4())
    token   = make_token(run_id)
    results:   list[dict]       = []
    latencies: dict[str, float] = {}

    async with httpx.AsyncClient() as client:
        await wait_for_app(client)
        for i, case in enumerate(gate_cases, 1):
            session_id = f"gate-{run_id[:8]}-{case['id']}"
            print(f"[{i}/{len(gate_cases)}] {case['id']} ({case['category']})...", end=" ", flush=True)

            for prior in case.get("prior_turns", []):
                await fire_query(client, prior["query"], session_id, token)

            body, status, latency_ms = await fire_query(client, case["query"], session_id, token)
            latencies[case["id"]] = latency_ms

            scored = await score_case(case, body, status)
            results.append(scored)

            print("PASS" if scored["passed"] else f"FAIL — {scored['failure_reason']}")

    comparison = compare_with_baseline(results, latencies)
    print_report(results, latencies, comparison)

    if comparison["has_regressions"]:
        print("\n❌ Gate FAILED — regressions detected. Merge blocked.")
        sys.exit(1)

    print("\n✅ Gate PASSED.")
    sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())
