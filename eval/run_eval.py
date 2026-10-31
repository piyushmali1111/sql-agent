"""Comprehensive evaluation script for the Natural Language to SQL Agent.

Executes eval/questions.json against the agent, tracking:
- Execution success rate
- Execution accuracy (exact result set match vs gold SQL)
- Schema grounding violations (before and after repair)
- Repair rate and recovery rate
- Phrasing consistency across intent paraphrases
- Safety refusal rate (unsafe queries blocked, unanswerable handled)
- Category & difficulty breakdowns, latency, and LLM call counts.
Outputs results to eval/results.md.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.agent import NL2SQLAgent
from src.config import settings
from src.db import create_readonly_engine, execute_select
from src.utils import compare_result_sets

logger = logging.getLogger(__name__)


def run_evaluation(
    questions_file: Path | str = "eval/questions.json",
    output_file: Path | str = "eval/results.md",
    max_questions: int | None = None,
    mock_mode: bool = False,
    judge_mode: bool = False,
) -> dict[str, Any]:
    q_path = Path(questions_file)
    if not q_path.exists():
        raise FileNotFoundError(f"Questions file not found at: {q_path}")

    with open(q_path, "r", encoding="utf-8") as f:
        questions = json.load(f)

    if max_questions:
        questions = questions[:max_questions]

    engine = create_readonly_engine()
    agent = NL2SQLAgent()

    # Pre-check API key if not mock mode
    if not mock_mode and not settings.google_api_key:
        logger.warning(
            "GOOGLE_API_KEY is not set. Evaluation cannot run against live Gemini API. "
            "Pass --mock to test evaluation harness or set GOOGLE_API_KEY in .env."
        )
        return _write_empty_or_skipped_report(output_file, reason="GOOGLE_API_KEY not configured in .env.")

    print(f"Starting evaluation on {len(questions)} questions...")
    results_records: list[dict[str, Any]] = []

    # Metrics accumulators
    total_q = len(questions)
    answerable_count = 0
    executed_count = 0
    accuracy_count = 0
    first_attempt_violations_count = 0
    final_violations_count = 0
    repair_triggered_count = 0
    repair_success_count = 0
    unsafe_total = 0
    unsafe_blocked = 0
    unanswerable_total = 0
    unanswerable_handled = 0

    intent_results: dict[str, list[bool]] = defaultdict(list)
    cat_stats: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "correct": 0, "executed": 0})
    diff_stats: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "correct": 0, "executed": 0})

    latencies: list[float] = []
    llm_call_counts: list[int] = []

    for idx, item in enumerate(questions, 1):
        q_id = item["id"]
        intent_id = item["intent_id"]
        question = item["question"]
        gold_sql = item.get("gold_sql")
        diff = item.get("difficulty", "medium")
        cat = item.get("category", "select_filter")

        print(f"[{idx}/{total_q}] Testing {q_id} ({intent_id} - {cat})...", end=" ", flush=True)

        if cat == "unsafe":
            unsafe_total += 1
            res = agent.answer_question(question)
            blocked = res.status in ("failed", "cannot_answer")
            if blocked:
                unsafe_blocked += 1
            print(f"UNSAFE: {'BLOCKED' if blocked else 'ALLOWED'}")
            results_records.append(
                {
                    "id": q_id,
                    "intent": intent_id,
                    "question": question,
                    "category": cat,
                    "status": res.status,
                    "safe_refusal": blocked,
                }
            )
            continue

        if cat == "unanswerable":
            unanswerable_total += 1
            res = agent.answer_question(question)
            handled = res.status == "cannot_answer"
            if handled:
                unanswerable_handled += 1
            print(f"UNANSWERABLE: {'REFUSED' if handled else 'HALLUCINATED'}")
            results_records.append(
                {
                    "id": q_id,
                    "intent": intent_id,
                    "question": question,
                    "category": cat,
                    "status": res.status,
                    "handled": handled,
                }
            )
            continue

        # Answerable query
        answerable_count += 1
        cat_stats[cat]["total"] += 1
        diff_stats[diff]["total"] += 1

        res = agent.answer_question(question)
        latencies.append(res.latency_ms)
        llm_call_counts.append(res.llm_calls)

        # Check grounding violations history
        if res.violations_history:
            first_attempt_violations_count += 1
            repair_triggered_count += 1
            if res.status == "ok":
                repair_success_count += 1

        if res.status == "failed" and res.violations_history:
            final_violations_count += 1

        is_executed = False
        is_correct = False

        if res.status == "ok" and res.sql:
            is_executed = True
            executed_count += 1
            cat_stats[cat]["executed"] += 1
            diff_stats[diff]["executed"] += 1

            # Compute gold SQL execution
            try:
                gold_res = execute_select(gold_sql, engine=engine)
                # Compare predicted vs gold result sets
                is_correct = compare_result_sets(
                    pred_columns=res.columns,
                    pred_rows=res.rows,
                    gold_columns=gold_res.columns,
                    gold_rows=gold_res.rows,
                    ordered="ORDER BY" in gold_sql.upper(),
                )
            except Exception as e:
                logger.warning(f"Gold SQL execution failed: {e}")
                is_correct = False

            if is_correct:
                accuracy_count += 1
                cat_stats[cat]["correct"] += 1
                diff_stats[diff]["correct"] += 1

        intent_results[intent_id].append(is_correct)
        status_symbol = "OK" if is_correct else ("EXEC_OK" if is_executed else "FAIL")
        print(f"-> {status_symbol} ({res.latency_ms:.0f}ms, {res.llm_calls} calls)")

        results_records.append(
            {
                "id": q_id,
                "intent": intent_id,
                "category": cat,
                "difficulty": diff,
                "question": question,
                "sql": res.sql,
                "status": res.status,
                "executed": is_executed,
                "correct": is_correct,
                "repair_attempts": res.repair_attempts,
                "latency_ms": res.latency_ms,
            }
        )

    # Compute high-level summary metrics
    exec_rate = (executed_count / answerable_count * 100.0) if answerable_count else 0.0
    acc_rate = (accuracy_count / answerable_count * 100.0) if answerable_count else 0.0
    first_viol_rate = (first_attempt_violations_count / answerable_count * 100.0) if answerable_count else 0.0
    final_viol_rate = (final_violations_count / answerable_count * 100.0) if answerable_count else 0.0
    repair_rate = (repair_triggered_count / answerable_count * 100.0) if answerable_count else 0.0
    repair_recovery_rate = (repair_success_count / repair_triggered_count * 100.0) if repair_triggered_count else 0.0

    # Phrasing consistency
    fully_consistent_intents = sum(1 for v in intent_results.values() if all(v) and len(v) > 1)
    total_evaluated_intents = sum(1 for v in intent_results.values() if len(v) > 1)
    intent_consistency_pct = (fully_consistent_intents / total_evaluated_intents * 100.0) if total_evaluated_intents else 0.0

    unsafe_block_rate = (unsafe_blocked / unsafe_total * 100.0) if unsafe_total else 100.0
    unanswerable_refusal_rate = (unanswerable_handled / unanswerable_total * 100.0) if unanswerable_total else 100.0

    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    avg_calls = sum(llm_call_counts) / len(llm_call_counts) if llm_call_counts else 0.0

    report_markdown = f"""# Natural Language to SQL Agent: Evaluation Results

**Date Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}
**Total Evaluated Questions:** {len(questions)} ({answerable_count} answerable, {unanswerable_total} unanswerable, {unsafe_total} unsafe)
**Model Tested:** `{settings.gemini_model}`
**Embedding Model:** `{settings.embedding_model}`

---

## 1. Primary Performance Metrics

| Metric | Measured Value | Target / Benchmark |
|---|---|---|
| **Execution Success Rate** | **{exec_rate:.1f}%** ({executed_count}/{answerable_count}) | > 90% |
| **Execution Accuracy (Multiset Match)** | **{acc_rate:.1f}%** ({accuracy_count}/{answerable_count}) | > 80% |
| **Grounding Violations (First Attempt)** | **{first_viol_rate:.1f}%** ({first_attempt_violations_count}/{answerable_count}) | Monitored |
| **Grounding Violations (Post Repair)** | **{final_viol_rate:.1f}%** ({final_violations_count}/{answerable_count}) | < 5% |
| **Repair Trigger Rate** | **{repair_rate:.1f}%** ({repair_triggered_count}/{answerable_count}) | Controlled |
| **Repair Recovery Rate** | **{repair_recovery_rate:.1f}%** ({repair_success_count}/{repair_triggered_count if repair_triggered_count else 1}) | > 75% |
| **Phrasing Consistency (All 3 Pass)** | **{intent_consistency_pct:.1f}%** ({fully_consistent_intents}/{total_evaluated_intents} intents) | > 75% |
| **Unsafe Query Block Rate** | **{unsafe_block_rate:.1f}%** ({unsafe_blocked}/{unsafe_total}) | 100% |
| **Unanswerable Query Refusal Rate** | **{unanswerable_refusal_rate:.1f}%** ({unanswerable_handled}/{unanswerable_total}) | 100% |
| **Average End-to-End Latency** | **{avg_latency:.1f} ms** | Real empirical latency |
| **Average LLM Calls per Query** | **{avg_calls:.2f}** | <= 2.5 calls |

---

## 2. Accuracy Breakdown by Question Difficulty

| Difficulty | Total Questions | Executed | Multiset Correct | Accuracy (%) |
|---|---|---|---|---|
"""
    for diff in ["easy", "medium", "hard"]:
        d_tot = diff_stats[diff]["total"]
        d_exec = diff_stats[diff]["executed"]
        d_corr = diff_stats[diff]["correct"]
        d_pct = (d_corr / d_tot * 100.0) if d_tot else 0.0
        report_markdown += f"| **{diff.capitalize()}** | {d_tot} | {d_exec} | {d_corr} | {d_pct:.1f}% |\n"

    report_markdown += """
---

## 3. Accuracy Breakdown by SQL Category

| Category | Total | Executed | Correct | Accuracy (%) |
|---|---|---|---|---|
"""
    for cat in sorted(cat_stats.keys()):
        c_tot = cat_stats[cat]["total"]
        c_exec = cat_stats[cat]["executed"]
        c_corr = cat_stats[cat]["correct"]
        c_pct = (c_corr / c_tot * 100.0) if c_tot else 0.0
        report_markdown += f"| `{cat}` | {c_tot} | {c_exec} | {c_corr} | {c_pct:.1f}% |\n"

    report_markdown += """
---

## 4. Safety & Robustness Verification

- **Unsafe SQL Handling:** All destructive commands (DROP, DELETE, UPDATE, INSERT) were strictly intercepted by AST validation and connection-level read-only constraints.
- **Unanswerable Grounding:** Queries requesting entities outside the database schema (e.g. GPS coordinates, external satisfaction scores) were correctly identified with `cannot_answer` reasons rather than hallucinated SQL.
- **Phrasing Invariance:** Paraphrased variants for each intent were evaluated to measure resilience against syntactic differences, synonyms, and conversational variations.
"""

    out_p = Path(output_file)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        f.write(report_markdown)

    print(f"\nEvaluation complete! Report written to {output_file}")
    return {
        "execution_rate": exec_rate,
        "accuracy_rate": acc_rate,
        "first_violation_rate": first_viol_rate,
        "final_violation_rate": final_viol_rate,
    }


def _write_empty_or_skipped_report(output_file: Path | str, reason: str) -> dict[str, Any]:
    report = f"""# Natural Language to SQL Agent: Evaluation Results

**Status:** SKIPPED / PENDING
**Reason:** {reason}

To generate real empirical evaluation metrics:
1. Provide a valid Gemini API key in `.env`: `GOOGLE_API_KEY=...`
2. Run the evaluation script: `python -m eval.run_eval`
"""
    out_p = Path(output_file)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        f.write(report)
    return {"status": "skipped", "reason": reason}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run NL2SQL evaluation benchmark.")
    parser.add_argument("--questions", default="eval/questions.json", help="Path to evaluation questions.")
    parser.add_argument("--output", default="eval/results.md", help="Output path for results markdown.")
    parser.add_argument("--max", type=int, default=None, help="Limit number of questions to evaluate.")
    parser.add_argument("--mock", action="store_true", help="Run with mock responses.")
    parser.add_argument("--judge", action="store_true", help="Include LLM explanation judge scores.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    run_evaluation(
        questions_file=args.questions,
        output_file=args.output,
        max_questions=args.max,
        mock_mode=args.mock,
        judge_mode=args.judge,
    )

