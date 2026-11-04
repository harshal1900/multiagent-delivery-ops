#!/usr/bin/env python3
"""End-to-end CLI demo: builds the pipeline, runs it over every shipment in
the sample dataset (or DATA_DIR, if set), prints a per-shipment report and
the aggregate metrics.

Usage:
    cp .env.example .env   # fill in OPENAI_API_KEY / OPENAI_API_BASE
    pip install -r requirements.txt
    python scripts/run_demo.py
"""
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from delivery_ops.evaluation import print_test_case_output, run_single_test_case  # noqa: E402
from delivery_ops.graph import build_graph  # noqa: E402
from delivery_ops.runtime import build_context  # noqa: E402


def main() -> None:
    ctx = build_context()
    app = build_graph(ctx)

    all_logs = ctx.tools.read_delivery_logs.invoke({})
    shipment_groups = defaultdict(list)
    for row in all_logs:
        shipment_groups[row["shipment_id"]].append(row)
    unique_shipment_ids = list(dict.fromkeys(row["shipment_id"] for row in all_logs))

    gt_by_shipment = defaultdict(list)
    for _, row in ctx.ingested.ground_truth_df.iterrows():
        gt_by_shipment[row["shipment_id"]].append(row.to_dict())

    gt_consolidated = {}
    for sid, rows in gt_by_shipment.items():
        exc_rows = [r for r in rows if r["is_exception"] == "YES"]
        gt_consolidated[sid] = exc_rows[-1] if exc_rows else rows[0]

    print(f"Shipments to process: {unique_shipment_ids}\n")

    all_results = {}
    for sid in unique_shipment_ids:
        print(f"\n===== {sid} =====")
        all_results[sid] = run_single_test_case(ctx, app, sid, shipment_groups[sid], gt_consolidated[sid])
        print_test_case_output(sid, all_results[sid], gt_consolidated[sid])

    task_flags = [r["task_completion"]["task_complete"] for r in all_results.values()]
    exc_flags = [r["task_completion"]["exception_correct"] for r in all_results.values()]
    res_flags = [r["task_completion"]["resolution_correct"] for r in all_results.values()]
    tone_flags = [r["task_completion"]["tone_correct"] for r in all_results.values()]
    esc_flags = [r["escalation_correct"] for r in all_results.values() if r["escalation_correct"] is not None]
    tool_flags = [r["tool_call_correct"] for r in all_results.values()]
    coherence_scores = [r["coherence"]["score"] for r in all_results.values()]
    latencies = [r["latency"] for r in all_results.values()]

    print("\n\n===== Aggregate metrics =====")
    print("Test cases run:", len(all_results))
    print("Task completion rate:", round(100 * sum(task_flags) / len(task_flags), 1), "percent")
    print("Tasks completed:", sum(task_flags), "of", len(task_flags))
    print("Exception classification correct:", sum(exc_flags), "of", len(exc_flags))
    print("Resolution action correct:", sum(res_flags), "of", len(res_flags))
    print("Communication tone correct:", sum(tone_flags), "of", len(tone_flags))
    if esc_flags:
        print("Escalation accuracy:", round(100 * sum(esc_flags) / len(esc_flags), 1), "percent")
    print("Escalation cases evaluated:", len(esc_flags))
    print("Tool call accuracy:", round(100 * sum(tool_flags) / len(tool_flags), 1), "percent")
    print("Mean coherence score:", round(sum(coherence_scores) / len(coherence_scores), 2), "out of 5")
    print("Mean end-to-end latency:", round(sum(latencies) / len(latencies), 2), "seconds")
    print("Maximum end-to-end latency:", round(max(latencies), 2), "seconds")
    print("Total pipeline time:", round(sum(latencies), 2), "seconds")


if __name__ == "__main__":
    main()
