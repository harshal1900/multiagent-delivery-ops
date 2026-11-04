"""The five evaluation metrics (task completion, escalation accuracy, tool
call accuracy, reasoning trajectory coherence, end-to-end latency) and the
single-shipment test runner used by the CLI and the test suite."""
import json

from langchain_core.messages import HumanMessage, SystemMessage

from . import config
from .prompts import REASONING_TRAJECTORY_COHERENCE_PROMPT
from .runtime import PipelineContext
from .state import UnifiedAgentState


def compute_task_completion(gt: dict, pred: dict) -> dict:
    """Compute task completion for a single shipment."""
    res = pred.get("resolution_output", {})
    comm = pred.get("communication_output", {})

    exception_correct = gt.get("is_exception") == res.get("is_exception")

    if gt.get("is_exception") == "YES":
        resolution_correct = gt.get("expected_resolution") == res.get("resolution")
        tone_correct = gt.get("expected_tone") == comm.get("tone_label", "N/A")
    else:
        resolution_correct = res.get("resolution") in ("N/A", None, "")
        tone_correct = True

    return {
        "exception_correct": exception_correct,
        "resolution_correct": resolution_correct,
        "tone_correct": tone_correct,
        "task_complete": exception_correct and resolution_correct and tone_correct,
    }


def compute_escalation_accuracy(gt: dict, pred: dict):
    """Check if escalation decision matches ground truth for a single shipment."""
    if gt.get("should_escalate") not in ("YES", "NO"):
        return None
    pred_escalated = "YES" if pred.get("escalated", False) else "NO"
    return gt.get("should_escalate") == pred_escalated


def compute_tool_call_accuracy(gt: dict, pred: dict) -> bool:
    """Check if correct tools were invoked for a single shipment."""
    tool_log_str = " ".join(pred.get("tool_calls_log", []))
    is_exception = gt.get("is_exception", "NO")
    noise_override = pred.get("noise_override", False)

    if noise_override:
        skip_tools = ["lookup_customer_profile", "check_locker_availability",
                      "search_playbook", "check_escalation_rules",
                      "resolution_agent", "communication_agent"]
        return not any(t in tool_log_str for t in skip_tools)

    required = ["lookup_customer_profile", "check_locker_availability",
                "search_playbook", "check_escalation_rules", "resolution_agent"]

    if is_exception == "YES":
        required.append("communication_agent")

    all_present = all(t in tool_log_str for t in required)

    if is_exception == "NO":
        all_present = all_present and "communication_agent" not in tool_log_str

    return all_present


def compute_coherence_score(ctx: PipelineContext, pred: dict) -> dict:
    """Use the evaluation LLM to score reasoning trajectory coherence."""
    trace = {
        "shipment_id": pred.get("shipment_id"),
        "consolidated_event": pred.get("consolidated_event", {}),
        "customer_tier": pred.get("customer_profile", {}).get("tier"),
        "escalation_signals": pred.get("escalation_signals", {}),
        "resolution_output": pred.get("resolution_output", {}),
        "critic_resolution_output": pred.get("critic_resolution_output", {}),
        "resolution_revision_count": pred.get("resolution_revision_count", 0),
        "communication_output": pred.get("communication_output", {}),
        "critic_communication_output": pred.get("critic_communication_output", {}),
        "trajectory_log": pred.get("trajectory_log", []),
    }

    try:
        response = ctx.eval_llm.invoke([
            SystemMessage(content=REASONING_TRAJECTORY_COHERENCE_PROMPT),
            HumanMessage(content=json.dumps(trace, indent=2)),
        ])
        content = response.content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        parsed = json.loads(content)
        return {"score": parsed.get("score", 0), "justification": parsed.get("justification", "")}
    except Exception as e:
        return {"score": 0, "justification": str(e)}


def run_single_test_case(ctx: PipelineContext, app, shipment_id: str, rows: list[dict], gt: dict) -> dict:
    """Execute the pipeline for a single shipment and compute all metrics."""
    initial_state: UnifiedAgentState = {
        "raw_rows": rows,
        "shipment_id": shipment_id,
        "consolidated_event": {},
        "customer_profile": {},
        "customer_profile_full": {},
        "locker_availability": [],
        "playbook_context": [],
        "escalation_signals": {},
        "resolution_output": {},
        "critic_resolution_output": {},
        "resolution_revision_count": 0,
        "critic_feedback": "",
        "communication_output": {},
        "critic_communication_output": {},
        "next_agent": "resolution_agent",
        "max_loops": config.MAX_REVISION_LOOPS,
        "escalated": False,
        "tool_calls_log": [],
        "trajectory_log": [],
        "start_time": None,
        "latency_sec": None,
        "final_actions": [],
        "noise_override": False,
        "guardrail_triggered": False,
    }

    result = app.invoke(initial_state)

    task = compute_task_completion(gt, result)
    esc_acc = compute_escalation_accuracy(gt, result)
    tool_acc = compute_tool_call_accuracy(gt, result)
    coherence = compute_coherence_score(ctx, result)

    citations = [f"Page {c['page']}" for c in result.get("playbook_context", [])]

    return {
        "state": result,
        "task_completion": task,
        "escalation_correct": esc_acc,
        "tool_call_correct": tool_acc,
        "coherence": coherence,
        "latency": result.get("latency_sec", 0.0),
        "citations": citations,
    }


def print_test_case_output(sid: str, result: dict, gt: dict) -> None:
    """Print a single test case result."""
    state = result["state"]
    res = state.get("resolution_output", {})
    comm = state.get("communication_output", {})
    tc = result["task_completion"]

    print("--- Predictions ---")
    print(f"  Exception:  {res.get('is_exception', 'N/A')} (GT: {gt.get('is_exception')})")
    print(f"  Resolution: {res.get('resolution', 'N/A')} (GT: {gt.get('expected_resolution')})")
    pred_esc = "YES" if state.get("escalated") else "NO"
    print(f"  Escalated:  {pred_esc} (GT: {gt.get('should_escalate')})")
    print(f"  Tone:       {comm.get('tone_label', 'N/A')} (GT: {gt.get('expected_tone')})")
    print(f"  Revisions:  {state.get('resolution_revision_count', 0)}")
    print(f"  Guardrail:  {'TRIGGERED' if state.get('guardrail_triggered') else 'CLEAR'}")

    print("\n--- Metrics ---")
    print(f"  Task Complete:       {'PASS' if tc['task_complete'] else 'FAIL'}")
    print(f"    Exception ID:      {'PASS' if tc['exception_correct'] else 'FAIL'}")
    print(f"    Resolution:        {'PASS' if tc['resolution_correct'] else 'FAIL'}")
    print(f"    Tone:              {'PASS' if tc['tone_correct'] else 'FAIL'}")
    esc_str = "N/A" if result["escalation_correct"] is None else ("PASS" if result["escalation_correct"] else "FAIL")
    print(f"  Escalation Accuracy: {esc_str}")
    print(f"  Tool Call Accuracy:  {'PASS' if result['tool_call_correct'] else 'FAIL'}")
    print(f"  Coherence Score:     {result['coherence']['score']}/5")
    print(f"  Latency:             {result['latency']:.2f}s")

    print("\n--- Trajectory ---")
    for entry in state.get("trajectory_log", []):
        print(f"  {entry}")

    print("\n--- Document Citations ---")
    print(f"  Playbook pages referenced: {', '.join(result['citations']) if result['citations'] else 'None'}")

    if comm.get("communication_message"):
        print("\n--- Customer Message ---")
        msg = comm["communication_message"]
        print(f"  {msg[:200]}{'...' if len(msg) > 200 else ''}")
