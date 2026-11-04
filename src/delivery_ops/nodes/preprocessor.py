"""Preprocessor node: deduplicates, consolidates, assembles context via
tools, and runs the input guardrails before any LLM call is made."""
import time

from langsmith import traceable

from ..guardrails import (
    check_noise_override,
    consolidate_event,
    deduplicate_rows,
    scan_chunks_for_injection,
    scan_inputs_for_injection,
)
from ..runtime import PipelineContext
from ..state import RouterView, UnifiedAgentState, merge_back, project_into


def _fetch_context(ctx: PipelineContext, consolidated: dict, tool_log: list[str]) -> dict:
    """Fetch all context via tools: customer profiles, lockers, playbook, escalation rules."""
    tools = ctx.tools
    customer_id = consolidated["customer_id"]

    customer_profile = tools.lookup_customer_profile.invoke(
        {"customer_id": customer_id, "include_pii": False}
    )
    tool_log.append(f"TOOL: lookup_customer_profile({customer_id}, pii=False)")

    customer_profile_full = tools.lookup_customer_profile.invoke(
        {"customer_id": customer_id, "include_pii": True}
    )
    tool_log.append(f"TOOL: lookup_customer_profile({customer_id}, pii=True)")

    address_parts = consolidated["delivery_address"].split(",")
    zip_code = address_parts[-1].strip() if address_parts else ""
    locker_availability = tools.check_locker_availability.invoke(
        {"zip_code": zip_code, "package_size": consolidated["package_size"]}
    )
    tool_log.append(f"TOOL: check_locker_availability({zip_code}, {consolidated['package_size']})")

    query = f"{consolidated['status_code']} {consolidated['package_type']} {consolidated['status_description'][:100]}"
    playbook_context = tools.search_playbook.invoke({"query": query})
    tool_log.append("TOOL: search_playbook(query)")

    escalation_signals = tools.check_escalation_rules.invoke({
        "customer_tier": customer_profile.get("tier", "STANDARD"),
        "exceptions_last_90d": customer_profile.get("exceptions_last_90d", 0),
        "attempt_number": consolidated["attempt_number"],
        "package_type": consolidated["package_type"],
        "status_code": consolidated["status_code"],
        "status_description": consolidated["status_description"],
    })
    tool_log.append("TOOL: check_escalation_rules(...)")

    return {
        "customer_profile": customer_profile,
        "customer_profile_full": customer_profile_full,
        "locker_availability": locker_availability,
        "playbook_context": playbook_context,
        "escalation_signals": escalation_signals,
    }


def make_preprocessor_node(ctx: PipelineContext):
    @traceable(name="preprocessor_node")
    def preprocessor_node(state: UnifiedAgentState) -> UnifiedAgentState:
        """Deduplicates, consolidates, assembles context, and runs input guardrails."""
        view = project_into(state, RouterView)
        tool_log = []
        trajectory = []
        start = time.time()

        unique_rows = deduplicate_rows(view["raw_rows"])
        tool_log.append("PREPROCESSOR: Deduplicated rows")
        trajectory.append(f"preprocessor: {len(view['raw_rows'])} raw rows -> {len(unique_rows)} after dedup")

        consolidated = consolidate_event(unique_rows, view["raw_rows"])

        # Scan driver notes for prompt injection before any LLM call.
        if scan_inputs_for_injection(consolidated, view["raw_rows"]):
            tool_log.append("GUARDRAIL: Injection detected in delivery input")
            trajectory.append("preprocessor: Guardrail triggered - prompt injection detected")
            output = {
                "consolidated_event": consolidated,
                "customer_profile": {}, "customer_profile_full": {},
                "locker_availability": [], "playbook_context": [], "escalation_signals": {},
                "tool_calls_log": tool_log, "trajectory_log": trajectory,
                "resolution_revision_count": 0, "critic_feedback": "",
                "noise_override": False, "guardrail_triggered": True,
                "escalated": True, "start_time": start, "next_agent": "finalize",
            }
            return merge_back(state, output, RouterView)

        # Check for routine noise before making any tool calls.
        noise_override = check_noise_override(consolidated)
        if noise_override:
            tool_log.append("PREPROCESSOR: Noise guardrail - routine status with no anomaly")
            trajectory.append(f"preprocessor: {consolidated['status_code']} flagged as noise by guardrail, skipping tool calls")
            output = {
                "consolidated_event": consolidated,
                "customer_profile": {}, "customer_profile_full": {},
                "locker_availability": [], "playbook_context": [],
                "escalation_signals": {},
                "tool_calls_log": tool_log, "trajectory_log": trajectory,
                "resolution_revision_count": 0, "critic_feedback": "",
                "noise_override": True, "guardrail_triggered": False,
                "escalated": False, "start_time": start, "next_agent": "orchestrator",
            }
            return merge_back(state, output, RouterView)

        context = _fetch_context(ctx, consolidated, tool_log)

        # Scan retrieved playbook chunks for injection.
        if scan_chunks_for_injection(context["playbook_context"]):
            tool_log.append("GUARDRAIL: Injection detected in retrieved playbook chunk")
            trajectory.append("preprocessor: Guardrail triggered - injection in RAG chunk")
            context["playbook_context"] = []
            output = {
                "consolidated_event": consolidated,
                **context,
                "tool_calls_log": tool_log, "trajectory_log": trajectory,
                "resolution_revision_count": 0, "critic_feedback": "",
                "noise_override": False, "guardrail_triggered": True,
                "escalated": True, "start_time": start, "next_agent": "finalize",
            }
            return merge_back(state, output, RouterView)

        output = {
            "consolidated_event": consolidated,
            **context,
            "tool_calls_log": tool_log, "trajectory_log": trajectory,
            "resolution_revision_count": 0, "critic_feedback": "",
            "noise_override": noise_override, "guardrail_triggered": False,
            "escalated": False, "start_time": start, "next_agent": "orchestrator",
        }
        return merge_back(state, output, RouterView)

    return preprocessor_node
