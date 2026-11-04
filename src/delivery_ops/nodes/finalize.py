"""Finalize node: packages the output and records end-to-end latency."""
import json
import time

from langsmith import traceable

from ..runtime import PipelineContext
from ..state import RouterView, UnifiedAgentState, merge_back, project_into


def make_finalize_node(ctx: PipelineContext):
    @traceable(name="finalize_node")
    def finalize_node(state: UnifiedAgentState) -> UnifiedAgentState:
        """Packages final results and records end time."""
        view = project_into(state, RouterView)

        if view.get("guardrail_triggered"):
            final = {
                "shipment_id": view["shipment_id"],
                "is_exception": "BLOCKED",
                "resolution": "ESCALATED",
                "escalated": True,
                "tone": "N/A",
                "message": "This shipment was flagged by the input guardrail and requires human review.",
                "revision_count": 0,
                "guardrail_blocked": True,
            }
        else:
            final = {
                "shipment_id": view["shipment_id"],
                "is_exception": view.get("resolution_output", {}).get("is_exception", "ERROR"),
                "resolution": view.get("resolution_output", {}).get("resolution", "ERROR"),
                "escalated": view["escalated"],
                "tone": view.get("communication_output", {}).get("tone_label", "N/A"),
                "message": view.get("communication_output", {}).get("communication_message", ""),
                "revision_count": view["resolution_revision_count"],
                "guardrail_blocked": False,
            }

        latency = time.time() - view["start_time"] if view.get("start_time") else 0.0

        output = {
            "final_actions": [final],
            "latency_sec": latency,
            "next_agent": "END",
        }

        state["trajectory_log"].append(
            f"finalize: actions={json.dumps(final)}; latency={latency:.3f}s"
        )

        return merge_back(state, output, RouterView)

    return finalize_node
