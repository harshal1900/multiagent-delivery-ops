"""Communication Agent node: writes the customer notification. This is the
only node whose view includes the customer's name (customer_profile_full)."""
import json

from langchain_core.messages import HumanMessage, SystemMessage
from langsmith import traceable

from .. import config
from ..prompts import COMMUNICATION_AGENT_SYSTEM_PROMPT
from ..runtime import PipelineContext
from ..schemas import CommunicationOutput
from ..state import CommunicationAgentView, UnifiedAgentState, merge_back, project_into


def _build_communication_context(view: dict) -> tuple[dict, str]:
    """Build the context dict and locker info string for the Communication
    Agent. Only uses fields available in CommunicationAgentView."""
    event = view["consolidated_event"]
    profile = view["customer_profile_full"]
    resolution = view["resolution_output"]
    lockers = view["locker_availability"]

    locker_info = ""
    if resolution.get("resolution") == "REROUTE_TO_LOCKER":
        eligible = [l for l in lockers if l.get("eligible")]
        if eligible:
            locker_info = f"\nLOCKER FOR REROUTE:\n{json.dumps(eligible[0], indent=2)}"

    comm_context = {
        "customer_name": profile.get("name", "Customer"),
        "customer_tier": profile.get("tier"),
        "preferred_channel": profile.get("preferred_channel"),
        "active_credit": profile.get("active_credit", 0),
        "exception_type": event["status_code"],
        "status_description": event["status_description"],
        "package_type": event["package_type"],
        "resolution": resolution.get("resolution"),
        "resolution_rationale": resolution.get("rationale"),
    }

    return comm_context, locker_info


def make_communication_agent_node(ctx: PipelineContext):
    @traceable(name="communication_agent_node")
    def communication_agent_node(state: UnifiedAgentState) -> UnifiedAgentState:
        """Communication Agent: generates customer notification. Only agent with PII access."""
        view = project_into(state, CommunicationAgentView)
        comm_context, locker_info = _build_communication_context(view)
        user_content = f"CONTEXT:\n{json.dumps(comm_context, indent=2)}\n{locker_info}"

        structured_llm = ctx.gen_llm.with_structured_output(CommunicationOutput)
        max_retries = config.COMMUNICATION_MAX_RETRIES
        result = None

        for attempt in range(max_retries):
            try:
                result = structured_llm.invoke([
                    SystemMessage(content=COMMUNICATION_AGENT_SYSTEM_PROMPT),
                    HumanMessage(content=user_content),
                ])
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    state["trajectory_log"].append(
                        f"communication_agent: Validation failed (attempt {attempt + 1}), retrying - {str(e)[:100]}"
                    )
                else:
                    tier = view["customer_profile_full"].get("tier", "STANDARD")
                    result = CommunicationOutput(
                        tone_label="FORMAL" if tier in ("VIP", "PREMIUM") else "CASUAL",
                        communication_message="We're aware of an issue with your delivery and are working to resolve it. A team member will follow up shortly.",
                    )
                    state["escalated"] = True
                    state["trajectory_log"].append(
                        f"communication_agent: All {max_retries} retries exhausted, "
                        f"defaulting to generic message with forced escalation"
                    )

        agent_output = {"communication_output": result.model_dump()}
        state = merge_back(state, agent_output, CommunicationAgentView)

        state["tool_calls_log"].append("AGENT: communication_agent invoked")
        state["trajectory_log"].append(f"communication_agent: tone={result.tone_label}")
        state["next_agent"] = "orchestrator"
        return state

    return communication_agent_node
