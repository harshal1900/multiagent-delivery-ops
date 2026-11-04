"""Critic nodes: validate the Resolution Agent's decision against the
playbook, and validate the customer message before it is sent."""
import json

from langchain_core.messages import HumanMessage, SystemMessage
from langsmith import traceable

from ..prompts import (
    CRITIC_COMMUNICATION_SYSTEM_PROMPT,
    CRITIC_RESOLUTION_SYSTEM_PROMPT,
    format_playbook_context,
)
from ..runtime import PipelineContext
from ..schemas import CriticCommunicationOutput, CriticResolutionOutput
from ..state import (
    CriticCommunicationView,
    CriticResolutionView,
    UnifiedAgentState,
    merge_back,
    project_into,
)


def _build_critic_resolution_context(view: dict) -> str:
    playbook_text = format_playbook_context(view["playbook_context"])
    return (
        f"DELIVERY EVENT:\n{json.dumps(view['consolidated_event'], indent=2)}\n\n"
        f"CUSTOMER PROFILE:\n{json.dumps(view['customer_profile'], indent=2)}\n\n"
        f"LOCKER AVAILABILITY:\n{json.dumps(view['locker_availability'], indent=2)}\n\n"
        f"ESCALATION SIGNALS:\n{json.dumps(view['escalation_signals'], indent=2)}\n\n"
        f"PLAYBOOK CONTEXT:\n{playbook_text}\n\n"
        f"RESOLUTION AGENT OUTPUT:\n{json.dumps(view['resolution_output'], indent=2)}"
    )


def _build_critic_communication_context(view: dict) -> str:
    validation_context = {
        "customer_tier": view["customer_profile"].get("tier"),
        "preferred_channel": view["customer_profile"].get("preferred_channel"),
        "active_credit": view["customer_profile"].get("active_credit", 0),
        "resolution": view["resolution_output"].get("resolution"),
        "exception_type": view["consolidated_event"]["status_code"],
        "package_type": view["consolidated_event"]["package_type"],
    }
    return (
        f"VALIDATION CONTEXT:\n{json.dumps(validation_context, indent=2)}\n\n"
        f"COMMUNICATION AGENT OUTPUT:\n{json.dumps(view['communication_output'], indent=2)}"
    )


def make_critic_resolution_node(ctx: PipelineContext):
    @traceable(name="critic_resolution_node")
    def critic_resolution_node(state: UnifiedAgentState) -> UnifiedAgentState:
        """Critic Agent: validates resolution decision against playbook and context."""
        view = project_into(state, CriticResolutionView)
        user_content = _build_critic_resolution_context(view)

        structured_llm = ctx.eval_llm.with_structured_output(CriticResolutionOutput)
        result = structured_llm.invoke([
            SystemMessage(content=CRITIC_RESOLUTION_SYSTEM_PROMPT),
            HumanMessage(content=user_content),
        ])

        agent_output = {"critic_resolution_output": result.model_dump()}
        state = merge_back(state, agent_output, CriticResolutionView)

        if result.decision == "ESCALATE":
            state["escalated"] = True
        if result.decision == "REVISE":
            state["resolution_revision_count"] += 1
            state["critic_feedback"] = result.rationale

        state["tool_calls_log"].append("AGENT: critic_resolution invoked")
        state["trajectory_log"].append(f"critic_resolution: decision={result.decision}")
        state["next_agent"] = "orchestrator"
        return state

    return critic_resolution_node


def make_critic_communication_node(ctx: PipelineContext):
    @traceable(name="critic_communication_node")
    def critic_communication_node(state: UnifiedAgentState) -> UnifiedAgentState:
        """Critic Agent: validates customer communication quality. No PII access."""
        view = project_into(state, CriticCommunicationView)
        user_content = _build_critic_communication_context(view)

        structured_llm = ctx.eval_llm.with_structured_output(CriticCommunicationOutput)
        result = structured_llm.invoke([
            SystemMessage(content=CRITIC_COMMUNICATION_SYSTEM_PROMPT),
            HumanMessage(content=user_content),
        ])

        agent_output = {"critic_communication_output": result.model_dump()}
        state = merge_back(state, agent_output, CriticCommunicationView)

        if result.decision == "ESCALATE":
            state["escalated"] = True

        state["tool_calls_log"].append("AGENT: critic_communication invoked")
        state["trajectory_log"].append(f"critic_communication: decision={result.decision}")
        state["next_agent"] = "orchestrator"
        return state

    return critic_communication_node
