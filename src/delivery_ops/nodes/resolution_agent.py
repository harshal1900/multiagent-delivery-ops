"""Resolution Agent node: classifies the delivery event and decides the
resolution action. No PII access."""
import json

from langchain_core.messages import HumanMessage, SystemMessage
from langsmith import traceable

from .. import config
from ..prompts import RESOLUTION_AGENT_SYSTEM_PROMPT, format_playbook_context
from ..runtime import PipelineContext
from ..schemas import ResolutionOutput
from ..state import ResolutionAgentView, UnifiedAgentState, merge_back, project_into


def make_resolution_agent_node(ctx: PipelineContext):
    @traceable(name="resolution_agent_node")
    def resolution_agent_node(state: UnifiedAgentState) -> UnifiedAgentState:
        """Resolution Agent: classifies exception and decides resolution action."""
        view = project_into(state, ResolutionAgentView)

        feedback = view.get("critic_feedback", "")
        feedback_section = ""
        if feedback:
            feedback_section = (
                f"\n\nPREVIOUS ATTEMPT WAS REJECTED. Critic feedback:\n{feedback}\n"
                f"Revise your decision based on this feedback."
            )

        system_prompt = RESOLUTION_AGENT_SYSTEM_PROMPT.format(critic_feedback=feedback_section)
        playbook_text = format_playbook_context(view["playbook_context"])

        user_content = (
            f"DELIVERY EVENT:\n{json.dumps(view['consolidated_event'], indent=2)}\n\n"
            f"CUSTOMER PROFILE (redacted):\n{json.dumps(view['customer_profile'], indent=2)}\n\n"
            f"LOCKER AVAILABILITY:\n{json.dumps(view['locker_availability'], indent=2)}\n\n"
            f"ESCALATION SIGNALS:\n{json.dumps(view['escalation_signals'], indent=2)}\n\n"
            f"RELEVANT PLAYBOOK SECTIONS:\n{playbook_text}"
        )

        structured_llm = ctx.gen_llm.with_structured_output(ResolutionOutput)
        max_retries = config.RESOLUTION_MAX_RETRIES
        result = None

        for attempt in range(max_retries):
            try:
                result = structured_llm.invoke([
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_content),
                ])
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    state["trajectory_log"].append(
                        f"resolution_agent: Validation failed (attempt {attempt + 1}), retrying - {str(e)[:100]}"
                    )
                else:
                    # All retries exhausted - safe fallback with forced escalation.
                    result = ResolutionOutput(
                        is_exception="YES",
                        resolution="RESCHEDULE",
                        rationale=(
                            f"Resolution agent failed after {max_retries} attempts. "
                            f"Defaulting to RESCHEDULE with escalation. Last error: {str(e)[:200]}"
                        ),
                    )
                    state["escalated"] = True
                    state["trajectory_log"].append(
                        f"resolution_agent: All {max_retries} retries exhausted, "
                        f"defaulting to RESCHEDULE with forced escalation"
                    )

        agent_output = {"resolution_output": result.model_dump()}
        state = merge_back(state, agent_output, ResolutionAgentView)

        state["tool_calls_log"].append("AGENT: resolution_agent invoked")
        state["trajectory_log"].append(
            f"resolution_agent: is_exception={result.is_exception}, resolution={result.resolution}"
        )
        state["next_agent"] = "orchestrator"
        return state

    return resolution_agent_node
