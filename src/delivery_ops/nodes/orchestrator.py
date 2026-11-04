"""Orchestrator node: the deterministic router. No LLM call of its own -
it only reads state set by other nodes and decides next_agent."""
from langsmith import traceable

from ..runtime import PipelineContext
from ..state import RouterView, UnifiedAgentState, project_into


def make_orchestrator_node(ctx: PipelineContext):
    @traceable(name="orchestrator_node")
    def orchestrator_node(state: UnifiedAgentState) -> UnifiedAgentState:
        """
        Central router. Determines next_agent based on current state.

        Routing order:
          0. Guardrail triggered                          -> finalize
          1. Noise override from preprocessor              -> finalize (skip LLM)
          2. Resolution not yet run                        -> resolution_agent
          3. Resolution done, critic not yet run            -> critic_resolution
          4. Critic returned REVISE and under loop limit    -> resolution_agent (reset)
          5. Critic returned REVISE and at loop limit       -> force ESCALATE -> communication/finalize
          6. Enforce automatic escalation triggers           (rule engine is authoritative)
          7. Not an exception                               -> finalize
          8. Communication not yet run                      -> communication_agent
          9. Communication done, critic not yet run         -> critic_communication
          10. All done                                      -> finalize
        """
        view = project_into(state, RouterView)

        # 0. Guardrail triggered - no LLM, force escalation, go to finalize.
        if view.get("guardrail_triggered"):
            state["resolution_output"] = {
                "is_exception": "YES",
                "resolution": "RESCHEDULE",
                "rationale": "Input flagged by guardrail - prompt injection detected. Defaulting to RESCHEDULE with forced escalation for human review.",
            }
            state["escalated"] = True
            state["next_agent"] = "finalize"
            state["trajectory_log"].append("orchestrator: Guardrail triggered, forcing escalation to finalize")
            return state

        # 1. Noise override - classify as non-exception, skip all agents.
        if view.get("noise_override") and not view.get("resolution_output"):
            state["resolution_output"] = {
                "is_exception": "NO",
                "resolution": "N/A",
                "rationale": f"Status code {view['consolidated_event']['status_code']} with routine description. No anomaly indicators. Classified as noise by preprocessor guardrail.",
            }
            state["trajectory_log"].append("orchestrator: Noise override from preprocessor, skipping to finalize")
            state["next_agent"] = "finalize"
            return state

        # 2. Resolution Agent hasn't run yet - send it there.
        if not view.get("resolution_output"):
            state["next_agent"] = "resolution_agent"
            return state

        # 3. Resolution done but Critic hasn't validated yet.
        if not view.get("critic_resolution_output"):
            state["next_agent"] = "critic_resolution"
            return state

        critic_decision = view["critic_resolution_output"].get("decision")

        # 4. Critic wants a revision and we're under the retry limit.
        if critic_decision == "REVISE" and view["resolution_revision_count"] < view["max_loops"]:
            state["resolution_output"] = {}
            state["critic_resolution_output"] = {}
            state["next_agent"] = "resolution_agent"
            state["trajectory_log"].append(
                f"orchestrator: REVISE loop {view['resolution_revision_count']}/{view['max_loops']}"
            )
            return state

        # 5. Revision limit exhausted - force escalate, still generate communication if exception.
        if critic_decision == "REVISE" and view["resolution_revision_count"] >= view["max_loops"]:
            state["escalated"] = True
            state["critic_resolution_output"] = {
                "decision": "ESCALATE",
                "rationale": "Max revision loops reached. Accepting current resolution with escalation.",
            }
            if view["resolution_output"].get("is_exception") == "YES":
                state["next_agent"] = "communication_agent"
            else:
                state["next_agent"] = "finalize"
            state["trajectory_log"].append("orchestrator: Max loops reached, forcing ESCALATE")
            return state

        # 6. Enforce automatic escalation triggers - rule engine is authoritative.
        if view.get("escalation_signals", {}).get("has_triggers"):
            automatic = [t for t in view["escalation_signals"].get("triggers", [])
                         if t.startswith("AUTOMATIC")]
            if automatic and view["resolution_output"].get("is_exception") == "YES":
                state["escalated"] = True
                state["trajectory_log"].append(
                    f"orchestrator: Forced escalation from rule engine - {automatic}"
                )

        # 7. Not an exception - no customer message needed, go to finalize.
        if view["resolution_output"].get("is_exception") == "NO":
            state["next_agent"] = "finalize"
            state["trajectory_log"].append("orchestrator: Not an exception, skipping to finalize")
            return state

        # 8. Communication Agent hasn't run yet.
        if not view.get("communication_output"):
            state["next_agent"] = "communication_agent"
            return state

        # 9. Communication done but Critic hasn't validated yet.
        if not view.get("critic_communication_output"):
            state["next_agent"] = "critic_communication"
            return state

        # 10. Everything complete - finalize.
        state["next_agent"] = "finalize"
        return state

    return orchestrator_node
