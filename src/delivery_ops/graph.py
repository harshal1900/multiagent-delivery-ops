"""Builds and compiles the LangGraph workflow from a PipelineContext."""
from langgraph.graph import END, START, StateGraph

from .nodes import (
    make_communication_agent_node,
    make_critic_communication_node,
    make_critic_resolution_node,
    make_finalize_node,
    make_orchestrator_node,
    make_preprocessor_node,
    make_resolution_agent_node,
)
from .runtime import PipelineContext
from .state import UnifiedAgentState


def route_from_orchestrator(state: UnifiedAgentState) -> str:
    """Return the next node name that the orchestrator stored in the state."""
    return state["next_agent"]


def build_graph(ctx: PipelineContext):
    """Compile the seven-node delivery-exception workflow:

        START -> preprocessor -> orchestrator -> {resolution_agent,
        critic_resolution, communication_agent, critic_communication,
        finalize} -> orchestrator (loop) -> ... -> finalize -> END
    """
    workflow = StateGraph(UnifiedAgentState)

    workflow.add_node("preprocessor", make_preprocessor_node(ctx))
    workflow.add_node("orchestrator", make_orchestrator_node(ctx))
    workflow.add_node("resolution_agent", make_resolution_agent_node(ctx))
    workflow.add_node("critic_resolution", make_critic_resolution_node(ctx))
    workflow.add_node("communication_agent", make_communication_agent_node(ctx))
    workflow.add_node("critic_communication", make_critic_communication_node(ctx))
    workflow.add_node("finalize", make_finalize_node(ctx))

    workflow.add_edge(START, "preprocessor")
    workflow.add_edge("preprocessor", "orchestrator")

    for agent_name in ["resolution_agent", "critic_resolution", "communication_agent", "critic_communication"]:
        workflow.add_edge(agent_name, "orchestrator")

    workflow.add_conditional_edges(
        "orchestrator",
        route_from_orchestrator,
        {
            "resolution_agent": "resolution_agent",
            "critic_resolution": "critic_resolution",
            "communication_agent": "communication_agent",
            "critic_communication": "critic_communication",
            "finalize": "finalize",
        },
    )

    workflow.add_edge("finalize", END)

    return workflow.compile()
