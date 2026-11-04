"""LangGraph node factories. Each make_*_node(ctx) closes over a
PipelineContext and returns a node function with the signature
(UnifiedAgentState) -> UnifiedAgentState that graph.py wires into the
StateGraph.
"""
from .preprocessor import make_preprocessor_node
from .orchestrator import make_orchestrator_node
from .resolution_agent import make_resolution_agent_node
from .communication_agent import make_communication_agent_node
from .critics import make_critic_resolution_node, make_critic_communication_node
from .finalize import make_finalize_node

__all__ = [
    "make_preprocessor_node",
    "make_orchestrator_node",
    "make_resolution_agent_node",
    "make_communication_agent_node",
    "make_critic_resolution_node",
    "make_critic_communication_node",
    "make_finalize_node",
]
