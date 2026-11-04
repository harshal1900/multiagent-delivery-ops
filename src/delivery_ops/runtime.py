"""Wires together the LLM clients, ingested data and tools into one
PipelineContext that every node factory takes as its first argument.
Build one with build_context() and reuse it across the whole process.
"""
from dataclasses import dataclass

from langchain_openai import ChatOpenAI

from . import config
from .ingestion import IngestedData, load_all
from .tools import Tools, make_tools


@dataclass
class PipelineContext:
    gen_llm: ChatOpenAI       # Resolution Agent and Communication Agent
    eval_llm: ChatOpenAI      # Critic agents and the coherence judge
    ingested: IngestedData
    tools: Tools


def build_context() -> PipelineContext:
    gen_llm = ChatOpenAI(model=config.GEN_MODEL, temperature=0)
    eval_llm = ChatOpenAI(model=config.EVAL_MODEL, temperature=0)
    ingested = load_all()
    tools = make_tools(ingested, config.DELIVERY_LOGS_PATH)
    return PipelineContext(gen_llm=gen_llm, eval_llm=eval_llm, ingested=ingested, tools=tools)
