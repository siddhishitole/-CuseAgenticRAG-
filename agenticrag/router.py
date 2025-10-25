from __future__ import annotations
from typing import Literal
from .agents import build_corrective_rag
from .config import settings


AgentName = Literal["corrective"]


def choose_agent(question: str) -> AgentName:
    return "corrective"


def get_graph(agent: AgentName):
    return build_corrective_rag()


def run_agent(question: str, agent: AgentName | None = None) -> str:
    agent_name = agent or choose_agent(question)
    graph = get_graph(agent_name)

    # For CLI, do not use checkpointer (avoids config errors)
    app = graph.compile()

    state = {"question": question}
    result = app.invoke(state)
    answer = result.get("generation", "No answer produced.")

    return answer
