from __future__ import annotations
from typing import Literal
from .agents import build_corrective_rag, build_preact_rag
from .config import settings


AgentName = Literal["corrective", "preact"]


def choose_agent(question: str) -> AgentName:
    """Choose agent based on question complexity."""
    # Simple heuristic: use preact for complex multi-step questions
    keywords = ["step by step", "first", "then", "after that", "plan", "how do i", "explain", "multiple"]
    question_lower = question.lower()
    
    if any(keyword in question_lower for keyword in keywords):
        return "preact"
    
    return "corrective"


def get_graph(agent: AgentName):
    """Get the graph for the specified agent."""
    if agent == "corrective":
        return build_corrective_rag()
    elif agent == "preact":
        return build_preact_rag()
    else:
        raise ValueError(f"Unknown agent: {agent}")


def run_agent(question: str, agent: AgentName | None = None) -> str:
    agent_name = agent or choose_agent(question)
    graph = get_graph(agent_name)

    # For CLI, do not use checkpointer (avoids config errors)
    app = graph.compile()

    state = {"question": question}
    result = app.invoke(state)
    answer = result.get("generation", "No answer produced.")

    return answer
