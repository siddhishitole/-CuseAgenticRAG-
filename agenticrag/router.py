from __future__ import annotations
from typing import Literal, Dict, Any
from .agents import build_corrective_rag, build_preact_rag
from .agents.workflow import WorkflowAgent
from .config import settings


AgentName = Literal["corrective", "preact", "workflow"]


def choose_agent(question: str) -> AgentName:
    """Choose agent based on question complexity and intent."""
    # Keywords that suggest workflow automation
    workflow_keywords = ["schedule", "meeting", "email", "send", "create task", "reminder", "automate"]
    question_lower = question.lower()

    # Check for workflow automation intent first
    if any(keyword in question_lower for keyword in workflow_keywords):
        return "workflow"
    
    # For complex multi-step questions use preact
    preact_keywords = ["step by step", "first", "then", "after that", "plan", "how do i", "explain", "multiple"]
    if any(keyword in question_lower for keyword in preact_keywords):
        return "preact"
    
    return "corrective"


def get_graph(agent: AgentName):
    """Get the graph for the specified agent."""
    if agent == "corrective":
        return build_corrective_rag()
    elif agent == "preact":
        return build_preact_rag()
    elif agent == "workflow":
        return WorkflowAgent()
    else:
        raise ValueError(f"Unknown agent: {agent}")


def run_agent(question: str, agent: AgentName | None = None) -> str | Dict[Any, Any]:
    agent_name = agent or choose_agent(question)
    graph = get_graph(agent_name)

    if agent_name == "workflow":
        # For workflow agent, directly process the question
        return graph(question)
    else:
        # For other agents, use the existing graph-based approach
        app = graph.compile()
        state = {"question": question}
        result = app.invoke(state)
        answer = result.get("generation", "No answer produced.")
        return answer
