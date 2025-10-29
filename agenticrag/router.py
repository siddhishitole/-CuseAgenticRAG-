from __future__ import annotations
from typing import Literal, Dict, Any
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage
from .agents import build_corrective_rag, build_preact_rag
from .agents.workflow import WorkflowAgent
from .llm import get_llm


# --- Types ---
AgentName = Literal["corrective", "preact", "workflow"]


# --- Models ---
class RouterDecision(BaseModel):
    """Decision model for routing queries to the appropriate agent."""
    agent: AgentName = Field(description="The selected agent: 'corrective', 'preact', or 'workflow'")
    reasoning: str = Field(description="Brief explanation for the routing decision")


# --- Constants ---
ROUTER_SYSTEM_PROMPT = """You are an intelligent routing system that determines which specialized agent should handle a user query.

**Agents:**
1. **CORRECTIVE** — For simple, single-question, factual or definitional queries.
2. **WORKFLOW** — For automation or action requests (emails, scheduling, tasks, reminders).
3. **PREACT** — For multi-step, analytical, or planning-based queries.

**Rules:**
- Simple single query → CORRECTIVE
- Automation/action request → WORKFLOW
- Multi-step, analytical, or reasoning-heavy query → PREACT
- Prefer WORKFLOW if query contains action verbs (send, schedule, automate).
- Prefer PREACT if query has multiple sub-questions or requires sequential steps.
"""


# --- Core Router ---
def choose_agent(question: str) -> AgentName:
    """Choose the agent based on question complexity and intent using LLM, with fallback."""
    try:
        llm = get_llm(provider="gemini").with_structured_output(RouterDecision)
        messages = [
            SystemMessage(content=ROUTER_SYSTEM_PROMPT),
            HumanMessage(content=f"Query: {question}")
        ]

        decision: RouterDecision = llm.invoke(messages)
        print(f"\n🔀 Agent: {decision.agent}\n💭 Reason: {decision.reasoning}\n")
        return decision.agent

    except Exception as e:
        print(f"⚠️ LLM routing failed: {e}. Falling back to heuristic routing.")
        return _fallback_choose_agent(question)


# --- Fallback Heuristic ---
def _fallback_choose_agent(question: str) -> AgentName:
    """Fast keyword-based routing used when LLM fails."""
    q = question.lower().strip()

    # 1️⃣ Workflow detection (intent + object)
    actions = ["schedule", "send", "create", "remind", "automate", "book", "set up", "notify"]
    objects = ["email", "meeting", "event", "task", "calendar", "reminder", "report"]
    if any(a in q for a in actions) and any(o in q for o in objects):
        return "workflow"

    # 2️⃣ Preact detection (reasoning, complexity)
    reasoning_phrases = [
        "step by step", "first", "then", "next", "after that", "plan",
        "analyze", "compare", "evaluate", "design", "explain how", "difference between"
    ]
    if any(p in q for p in reasoning_phrases) or q.count("?") > 1 or len(q.split()) > 40:
        return "preact"

    # 3️⃣ Default → Corrective
    return "corrective"


# --- Agent Graph Selector ---
def get_graph(agent: AgentName):
    """Return graph or agent class based on selection."""
    match agent:
        case "corrective":
            return build_corrective_rag()
        case "preact":
            return build_preact_rag()
        case "workflow":
            return WorkflowAgent()
        case _:
            raise ValueError(f"Unknown agent type: {agent}")


# --- Runner ---
def run_agent(question: str, agent: AgentName | None = None) -> str | Dict[Any, Any]:
    """Run the selected agent and return its output."""
    agent_name = agent or choose_agent(question)
    graph = get_graph(agent_name)

    if agent_name == "workflow":
        return graph(question)

    app = graph.compile()
    result = app.invoke({"question": question})
    return result.get("generation", "No answer produced.")
