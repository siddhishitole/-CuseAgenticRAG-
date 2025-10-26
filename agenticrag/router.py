from __future__ import annotations
from typing import Literal, Dict, Any
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage
from .agents import build_corrective_rag, build_preact_rag
from .agents.workflow import WorkflowAgent
from .config import settings
from .llm import get_llm


AgentName = Literal["corrective", "preact", "workflow"]


class RouterDecision(BaseModel):
    """Decision model for routing queries to the appropriate agent."""
    agent: AgentName = Field(
        description="The selected agent to handle the query: 'corrective', 'preact', or 'workflow'"
    )
    reasoning: str = Field(
        description="Brief explanation of why this agent was chosen"
    )


ROUTER_SYSTEM_PROMPT = """You are an intelligent routing system that analyzes user queries and determines which specialized agent should handle them. Your task is to carefully analyze the query's characteristics and route it to the most appropriate agent.

**Available Agents:**

1. **CORRECTIVE AGENT** - Best for: Simple, focused queries
   - Handles relatively small and single-question queries
   - Ideal for straightforward information retrieval
   - Examples:
     * "What is the capital of France?"
     * "Explain the concept of machine learning"
     * "Who wrote the book 1984?"
     * "What are the benefits of exercise?"
   - Use when: Query asks ONE specific question that can be answered directly

2. **WORKFLOW AGENT** - Best for: Task automation and action-oriented queries
   - Handles queries related to setting up emails, creating calendar events, scheduling, and task automation
   - Designed for actionable requests that interact with external services
   - Examples:
     * "Send an email to john@example.com regarding the project deadline"
     * "Schedule a meeting with the team for next Monday at 3 PM"
     * "Create a calendar event for the client presentation"
     * "Set up a reminder for tomorrow's standup"
     * "Automate sending weekly reports to stakeholders"
   - Use when: Query contains action verbs like send, schedule, create, automate, set up, remind AND involves external tools/services

3. **PREACT AGENT** - Best for: Complex, multi-faceted queries requiring planning
   - Handles large queries with multiple questions or sub-questions
   - Manages complex reasoning tasks requiring step-by-step planning
   - Ideal for queries that need decomposition into smaller tasks
   - Examples:
     * "First, research the history of AI, then explain its current applications, and finally predict its future impact"
     * "I need to understand quantum computing. Start with basic principles, then explain qubits, and show me how quantum algorithms differ from classical ones"
     * "Plan a complete marketing strategy: analyze target audience, design campaign themes, and suggest distribution channels"
     * "How do I build a web application? Explain architecture, backend setup, frontend development, and deployment"
   - Use when: Query contains multiple distinct questions OR requires sequential reasoning OR explicitly mentions planning/steps OR is notably complex/long

**Routing Decision Criteria:**

- **Size & Complexity**: Single simple question → CORRECTIVE | Multiple questions or complex reasoning → PREACT
- **Intent**: Asking for information → CORRECTIVE | Requesting an action/automation → WORKFLOW | Requiring planning/analysis → PREACT
- **Scope**: Narrow focus → CORRECTIVE | Broad, multi-part query → PREACT | Task execution → WORKFLOW

**Instructions:**
1. Carefully read and analyze the entire user query
2. Identify the number of distinct questions or sub-tasks
3. Assess the complexity and whether step-by-step reasoning is needed
4. Check if the query requires external actions (email, scheduling, etc.)
5. Select the most appropriate agent based on the criteria above
6. Provide clear reasoning for your decision

Remember: When in doubt between CORRECTIVE and PREACT, consider the query length and number of questions. For any automation/action requests, always prefer WORKFLOW."""


def choose_agent(question: str) -> AgentName:
    """Choose agent based on question complexity and intent using LLM."""
    try:
        # Get LLM instance
        llm = get_llm(provider="gemini")
        
        # Use structured output with Pydantic model
        structured_llm = llm.with_structured_output(RouterDecision)
        
        # Create messages
        messages = [
            SystemMessage(content=ROUTER_SYSTEM_PROMPT),
            HumanMessage(content=f"Query to route: {question}")
        ]
        
        # Get routing decision
        decision: RouterDecision = structured_llm.invoke(messages)
        
        # Log the decision (optional, can be removed in production)
        print(f"\n🔀 Router Decision: {decision.agent}")
        print(f"💭 Reasoning: {decision.reasoning}\n")
        
        return decision.agent
        
    except Exception as e:
        # Fallback to simple heuristic if LLM routing fails
        print(f"⚠️  LLM routing failed: {e}. Using fallback heuristic.")
        return _fallback_choose_agent(question)


def _fallback_choose_agent(question: str) -> AgentName:
    """Fallback routing based on keywords if LLM routing fails."""
    question_lower = question.lower()
    
    # Check for workflow automation intent
    workflow_keywords = ["schedule", "meeting", "email", "send", "create task", "reminder", "automate", "calendar"]
    if any(keyword in question_lower for keyword in workflow_keywords):
        return "workflow"
    
    # Check for complex multi-step questions
    preact_indicators = ["step by step", "first", "then", "after that", "plan", "multiple", "?.*?"]
    question_marks = question.count("?")
    if question_marks >= 2 or any(indicator in question_lower for indicator in preact_indicators) or len(question.split()) > 50:
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

# def run_agent(question: str, agent: AgentName | None = None) -> str:
#     agent_name = agent or choose_agent(question)
#     graph = get_graph(agent_name)

#     # For CLI, do not use checkpointer (avoids config errors)
#     app = graph.compile()

#     state = {"question": question}
#     result = app.invoke(state)
#     answer = result.get("generation", "No answer produced.")

#     return answer