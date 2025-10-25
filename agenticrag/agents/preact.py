from __future__ import annotations
from typing import List, TypedDict, Optional, Dict, Any
from langgraph.graph import StateGraph, END, START
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from langchain_core.documents import Document

from ..llm import get_llm
from ..vectorstore import get_retriever
from ..tools.websearch import TOOLS


class GraphState(TypedDict):
    """State for Pre-Act RAG agent."""
    question: str
    generation: str
    documents: List[Document]
    plan: List[Dict[str, Any]]  # List of planned steps
    current_step: int  # Current step index
    previous_steps: List[Dict[str, Any]]  # History of executed steps
    context: List[tuple]  # Accumulated (action, observation) pairs
    sub_questions: List[str]  # Decomposed sub-questions
    sub_answers: List[Dict[str, Any]]  # Answers to sub-questions


class PlanStep(BaseModel):
    """A single step in the Pre-Act plan."""
    step_number: int = Field(description="Sequential step number")
    action_type: str = Field(description="Type of action: 'retrieve', 'web_search', 'grade', or 'answer'")
    reasoning: str = Field(description="Detailed reasoning for this step")
    arguments: Optional[Dict[str, Any]] = Field(default=None, description="Arguments for the action")


class Plan(BaseModel):
    """Multi-step plan for Pre-Act."""
    previous_steps_summary: str = Field(description="Summary of previously executed steps")
    next_steps: List[PlanStep] = Field(description="List of next steps to execute")


class QuestionDecomposition(BaseModel):
    """Decomposition of a complex question into sub-questions."""
    is_multi_part: bool = Field(description="Whether the question contains multiple distinct parts")
    sub_questions: List[str] = Field(description="List of individual sub-questions (empty if single question)")
    reasoning: str = Field(description="Explanation of the decomposition")


def _get_planner():
    """Create a planner that generates multi-step plans."""
    llm = get_llm(provider="gemini").bind(temperature=0.2)
    structured = llm.with_structured_output(Plan)
    
    system = """You are a planning agent that creates comprehensive multi-step plans for answering questions using RAG (Retrieval Augmented Generation).

Your task is to create a detailed plan that includes:
1. Previous Steps: Summarize what has been done so far (if any)
2. Next Steps: List all remaining steps needed to reach the final answer

Available actions:
- retrieve: Search the vector store for relevant documents
- grade: Evaluate if retrieved documents are relevant
- web_search: Search the web for additional information
- answer: Generate the final answer based on available information

For each step, provide:
- Clear reasoning that references previous steps
- Specific action type
- Required arguments (if any)

Be strategic and plan ahead - consider what might go wrong and have contingency steps."""

    plan_prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", """Question: {question}

Previous Context:
{context}

Current Situation:
{situation}

Create a comprehensive plan to answer this question.""")
    ])
    
    return plan_prompt | structured


def _get_question_decomposer():
    """Create a question decomposer that identifies multiple sub-questions."""
    llm = get_llm(provider="gemini").bind(temperature=0.1)
    structured = llm.with_structured_output(QuestionDecomposition)
    
    system = """You are a question analyzer. Your task is to determine if a user's question contains multiple distinct parts that should be answered separately.

Identify questions that:
- Ask about multiple different topics
- Use "and" to connect separate questions
- Contain numbered/bulleted sub-questions
- Request information about different aspects that require separate research

Examples of multi-part questions:
- "What is machine learning and how does it differ from deep learning?"
- "Explain RAG systems and what are their main components?"
- "Who is the CEO of OpenAI and what are their recent announcements?"

Examples of single questions (even if complex):
- "How does machine learning work?"
- "Explain the architecture of transformer models step by step"
- "What are the components needed to build a RAG system?"

If multi-part, decompose into clear, standalone sub-questions that can be answered independently."""

    decompose_prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "Question: {question}\n\nAnalyze if this contains multiple distinct questions that should be answered separately.")
    ])
    
    return decompose_prompt | structured


def _get_answer_combiner():
    """Create a combiner that synthesizes multiple sub-answers."""
    template = """You are tasked with combining multiple answers into a comprehensive, coherent response.

Original Question: {original_question}

Sub-Questions and Their Answers:
{sub_answers}

Synthesize these answers into a single, well-structured response that:
1. Addresses all parts of the original question
2. Flows naturally and coherently
3. Avoids redundancy
4. Maintains all important information from sub-answers
5. Provides clear transitions between topics

Combined Answer:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | get_llm(provider="gemini") | StrOutputParser()
    return chain


def _get_revisor():
    """Create a plan revisor that updates the plan based on observations."""
    llm = get_llm(provider="gemini").bind(temperature=0.2)  # type: ignore[attr-defined]
    structured = llm.with_structured_output(Plan)
    
    system = """You are a plan revision agent. Based on the outcome of the previous action, revise the plan for the remaining steps.

Analyze:
1. What was accomplished in the previous step
2. Whether the result was successful or requires adjustments
3. What steps remain to reach the final answer

Update the plan accordingly, adjusting for:
- Unexpected results
- Missing information
- Better strategies based on observations

Be adaptive and ensure the plan leads to a complete answer."""

    revise_prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", """Question: {question}

Previous Steps Executed:
{previous_steps}

Last Action: {last_action}
Last Observation: {last_observation}

Current Documents Available: {num_docs}

Revise the plan for remaining steps.""")
    ])
    
    return revise_prompt | structured


def _get_rag_chain():
    """Create RAG chain for generating answers."""
    template = """You are answering a question using the provided context.

Previous reasoning and actions taken:
{previous_reasoning}

Context documents:
{context}

Question: {question}

Based on all the information gathered through the planned steps above, provide a comprehensive answer:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | get_llm(provider="gemini") | StrOutputParser()
    return chain


def _get_document_grader():
    """Create document relevance grader."""
    llm = get_llm(provider="gemini").bind(temperature=0)  # type: ignore[attr-defined]
    
    class GradeResult(BaseModel):
        binary_score: str = Field(description="'yes' if relevant, 'no' if not relevant")
        reasoning: str = Field(description="Brief explanation of the grade")
    
    structured = llm.with_structured_output(GradeResult)
    
    system = """You are a document grader. Evaluate if the retrieved document is relevant to the question.

Consider:
- Does it contain keywords related to the question?
- Is it semantically related to the topic?
- Could it help answer the question?

Provide a binary score ('yes' or 'no') and brief reasoning."""

    grade_prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "Document:\n{document}\n\nQuestion: {question}")
    ])
    
    return grade_prompt | structured


def _format_docs(docs: List[Document]) -> str:
    """Format documents for context."""
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


def _format_context(context: List[tuple]) -> str:
    """Format accumulated context of (action, observation) pairs."""
    if not context:
        return "No previous actions."
    
    formatted = []
    for i, (action, observation) in enumerate(context, 1):
        formatted.append(f"Step {i}:")
        formatted.append(f"  Action: {action}")
        formatted.append(f"  Observation: {observation[:200]}..." if len(observation) > 200 else f"  Observation: {observation}")
    
    return "\n".join(formatted)


def _format_previous_steps(previous_steps: List[Dict[str, Any]]) -> str:
    """Format previous steps for display."""
    if not previous_steps:
        return "None"
    
    formatted = []
    for step in previous_steps:
        formatted.append(f"- Step {step['step_number']}: {step['action_type']}")
        formatted.append(f"  Reasoning: {step['reasoning']}")
        if step.get('result'):
            formatted.append(f"  Result: {step['result'][:150]}...")
    
    return "\n".join(formatted)


def build_preact_rag() -> StateGraph:
    """Build Pre-Act RAG agent with multi-step planning and question decomposition."""
    
    retriever = get_retriever()
    planner = _get_planner()
    revisor = _get_revisor()
    rag_chain = _get_rag_chain()
    grader = _get_document_grader()
    decomposer = _get_question_decomposer()
    combiner = _get_answer_combiner()
    
    def decompose_question(state: GraphState):
        """Analyze if question has multiple parts and decompose if needed."""
        question = state["question"]
        
        print(f"\n=== QUESTION DECOMPOSITION ===")
        print(f"Original Question: {question}")
        
        # Decompose the question
        decomposition = decomposer.invoke({"question": question})
        
        if decomposition.is_multi_part and decomposition.sub_questions:
            print(f"\nMulti-part question detected!")
            print(f"Reasoning: {decomposition.reasoning}")
            print(f"\nSub-questions ({len(decomposition.sub_questions)}):")
            for i, sq in enumerate(decomposition.sub_questions, 1):
                print(f"  {i}. {sq}")
            print("=" * 50)
            
            return {
                "question": question,
                "sub_questions": decomposition.sub_questions,
                "sub_answers": [],
                "documents": [],
                "context": [],
                "previous_steps": [],
                "current_step": 0,
                "plan": [],
                "generation": state.get("generation", "")
            }
        else:
            print(f"\nSingle question - no decomposition needed")
            print(f"Reasoning: {decomposition.reasoning}")
            print("=" * 50)
            
            return {
                "question": question,
                "sub_questions": [],
                "sub_answers": [],
                "documents": [],
                "context": [],
                "previous_steps": [],
                "current_step": 0,
                "plan": [],
                "generation": state.get("generation", "")
            }
    
    def run_preact_for_subquestion(sub_question: str):
        """Run the standard Pre-Act flow for a single sub-question."""
        local_state: GraphState = {
            "question": sub_question,
            "generation": "",
            "documents": [],
            "plan": [],
            "current_step": 0,
            "previous_steps": [],
            "context": [],
            "sub_questions": [],
            "sub_answers": []
        }

        # Initialize plan
        updates = create_initial_plan(local_state)
        local_state.update(updates)

        max_iterations = 12
        iterations = 0

        while iterations < max_iterations:
            iterations += 1

            updates = execute_step(local_state)
            local_state.update(updates)
            decision = should_revise_plan(local_state)

            if decision == "execute":
                continue
            if decision == "revise":
                updates = revise_plan(local_state)
                local_state.update(updates)
                continue
            if decision == "end":
                return local_state

        raise RuntimeError(
            "Sub-question processing exceeded maximum iterations; possible infinite loop in plan execution."
        )

    def process_sub_questions(state: GraphState):
        """Process each sub-question using Pre-Act and collect answers."""
        sub_questions = state["sub_questions"]
        original_question = state["question"]
        
        print(f"\n=== PROCESSING SUB-QUESTIONS ===")
        
        sub_answers = []
        
        for i, sub_q in enumerate(sub_questions, 1):
            print(f"\n--- Sub-Question {i}/{len(sub_questions)} ---")
            print(f"Question: {sub_q}")
            
            # Run Pre-Act for this sub-question
            sub_result = run_preact_for_subquestion(sub_q)
            
            sub_answers.append({
                "question": sub_q,
                "answer": sub_result["generation"],
                "documents_used": len(sub_result.get("documents", [])),
                "steps_taken": len(sub_result.get("previous_steps", []))
            })
            
            print(f"\nSub-Answer {i}: {sub_result['generation'][:150]}...")
        
        print(f"\n=== COMPLETED {len(sub_answers)} SUB-QUESTIONS ===")
        print("=" * 50)
        
        return {
            "question": original_question,
            "sub_questions": sub_questions,
            "sub_answers": sub_answers,
            "documents": [],
            "context": [],
            "previous_steps": [],
            "current_step": 0,
            "plan": [],
            "generation": state.get("generation", "")
        }
    
    def combine_answers(state: GraphState):
        """Combine all sub-answers into a coherent final answer."""
        original_question = state["question"]
        sub_answers = state["sub_answers"]
        
        print(f"\n=== COMBINING ANSWERS ===")
        
        # Format sub-answers for the combiner
        formatted_answers = []
        for i, sa in enumerate(sub_answers, 1):
            formatted_answers.append(f"Sub-Question {i}: {sa['question']}")
            formatted_answers.append(f"Answer: {sa['answer']}")
            formatted_answers.append("")
        
        sub_answers_text = "\n".join(formatted_answers)
        
        # Combine answers
        combined = combiner.invoke({
            "original_question": original_question,
            "sub_answers": sub_answers_text
        })
        
        print(f"Combined answer generated")
        print("=" * 50)
        
        return {
            "question": original_question,
            "generation": combined,
            "sub_questions": state["sub_questions"],
            "sub_answers": sub_answers,
            "documents": [],
            "context": [],
            "previous_steps": [],
            "current_step": 0,
            "plan": []
        }
    
    def should_decompose(state: GraphState) -> str:
        """Route based on whether question was decomposed."""
        sub_questions = state.get("sub_questions", [])
        
        if sub_questions:
            # Multi-part question - process sub-questions
            return "multi"
        else:
            # Single question - use normal Pre-Act flow
            return "single"
    
    def create_initial_plan(state: GraphState):
        """Create initial multi-step plan."""
        question = state["question"]
        
        # Create initial plan
        plan_result = planner.invoke({
            "question": question,
            "context": "Starting fresh - no previous actions.",
            "situation": "Initial planning phase. Need to determine strategy to answer the question."
        })
        
        # Convert to dict format
        plan_steps = []
        for step in plan_result.next_steps:
            plan_steps.append({
                "step_number": step.step_number,
                "action_type": step.action_type,
                "reasoning": step.reasoning,
                "arguments": step.arguments or {}
            })
        
        print(f"\n=== INITIAL PLAN ===")
        print(f"Question: {question}")
        print(f"\nPlanned Steps:")
        for step in plan_steps:
            print(f"  {step['step_number']}. {step['action_type']}: {step['reasoning']}")
        print("=" * 50)
        
        return {
            "question": question,
            "plan": plan_steps,
            "current_step": 0,
            "previous_steps": [],
            "context": [],
            "documents": [],
            "generation": state.get("generation", ""),
            "sub_questions": state.get("sub_questions", []),
            "sub_answers": state.get("sub_answers", [])
        }
    
    def execute_step(state: GraphState):
        """Execute the current step in the plan."""
        plan = state["plan"]
        current_step = state["current_step"]
        question = state["question"]
        documents = state.get("documents", [])
        context = state.get("context", [])
        previous_steps = state.get("previous_steps", [])
        generation = state.get("generation", "")
        sub_questions_state = state.get("sub_questions", [])
        sub_answers_state = state.get("sub_answers", [])
        
        if current_step >= len(plan):
            # Plan complete
            return {
                "question": question,
                "documents": documents,
                "context": context,
                "previous_steps": previous_steps,
                "current_step": current_step,
                "generation": generation,
                "sub_questions": sub_questions_state,
                "sub_answers": sub_answers_state
            }
        
        step = plan[current_step]
        action_type = step["action_type"]
        
        print(f"\n--- Executing Step {current_step + 1}: {action_type} ---")
        print(f"Reasoning: {step['reasoning']}")
        
        action_desc = f"{action_type}"
        observation = ""
        
        # Execute action based on type
        if action_type == "retrieve":
            # Retrieve documents from vector store
            docs = retriever.invoke(question)
            documents = list(docs)
            observation = f"Retrieved {len(documents)} documents from vector store"
            
        elif action_type == "grade":
            # Grade documents for relevance
            filtered_docs = []
            grades = []
            
            for doc in documents:
                grade_result = grader.invoke({
                    "question": question,
                    "document": doc.page_content
                })
                
                score = grade_result.binary_score.lower().strip()
                grades.append(score)
                
                if score == "yes":
                    filtered_docs.append(doc)
            
            relevant_count = sum(1 for g in grades if g == "yes")
            documents = filtered_docs
            observation = f"Graded {len(grades)} documents. {relevant_count} relevant, {len(grades) - relevant_count} filtered out"
            
        elif action_type == "web_search":
            # Search web for additional information
            if TOOLS:
                try:
                    tool = TOOLS[1] if len(TOOLS) > 1 else TOOLS[0]  # Prefer perplexity
                    web_result = str(tool.invoke(question))
                    documents = list(documents) + [Document(page_content=web_result)]
                    observation = f"Web search completed. Added web content to documents"
                except Exception as e:
                    observation = f"Web search failed: {e}"
            else:
                observation = "No web search tools available"
                
        elif action_type == "answer":
            # Generate final answer
            previous_reasoning = _format_context(context)
            generation = rag_chain.invoke({
                "previous_reasoning": previous_reasoning,
                "context": _format_docs(documents),
                "question": question
            })
            
            observation = f"Generated answer: {generation}"
            
            return {
                "question": question,
                "generation": generation,
                "documents": documents,
                "context": context + [(action_desc, observation)],
                "previous_steps": previous_steps + [step],
                "current_step": current_step + 1,
                "sub_questions": sub_questions_state,
                "sub_answers": sub_answers_state
            }
        
        # Update context and previous steps
        new_context = context + [(action_desc, observation)]
        new_previous_steps = previous_steps + [{**step, "result": observation}]
        
        print(f"Observation: {observation}")
        
        return {
            "question": question,
            "documents": documents,
            "context": new_context,
            "previous_steps": new_previous_steps,
            "current_step": current_step + 1,
            "generation": generation,
            "sub_questions": sub_questions_state,
            "sub_answers": sub_answers_state
        }
    
    def revise_plan(state: GraphState):
        """Revise the plan based on execution results."""
        question = state["question"]
        previous_steps = state["previous_steps"]
        context = state["context"]
        documents = state["documents"]
        current_step = state["current_step"]
        generation = state.get("generation", "")
        sub_questions_state = state.get("sub_questions", [])
        sub_answers_state = state.get("sub_answers", [])
        
        # Get last action and observation
        if context:
            last_action, last_observation = context[-1]
        else:
            last_action = "None"
            last_observation = "None"
        
        # Revise plan
        revised_plan = revisor.invoke({
            "question": question,
            "previous_steps": _format_previous_steps(previous_steps),
            "last_action": last_action,
            "last_observation": last_observation,
            "num_docs": len(documents)
        })
        
        # Convert revised plan to dict format
        new_plan_steps = []
        for step in revised_plan.next_steps:
            new_plan_steps.append({
                "step_number": len(previous_steps) + step.step_number,
                "action_type": step.action_type,
                "reasoning": step.reasoning,
                "arguments": step.arguments or {}
            })
        
        print(f"\n=== PLAN REVISED ===")
        print(f"Previous steps completed: {len(previous_steps)}")
        print(f"\nRevised next steps:")
        for step in new_plan_steps:
            print(f"  {step['step_number']}. {step['action_type']}: {step['reasoning']}")
        print("=" * 50)
        
        return {
            "question": question,
            "plan": new_plan_steps,
            "documents": documents,
            "context": context,
            "previous_steps": previous_steps,
            "current_step": 0,  # Reset to execute revised plan
            "generation": generation,
            "sub_questions": sub_questions_state,
            "sub_answers": sub_answers_state
        }
    
    def should_revise_plan(state: GraphState) -> str:
        """Decide whether to revise the plan or continue execution."""
        current_step = state["current_step"]
        plan = state["plan"]
        
        # Check if we've completed all steps in current plan
        if current_step >= len(plan):
            # Check if last step was 'answer'
            if plan and plan[-1]["action_type"] == "answer":
                return "end"
            else:
                # Need to revise plan to add answer step
                return "revise"
        
        # Continue executing current plan
        return "execute"
    
    # Build the graph
    workflow = StateGraph(GraphState)
    
    # Add nodes
    workflow.add_node("decompose_question", decompose_question)
    workflow.add_node("process_sub_questions", process_sub_questions)
    workflow.add_node("combine_answers", combine_answers)
    workflow.add_node("create_plan", create_initial_plan)
    workflow.add_node("execute_step", execute_step)
    workflow.add_node("revise_plan", revise_plan)
    
    # Add edges
    workflow.add_edge(START, "decompose_question")
    workflow.add_conditional_edges(
        "decompose_question",
        should_decompose,
        {
            "multi": "process_sub_questions",
            "single": "create_plan",
        },
    )
    workflow.add_edge("process_sub_questions", "combine_answers")
    workflow.add_edge("combine_answers", END)
    workflow.add_edge("create_plan", "execute_step")
    workflow.add_edge("revise_plan", "execute_step")
    
    # Conditional routing after execution
    workflow.add_conditional_edges(
        "execute_step",
        should_revise_plan,
        {
            "execute": "execute_step",  # Continue with next step
            "revise": "revise_plan",    # Revise the plan
            "end": END                   # Finished
        }
    )
    
    return workflow
