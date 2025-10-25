from __future__ import annotations
from typing import List, TypedDict

from langgraph.graph import StateGraph, END, START
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from langchain_core.documents import Document

from ..llm import get_llm
from ..vectorstore import get_retriever
from ..tools.websearch import TOOLS


class GraphState(TypedDict):
    question: str
    generation: str
    web_search: str
    documents: List[Document]


class GradeDocuments(BaseModel):
    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


def _get_retrieval_grader():
    # Use a deterministic LLM for grading
    llm = get_llm(provider="gemini").bind(temperature=0)  # type: ignore[attr-defined]
    structured = llm.with_structured_output(GradeDocuments)
    system = (
        "As a grader, evaluate whether the retrieved document is relevant to the user's question.\n"
        "This is a simple check to filter out incorrect results, not a stringent test.\n"
        "If the document contains keywords or is semantically related to the question, grade it as relevant.\n"
        "Respond with a binary score of 'yes' or 'no'."
    )
    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Retrieved document:\n\n{document}\n\nUser question: {question}"),
        ]
    )
    return grade_prompt | structured


def _get_rag_chain():
    template = (
        "Generate an answer to the question using only the information contained within the provided context.\n"
        "Answer the question using the provided context.\n"
        "Context: {context}\n"
        "Question: {question}\n"
        "Answer:"
    )
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | get_llm() | StrOutputParser()
    return chain


def _format_docs(docs: List[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def build_corrective_rag() -> StateGraph:
    retriever = get_retriever()
    rag_chain = _get_rag_chain()
    retrieval_grader = _get_retrieval_grader()

    def retrieve(state: GraphState):
        question = state["question"]
        documents = retriever.invoke(question)
        return {"documents": documents, "question": question}

    def generate(state: GraphState):
        question = state["question"]
        documents = state["documents"]
        generation = rag_chain.invoke({"context": _format_docs(documents), "question": question})
        return {"documents": documents, "question": question, "generation": generation}

    def grade_documents(state: GraphState):
        question = state["question"]
        documents = state["documents"]
        filtered_docs: List[Document] = []
        web_search = "No"
        for d in documents:
            score = retrieval_grader.invoke({"question": question, "document": d.page_content})
            grade = getattr(score, "binary_score", None) or (
                score.get("binary_score") if isinstance(score, dict) else None
            )
            if isinstance(grade, str) and grade.lower().strip() == "yes":
                filtered_docs.append(d)
            else:
                web_search = "Yes"
        return {"documents": filtered_docs, "question": question, "web_search": web_search}

    def web_search_node(state: GraphState):
        question = state["question"]
        documents = state["documents"]
        web_text = ""
        if TOOLS:
            # Use first available tool
            try:
                # hardcode to use perplexity_search for now
                tool = TOOLS[1]
                web_text = str(tool.invoke(question))
            except Exception as e:
                web_text = f"Web search error: {e}"
        if web_text:
            documents = list(documents) + [Document(page_content=web_text)]
        return {"documents": documents, "question": question}

    def decide_to_generate(state: GraphState):
        return "web_search_node" if state.get("web_search") == "Yes" else "generate"

    workflow = StateGraph(GraphState)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate", generate)
    workflow.add_node("web_search_node", web_search_node)

    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "web_search_node": "web_search_node",
            "generate": "generate",
        },
    )
    workflow.add_edge("web_search_node", "generate")
    workflow.add_edge("generate", END)

    return workflow
