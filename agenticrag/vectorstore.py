from __future__ import annotations
from typing import Optional
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from .llm import get_embeddings
from .config import settings
import os

_vs: Optional[Chroma] = None


def get_vectorstore() -> Chroma:
    global _vs
    if _vs is None:
        os.makedirs(settings.chroma_path, exist_ok=True)
        _vs = Chroma(
            collection_name="agenticrag",
            persist_directory=settings.chroma_path,
            embedding_function=get_embeddings(),
        )
    return _vs


def get_retriever(k: int = 4) -> VectorStoreRetriever:
    return get_vectorstore().as_retriever(search_kwargs={"k": k})


def add_texts(texts: list[str], metadatas: Optional[list[dict]] = None):
    vs = get_vectorstore()
    vs.add_texts(texts, metadatas=metadatas)


def add_documents(docs: list[Document]):
    vs = get_vectorstore()
    vs.add_documents(docs)
    try:
        vs.persist()
    except Exception:
        pass
