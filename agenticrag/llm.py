from __future__ import annotations
from typing import Optional, Literal, Tuple, Dict
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from .config import settings

Provider = Literal["openai", "gemini"]

# Simple cache to reuse client instances per (provider, model)
_llm_cache: Dict[Tuple[str, str], BaseChatModel] = {}
_embeddings: Optional[OpenAIEmbeddings] = None


def get_llm(provider: Optional[Provider] = None, model: Optional[str] = None) -> BaseChatModel:
    """Return a chat LLM instance."""
    # Default selection
    resolved_provider: Provider = provider or "openai"
    if resolved_provider == "gemini":
        # Choose Gemini model
        resolved_model = model or settings.gemini_model
        cache_key = (resolved_provider, resolved_model)
        if cache_key in _llm_cache:
            return _llm_cache[cache_key]

        if not settings.google_api_key:
            raise RuntimeError(
                "GOOGLE_API_KEY is required to use Gemini. Set it in your environment or .env file."
            )
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "Missing or incompatible dependency: langchain-google-genai. Install it to use Gemini."
            ) from e

        client = ChatGoogleGenerativeAI(
            model=resolved_model,
            temperature=0.5,
            api_key=settings.google_api_key,
        )
        _llm_cache[cache_key] = client
        return client

    # Default and fallback: OpenAI
    resolved_model = model or settings.openai_model
    cache_key = ("openai", resolved_model)
    if cache_key in _llm_cache:
        return _llm_cache[cache_key]

    if not settings.openai_api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is required for OpenAI. To use Gemini instead, call get_llm(provider='gemini')."
        )

    client = ChatOpenAI(
        model=resolved_model,
        temperature=0.1,
        openai_api_key=settings.openai_api_key,
    )
    _llm_cache[cache_key] = client
    return client


def get_embeddings() -> OpenAIEmbeddings:
    global _embeddings
    if _embeddings is None:
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is required. Set it in your environment or .env file.")
        _embeddings = OpenAIEmbeddings(
            model=settings.embedding_model,
            openai_api_key=settings.openai_api_key,
        )
    return _embeddings
