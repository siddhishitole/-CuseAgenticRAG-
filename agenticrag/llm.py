from __future__ import annotations
from typing import Optional, Literal, Tuple, Dict, Union
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from .config import settings

Provider = Literal["openai", "gemini"]

# Caches
_llm_cache: Dict[Tuple[str, str], BaseChatModel] = {}
_embeddings: Optional[OpenAIEmbeddings] = None


def _require_env(var_name: str, value: Optional[str]) -> None:
    """Raise an error if an expected environment variable is missing."""
    if not value:
        raise RuntimeError(f"{var_name} is required. Set it in your environment or .env file.")


def _get_cached_llm(provider: str, model: str) -> Optional[BaseChatModel]:
    """Return cached LLM client if available."""
    return _llm_cache.get((provider, model))


def _cache_llm(provider: str, model: str, client: BaseChatModel) -> None:
    """Cache an LLM client."""
    _llm_cache[(provider, model)] = client


def get_llm(provider: Optional[Provider] = None, model: Optional[str] = None) -> BaseChatModel:
    """Return a cached or newly created chat LLM instance."""
    provider = provider or "openai"
    model = model or (settings.gemini_model if provider == "gemini" else settings.openai_model)

    cached = _get_cached_llm(provider, model)
    if cached:
        return cached

    if provider == "gemini":
        _require_env("GOOGLE_API_KEY", settings.google_api_key)
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI  # Lazy import
        except ImportError as e:
            raise RuntimeError("Install 'langchain-google-genai' to use Gemini.") from e

        client = ChatGoogleGenerativeAI(
            model=model,
            temperature=0.5,
            api_key=settings.google_api_key,
        )

    elif provider == "openai":
        _require_env("OPENAI_API_KEY", settings.openai_api_key)
        client = ChatOpenAI(
            model=model,
            temperature=0.1,
            openai_api_key=settings.openai_api_key,
        )

    else:
        raise ValueError(f"Unsupported provider: {provider}")

    _cache_llm(provider, model, client)
    return client


def get_embeddings() -> OpenAIEmbeddings:
    """Return cached or newly created OpenAI embedding model."""
    global _embeddings
    if _embeddings:
        return _embeddings

    _require_env("OPENAI_API_KEY", settings.openai_api_key)
    _embeddings = OpenAIEmbeddings(
        model=settings.embedding_model,
        openai_api_key=settings.openai_api_key,
    )
    return _embeddings
