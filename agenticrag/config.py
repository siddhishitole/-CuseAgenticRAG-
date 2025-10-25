from __future__ import annotations
import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass(frozen=True)
class Settings:
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-5-nano-2025-08-07")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    chroma_path: str = os.getenv("CHROMA_PATH", os.path.join(os.getcwd(), "chromadb"))
    memory_namespace: str = os.getenv("MEMORY_NAMESPACE", "default")
    session_id: str = os.getenv("SESSION_ID", "local-dev")

    # Google Gemini
    google_api_key: str | None = os.getenv("GOOGLE_API_KEY")
    gemini_model: str = os.getenv(
        "GEMINI_MODEL", "gemini-2.5-flash-lite-preview-09-2025"
    )

    # Web search
    tavily_api_key: str | None = os.getenv("TAVILY_API_KEY")
    perplexity_api_key: str | None = os.getenv("PERPLEXITY_API_KEY")
    perplexity_base_url: str = os.getenv("PERPLEXITY_BASE_URL", "https://api.perplexity.ai")

    sonar_api_key: str | None = os.getenv("SONAR_API_KEY")
    sonar_base_url: str = os.getenv("SONAR_BASE_URL", "https://api.sonar.workers.dev/v1")


settings = Settings()
