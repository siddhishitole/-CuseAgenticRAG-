from __future__ import annotations
from typing import Optional
import httpx
from langchain_core.tools import tool
from ..config import settings


@tool("tavily_search", return_direct=False)
def tavily_search(urls: list[str], max_results: int = 5) -> str:
    """Extract raw content from URLs using Tavily's extract endpoint. Returns raw_content for each URL."""
    if not settings.tavily_api_key:
        return "Tavily API key not configured."
    endpoint = "https://api.tavily.com/extract"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {settings.tavily_api_key}"
    }
    payload = {"urls": urls[:max_results]}
    try:
        resp = httpx.post(endpoint, headers=headers, json=payload, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", [])
        if not results:
            return "No extract results found."
        # Return raw_content for each URL
        output = []
        for r in results:
            url = r.get("url", "")
            raw_content = r.get("raw_content", "")
            output.append(f"URL: {url}\nRaw Content:\n{raw_content}\n")
        return "\n---\n".join(output)
    except Exception as e:
        return f"Tavily extract error: {e}"


@tool("perplexity_search", return_direct=False)
def perplexity_search(query: str) -> str:
    """Search web using Perplexity's Search API, then extract raw content from the first result using Tavily."""
    if not settings.perplexity_api_key:
        return "Perplexity API key not configured."
    headers = {
        "Authorization": f"Bearer {settings.perplexity_api_key}",
        "Content-Type": "application/json",
    }
    url = f"{settings.perplexity_base_url.rstrip('/')}/search"
    payload = {
        "query": query,
        "max_results": 5,
        "max_tokens_per_page": 1024
    }
    try:
        resp = httpx.post(url, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", [])
        if not results:
            return "No search results found."
        first_url = results[0].get("url")
        if not first_url:
            return "No URL found in Perplexity results."
        # Now call Tavily extract for raw content
        from .websearch import tavily_search
        raw_content = tavily_search.invoke({"urls": [first_url]})
        return f"Raw content from first Perplexity result ({first_url}):\n\n{raw_content}"
    except Exception as e:
        return f"Perplexity search error: {e}"


TOOLS = []
if settings.tavily_api_key:
    TOOLS.append(tavily_search)
if settings.perplexity_api_key:
    TOOLS.append(perplexity_search)
