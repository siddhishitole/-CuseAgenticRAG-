from __future__ import annotations
from typing import Optional
import httpx
from langchain_core.tools import tool
from ..config import settings

# -----------------------------------
# Tavily extract tool
# -----------------------------------
@tool("tavily_search", return_direct=False)
def tavily_search(urls: list[str], max_results: int = 5) -> str:
    """Extract raw content from URLs using Tavily's extract endpoint."""
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
        results = resp.json().get("results", [])
        if not results:
            return "No extract results found."
        output = [f"URL: {r.get('url', '')}\nRaw Content:\n{r.get('raw_content', '')}" for r in results]
        return "\n---\n".join(output)
    except Exception as e:
        return f"Tavily extract error: {e}"

# -----------------------------------
# Sonar summarize tool
# -----------------------------------
@tool("sonar_search", return_direct=False)
def sonar_search(query: str, model: str = "sonar-pro") -> str:
    """Summarize content using Sonar Pro API."""
    if not settings.sonar_api_key:
        return "Sonar API key not configured."
    
    endpoint = f"{settings.sonar_base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {settings.sonar_api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful web research assistant."},
            {"role": "user", "content": query}
        ]
    }
    try:
        resp = httpx.post(endpoint, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        if not content:
            return "Sonar API returned empty content."
        return content
    except Exception as e:
        return f"Sonar API error: {e}"

# -----------------------------------
# Perplexity + Tavily + Sonar combined
# -----------------------------------
@tool("perplexity_search", return_direct=False)
def perplexity_search(query: str) -> str:
    """Search web using Perplexity, extract with Tavily, summarize with Sonar Pro."""
    
    debug_logs = []
    
    if not settings.perplexity_api_key:
        return "Perplexity API key not configured."
    
    # --- Perplexity search ---
    try:
        url = f"{settings.perplexity_base_url.rstrip('/')}/search"
        headers = {
            "Authorization": f"Bearer {settings.perplexity_api_key}",
            "Content-Type": "application/json"
        }
        payload = {"query": query, "max_results": 5, "max_tokens_per_page": 1024}
        resp = httpx.post(url, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        debug_logs.append(f"Perplexity search response: {data}")
        results = data.get("results", [])
        if not results:
            return "\n".join(debug_logs + ["No search results found on Perplexity."])
        first_url = results[0].get("url")
        if not first_url:
            return "\n".join(debug_logs + ["First URL not found in Perplexity results."])
        debug_logs.append(f"First URL from Perplexity: {first_url}")
    except Exception as e:
        return "\n".join(debug_logs + [f"Perplexity search error: {e}"])
    
    # --- Tavily extract ---
    try:
        raw_content = tavily_search([first_url])
        debug_logs.append(f"Tavily raw content extracted (truncated 500 chars): {raw_content[:500]}")
    except Exception as e:
        raw_content = f"Tavily extract error: {e}"
        debug_logs.append(raw_content)
    
    # --- Sonar summarize ---
    try:
        prompt = (
            f"User query: {query}\n\n"
            f"Source URL: {first_url}\n\n"
            f"Extracted content:\n{raw_content}\n\n"
            "Summarize the key points relevant to the user query."
        )
        summary = sonar_search(prompt)
        debug_logs.append(f"Sonar summary (truncated 500 chars): {summary[:500]}")
    except Exception as e:
        summary = f"Sonar API error: {e}"
        debug_logs.append(summary)
    
    # --- Return combined response ---
    return "\n--- DEBUG LOGS ---\n" + "\n".join(debug_logs) + "\n--- RAW EXTRACT ---\n" + raw_content + "\n--- SONAR SUMMARY ---\n" + summary

# -----------------------------------
# Register tools dynamically
# -----------------------------------
TOOLS = []
if settings.tavily_api_key:
    TOOLS.append(tavily_search)
if settings.perplexity_api_key:
    TOOLS.append(perplexity_search)
if settings.sonar_api_key:
    TOOLS.append(sonar_search)
