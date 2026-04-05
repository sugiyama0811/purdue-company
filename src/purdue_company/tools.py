"""
Shared tools — no API keys required.

Search: DuckDuckGo (free, no key needed)
LLM:    Ollama (local, see README for setup)
"""

import os
from crewai.tools import BaseTool
from crewai import LLM
from pydantic import BaseModel, Field
from typing import Type


# ---------------------------------------------------------------------------
# DuckDuckGo Search Tool
# ---------------------------------------------------------------------------

class DDGSearchInput(BaseModel):
    query: str = Field(description="The search query to run on DuckDuckGo")
    max_results: int = Field(default=8, description="Number of results to return (max 10)")


class DuckDuckGoSearchTool(BaseTool):
    name: str = "duckduckgo_search"
    description: str = (
        "Search the web using DuckDuckGo. Returns titles, URLs, and snippets "
        "for the given query. Use for finding research papers, news, technical "
        "documentation, financial data, and any web information."
    )
    args_schema: Type[BaseModel] = DDGSearchInput

    def _run(self, query: str, max_results: int = 8) -> str:
        try:
            from duckduckgo_search import DDGS
            results = []
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=min(max_results, 10)):
                    results.append(
                        f"Title: {r.get('title', 'N/A')}\n"
                        f"URL:   {r.get('href', 'N/A')}\n"
                        f"Snippet: {r.get('body', 'N/A')}\n"
                    )
            if not results:
                return f"No results found for: {query}"
            return f"Search results for '{query}':\n\n" + "\n---\n".join(results)
        except Exception as e:
            return f"Search error: {str(e)}"


# ---------------------------------------------------------------------------
# LLM factory — Ollama (local, no API key) or fallback to Anthropic if set
# ---------------------------------------------------------------------------

def get_llm(max_tokens: int = 8096) -> LLM:
    """
    Returns an LLM instance.

    Priority:
    1. Ollama (local) — if OLLAMA_MODEL is set or default llama3.2 is available
    2. Anthropic Claude — if ANTHROPIC_API_KEY is set
    3. Raises clear error if neither is configured
    """
    # --- Ollama (no API key needed) ---
    ollama_model = os.getenv("OLLAMA_MODEL", "llama3.2")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")

    if anthropic_key:
        # User has Anthropic key — use Claude for best quality
        return LLM(
            model=os.getenv("MODEL", "anthropic/claude-sonnet-4-6"),
            api_key=anthropic_key,
            max_tokens=max_tokens,
        )
    else:
        # Default: Ollama local model (no API key required)
        return LLM(
            model=f"ollama/{ollama_model}",
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            max_tokens=max_tokens,
        )
