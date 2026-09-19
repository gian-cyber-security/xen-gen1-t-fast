"""Small optional web-search tool for XEN text models."""
from __future__ import annotations

from ddgs import DDGS


def should_search(prompt: str) -> bool:
    p = prompt.lower()
    triggers = (
        "latest", "current", "today", "right now", "recent",
        "news", "this week", "this month", "who is", "what happened",
        "price", "weather", "stock", "score", "update", "2026",
    )
    return any(x in p for x in triggers)


def search_web(query: str, max_results: int = 3) -> list[dict[str, str]]:
    try:
        results = DDGS(timeout=8).text(
            query,
            region="us-en",
            safesearch="moderate",
            max_results=max_results,
        )
        return [
            {
                "title": r.get("title", ""),
                "url": r.get("href", ""),
                "snippet": r.get("body", ""),
            }
            for r in results
        ]
    except Exception:
        return []


def format_results(results: list[dict[str, str]], max_chars: int = 1800) -> str:
    if not results:
        return ""
    parts = ["WEB SEARCH RESULTS:"]
    used = len(parts[0])
    for i, r in enumerate(results, 1):
        item = (
            f"{i}. {r['title']}\n"
            f"URL: {r['url']}\n"
            f"Summary: {r['snippet']}\n"
        )
        if used + len(item) > max_chars:
            break
        parts.append(item)
        used += len(item)
    parts.append("Use these results as current context. Do not invent facts not supported by them.")
    return "\n".join(parts)
