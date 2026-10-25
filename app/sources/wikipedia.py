import httpx
from typing import Any


WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
WIKI_HEADERS = {
    "User-Agent": "DetectorSherlock/0.3 (research@sherlock.org)",
    "Accept": "application/json",
}


async def search_wikipedia(query: str, limit: int = 5) -> list[dict[str, Any]]:
    """
    Search Wikipedia for encyclopedic knowledge, historical events,
    geography, biographies, and factual entity definitions.
    """
    if not query or not query.strip():
        return []

    limit = max(1, min(limit, 10))

    params = {
        "action": "query",
        "generator": "search",
        "gsrsearch": query.strip(),
        "gsrlimit": limit,
        "prop": "extracts",
        "exintro": 1,
        "explaintext": 1,
        "format": "json",
        "utf8": 1,
    }

    try:
        async with httpx.AsyncClient(timeout=12.0, headers=WIKI_HEADERS) as client:
            response = await client.get(WIKIPEDIA_API, params=params)
            if response.status_code != 200:
                return []

            data = response.json()
            pages = data.get("query", {}).get("pages", {})
    except Exception:
        return []

    results: list[dict[str, Any]] = []

    for _, page in pages.items():
        title = page.get("title", "").strip()
        extract = page.get("extract", "").strip()

        if not title or not extract:
            continue

        url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"

        results.append(
            {
                "title": f"Wikipedia — {title}",
                "abstract": extract,
                "url": url,
                "doi": None,
                "sources": ["Wikipedia"],
            }
        )

    return results
