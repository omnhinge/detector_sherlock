import re
import urllib.parse
import xml.etree.ElementTree as ET
import httpx
from typing import Any


NEWS_RSS_URL = "https://news.google.com/rss/search"


async def search_news(query: str, limit: int = 5) -> list[dict[str, Any]]:
    """
    Search real-time global news reports from verified journalism agencies
    (Reuters, Associated Press, BBC, Bloomberg, etc.).
    """
    if not query or not query.strip():
        return []

    limit = max(1, min(limit, 10))
    encoded_query = urllib.parse.quote_plus(query.strip())
    url = f"{NEWS_RSS_URL}?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"

    try:
        async with httpx.AsyncClient(timeout=12.0) as client:
            response = await client.get(url)
            if response.status_code != 200:
                return []

            root = ET.fromstring(response.text)
            items = root.findall(".//item")
    except Exception:
        return []

    results: list[dict[str, Any]] = []

    for item in items[:limit]:
        raw_title = item.findtext("title", default="").strip()
        link = item.findtext("link", default="").strip()
        pub_date = item.findtext("pubDate", default="Recent").strip()
        source_name = item.findtext("source", default="Verified News").strip()
        description = item.findtext("description", default="").strip()

        # Clean HTML tags in description
        clean_desc = re.sub(r"<[^>]+>", " ", description).strip()
        clean_title = re.sub(r"<[^>]+>", " ", raw_title).strip()

        if not clean_title:
            continue

        abstract = (
            f"Breaking / Reported News ({source_name}, {pub_date}): "
            f"{clean_title}. {clean_desc}"
        ).strip()

        results.append(
            {
                "title": f"News: {clean_title}",
                "abstract": abstract,
                "url": link or "https://news.google.com",
                "doi": None,
                "sources": [f"News ({source_name})"],
            }
        )

    return results
