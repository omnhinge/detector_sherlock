import os
import re
import urllib.parse
import xml.etree.ElementTree as ET
import httpx
from typing import Any


FACTCHECK_API_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
FACTCHECK_RSS_URL = "https://news.google.com/rss/search"


async def search_factcheck(query: str, limit: int = 5) -> list[dict[str, Any]]:
    """
    Search verified fact-checking databases (Snopes, Reuters Fact Check,
    PolitiFact, AP Fact Check, BoomLive, etc.) to detect debunked rumors
    or verified viral assertions.
    """
    if not query or not query.strip():
        return []

    limit = max(1, min(limit, 10))
    api_key = os.getenv("FACTCHECK_API_KEY")

    # If Google Fact Check API key is provided, use official API
    if api_key:
        try:
            async with httpx.AsyncClient(timeout=12.0) as client:
                res = await client.get(
                    FACTCHECK_API_URL,
                    params={"query": query.strip(), "key": api_key},
                )
                if res.status_code == 200:
                    data = res.json()
                    claims = data.get("claims", [])
                    results = []
                    for c in claims[:limit]:
                        claim_text = c.get("text", "")
                        reviews = c.get("claimReview", [])
                        if not reviews:
                            continue
                        rev = reviews[0]
                        publisher = rev.get("publisher", {}).get("name", "Fact Checker")
                        rating = rev.get("textualRating", "Evaluated")
                        url = rev.get("url", "")
                        title = rev.get("title", f"Fact Check: {claim_text}")

                        abstract = (
                            f"Verified Fact Check by {publisher}: Claim asserts '{claim_text}'. "
                            f"Official Rating / Verdict: {rating}."
                        )
                        results.append({
                            "title": f"Fact Check [{rating}]: {title}",
                            "abstract": abstract,
                            "url": url,
                            "doi": None,
                            "sources": [f"FactCheck ({publisher})"],
                        })
                    if results:
                        return results
        except Exception:
            pass

    # Open verified fact-checking stream fallback
    encoded = urllib.parse.quote_plus(f"{query.strip()} fact check")
    url = f"{FACTCHECK_RSS_URL}?q={encoded}&hl=en-US&gl=US&ceid=US:en"

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
        source_name = item.findtext("source", default="Fact Checker").strip()
        description = item.findtext("description", default="").strip()

        clean_desc = re.sub(r"<[^>]+>", " ", description).strip()
        clean_title = re.sub(r"<[^>]+>", " ", raw_title).strip()

        if not clean_title:
            continue

        abstract = (
            f"Fact-Checking Report ({source_name}, {pub_date}): "
            f"{clean_title}. {clean_desc}"
        ).strip()

        results.append(
            {
                "title": f"Fact Check: {clean_title}",
                "abstract": abstract,
                "url": link or "https://news.google.com",
                "doi": None,
                "sources": [f"FactCheck ({source_name})"],
            }
        )

    return results
