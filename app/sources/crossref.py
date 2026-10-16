
import httpx

CROSSREF_API = "https://api.crossref.org/works"


async def search_crossref(query: str, limit: int = 5) -> list[dict]:
    """Search scholarly publication metadata through Crossref."""

    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.get(
            CROSSREF_API,
            params={
                "query.bibliographic": query,
                "rows": limit,
                "select": "DOI,title,author,published,container-title,URL,abstract",
            },
            headers={
                "User-Agent": "DetectorSherlock/0.1 (AI evidence verification)"
            },
        )
        response.raise_for_status()
        items = response.json()["message"]["items"]

    results = []

    for item in items:
        titles = item.get("title") or []
        authors = item.get("author") or []
        dates = item.get("published", {}).get("date-parts", [[]])
        date_parts = dates[0] if dates else []

        results.append({
            "title": titles[0] if titles else "Untitled",
            "abstract": item.get("abstract", ""),
            "url": item.get("URL"),
            "doi": item.get("DOI"),
            "authors": [
                " ".join(
                    part for part in [
                        author.get("given", ""),
                        author.get("family", ""),
                    ] if part
                )
                for author in authors[:10]
            ],
            "published_year": date_parts[0] if date_parts else None,
            "source": "Crossref",
        })

    return results
