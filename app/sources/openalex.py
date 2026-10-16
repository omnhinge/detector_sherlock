
import httpx

OPENALEX_API = "https://api.openalex.org/works"


def reconstruct_abstract(inverted_index: dict | None) -> str:
    """Reconstruct plain-text abstract from OpenAlex's abstract_inverted_index format."""
    if not inverted_index or not isinstance(inverted_index, dict):
        return ""

    positions = []
    for word, idx_list in inverted_index.items():
        for idx in idx_list:
            positions.append((idx, word))

    # Sort tokens by word position index
    positions.sort(key=lambda x: x[0])
    return " ".join(word for _, word in positions)


async def search_openalex(
    query: str,
    limit: int = 5,
) -> list[dict]:
    """Search OpenAlex for scholarly works."""

    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.get(
            OPENALEX_API,
            params={
                "search": query,
                "per_page": limit,
            },
        )
        response.raise_for_status()

        items = response.json().get("results", [])

    results = []

    for item in items:
        authorships = item.get("authorships") or []

        authors = [
            author.get("author", {}).get("display_name", "")
            for author in authorships[:10]
        ]

        doi = item.get("doi")
        openalex_id = item.get("id")
        publication_date = item.get("publication_date")
        abstract = reconstruct_abstract(item.get("abstract_inverted_index"))

        results.append({
            "title": item.get("title") or "Untitled",
            "abstract": abstract,
            "url": doi or openalex_id,
            "doi": doi,
            "authors": authors,
            "published_year": (
                int(publication_date[:4])
                if publication_date
                and publication_date[:4].isdigit()
                else None
            ),
            "source": "OpenAlex",
        })

    return results
