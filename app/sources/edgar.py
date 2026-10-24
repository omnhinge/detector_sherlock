import httpx
from typing import Any


EDGAR_SEARCH_API = "https://efts.sec.gov/LATEST/search-index"
EDGAR_HEADERS = {
    "User-Agent": "DetectorSherlock research@sherlock.org",
    "Accept": "application/json",
}


async def search_edgar(query: str, limit: int = 5) -> list[dict[str, Any]]:
    """
    Search the U.S. Securities and Exchange Commission (SEC) EDGAR public records.
    Finds corporate filings (10-K, 10-Q, 8-K) containing corporate earnings,
    production, executive announcements, and financial facts.
    """
    if not query or not query.strip():
        return []

    limit = max(1, min(limit, 10))

    try:
        async with httpx.AsyncClient(timeout=15.0, headers=EDGAR_HEADERS) as client:
            response = await client.get(
                EDGAR_SEARCH_API,
                params={"q": query.strip()},
            )
            if response.status_code != 200:
                return []

            data = response.json()
            hits = data.get("hits", {}).get("hits", [])
    except Exception:
        return []

    results: list[dict[str, Any]] = []

    for hit in hits[:limit]:
        src = hit.get("_source", {})
        form = src.get("form", "Filing")
        file_date = src.get("file_date", "Unknown date")
        ciks = src.get("ciks", [])
        cik = ciks[0] if ciks else ""
        adsh = src.get("adsh", "")
        file_num = src.get("file_num", [])
        items = src.get("items", [])

        # Extract entity display name
        display_names = src.get("display_names", [])
        entity = display_names[0] if display_names else "Public Company"
        # Clean entity name if it contains CIK suffix
        entity_clean = entity.split("  (")[0] if "  (" in entity else entity

        title = f"SEC Form {form} — {entity_clean} ({file_date})"

        # Construct filing URL
        if cik and adsh:
            clean_adsh = adsh.replace("-", "")
            url = f"https://www.sec.gov/edgar/browse/?CIK={cik}"
        else:
            url = "https://www.sec.gov/edgar/searchedgar/companysearch"

        # Construct abstract excerpt
        items_desc = f" Items: {', '.join(items)}." if items else ""
        file_num_str = f" File #{file_num[0]}." if file_num else ""
        abstract = (
            f"Official SEC EDGAR filing (Form {form}) for {entity_clean}, "
            f"period filed on {file_date}.{file_num_str}{items_desc} "
            f"Accession number: {adsh}."
        )

        results.append(
            {
                "title": title,
                "abstract": abstract,
                "url": url,
                "doi": None,
                "sources": ["SEC EDGAR"],
            }
        )

    return results
