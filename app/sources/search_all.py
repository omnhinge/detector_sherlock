
import asyncio
import re
import unicodedata

from app.sources.pubmed import search_pubmed
from app.sources.crossref import search_crossref
from app.sources.openalex import search_openalex


def normalize_doi(doi: str | None) -> str | None:
    """Normalize DOI values from different databases."""
    if not doi:
        return None

    doi = doi.strip().lower()
    doi = re.sub(r"^https?://(dx\.)?doi\.org/", "", doi)
    doi = re.sub(r"^doi:\s*", "", doi)

    return doi or None


def normalize_title(title: str) -> str:
    """Normalize titles for fallback deduplication."""
    title = unicodedata.normalize("NFKD", title)
    title = "".join(
        char for char in title
        if not unicodedata.combining(char)
    )
    return re.sub(r"[^a-z0-9]+", "", title.lower())


def normalize_record(record: dict) -> dict:
    """Convert a source record to our shared evidence format."""
    doi = normalize_doi(record.get("doi"))
    url = record.get("url")

    if doi:
        url = f"https://doi.org/{doi}"

    return {
        "title": (record.get("title") or "Untitled").strip(),
        "abstract": record.get("abstract") or "",
        "url": url,
        "doi": doi,
        "authors": record.get("authors") or [],
        "published_year": record.get("published_year"),
        "sources": [record.get("source", "Unknown")],
    }


def deduplicate_records(records: list[dict]) -> list[dict]:
    """Merge duplicate records while preserving source information."""
    unique = {}

    for record in records:
        doi = normalize_doi(record.get("doi"))
        title_key = normalize_title(record.get("title", ""))

        # Prefer DOI matching; use title only when DOI is unavailable.
        key = f"doi:{doi}" if doi else f"title:{title_key}"

        if key in unique:
            existing = unique[key]

            for source in record["sources"]:
                if source not in existing["sources"]:
                    existing["sources"].append(source)

            # Preserve an available abstract if the first record lacks one.
            if not existing["abstract"] and record["abstract"]:
                existing["abstract"] = record["abstract"]

            if not existing["authors"] and record["authors"]:
                existing["authors"] = record["authors"]

            if not existing["published_year"] and record["published_year"]:
                existing["published_year"] = record["published_year"]

            if not existing["doi"] and doi:
                existing["doi"] = doi

            if not existing["url"] and record["url"]:
                existing["url"] = record["url"]

        else:
            unique[key] = record.copy()

    return list(unique.values())


async def search_all_sources(
    query: str,
    limit_per_source: int = 5,
) -> dict:
    """Search research sources concurrently and merge their records."""

    source_functions = {
        "PubMed": search_pubmed,
        "Crossref": search_crossref,
        "OpenAlex": search_openalex,
    }

    names = list(source_functions)

    tasks = [
        source_functions[name](query, limit_per_source)
        for name in names
    ]

    responses = await asyncio.gather(
        *tasks,
        return_exceptions=True,
    )

    source_status = {}
    all_records = []

    for name, response in zip(names, responses):
        if isinstance(response, Exception):
            source_status[name] = {
                "status": "error",
                "count": 0,
                "error": str(response),
            }
            continue

        source_status[name] = {
            "status": "success",
            "count": len(response),
        }

        for item in response:
            all_records.append(normalize_record(item))

    unique_records = deduplicate_records(all_records)

    return {
        "query": query,
        "sources": source_status,
        "records_found": len(all_records),
        "unique_records": len(unique_records),
        "duplicates_removed": len(all_records) - len(unique_records),
        "results": unique_records,
    }
