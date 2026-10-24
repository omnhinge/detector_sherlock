import asyncio
import re
import unicodedata

from app.sources.pubmed import search_pubmed
from app.sources.crossref import search_crossref
from app.sources.openalex import search_openalex
from app.sources.edgar import search_edgar
from app.sources.datagov import search_datagov


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

    sources = record.get("sources")
    if not sources:
        sources = [record.get("source", "Unknown")]

    return {
        "title": (record.get("title") or "Untitled").strip(),
        "abstract": record.get("abstract") or "",
        "url": url,
        "doi": doi,
        "authors": record.get("authors") or [],
        "published_year": record.get("published_year"),
        "sources": sources,
    }


def deduplicate_records(records: list[dict]) -> list[dict]:
    """Merge duplicate records while preserving source information."""
    unique = {}

    for record in records:
        doi = normalize_doi(record.get("doi"))
        title_key = normalize_title(record.get("title", ""))

        # Prefer DOI matching; use title or url when DOI is unavailable.
        if doi:
            key = f"doi:{doi}"
        elif title_key:
            key = f"title:{title_key}"
        elif record.get("url"):
            key = f"url:{record.get('url')}"
        else:
            key = f"raw:{id(record)}"

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
    selected_sources: list[str] | None = None,
) -> dict:
    """Search research and public record sources concurrently and merge their records."""

    source_functions = {
        "PubMed": search_pubmed,
        "Crossref": search_crossref,
        "OpenAlex": search_openalex,
        "SEC EDGAR": search_edgar,
        "World Bank / GovData": search_datagov,
    }

    if selected_sources:
        names = [s for s in selected_sources if s in source_functions]
    else:
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
    merged_records = []

    for name, response in zip(names, responses):
        if isinstance(response, Exception):
            source_status[name] = {
                "status": "error",
                "count": 0,
                "error": str(response),
            }
        else:
            source_status[name] = {
                "status": "ok",
                "count": len(response),
            }

            for record in response:
                merged_records.append(normalize_record(record))

    deduped = deduplicate_records(merged_records)

    return {
        "query": query,
        "total": len(deduped),
        "sources": source_status,
        "results": deduped,
    }
