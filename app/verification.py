
import asyncio

from app.sources.search_all import search_all_sources


def clean_abstract(text: str) -> str:
    """Remove simple XML/HTML tags from abstracts."""
    import re
    from html import unescape

    text = re.sub(r"<[^>]+>", " ", text or "")
    return re.sub(r"\s+", " ", unescape(text)).strip()


async def find_evidence(claim: str) -> dict:
    """Retrieve and clean candidate research evidence for a claim."""

    try:
        result = await search_all_sources(
            query=claim,
            limit_per_source=5,
        )

        evidence = []
        seen = set()

        for record in result.get("results", []):
            title = (record.get("title") or "").strip()
            url = record.get("url") or ""

            # Skip records without a title and avoid duplicate links.
            if not title:
                continue

            key = url.lower() or title.lower()
            if key in seen:
                continue
            seen.add(key)

            abstract = clean_abstract(record.get("abstract", ""))

            evidence.append({
                "title": title,
                "url": url,
                "abstract": abstract,
                "sources": record.get("sources", []),
                "published_year": record.get("published_year"),
                "has_abstract": bool(abstract),
            })

        # Put records with abstracts first.
        evidence.sort(
            key=lambda item: item["has_abstract"],
            reverse=True,
        )

        useful_count = sum(
            1 for item in evidence if item["has_abstract"]
        )

        return {
            "claim": claim,
            "evidence": evidence,
            "evidence_count": len(evidence),
            "records_with_abstracts": useful_count,
            "status": (
                "evidence_found"
                if evidence
                else "no_records_found"
            ),
        }

    except Exception:
        return {
            "claim": claim,
            "evidence": [],
            "evidence_count": 0,
            "records_with_abstracts": 0,
            "status": "search_error",
        }


async def verify_claims(claims: list[str]) -> list[dict]:
    """Search for candidate evidence for every extracted claim."""

    return await asyncio.gather(
        *(find_evidence(claim) for claim in claims)
    )
