
import httpx
import xml.etree.ElementTree as ET

PUBMED_API = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


async def search_pubmed(query: str, limit: int = 5) -> list[dict]:
    """Search PubMed and return article metadata and abstracts when available."""

    async with httpx.AsyncClient(timeout=20.0) as client:
        search_response = await client.get(
            f"{PUBMED_API}/esearch.fcgi",
            params={
                "db": "pubmed",
                "term": query,
                "retmax": limit,
                "retmode": "json",
                "sort": "relevance",
            },
        )
        search_response.raise_for_status()

        ids = search_response.json()["esearchresult"]["idlist"]

        if not ids:
            return []

        fetch_response = await client.get(
            f"{PUBMED_API}/efetch.fcgi",
            params={
                "db": "pubmed",
                "id": ",".join(ids),
                "retmode": "xml",
            },
        )
        fetch_response.raise_for_status()

    root = ET.fromstring(fetch_response.text)
    articles = []

    for article in root.findall(".//PubmedArticle"):
        title = article.findtext(
            ".//ArticleTitle", default="Untitled"
        )

        abstract_parts = [
            " ".join(part.itertext()).strip()
            for part in article.findall(".//Abstract/AbstractText")
        ]
        abstract = " ".join(part for part in abstract_parts if part)

        pmid = article.findtext(".//PMID")

        articles.append({
            "title": title,
            "abstract": abstract,
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
            if pmid else None,
            "source": "PubMed",
        })

    return articles
