from app.sources.pubmed import search_pubmed
from fastapi import FastAPI
from pydantic import BaseModel, Field
from app.sources.crossref import search_crossref
from app.sources.openalex import search_openalex
from app.sources.search_all import search_all_sources


app = FastAPI(
    title="Detector Sherlock",
    description="Evidence-based AI response verification API",
    version="0.1.0",
)


class VerifyRequest(BaseModel):
    response: str = Field(
        min_length=1,
        max_length=20000,
        description="The AI-generated response to verify",
    )


@app.get("/")
def home():
    return {
        "project": "Detector Sherlock",
        "message": "AI response verification engine is running",
        "version": "0.1.0",
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/verify")
def verify(request: VerifyRequest):
    return {
        "status": "received",
        "message": "Response received. Evidence verification is the next step.",
        "response_length": len(request.response),
    }


@app.get("/sources/pubmed")
async def pubmed_search(q: str, limit: int = 5):
    if not q.strip():
        return {"results": []}

    limit = max(1, min(limit, 10))
    results = await search_pubmed(q.strip(), limit)

    return {
        "source": "PubMed",
        "query": q,
        "count": len(results),
        "results": results,
    }


@app.get("/sources/crossref")
async def crossref_search(q: str, limit: int = 5):
    if not q.strip():
        return {"results": []}

    limit = max(1, min(limit, 10))
    results = await search_crossref(q.strip(), limit)

    return {
        "source": "Crossref",
        "query": q,
        "count": len(results),
        "results": results,
    }

@app.get("/sources/openalex")
async def openalex_search(q: str, limit: int = 5):
    if not q.strip():
        return {"results": []}

    limit = max(1, min(limit, 10))

    results = await search_openalex(
        q.strip(),
        limit,
    )

    return {
        "source": "OpenAlex",
        "query": q,
        "count": len(results),
        "results": results,
    }


@app.get("/sources/search-all")
async def search_all(q: str, limit: int = 5):
    if not q.strip():
        return {"results": []}

    limit = max(1, min(limit, 10))

    return await search_all_sources(
        q.strip(),
        limit_per_source=limit,
    )


