from app.claims import extract_claims
from app.router import detect_domain, get_sources_for_domain
from app.sources.pubmed import search_pubmed
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from app.sources.crossref import search_crossref
from app.sources.openalex import search_openalex
from app.sources.search_all import search_all_sources


app = FastAPI(
    title="Detector Sherlock",
    description="Evidence-based AI response verification API",
    version="0.1.0",
)

# Allow cross-origin requests from Chrome extensions and local tools
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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


import asyncio
from app.evaluator import evaluate_claim_against_evidence


@app.post("/verify")
async def verify(request: VerifyRequest):
    claims = extract_claims(request.response)
    if not claims:
        return {
            "status": "completed",
            "message": "No candidate claims detected in the response.",
            "response_length": len(request.response),
            "claims_count": 0,
            "summary": {
                "supported": 0,
                "unverified": 0,
                "reliability_score": 0.0,
            },
            "claims": [],
        }

    # Fetch evidence and evaluate each claim concurrently
    async def process_single_claim(claim_text: str) -> dict:
        domain = detect_domain(claim_text)
        sources_to_check = get_sources_for_domain(domain)

        search_res = await search_all_sources(
            claim_text,
            limit_per_source=3,
            selected_sources=sources_to_check,
        )
        evidence_list = search_res.get("results", [])

        # Evaluate claim against retrieved evidence
        evaluation = evaluate_claim_against_evidence(claim_text, evidence_list)

        return {
            "claim": claim_text,
            "domain": domain,
            "sources_targeted": sources_to_check,
            "verdict": evaluation["verdict"],
            "confidence": evaluation["confidence"],
            "explanation": evaluation["explanation"],
            "best_evidence": evaluation["best_evidence"],
            "evidence_count": len(evidence_list),
            "sources": search_res.get("sources", {}),
            "evidence": evidence_list,
        }

    claim_results = await asyncio.gather(*(process_single_claim(c) for c in claims))

    # Calculate overall reliability and risk metrics
    total_claims = len(claim_results)
    supported_count = sum(1 for c in claim_results if c["verdict"] == "supported")
    contradicted_count = sum(1 for c in claim_results if c["verdict"] == "contradicted")
    unverified_count = sum(1 for c in claim_results if c["verdict"] == "unverified")
    claims_with_evidence = sum(1 for c in claim_results if c["evidence_count"] > 0)

    reliability_score = round((supported_count / total_claims) * 100, 1) if total_claims else 0.0
    evidence_coverage = round((claims_with_evidence / total_claims) * 100, 1) if total_claims else 0.0

    # Contradicted claims immediately push hallucination risk higher
    if contradicted_count > 0:
        hallucination_risk = "high"
    elif reliability_score >= 75.0:
        hallucination_risk = "low"
    elif reliability_score >= 40.0:
        hallucination_risk = "medium"
    else:
        hallucination_risk = "high"

    return {
        "status": "completed",
        "message": "Claims extracted, evidence retrieved, and verdicts evaluated.",
        "response_length": len(request.response),
        "claims_count": total_claims,
        "summary": {
            "supported": supported_count,
            "contradicted": contradicted_count,
            "unverified": unverified_count,
            "reliability_score": reliability_score,
            "hallucination_risk": hallucination_risk,
            "evidence_coverage": evidence_coverage,
            "caveat": (
                "An unverified verdict indicates that no authoritative literature in the queried "
                "databases corroborated the claim. A contradicted verdict indicates that retrieved "
                "evidence asserts opposing findings."
            ),
        },
        "claims": claim_results,
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


