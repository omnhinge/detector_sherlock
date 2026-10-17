from app.sources.pubmed import search_pubmed
from fastapi import FastAPI
from pydantic import BaseModel, Field
from app.sources.crossref import search_crossref
from app.sources.openalex import search_openalex
from app.sources.search_all import search_all_sources
from app.claims import extract_claims
from pydantic import BaseModel, Field
from app.verification import verify_claims
from app.assessor import assess_claims
from app.scoring import calculate_score


app = FastAPI(
    title="Detector Sherlock",
    description="Evidence-based AI response verification API",
    version="0.1.0",
)


class VerifyRequest(BaseModel):
    response: str = Field(min_length=1, max_length=30000)


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
async def verify_response(request: VerifyRequest):
    # 1. Extract candidate claims from the AI response.
    claims = extract_claims(request.response)

    if not claims:
        return {
            "status": "no_claims_found",
            "total_claims": 0,
            "claims": [],
            "score": None,
            "score_status": "insufficient_evidence",
            "evidence_coverage": 0.0,
        }

    # 2. Retrieve research evidence for every claim.
    evidence_results = await verify_claims(claims)

    # 3. Ask Gemini to assess the retrieved evidence.
    assessments = await assess_claims(claims, evidence_results)

    # 4. Combine claims, assessments, and their evidence.
    assessed_claims = []

    for index, claim in enumerate(claims):
        assessment = assessments[index]
        evidence_result = evidence_results[index]

        evidence = evidence_result.get("evidence", [])

        # Include the evidence items Gemini cited in its assessment.
        cited_evidence = [
            evidence[evidence_index]
            for evidence_index in assessment["evidence_indices"]
            if 0 <= evidence_index < len(evidence)
        ]

        assessed_claims.append({
            "id": index + 1,
            "text": claim,
            "verdict": assessment["verdict"],
            "explanation": assessment["explanation"],
            "evidence_indices": assessment["evidence_indices"],
            "cited_evidence": cited_evidence,
            "evidence": evidence,
            "evidence_status": evidence_result["status"],
        })

    # 5. Calculate the score from assessed verdicts.
    score_result = calculate_score(assessed_claims)

    return {
        "status": "assessment_complete",
        "total_claims": len(assessed_claims),
        "claims": assessed_claims,
        **score_result,
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


