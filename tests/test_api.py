import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_home_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["project"] == "Detector Sherlock"
    assert data["version"] == "0.3.0"
    assert "PubMed (biomedical research)" in data["sources"]
    assert "neural+lexical (DeBERTa NLI)" in data["inference_modes"]


def test_health_endpoint_reports_nli_status(monkeypatch):
    nli_status = {
        "model": "cross-encoder/nli-deberta-v3-small",
        "available": False,
        "mode": "lexical-only",
    }
    monkeypatch.setattr("app.main.nli_engine.get_status", lambda: nli_status)

    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy",
        "version": "0.3.0",
        "nli_engine": nli_status,
    }


def test_verify_empty():
    response = client.post("/verify", json={"response": "Short."})
    assert response.status_code == 200
    data = response.json()
    assert data["claims_count"] == 0
    assert data["summary"]["reliability_score"] == 0.0
    assert data["claims"] == []


def test_verify_validates_response_length():
    empty_response = client.post("/verify", json={"response": ""})
    oversized_response = client.post("/verify", json={"response": "x" * 20001})

    assert empty_response.status_code == 422
    assert oversized_response.status_code == 422


def test_verify_returns_all_verdicts_and_extension_fields(monkeypatch):
    claims = [
        "Metformin is a first-line medication for type 2 diabetes.",
        "Smoking cigarettes reduces risk of lung cancer.",
        "The ancient city of Aster was founded in 411 BCE.",
    ]
    evidence_by_claim = {
        claims[0]: [
            {
                "title": "Metformin is a first-line medication",
                "abstract": "Metformin is a first-line medication for type 2 diabetes.",
                "url": "https://example.test/metformin",
                "sources": ["Test"],
            }
        ],
        claims[1]: [
            {
                "title": "Smoking cigarettes increases risk of lung cancer",
                "abstract": "Smoking cigarettes increases risk of lung cancer.",
                "url": "https://example.test/smoking",
                "sources": ["Test"],
            }
        ],
        claims[2]: [],
    }

    monkeypatch.setattr("app.main.extract_claims", lambda response: claims)

    async def fake_search(query, limit_per_source, selected_sources):
        results = evidence_by_claim[query]
        return {
            "results": results,
            "sources": {
                source: {"status": "success", "count": len(results)}
                for source in selected_sources
            },
        }

    monkeypatch.setattr("app.main.search_all_sources", fake_search)

    response = client.post("/verify", json={"response": "response under test"})

    assert response.status_code == 200
    data = response.json()
    assert {claim["verdict"] for claim in data["claims"]} == {
        "supported",
        "contradicted",
        "unverified",
    }
    assert data["summary"]["supported"] == 1
    assert data["summary"]["contradicted"] == 1
    assert data["summary"]["unverified"] == 1
    assert data["inference_mode"] == "lexical-only"
    assert all(
        {
            "claim",
            "domain",
            "sources_targeted",
            "verdict",
            "confidence",
            "explanation",
            "best_evidence",
            "nli_scores",
            "inference_mode",
            "evidence_count",
            "sources",
            "evidence",
        }
        <= claim.keys()
        for claim in data["claims"]
    )


@pytest.mark.parametrize(
    ("path", "source_name", "search_function"),
    [
        ("/sources/pubmed", "PubMed", "search_pubmed"),
        ("/sources/crossref", "Crossref", "search_crossref"),
        ("/sources/openalex", "OpenAlex", "search_openalex"),
    ],
)
def test_individual_source_endpoints(path, source_name, search_function, monkeypatch):
    calls = []
    results = [{"title": f"{source_name} record"}]

    async def fake_search(query, limit):
        calls.append((query, limit))
        return results

    monkeypatch.setattr(f"app.main.{search_function}", fake_search)

    response = client.get(path, params={"q": "  test query  ", "limit": 99})

    assert response.status_code == 200
    assert response.json() == {
        "source": source_name,
        "query": "  test query  ",
        "count": 1,
        "results": results,
    }
    assert calls == [("test query", 10)]


def test_source_search_all_skips_blank_query_and_limits_results(monkeypatch):
    calls = []
    result = {"query": "test query", "results": []}

    async def fake_search(query, limit_per_source):
        calls.append((query, limit_per_source))
        return result

    monkeypatch.setattr("app.main.search_all_sources", fake_search)

    blank_response = client.get("/sources/search-all", params={"q": "   "})
    response = client.get(
        "/sources/search-all", params={"q": "  test query  ", "limit": 99}
    )

    assert blank_response.status_code == 200
    assert blank_response.json() == {"results": []}
    assert response.status_code == 200
    assert response.json() == result
    assert calls == [("test query", 10)]
