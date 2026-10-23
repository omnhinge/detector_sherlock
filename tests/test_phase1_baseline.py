import asyncio

from app import evaluator
from app.sources import search_all


def test_search_all_reports_source_failure_and_keeps_successful_results(monkeypatch):
    async def failing_search(query, limit):
        raise RuntimeError("source unavailable")

    async def successful_search(query, limit):
        return [
            {
                "title": "A test record",
                "abstract": "A deterministic record returned by the test.",
                "url": "https://example.test/record",
                "source": "Crossref",
            }
        ]

    monkeypatch.setattr(search_all, "search_pubmed", failing_search)
    monkeypatch.setattr(search_all, "search_crossref", successful_search)

    result = asyncio.run(
        search_all.search_all_sources(
            "a test claim",
            limit_per_source=2,
            selected_sources=["PubMed", "Crossref"],
        )
    )

    assert result["sources"]["PubMed"] == {
        "status": "error",
        "count": 0,
        "error": "source unavailable",
    }
    assert result["sources"]["Crossref"] == {"status": "success", "count": 1}
    assert result["results"][0]["title"] == "A test record"
    assert result["results"][0]["sources"] == ["Crossref"]
    assert result["records_found"] == result["unique_records"] == 1


def test_evaluator_uses_lexical_only_when_nli_is_unavailable():
    claim = "Metformin is a first-line medication for type 2 diabetes."
    evidence = [
        {
            "title": "Metformin is a first-line medication",
            "abstract": "Metformin is a first-line medication for type 2 diabetes.",
            "url": "https://example.test/metformin",
            "sources": ["Test"],
        }
    ]

    result = evaluator.evaluate_claim_against_evidence(claim, evidence)

    assert result["verdict"] == "supported"
    assert result["inference_mode"] == "lexical-only"
    assert result["nli_scores"] is None
