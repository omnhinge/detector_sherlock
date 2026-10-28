import pytest
import asyncio
from unittest.mock import patch, AsyncMock

from app.router import route_claim, detect_domain, get_sources_for_domain, get_sources_for_domains
from app.domain_classifier import (
    DOMAIN_BIOMEDICAL,
    DOMAIN_COMPUTING,
    DOMAIN_FINANCIAL,
    DOMAIN_GENERAL,
    DOMAIN_GOVERNMENT_ECONOMIC,
    DOMAIN_NEWS,
    DOMAIN_SPACE,
)
from app.main import app, verify, VerifyRequest
from app.sources.search_all import search_all_sources


def _run_sync(coro):
    return asyncio.run(coro)


# --- Source Selection and Deduplication Tests ---

def test_sources_for_all_domains():
    """Verify that source selection covers required connectors per domain."""
    bio_sources = get_sources_for_domain(DOMAIN_BIOMEDICAL)
    assert "PubMed" in bio_sources
    assert "Crossref" in bio_sources
    assert "OpenAlex" in bio_sources

    fin_sources = get_sources_for_domain(DOMAIN_FINANCIAL)
    assert "SEC EDGAR" in fin_sources
    assert "Crossref" in fin_sources

    gov_sources = get_sources_for_domain(DOMAIN_GOVERNMENT_ECONOMIC)
    assert "World Bank / GovData" in gov_sources
    assert "Crossref" in gov_sources

    comp_sources = get_sources_for_domain(DOMAIN_COMPUTING)
    assert "Crossref" in comp_sources
    assert "OpenAlex" in comp_sources

    gen_sources = get_sources_for_domain(DOMAIN_GENERAL)
    assert "Wikipedia" in gen_sources
    assert "FactCheck" in gen_sources
    assert "News" in gen_sources


def test_sources_for_domains_deduplication():
    """Verify that multi-domain routing merges sources without duplicate entries."""
    domains = [DOMAIN_BIOMEDICAL, DOMAIN_FINANCIAL]
    sources = get_sources_for_domains(domains)

    # Must contain sources from both domains
    assert "PubMed" in sources
    assert "SEC EDGAR" in sources

    # No duplicates allowed
    assert len(sources) == len(set(sources))


def test_sources_for_empty_domain_list():
    """Verify fallback when an empty domain list is passed."""
    sources = get_sources_for_domains([])
    assert "Wikipedia" in sources
    assert "Crossref" in sources


# --- Paraphrase and Semantic Classification Tests (Offline Cached Model) ---

def test_semantic_routing_biomedical_paraphrase():
    """Biomedical claim with specialized terms not in standard keyword list."""
    claim = "Renal clearance of endogenous metabolites was impaired in glomerulonephritis."
    result = route_claim(claim)
    assert result.primary_domain == DOMAIN_BIOMEDICAL
    assert "PubMed" in result.sources


def test_semantic_routing_financial_paraphrase():
    """Financial claim without generic keywords like revenue or tesla."""
    claim = "The conglomerate completed a leveraged buyout funded through high-yield corporate debentures."
    result = route_claim(claim)
    assert result.primary_domain == DOMAIN_FINANCIAL
    assert "SEC EDGAR" in result.sources


def test_semantic_routing_government_economic_paraphrase():
    """Macroeconomics claim without standard gdp or inflation words."""
    claim = "Sovereign treasury yields inverted along the maturity curve signalling macroeconomic contraction."
    result = route_claim(claim)
    assert result.primary_domain in (DOMAIN_GOVERNMENT_ECONOMIC, "government")
    assert "World Bank / GovData" in result.sources


def test_semantic_routing_computing_paraphrase():
    """Computing claim with algorithm and system design concepts."""
    claim = "Asymptotic time complexity of the sorting procedure is bounded by big-O of N log N."
    result = route_claim(claim)
    assert result.primary_domain == DOMAIN_COMPUTING
    assert "Crossref" in result.sources


def test_semantic_routing_multi_domain_claim():
    """Legitimate multi-domain claim spanning biomedical and financial."""
    claim = "Pfizer reported fourteen billion dollars in quarterly revenue from its COVID-19 vaccine."
    result = route_claim(claim)
    # Primary must be biomedical or financial
    assert result.primary_domain in (DOMAIN_BIOMEDICAL, DOMAIN_FINANCIAL)
    # Multi-domain sources must target both PubMed and SEC EDGAR
    assert "PubMed" in result.sources
    assert "SEC EDGAR" in result.sources


def test_semantic_routing_general_fallback():
    """Vague or architectural historical fact routes to general sources."""
    claim = "The ancient Library of Alexandria was one of the largest archives of antiquity."
    result = route_claim(claim)
    assert result.primary_domain == DOMAIN_GENERAL
    assert "Wikipedia" in result.sources


# --- Source Connector Failure Handling ---

def test_search_all_sources_resilience_to_connector_failure():
    """Verify that a single source exception does not crash the verification pipeline."""
    async def run_resilience_test():
        with patch("app.sources.search_all.search_pubmed", new_callable=AsyncMock) as mock_pubmed, \
             patch("app.sources.search_all.search_crossref", new_callable=AsyncMock) as mock_crossref:
            mock_pubmed.side_effect = ConnectionError("PubMed timeout")
            mock_crossref.return_value = [
                {
                    "title": "Coronary Artery Disease Study",
                    "abstract": "Clinical analysis of arterial intervention.",
                    "url": "https://doi.org/10.1000/182",
                    "doi": "10.1000/182",
                    "source": "Crossref",
                }
            ]
            res = await search_all_sources(
                "coronary intervention",
                limit_per_source=3,
                selected_sources=["PubMed", "Crossref"],
            )

            # PubMed failed, but status recorded gracefully
            assert res["sources"]["PubMed"]["status"] == "error"
            # Crossref succeeded
            assert res["sources"]["Crossref"]["status"] == "ok"
            # Merged results preserved Crossref record
            assert len(res["results"]) == 1
            assert res["results"][0]["title"] == "Coronary Artery Disease Study"

    asyncio.run(run_resilience_test())


# --- API and Chrome Extension Compatibility Tests ---

def test_verify_endpoint_schema_compatibility():
    """Verify /verify API output conforms to the Chrome extension contract."""
    async def run_verify_test():
        # Mock search_all_sources to avoid external network calls during unit test
        mock_search_res = {
            "query": "Test claim",
            "total": 1,
            "sources": {"Wikipedia": {"status": "ok", "count": 1}},
            "results": [
                {
                    "title": "Mount Everest",
                    "abstract": "Mount Everest is Earth's highest mountain above sea level.",
                    "url": "https://en.wikipedia.org/wiki/Mount_Everest",
                    "doi": None,
                    "authors": [],
                    "published_year": None,
                    "sources": ["Wikipedia"],
                }
            ],
        }
        mock_eval = {
            "verdict": "supported",
            "confidence": 0.95,
            "explanation": "Corroborated by Wikipedia",
            "best_evidence": {
                "snippet": "Mount Everest is Earth's highest mountain above sea level.",
                "sources": ["Wikipedia"],
                "url": "https://en.wikipedia.org/wiki/Mount_Everest",
            },
            "nli": {"available": True, "entailment": 0.95, "contradiction": 0.01, "neutral": 0.04},
        }

        with patch("app.main.search_all_sources", new_callable=AsyncMock, return_value=mock_search_res):
            with patch("app.main.evaluate_claim_against_evidence", return_value=mock_eval):
                request = VerifyRequest(response="Mount Everest is the highest terrestrial mountain peak above sea level.")
                response = await verify(request)

                assert response["status"] == "completed"
                assert response["claims_count"] == 1
                assert "summary" in response
                assert "reliability_score" in response["summary"]
                assert "hallucination_risk" in response["summary"]
                assert "evidence_coverage" in response["summary"]

                claim_data = response["claims"][0]
                # Essential fields required by popup.js
                assert "claim" in claim_data
                assert "domain" in claim_data
                assert "sources_targeted" in claim_data
                assert "verdict" in claim_data
                assert "confidence" in claim_data
                assert "best_evidence" in claim_data
                # New non-breaking routing diagnostics
                assert "routing" in claim_data
                assert "domains" in claim_data["routing"]

    asyncio.run(run_verify_test())
