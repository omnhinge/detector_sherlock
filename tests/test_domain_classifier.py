import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from app.domain_classifier import (
    DOMAIN_BIOMEDICAL,
    DOMAIN_COMPUTING,
    DOMAIN_FINANCIAL,
    DOMAIN_GENERAL,
    DOMAIN_GOVERNMENT_ECONOMIC,
    DOMAIN_UNKNOWN,
    DomainStr,
    classify_claim,
    count_lexical_matches,
    keyword_only_classify,
    MIN_SEMANTIC_THRESHOLD,
    MULTI_DOMAIN_MARGIN,
    MULTI_DOMAIN_MIN_SCORE,
)
from app.router import route_claim, detect_domain, get_sources_for_domains, get_sources_for_domain


# --- Fixtures and Mocks ---

@pytest.fixture
def mock_domain_embeddings():
    """
    Deterministic orthogonal-like domain anchor embeddings for unit testing without neural models.
    Dimension = 5 (biomedical, financial, government_economic, computing, general).
    """
    return {
        DOMAIN_BIOMEDICAL: np.array([[1.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        DOMAIN_FINANCIAL: np.array([[0.0, 1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        DOMAIN_GOVERNMENT_ECONOMIC: np.array([[0.0, 0.0, 1.0, 0.0, 0.0]], dtype=np.float32),
        DOMAIN_COMPUTING: np.array([[0.0, 0.0, 0.0, 1.0, 0.0]], dtype=np.float32),
        DOMAIN_GENERAL: np.array([[0.0, 0.0, 0.0, 0.0, 1.0]], dtype=np.float32),
    }


def test_domain_str_equality_alias():
    """Verify DomainStr transparently aliases government and government_economic."""
    d = DomainStr("government_economic")
    assert d == "government_economic"
    assert d == "government"
    assert "government" == d

    # Still distinguishes other domains
    assert d != "biomedical"
    assert d != "financial"


def test_count_lexical_matches():
    """Verify exact lexical matching counts."""
    claim = "Tesla reported quarterly revenue and earnings."
    matches = count_lexical_matches(claim)
    assert matches[DOMAIN_FINANCIAL] >= 3
    assert matches[DOMAIN_BIOMEDICAL] == 0


def test_keyword_only_classify_fallback():
    """Verify keyword-only fallback behavior when embeddings are unavailable."""
    claim = "Metformin is used to treat patients with type 2 diabetes."
    res = keyword_only_classify(claim)
    assert res.method == "keyword_fallback"
    assert res.primary_domain == DOMAIN_BIOMEDICAL
    assert res.is_confident is True

    # Claim with zero keywords defaults to general
    empty_claim_res = keyword_only_classify("The object was situated beside the path.")
    assert empty_claim_res.primary_domain == DOMAIN_GENERAL
    assert empty_claim_res.is_confident is False


def test_classify_claim_with_mock_embedder_single_domain(mock_domain_embeddings):
    """Test deterministic single domain classification with mock embedder."""
    # Vector pointing directly to biomedical
    def mock_embedder(text: str) -> np.ndarray:
        return np.array([0.9, 0.1, 0.0, 0.0, 0.0], dtype=np.float32)

    res = classify_claim(
        "A clinical test hypothesis",
        embedder=mock_embedder,
        domain_embeddings=mock_domain_embeddings,
    )
    assert res.primary_domain == DOMAIN_BIOMEDICAL
    assert res.domains == [DOMAIN_BIOMEDICAL]
    assert res.confidence > 0.5
    assert res.is_confident is True
    assert res.method == "hybrid_semantic"


def test_classify_claim_with_mock_embedder_multi_domain(mock_domain_embeddings):
    """Test deterministic multi-domain triggering when two domains score close to each other."""
    # Vector pointing equally to biomedical (0.45) and financial (0.44)
    def mock_embedder(text: str) -> np.ndarray:
        return np.array([0.45, 0.44, 0.0, 0.0, 0.0], dtype=np.float32)

    res = classify_claim(
        "Alpha beta gamma delta epsilon",
        embedder=mock_embedder,
        domain_embeddings=mock_domain_embeddings,
    )
    assert DOMAIN_BIOMEDICAL in res.domains
    assert DOMAIN_FINANCIAL in res.domains
    assert len(res.domains) == 2


def test_classify_claim_low_confidence_fallback(mock_domain_embeddings):
    """Test fallback to general when all similarities fall below threshold."""
    # Vector with very weak similarity to all specialized domains
    def mock_embedder(text: str) -> np.ndarray:
        return np.array([0.05, 0.05, 0.05, 0.05, 0.08], dtype=np.float32)

    res = classify_claim(
        "Some vague ambiguous phrase",
        embedder=mock_embedder,
        domain_embeddings=mock_domain_embeddings,
    )
    assert res.primary_domain == DOMAIN_GENERAL
    assert res.is_confident is False
    assert res.method == "low_confidence_fallback"


def test_classify_claim_empty_string():
    """Verify empty or whitespace-only claim handling."""
    res = classify_claim("")
    assert res.primary_domain == DOMAIN_GENERAL
    assert res.is_confident is False
    assert res.method == "empty_claim"


def test_classify_claim_model_failure_graceful_fallback():
    """Verify that if get_embedding_model fails, it gracefully falls back to keyword classifier."""
    with patch("app.domain_classifier.get_embedding_model", return_value=None):
        claim = "Tesla reported delivery figures for its electric vehicles."
        res = classify_claim(claim)
        assert res.method == "keyword_fallback"
        assert res.primary_domain == DOMAIN_FINANCIAL


def test_classify_claim_model_encode_exception_fallback():
    """Verify that if model.encode raises an unexpected exception, keyword fallback handles it."""
    mock_model = MagicMock()
    mock_model.encode.side_effect = RuntimeError("CUDA out of memory simulation")

    with patch("app.domain_classifier.get_embedding_model", return_value=mock_model):
        claim = "Metformin patient treatment."
        res = classify_claim(claim)
        assert res.method == "keyword_fallback"
        assert res.primary_domain == DOMAIN_BIOMEDICAL
