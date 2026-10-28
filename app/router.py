from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Set

from app.domain_classifier import (
    DOMAIN_BIOMEDICAL,
    DOMAIN_COMPUTING,
    DOMAIN_FINANCIAL,
    DOMAIN_GENERAL,
    DOMAIN_GOVERNMENT_ECONOMIC,
    DOMAIN_NEWS,
    DOMAIN_SPACE,
    DOMAIN_UNKNOWN,
    DomainStr,
    LEXICAL_KEYWORDS,
    ClassificationResult,
    classify_claim,
    keyword_only_classify,
)

# Export legacy keyword sets for full backward compatibility
BIOMEDICAL_KEYWORDS: Set[str] = LEXICAL_KEYWORDS[DOMAIN_BIOMEDICAL]
COMPUTING_KEYWORDS: Set[str] = LEXICAL_KEYWORDS[DOMAIN_COMPUTING]
SPACE_KEYWORDS: Set[str] = LEXICAL_KEYWORDS[DOMAIN_SPACE]
FINANCIAL_KEYWORDS: Set[str] = LEXICAL_KEYWORDS[DOMAIN_FINANCIAL]
GOVERNMENT_KEYWORDS: Set[str] = LEXICAL_KEYWORDS[DOMAIN_GOVERNMENT_ECONOMIC]
NEWS_KEYWORDS: Set[str] = LEXICAL_KEYWORDS[DOMAIN_NEWS]


@dataclass
class RoutingResult:
    """Structured internal routing outcome."""
    primary_domain: str
    domains: List[str]
    sources: List[str]
    confidence: float
    is_confident: bool
    method: str
    scores: Dict[str, float] = field(default_factory=dict)
    details: Dict[str, Any] = field(default_factory=dict)


def get_sources_for_domain(domain: str) -> list[str]:
    """
    Returns the list of source names appropriate for a given domain.
    Preserves all existing source mappings and handles domain aliases.
    """
    normalized = str(domain).lower().strip()

    if normalized == DOMAIN_BIOMEDICAL:
        return ["PubMed", "Wikipedia", "Crossref", "OpenAlex"]
    elif normalized == DOMAIN_FINANCIAL:
        return ["SEC EDGAR", "News", "Wikipedia", "Crossref", "OpenAlex"]
    elif normalized in (DOMAIN_GOVERNMENT_ECONOMIC, "government"):
        return ["World Bank / GovData", "News", "Wikipedia", "Crossref"]
    elif normalized == DOMAIN_NEWS:
        return ["News", "FactCheck", "Wikipedia"]
    elif normalized in (DOMAIN_COMPUTING, DOMAIN_SPACE):
        return ["Wikipedia", "Crossref", "OpenAlex"]
    else:
        # General or unknown claims check broad encyclopedia, fact checks, news, and academic indexes
        return ["Wikipedia", "FactCheck", "News", "Crossref", "OpenAlex"]


def get_sources_for_domains(domains: list[str]) -> list[str]:
    """
    Merges sources across multiple detected domains while maintaining
    deterministic priority and eliminating duplicate connectors.
    """
    if not domains:
        return get_sources_for_domain(DOMAIN_GENERAL)

    merged: list[str] = []
    seen: set[str] = set()

    for dom in domains:
        for src in get_sources_for_domain(dom):
            if src not in seen:
                seen.add(src)
                merged.append(src)

    return merged


def route_claim(claim: str) -> RoutingResult:
    """
    Routes a claim by executing hybrid semantic-lexical domain classification,
    selecting relevant evidence sources (with multi-domain deduplication),
    and providing confidence and fallback diagnostics.
    """
    classification: ClassificationResult = classify_claim(claim)
    sources = get_sources_for_domains(classification.domains)

    return RoutingResult(
        primary_domain=classification.primary_domain,
        domains=classification.domains,
        sources=sources,
        confidence=classification.confidence,
        is_confident=classification.is_confident,
        method=classification.method,
        scores=classification.scores,
        details=classification.details,
    )


def detect_domain(claim: str) -> str:
    """
    Identifies the subject domain of a claim.
    Maintains full backward-compatible signature and returns the primary domain.
    """
    result = route_claim(claim)
    return result.primary_domain
