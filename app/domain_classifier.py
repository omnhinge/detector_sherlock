"""
Semantic and hybrid domain classifier module for Detector Sherlock.

Combines sentence-transformer embeddings (cosine similarity against multi-facet
domain anchor descriptions) with lexical keyword signals and confidence margin
analysis to provide robust multi-domain classification and fallback behavior.
"""

from __future__ import annotations

import os
import re
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np


# Canonical domain labels
DOMAIN_BIOMEDICAL = "biomedical"
DOMAIN_FINANCIAL = "financial"
DOMAIN_GOVERNMENT_ECONOMIC = "government_economic"
DOMAIN_COMPUTING = "computing"
DOMAIN_GENERAL = "general"
DOMAIN_UNKNOWN = "unknown"

# Additional supported domains for full compatibility
DOMAIN_SPACE = "space"
DOMAIN_NEWS = "news"

SUPPORTED_DOMAINS = [
    DOMAIN_BIOMEDICAL,
    DOMAIN_FINANCIAL,
    DOMAIN_GOVERNMENT_ECONOMIC,
    DOMAIN_COMPUTING,
    DOMAIN_GENERAL,
]

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Confidence and multi-domain routing thresholds (calibrated against routing benchmark)
MIN_SEMANTIC_THRESHOLD = 0.18    # Minimum similarity to be considered a specialized domain
CONFIDENCE_MARGIN_THRESHOLD = 0.05 # Margin between top and runner-up for single-domain certainty
MULTI_DOMAIN_MARGIN = 0.10       # Runner-up within this margin of top score triggers multi-domain
MULTI_DOMAIN_MIN_SCORE = 0.24    # Runner-up must also meet this absolute floor
LEXICAL_BOOST_FACTOR = 0.08      # Lexical signal weight per matched term (capped)
MAX_LEXICAL_BOOST = 0.16         # Maximum lexical boost added to semantic similarity


class DomainStr(str):
    """
    String subclass ensuring backwards-compatibility for 'government' and 'government_economic'.
    Serializes to JSON as a regular string.
    """
    def __eq__(self, other: Any) -> bool:
        if super().__eq__(other):
            return True
        s = str(self)
        o = str(other)
        if s == DOMAIN_GOVERNMENT_ECONOMIC and o == "government":
            return True
        if s == "government" and o == DOMAIN_GOVERNMENT_ECONOMIC:
            return True
        return False

    def __hash__(self) -> int:
        return super().__hash__()


@dataclass
class DomainScore:
    domain: str
    semantic_similarity: float
    lexical_matches: int
    combined_score: float


@dataclass
class ClassificationResult:
    primary_domain: str
    domains: List[str]
    confidence: float
    is_confident: bool
    method: str
    scores: Dict[str, float]
    details: Dict[str, Any] = field(default_factory=dict)


# Multi-facet domain anchor descriptions covering scope, typical claims, and terminology
DOMAIN_DESCRIPTIONS: Dict[str, List[str]] = {
    DOMAIN_BIOMEDICAL: [
        "Clinical trials, randomized controlled studies, medical therapies, patient healthcare diagnoses, and pharmacotherapy.",
        "Molecular genetics, CRISPR gene editing, cellular biology, mRNA, DNA, proteins, immunology, and oncology neoplasms.",
        "Cardiology, myocardial infarction, coronary intervention, renal failure, creatinine, and clinical diagnostic laboratory markers.",
        "Pathology, bacterial infections, antimicrobial resistance, viral transmission, vaccines, and disease symptoms.",
        "Neurobiology, neurotransmitters, dopamine, brain receptors, striatum, and central nervous system physiology.",
        "Pharmacology, prescription medications, drug dosages, pharmacokinetics, and adverse therapeutic effects.",
    ],
    DOMAIN_FINANCIAL: [
        "Corporate earnings reports, quarterly revenue, operating profit margins, EBITDA, free cash flow, and balance sheets.",
        "Stock market equities, shares, initial public offerings, shareholder dividends, corporate valuation, and capital markets.",
        "Corporate debt, senior unsecured debentures, credit ratings, leveraged buyouts, working capital, and SEC filings.",
        "Securities and Exchange Commission disclosures, Form 10-K, annual reports, corporate governance, and asset management.",
        "Derivatives, equity options, convertible bonds, institutional investors, and financial exchange trading.",
    ],
    DOMAIN_GOVERNMENT_ECONOMIC: [
        "Macroeconomic indicators, gross domestic product GDP growth, national inflation rates, and consumer price index CPI.",
        "Central bank monetary policy, benchmark interest rates, repo rates, banking reserves, and currency money supply.",
        "Public finance, national union budget, fiscal deficits, sovereign debt yields, and sovereign treasury bond curves.",
        "International trade policy, customs tariffs, public welfare subsidies, demographic census, and national statistical offices.",
        "World Bank, multilateral development assistance loans, and government regulatory public sector administration.",
    ],
    DOMAIN_COMPUTING: [
        "Distributed systems, consensus protocols, Raft, fault tolerance, replicated state machines, and microkernels.",
        "Computer software engineering, programming languages, compilers, abstract syntax trees, bytecode, and algorithms.",
        "Relational databases, SQL, B-tree indexes, ACID transactions, join optimization, and vector search indices.",
        "Artificial intelligence, machine learning, deep learning neural networks, transformers, and computational complexity.",
        "Computer hardware architecture, microprocessors, CPU, GPU acceleration, register sets, and operating systems.",
        "Web architecture, RESTful API endpoints, HTTP protocols, serialized JSON payloads, and network sockets.",
    ],
    DOMAIN_GENERAL: [
        "Historical architecture, monuments, the Eiffel Tower, world heritage sites, and structural engineering history.",
        "World history, ancient civilizations, the Library of Alexandria, archaeological discoveries, and international treaties.",
        "Classical literature, playwrights, William Shakespeare, poetry, and classical music symphony composers.",
        "Physical geography, terrestrial mountain peaks, Mount Everest, oceans, Mariana Trench, and natural ecology.",
        "General cultural events, non-specialized general knowledge, everyday trivia, and encyclopedic facts.",
    ],
    DOMAIN_SPACE: [
        "Astronomy, astrophysics, planetary science, orbital mechanics, stars, galaxies, and celestial bodies.",
        "Space exploration missions, spacecraft, NASA, ESA, telescopes, and cosmological observations.",
    ],
    DOMAIN_NEWS: [
        "Breaking news events, international diplomacy, geopolitical conflicts, protests, elections, and summits.",
        "Recent headlines, government press releases, investigative reports, and journalistic coverage of current events.",
    ],
}


# Lexical keywords for domain-specific acronyms, named entities, and specialized vocabulary
LEXICAL_KEYWORDS: Dict[str, Set[str]] = {
    DOMAIN_BIOMEDICAL: {
        "patient", "patients", "disease", "diseases", "treatment", "treatments",
        "medication", "medications", "drug", "drugs", "clinical", "therapy",
        "therapies", "vaccine", "vaccines", "cancer", "diabetes", "cardiac",
        "blood", "cell", "cells", "metformin", "symptom", "symptoms", "syndrome",
        "infection", "infectious", "virus", "viral", "bacteria", "bacterial",
        "dna", "rna", "mrna", "gene", "genes", "protein", "proteins", "organ",
        "dosage", "pharmacology", "physician", "hospital", "diagnosis", "surgery",
        "oncology", "chemotherapy", "myocardial", "infarction", "renal", "creatinine",
        "statin", "sublingual", "dopamine", "striatum", "crispr", "oligonucleotide",
    },
    DOMAIN_FINANCIAL: {
        "tesla", "revenue", "quarterly", "earnings", "fiscal", "delivery", "deliveries",
        "stock", "shares", "dividend", "sec", "nasdaq", "nyse", "ipo", "profit",
        "margin", "billion", "million", "valuation", "apple", "microsoft", "amazon",
        "google", "meta", "nvidia", "gigafactory", "roadster", "q1", "q2", "q3", "q4",
        "ebitda", "10-k", "debenture", "buyout", "derivatives", "shareholders", "bondholders",
        "warrants", "liabilities", "equity",
    },
    DOMAIN_GOVERNMENT_ECONOMIC: {
        "gdp", "inflation", "cpi", "rbi", "reserve", "monetary", "repo", "budget",
        "deficit", "unemployment", "population", "census", "treaty", "parliament",
        "ministry", "tax", "tariff", "economic", "economy", "treasury", "macroeconomic",
        "sovereign", "tariffs", "yields", "subsidies", "statutory", "fdi",
    },
    DOMAIN_COMPUTING: {
        "software", "hardware", "programming", "compiler", "algorithm", "algorithms",
        "code", "coding", "database", "databases", "network", "networks", "python",
        "java", "javascript", "c++", "rust", "cpu", "gpu", "api", "apis", "framework",
        "git", "linux", "operating system", "neural network", "machine learning",
        "deep learning", "transformer", "artificial intelligence", "server", "servers",
        "raft", "microkernel", "json", "rest", "restful", "b-tree", "bytecode", "hashing",
        "sha-256", "risc-v", "concurrency", "socket", "asynchronous",
    },
    DOMAIN_SPACE: {
        "planet", "planets", "mars", "jupiter", "saturn", "moon", "orbit", "orbital",
        "galaxy", "galaxies", "asteroid", "asteroids", "star", "stars", "telescope",
        "astronomy", "astrophysics", "solar", "spacecraft", "nasa", "esa", "cosmic",
    },
    DOMAIN_NEWS: {
        "breaking", "announced", "reported", "protest", "election", "war", "conflict",
        "summit", "accident", "strike", "fire", "earthquake", "resigned", "killed",
        "arrested", "speech", "minister", "president", "police", "investigation",
        "scandal", "headline", "spokesperson", "sanctions", "casualty", "casualties",
        "ceasefire", "yesterday", "today", "crisis", "diplomat", "press",
    },
}


# Thread-safe model singleton
_MODEL_LOCK = threading.Lock()
_EMBEDDING_MODEL: Any = None
_EMBEDDING_MODEL_LOADED: bool = False
_DOMAIN_EMBEDDINGS_CACHE: Optional[Dict[str, np.ndarray]] = None


def get_embedding_model(model_name: str = DEFAULT_MODEL_NAME) -> Any:
    """
    Lazy thread-safe loader for SentenceTransformer.
    Returns None gracefully if the model or sentence_transformers library cannot be loaded.
    """
    global _EMBEDDING_MODEL, _EMBEDDING_MODEL_LOADED
    if _EMBEDDING_MODEL_LOADED:
        return _EMBEDDING_MODEL

    with _MODEL_LOCK:
        if _EMBEDDING_MODEL_LOADED:
            return _EMBEDDING_MODEL
        try:
            # Force offline mode if HuggingFace Hub is unreachable in sandboxed environments
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            from sentence_transformers import SentenceTransformer

            # Attempt local files first, then fallback to standard init
            try:
                _EMBEDDING_MODEL = SentenceTransformer(model_name, local_files_only=True)
            except Exception:
                _EMBEDDING_MODEL = SentenceTransformer(model_name)

        except Exception:
            _EMBEDDING_MODEL = None

        _EMBEDDING_MODEL_LOADED = True
        return _EMBEDDING_MODEL


def get_domain_embeddings(model: Any) -> Dict[str, np.ndarray]:
    """
    Precomputes and caches normalized embeddings for all domain description anchors.
    Reuses the precomputed cache across all subsequent classifications.
    """
    global _DOMAIN_EMBEDDINGS_CACHE
    if _DOMAIN_EMBEDDINGS_CACHE is not None:
        return _DOMAIN_EMBEDDINGS_CACHE

    if model is None:
        return {}

    cache: Dict[str, np.ndarray] = {}
    for domain, descs in DOMAIN_DESCRIPTIONS.items():
        embs = model.encode(descs, normalize_embeddings=True)
        cache[domain] = np.array(embs, dtype=np.float32)

    _DOMAIN_EMBEDDINGS_CACHE = cache
    return _DOMAIN_EMBEDDINGS_CACHE


def count_lexical_matches(claim: str) -> Dict[str, int]:
    """Counts lexical keyword matches in the claim across supported domains."""
    words = set(re.findall(r"\b[a-z0-9_-]+\b", claim.lower()))
    matches: Dict[str, int] = {}
    for domain, kws in LEXICAL_KEYWORDS.items():
        matches[domain] = len(words & kws)
    return matches


def keyword_only_classify(claim: str) -> ClassificationResult:
    """
    Legacy keyword-only classifier preserving existing deterministic routing.
    Used as an immediate offline fallback if the neural embedding model is unavailable.
    """
    matches = count_lexical_matches(claim)
    best_domain = max(matches, key=matches.get)
    best_count = matches[best_domain]

    if best_count > 0:
        primary = DomainStr(best_domain)
        # Check if multiple domains have significant keyword matches
        top_domains = [DomainStr(d) for d, cnt in matches.items() if cnt == best_count]
        return ClassificationResult(
            primary_domain=primary,
            domains=top_domains,
            confidence=min(1.0, 0.4 + 0.15 * best_count),
            is_confident=True,
            method="keyword_fallback",
            scores={d: float(cnt) for d, cnt in matches.items()},
            details={"keyword_counts": matches},
        )

    return ClassificationResult(
        primary_domain=DomainStr(DOMAIN_GENERAL),
        domains=[DomainStr(DOMAIN_GENERAL)],
        confidence=0.2,
        is_confident=False,
        method="keyword_fallback",
        scores={d: 0.0 for d in matches},
        details={"keyword_counts": matches},
    )


def classify_claim(
    claim: str,
    embedder: Optional[Callable[[str], np.ndarray]] = None,
    domain_embeddings: Optional[Dict[str, np.ndarray]] = None,
) -> ClassificationResult:
    """
    Hybrid semantic, confidence-aware, multi-domain classifier.
    
    1. Obtains normalized sentence embedding for the claim.
    2. Computes cosine similarity against cached multi-facet domain descriptions.
    3. Blends in lexical keyword matches as supplementary signals.
    4. Evaluates score separation margin to detect legitimate multi-domain claims.
    5. Applies calibrated confidence thresholds to fallback to general/unknown when uncertain.
    """
    if not claim or not claim.strip():
        return ClassificationResult(
            primary_domain=DomainStr(DOMAIN_GENERAL),
            domains=[DomainStr(DOMAIN_GENERAL)],
            confidence=0.0,
            is_confident=False,
            method="empty_claim",
            scores={},
        )

    # Allow custom embedder injection for deterministic offline unit testing
    claim_emb = None
    if embedder is not None and domain_embeddings is not None:
        try:
            claim_emb = embedder(claim)
            domain_embs = domain_embeddings
        except Exception:
            claim_emb = None

    if claim_emb is None:
        model = get_embedding_model()
        if model is None:
            return keyword_only_classify(claim)

        try:
            claim_emb = model.encode(claim, normalize_embeddings=True)
            domain_embs = get_domain_embeddings(model)
        except Exception:
            return keyword_only_classify(claim)

    # 1. Semantic similarities (max cosine similarity across facets for each domain)
    semantic_scores: Dict[str, float] = {}
    for dom, anchors in domain_embs.items():
        sims = np.dot(anchors, claim_emb)
        semantic_scores[dom] = float(np.max(sims))

    # 2. Lexical counts
    lexical_counts = count_lexical_matches(claim)

    # 3. Hybrid scoring: semantic similarity is primary; lexical keywords provide targeted boost
    combined_scores: Dict[str, float] = {}
    for dom, sem_sim in semantic_scores.items():
        lex_count = lexical_counts.get(dom, 0)
        boost = min(MAX_LEXICAL_BOOST, lex_count * LEXICAL_BOOST_FACTOR)
        combined_scores[dom] = round(sem_sim + boost, 4)

    # Rank domains by combined score
    ranked = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
    top_domain, top_score = ranked[0]
    runner_up_domain, runner_up_score = ranked[1] if len(ranked) > 1 else (DOMAIN_GENERAL, 0.0)
    score_margin = round(top_score - runner_up_score, 4)

    # Calibrated confidence calculation (clamped between 0.0 and 1.0)
    # Cosine similarities for all-MiniLM typically range from 0.05 (unrelated) to 0.70+ (exact match)
    raw_confidence = (top_score - MIN_SEMANTIC_THRESHOLD) / 0.40
    confidence = round(max(0.0, min(1.0, raw_confidence)), 3)

    # Check for low-confidence or ambiguous claims
    if top_score < MIN_SEMANTIC_THRESHOLD:
        return ClassificationResult(
            primary_domain=DomainStr(DOMAIN_GENERAL),
            domains=[DomainStr(DOMAIN_GENERAL)],
            confidence=confidence,
            is_confident=False,
            method="low_confidence_fallback",
            scores=combined_scores,
            details={
                "top_score": top_score,
                "semantic_scores": semantic_scores,
                "lexical_counts": lexical_counts,
                "reason": "Top domain score below minimum semantic threshold",
            },
        )

    # Determine multi-domain inclusion
    # If the runner-up score is close to the top score and exceeds the multi-domain floor, include both
    selected_domains = [DomainStr(top_domain)]
    if (
        score_margin <= MULTI_DOMAIN_MARGIN
        and runner_up_score >= MULTI_DOMAIN_MIN_SCORE
        and runner_up_domain != DOMAIN_GENERAL
        and runner_up_domain != top_domain
    ):
        selected_domains.append(DomainStr(runner_up_domain))

    # Check if a 3rd domain also qualifies for strong tripartite claims
    if len(ranked) > 2:
        third_domain, third_score = ranked[2]
        if (
            (top_score - third_score) <= MULTI_DOMAIN_MARGIN
            and third_score >= MULTI_DOMAIN_MIN_SCORE
            and third_domain not in (DOMAIN_GENERAL, top_domain, runner_up_domain)
        ):
            selected_domains.append(DomainStr(third_domain))

    is_confident = confidence >= 0.35 and score_margin >= CONFIDENCE_MARGIN_THRESHOLD

    return ClassificationResult(
        primary_domain=DomainStr(top_domain),
        domains=selected_domains,
        confidence=confidence,
        is_confident=is_confident,
        method="hybrid_semantic",
        scores=combined_scores,
        details={
            "score_margin": score_margin,
            "semantic_scores": semantic_scores,
            "lexical_counts": lexical_counts,
        },
    )
