import re
from typing import Any


# Common English words that carry little specific factual meaning.
# Filtering these out lets us focus on the "keywords" (like 'Metformin', 'diabetes', 'Guido', 'Python').
STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can", "cannot", "could", "did", "do",
    "does", "doing", "down", "during", "each", "few", "for", "from", "further",
    "had", "has", "have", "having", "he", "her", "here", "hers", "herself", "him",
    "himself", "his", "how", "i", "if", "in", "into", "is", "isn't", "it", "its",
    "itself", "me", "more", "most", "my", "myself", "no", "nor", "not", "of", "off",
    "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out",
    "over", "own", "same", "she", "should", "so", "some", "such", "than", "that",
    "the", "their", "theirs", "them", "themselves", "then", "there", "these", "they",
    "this", "those", "through", "to", "too", "under", "until", "up", "very", "was",
    "wasn't", "we", "were", "weren't", "what", "when", "where", "which", "while",
    "who", "whom", "why", "with", "won't", "would", "you", "your", "yours", "yourself",
}


def extract_keywords(text: str) -> set[str]:
    """
    Extract meaningful, lowercased keywords from a piece of text,
    ignoring common filler words and short symbols.
    """
    words = re.findall(r"\b[a-zA-Z0-9_-]{3,}\b", text.lower())
    return {w for w in words if w not in STOPWORDS}


def find_best_passage(claim_keywords: set[str], abstract: str) -> tuple[str, float]:
    """
    Look through every sentence in a paper's abstract to find the single
    sentence that has the highest overlap with the claim's keywords.
    Returns: (best_matching_sentence, match_ratio)
    """
    if not abstract or not claim_keywords:
        return ("", 0.0)

    # Split abstract into individual sentences
    sentences = re.split(r"(?<=[.!?])\s+", abstract.strip())
    best_sentence = ""
    best_overlap = 0.0

    for sentence in sentences:
        sentence_clean = sentence.strip()
        if len(sentence_clean) < 15:
            continue

        sentence_words = extract_keywords(sentence_clean)
        if not sentence_words:
            continue

        overlap_count = len(claim_keywords & sentence_words)
        overlap_ratio = overlap_count / len(claim_keywords)

        if overlap_ratio > best_overlap:
            best_overlap = overlap_ratio
            best_sentence = sentence_clean

    return (best_sentence, round(best_overlap, 2))


def evaluate_claim_against_evidence(
    claim: str,
    evidence_records: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Compares a single claim against all retrieved evidence papers.

    Determines:
    - verdict: 'supported' or 'unverified'
    - confidence: float between 0.0 and 1.0
    - best_evidence: the paper and specific sentence that best supports the claim
    - explanation: a clear, human-readable summary of why this verdict was given
    """
    claim_keywords = extract_keywords(claim)

    if not claim_keywords:
        return {
            "verdict": "unverified",
            "confidence": 0.0,
            "explanation": "Claim does not contain sufficient specific factual keywords to verify.",
            "best_evidence": None,
        }

    if not evidence_records:
        return {
            "verdict": "unverified",
            "confidence": 0.0,
            "explanation": "No relevant authoritative records were found for this claim.",
            "best_evidence": None,
        }

    best_match_record = None
    best_snippet = ""
    highest_score = 0.0

    for record in evidence_records:
        title = record.get("title", "")
        abstract = record.get("abstract", "")

        # 1. Check title keyword overlap
        title_words = extract_keywords(title)
        title_overlap = (
            len(claim_keywords & title_words) / len(claim_keywords)
            if claim_keywords
            else 0.0
        )

        # 2. Check abstract sentence-level overlap
        snippet, passage_overlap = find_best_passage(claim_keywords, abstract)

        # Combined score gives strong weight to passage matches, with title confirmation
        record_score = max(passage_overlap, title_overlap * 0.8)

        if record_score > highest_score:
            highest_score = record_score
            best_snippet = snippet if snippet else title
            best_match_record = record

    # Verdict threshold logic:
    # If key terms overlap substantially (>= 60%), we classify as 'supported'.
    # Otherwise, it remains 'unverified' (insufficient proof).
    if highest_score >= 0.60:
        verdict = "supported"
        explanation = (
            f"Supported by evidence from {best_match_record.get('sources', ['credible source'])} "
            f"in paper '{best_match_record.get('title')}': \"{best_snippet}\""
        )
    else:
        verdict = "unverified"
        if best_match_record:
            explanation = (
                f"Related literature was found ('{best_match_record.get('title')}'), "
                f"but evidence overlap ({int(highest_score * 100)}%) is too low to definitively corroborate."
            )
        else:
            explanation = "No sufficiently relevant evidence found in queried databases."

    return {
        "verdict": verdict,
        "confidence": round(highest_score, 2),
        "explanation": explanation,
        "best_evidence": (
            {
                "title": best_match_record.get("title"),
                "url": best_match_record.get("url"),
                "sources": best_match_record.get("sources"),
                "snippet": best_snippet,
            }
            if best_match_record
            else None
        ),
    }
