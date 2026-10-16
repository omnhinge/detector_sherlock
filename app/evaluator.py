import re
from typing import Any


# Common English words that carry little specific factual meaning.
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

# Opposing directional / polarity keyword pairs: (positive_word_set, negative_word_set)
POLARITY_PAIRS = [
    ({"reduces", "decreases", "lowers", "prevents", "cures", "treats", "protects", "lowered"},
     {"causes", "increases", "induces", "aggravates", "triggers", "elevates", "increased"}),
    ({"safe", "beneficial", "effective", "approved", "recommended", "essential", "first-line"},
     {"dangerous", "harmful", "toxic", "banned", "ineffective", "hazardous", "contraindicated"}),
    ({"causes", "creates", "generates", "increases"},
     {"prevents", "inhibits", "blocks", "reduces", "protects"}),
]


def extract_keywords(text: str) -> set[str]:
    """
    Extract meaningful, lowercased keywords from a piece of text,
    ignoring common filler words and short symbols.
    """
    words = re.findall(r"\b[a-zA-Z0-9_-]{3,}\b", text.lower())
    return {w for w in words if w not in STOPWORDS}


def check_for_contradiction(claim_text: str, candidate_text: str) -> tuple[bool, str]:
    """
    Detect if a candidate text (title or passage) directly opposes/contradicts
    the assertion in the claim.
    Returns: (is_contradicted, explanation)
    """
    claim_words = set(re.findall(r"\b[a-z0-9_-]+\b", claim_text.lower()))
    cand_words = set(re.findall(r"\b[a-z0-9_-]+\b", candidate_text.lower()))

    # Check polarity flip pairs
    for group_a, group_b in POLARITY_PAIRS:
        # Case 1: Claim has word from group A, Candidate has opposing word from group B
        claim_has_a = claim_words & group_a
        cand_has_b = cand_words & group_b
        if claim_has_a and cand_has_b:
            return (
                True,
                f"Claim asserts '{', '.join(claim_has_a)}' whereas evidence asserts opposing '{', '.join(cand_has_b)}'.",
            )

        # Case 2: Claim has word from group B, Candidate has opposing word from group A
        claim_has_b = claim_words & group_b
        cand_has_a = cand_words & group_a
        if claim_has_b and cand_has_a:
            return (
                True,
                f"Claim asserts '{', '.join(claim_has_b)}' whereas evidence asserts opposing '{', '.join(cand_has_a)}'.",
            )

    return (False, "")


def find_best_passage(claim_keywords: set[str], abstract: str) -> tuple[str, float]:
    """
    Look through every sentence in a paper's abstract to find the single
    sentence that has the highest overlap with the claim's keywords.
    Returns: (best_matching_sentence, match_ratio)
    """
    if not abstract or not claim_keywords:
        return ("", 0.0)

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
    - verdict: 'supported', 'contradicted', or 'unverified'
    - confidence: float between 0.0 and 1.0
    - best_evidence: the paper and specific sentence that relates to the claim
    - explanation: human-readable explanation of the verdict
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

    # First, scan all records for direct contradictions
    # (A contradiction exists when a paper discusses the subject but asserts opposing polarity)
    for record in evidence_records:
        title = record.get("title", "")
        abstract = record.get("abstract", "")

        # Check title
        title_keywords = extract_keywords(title)
        topic_overlap = (
            len(claim_keywords & title_keywords) / len(claim_keywords)
            if claim_keywords
            else 0.0
        )

        # Check candidate snippets in abstract as well
        cand_snippet, passage_overlap = find_best_passage(claim_keywords, abstract)
        combined_overlap = max(topic_overlap, passage_overlap)

        # If the paper is clearly discussing this subject (overlap >= 35%)
        if combined_overlap >= 0.35:
            # Check contradiction against title
            is_contra_t, reason_t = check_for_contradiction(claim, title)
            if is_contra_t:
                return {
                    "verdict": "contradicted",
                    "confidence": round(combined_overlap, 2),
                    "explanation": f"Contradicted by evidence in '{title}': {reason_t}",
                    "best_evidence": {
                        "title": title,
                        "url": record.get("url"),
                        "sources": record.get("sources"),
                        "snippet": title,
                    },
                }

            # Check contradiction against snippet
            if cand_snippet:
                is_contra_s, reason_s = check_for_contradiction(claim, cand_snippet)
                if is_contra_s:
                    return {
                        "verdict": "contradicted",
                        "confidence": round(combined_overlap, 2),
                        "explanation": f"Contradicted by evidence in '{title}': {reason_s} Snippet: \"{cand_snippet}\"",
                        "best_evidence": {
                            "title": title,
                            "url": record.get("url"),
                            "sources": record.get("sources"),
                            "snippet": cand_snippet,
                        },
                    }

        # Track best matching record for support/unverified
        if combined_overlap > highest_score:
            highest_score = combined_overlap
            best_snippet = cand_snippet if cand_snippet else title
            best_match_record = record

    # Verdict threshold logic for support:
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
