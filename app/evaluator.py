import re
from typing import Any
from app.nli_engine import score_nli


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

# Generic words that can be capitalized at the beginning of a sentence but are not entities
GENERIC_SENTENCE_STARTERS = {
    "the", "a", "an", "in", "on", "at", "during", "for", "some", "all", "each",
    "this", "that", "these", "those", "when", "while", "if", "regular", "recent",
    "many", "most", "several", "overall", "general", "high", "low", "good", "bad",
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
    ignoring HTML tags and common filler words.
    """
    clean_text = re.sub(r"<[^>]+>", " ", text)
    words = re.findall(r"\b[a-zA-Z0-9_-]{3,}\b", clean_text.lower())
    return {w for w in words if w not in STOPWORDS}


def check_for_contradiction(claim_text: str, candidate_text: str) -> tuple[bool, str]:
    """
    Detect if a candidate text (title or passage) directly opposes/contradicts
    the assertion in the claim.
    Returns: (is_contradicted, explanation)
    """
    clean_cand = re.sub(r"<[^>]+>", " ", candidate_text)
    claim_words = set(re.findall(r"\b[a-z0-9_-]+\b", claim_text.lower()))
    cand_words = set(re.findall(r"\b[a-z0-9_-]+\b", clean_cand.lower()))

    # Check polarity flip pairs
    for group_a, group_b in POLARITY_PAIRS:
        claim_has_a = claim_words & group_a
        cand_has_b = cand_words & group_b
        if claim_has_a and cand_has_b:
            return (
                True,
                f"Claim asserts '{', '.join(claim_has_a)}' whereas evidence asserts opposing '{', '.join(cand_has_b)}'.",
            )

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
    Strips raw XML/HTML tags like <jats:p>.
    Returns: (best_matching_sentence, match_ratio)
    """
    if not abstract or not claim_keywords:
        return ("", 0.0)

    clean_abstract = re.sub(r"<[^>]+>", " ", abstract).strip()
    sentences = re.split(r"(?<=[.!?])\s+", clean_abstract)
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
    Compares a single claim against retrieved evidence records.
    Filters out noise documents that do not mention core entities.
    """
    claim_keywords = extract_keywords(claim)

    if not claim_keywords:
        return {
            "verdict": "unverified",
            "confidence": 0.0,
            "explanation": "Claim does not contain sufficient specific factual keywords to verify.",
            "best_evidence": None,
            "nli": {"available": False},
        }

    if not evidence_records:
        return {
            "verdict": "unverified",
            "confidence": 0.0,
            "explanation": "No relevant authoritative records were found for this claim.",
            "best_evidence": None,
            "nli": {"available": False},
        }

    # Extract capitalized proper entities from the claim (e.g. 'Tesla', 'Roadster', 'Metformin')
    claim_tokens = re.findall(r"\b[A-Za-z0-9_-]+\b", claim)
    claim_entities = {
        t.lower() for t in claim_tokens
        if t[0].isupper() and t.lower() not in GENERIC_SENTENCE_STARTERS and t.lower() not in STOPWORDS
    }

    best_match_record = None
    best_snippet = ""
    highest_score = 0.0

    # Scan records
    for record in evidence_records:
        title = re.sub(r"<[^>]+>", " ", record.get("title", "")).strip()
        abstract = record.get("abstract", "")
        cand_text = f"{title} {abstract}".lower()

        # ENTITY GATE: If the claim mentions specific named entities, the paper MUST mention at least one!
        if claim_entities and not any(entity in cand_text for entity in claim_entities):
            continue

        # Check title
        title_keywords = extract_keywords(title)
        topic_overlap = (
            len(claim_keywords & title_keywords) / len(claim_keywords)
            if claim_keywords
            else 0.0
        )

        # Check candidate snippets in abstract
        cand_snippet, passage_overlap = find_best_passage(claim_keywords, abstract)
        combined_overlap = max(topic_overlap, passage_overlap)

        # If the paper has meaningful topical overlap (>= 30%)
        if combined_overlap >= 0.30:
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
                    "nli": {"available": False},
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
                        "nli": {"available": False},
                    }

        # Track best matching record
        if combined_overlap > highest_score:
            highest_score = combined_overlap
            best_snippet = cand_snippet if cand_snippet else title
            best_match_record = record

    if best_snippet:
        best_snippet = re.sub(r"<[^>]+>", " ", best_snippet).strip()

    # RELEVANCE THRESHOLD:
    # If highest overlap is below 20% or no entity-matched record was found, it is unverified
    if highest_score < 0.20 or not best_match_record:
        return {
            "verdict": "unverified",
            "confidence": 0.0,
            "explanation": "No sufficiently relevant records were found in queried databases.",
            "best_evidence": None,
            "nli": {"available": False},
        }

    # Tier 2: Neural Natural Language Inference check ONLY on relevant documents
    nli_res = {"available": False}
    if best_match_record and best_snippet:
        nli_res = score_nli(premise=best_snippet, hypothesis=claim)

    if nli_res.get("available") and nli_res.get("contradiction") is not None:
        p_contra = nli_res["contradiction"]
        p_entail = nli_res["entailment"]

        if p_contra >= 0.65:
            return {
                "verdict": "contradicted",
                "confidence": round(p_contra, 2),
                "explanation": (
                    f"Contradicted by evidence in '{best_match_record.get('title')}': "
                    f"Neural NLI detected high semantic contradiction ({int(p_contra * 100)}%). "
                    f"Snippet: \"{best_snippet}\""
                ),
                "best_evidence": {
                    "title": best_match_record.get("title"),
                    "url": best_match_record.get("url"),
                    "sources": best_match_record.get("sources"),
                    "snippet": best_snippet,
                },
                "nli": nli_res,
            }
        elif p_entail >= 0.65:
            blended_conf = round(0.4 * highest_score + 0.6 * p_entail, 2)
            return {
                "verdict": "supported",
                "confidence": blended_conf,
                "explanation": (
                    f"Supported by evidence from {best_match_record.get('sources', ['credible source'])} "
                    f"in '{best_match_record.get('title')}': \"{best_snippet}\" "
                    f"(Neural entailment: {int(p_entail * 100)}%)"
                ),
                "best_evidence": {
                    "title": best_match_record.get("title"),
                    "url": best_match_record.get("url"),
                    "sources": best_match_record.get("sources"),
                    "snippet": best_snippet,
                },
                "nli": nli_res,
            }

    # Lexical threshold logic for support / unverified
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
            if best_match_record and verdict in ("supported", "contradicted")
            else None
        ),
        "nli": nli_res,
    }
