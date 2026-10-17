
def calculate_score(claims: list[dict]) -> dict:
    """Calculate evidence support and coverage from assessed claims."""

    supported = sum(
        1 for claim in claims
        if claim.get("verdict") == "supported"
    )

    contradicted = sum(
        1 for claim in claims
        if claim.get("verdict") == "contradicted"
    )

    insufficient = sum(
        1 for claim in claims
        if claim.get("verdict") == "insufficient_evidence"
    )

    total = len(claims)
    assessed = supported + contradicted

    score = (
        round(supported / assessed * 100, 1)
        if assessed > 0
        else None
    )

    coverage = (
        round(assessed / total * 100, 1)
        if total > 0
        else 0.0
    )

    return {
        "score": score,
        "score_status": (
            "calculated" if assessed > 0
            else "insufficient_evidence"
        ),
        "evidence_coverage": coverage,
        "total_claims": total,
        "supported": supported,
        "contradicted": contradicted,
        "insufficient_evidence": insufficient,
        "not_checked": total - supported - contradicted - insufficient,
    }
