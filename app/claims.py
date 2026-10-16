
import re


def extract_claims(text: str) -> list[str]:
    """Split an AI-generated answer into candidate claims."""

    if not text or not text.strip():
        return []

    # Split text at sentence-ending punctuation or line breaks.
    parts = re.split(r'(?<=[.!?])\s+|\n+', text.strip())

    claims = []

    for part in parts:
        claim = part.strip()

        # Remove empty fragments and very short pieces.
        if len(claim) < 15:
            continue

        claims.append(claim)

    return claims
