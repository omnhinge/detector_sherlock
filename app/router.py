import re


# Keywords for domain identification
BIOMEDICAL_KEYWORDS = {
    "patient", "patients", "disease", "diseases", "treatment", "treatments",
    "medication", "medications", "drug", "drugs", "clinical", "therapy",
    "therapies", "vaccine", "vaccines", "cancer", "diabetes", "cardiac",
    "blood", "cell", "cells", "metformin", "symptom", "symptoms", "syndrome",
    "infection", "infectious", "virus", "viral", "bacteria", "bacterial",
    "dna", "rna", "gene", "genes", "protein", "proteins", "organ", "organs",
    "dosage", "pharmacology", "physician", "hospital", "diagnosis", "surgery",
}

COMPUTING_KEYWORDS = {
    "software", "hardware", "programming", "compiler", "algorithm", "algorithms",
    "code", "coding", "database", "databases", "network", "networks", "python",
    "java", "javascript", "c++", "rust", "cpu", "gpu", "api", "apis", "framework",
    "git", "linux", "operating system", "neural network", "machine learning",
    "deep learning", "transformer", "artificial intelligence", "server", "servers",
}

SPACE_KEYWORDS = {
    "planet", "planets", "mars", "jupiter", "saturn", "moon", "orbit", "orbital",
    "galaxy", "galaxies", "asteroid", "asteroids", "star", "stars", "telescope",
    "astronomy", "astrophysics", "solar", "spacecraft", "nasa", "esa", "cosmic",
}

FINANCIAL_KEYWORDS = {
    "tesla", "revenue", "quarterly", "earnings", "fiscal", "delivery", "deliveries",
    "stock", "shares", "dividend", "sec", "nasdaq", "nyse", "ipo", "profit",
    "margin", "billion", "million", "valuation", "apple", "microsoft", "amazon",
    "google", "meta", "nvidia", "gigafactory", "roadster", "q1", "q2", "q3", "q4",
}

GOVERNMENT_KEYWORDS = {
    "gdp", "inflation", "cpi", "rbi", "reserve", "monetary", "repo", "budget",
    "deficit", "unemployment", "population", "census", "treaty", "parliament",
    "ministry", "tax", "tariff", "economic", "economy",
}


def detect_domain(claim: str) -> str:
    """
    Identifies the subject domain of a claim based on keywords.
    Returns: 'biomedical', 'computing', 'space', 'financial', 'government', or 'general'.
    """
    text_lower = claim.lower()
    words = set(re.findall(r"\b[a-z0-9_-]+\b", text_lower))

    # Calculate match count for each domain
    bio_matches = len(words & BIOMEDICAL_KEYWORDS)
    comp_matches = len(words & COMPUTING_KEYWORDS)
    space_matches = len(words & SPACE_KEYWORDS)
    fin_matches = len(words & FINANCIAL_KEYWORDS)
    gov_matches = len(words & GOVERNMENT_KEYWORDS)

    counts = {
        "biomedical": bio_matches,
        "computing": comp_matches,
        "space": space_matches,
        "financial": fin_matches,
        "government": gov_matches,
    }

    best_domain = max(counts, key=counts.get)

    # Require at least one keyword match to specialize; otherwise default to general
    if counts[best_domain] > 0:
        return best_domain
    return "general"


def get_sources_for_domain(domain: str) -> list[str]:
    """
    Returns the list of source names appropriate for a given domain.
    """
    if domain == "biomedical":
        # Biomedical relies heavily on PubMed alongside Crossref and OpenAlex
        return ["PubMed", "Crossref", "OpenAlex"]
    elif domain == "financial":
        # Corporate disclosures and financial filings
        return ["SEC EDGAR", "Crossref", "OpenAlex"]
    elif domain == "government":
        # Macroeconomic and official government data
        return ["World Bank / GovData", "Crossref", "OpenAlex"]
    elif domain in ("computing", "space"):
        # Technical/scientific computing doesn't belong in medical PubMed
        return ["Crossref", "OpenAlex"]
    else:
        # General claims check broad cross-disciplinary indexes
        return ["Crossref", "OpenAlex"]
