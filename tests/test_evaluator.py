from app.evaluator import (
    evaluate_claim_against_evidence,
    check_for_contradiction,
    verify_numerical_grounding,
)


def test_evaluator_empty_evidence():
    claim = "Some claim without any evidence."
    result = evaluate_claim_against_evidence(claim, [])
    assert result["verdict"] == "inconclusive"
    assert result["confidence"] == 0.0


def test_evaluator_supported():
    claim = "Metformin is a first-line medication for type 2 diabetes."
    evidence = [
        {
            "title": "Metformin therapy",
            "abstract": "Currently, metformin is the first-line medication to treat type 2 diabetes mellitus.",
            "url": "https://doi.org/10.1234/test",
            "sources": ["PubMed"],
        }
    ]
    result = evaluate_claim_against_evidence(claim, evidence)
    assert result["verdict"] == "supported"
    assert result["confidence"] >= 0.60
    assert result["best_evidence"] is not None
    assert "metformin is the first-line medication" in result["best_evidence"]["snippet"].lower()


def test_evaluator_contradicted():
    claim = "Smoking cigarettes reduces the risk of lung cancer."
    evidence = [
        {
            "title": "Smoking cigarettes increases risk of lung cancer",
            "abstract": "Extensive studies confirm that cigarette smoking substantially increases the risk of developing lung cancer.",
            "url": "https://doi.org/10.1234/cancer",
            "sources": ["PubMed"],
        }
    ]
    result = evaluate_claim_against_evidence(claim, evidence)
    assert result["verdict"] == "contradicted"
    assert "opposing" in result["explanation"]


def test_numerical_grounding_rejects_hallucinated_percentage():
    claim = "India's annual inflation rate exceeded 85 % in 2023."
    evidence = [
        {
            "title": "India: Average annual inflation rate, 2001-2007",
            "abstract": "The average annual inflation rate between 2001 and 2007 in India was reported.",
            "url": "https://doi.org/10.1234/econ",
            "sources": ["Crossref"],
        }
    ]
    result = evaluate_claim_against_evidence(claim, evidence)
    assert result["verdict"] == "inconclusive"
    assert "quantitative figures" in result["explanation"] or "substantiated" in result["explanation"]


def test_verify_numerical_grounding_direct():
    # Percentage check
    ok, reason = verify_numerical_grounding(
        "Inflation rose by 85 % this quarter.",
        "Inflation rose by 5.2% this quarter.",
    )
    assert not ok
    assert "85%" in reason

    # Scale check
    ok, reason = verify_numerical_grounding(
        "The GDP surpassed 3 trillion USD.",
        "Economic projections for the period.",
    )
    assert not ok
    assert "3 trillion" in reason

    # Matching figures
    ok, reason = verify_numerical_grounding(
        "Quarterly revenue was 30 billion dollars.",
        "The company reported quarterly revenue of 30 billion dollars in fiscal 2021.",
    )
    assert ok
