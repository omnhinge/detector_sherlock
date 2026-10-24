from app.evaluator import evaluate_claim_against_evidence, check_for_contradiction


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
