import pytest
from app import nli_engine
from app.nli_engine import _softmax, score_nli
from app.evaluator import evaluate_claim_against_evidence


class FakeCrossEncoder:
    def __init__(self, logits):
        self.logits = logits
        self.config = type("Config", (), {"id2label": {0: "contradiction", 1: "entailment", 2: "neutral"}})()

    def predict(self, pairs):
        return [self.logits]


def test_softmax_distribution():
    probs = _softmax([1.0, 2.0, 3.0])
    assert len(probs) == 3
    assert abs(sum(probs) - 1.0) < 0.01
    assert probs[2] > probs[1] > probs[0]


def test_score_nli_empty_inputs():
    res = score_nli("", "Some claim")
    assert res["available"] is False
    assert res["neutral"] == 1.0


def test_evaluator_with_mock_nli_entailment(monkeypatch):
    # Mock NLI model returning strong entailment (index 1)
    monkeypatch.setattr(
        nli_engine, "_get_nli_model", lambda: FakeCrossEncoder([-2.0, 5.0, -1.0])
    )

    claim = "Regular exercise improves cardiac health."
    evidence = [
        {
            "title": "Cardiovascular exercise benefits",
            "abstract": "Physical activity protects cardiac function in humans.",
            "url": "https://doi.org/10.1234/heart",
            "sources": ["Crossref"],
        }
    ]

    result = evaluate_claim_against_evidence(claim, evidence)
    assert result["verdict"] == "supported"
    assert result["nli"]["available"] is True
    assert result["nli"]["entailment"] > 0.80


def test_evaluator_with_mock_nli_contradiction(monkeypatch):
    # Mock NLI model returning strong contradiction (index 0)
    monkeypatch.setattr(
        nli_engine, "_get_nli_model", lambda: FakeCrossEncoder([6.0, -3.0, -1.0])
    )

    claim = "Sugar consumption prevents dental cavities."
    evidence = [
        {
            "title": "Dental health and nutrition",
            "abstract": "High intake of refined sugars accelerates cavity formation and tooth decay.",
            "url": "https://doi.org/10.1234/teeth",
            "sources": ["PubMed"],
        }
    ]

    result = evaluate_claim_against_evidence(claim, evidence)
    assert result["verdict"] == "contradicted"
    assert result["nli"]["available"] is True
    assert result["nli"]["contradiction"] > 0.80
