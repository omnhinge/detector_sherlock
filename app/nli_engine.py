from typing import Any, Optional
import math


_NLI_MODEL: Any = None
_NLI_LOAD_ATTEMPTED: bool = False
DEFAULT_MODEL_NAME = "cross-encoder/nli-deberta-v3-small"


def _softmax(scores: list[float]) -> list[float]:
    """Compute softmax probabilities from raw model logits."""
    max_score = max(scores) if scores else 0.0
    exp_scores = [math.exp(s - max_score) for s in scores]
    total = sum(exp_scores) or 1.0
    return [round(s / total, 4) for s in exp_scores]


def _get_nli_model():
    """
    Lazy load of the CrossEncoder NLI model (singleton).
    Returns None if sentence_transformers or the model weights are not available.
    """
    global _NLI_MODEL, _NLI_LOAD_ATTEMPTED
    if _NLI_MODEL is not None:
        return _NLI_MODEL
    if _NLI_LOAD_ATTEMPTED:
        return None

    _NLI_LOAD_ATTEMPTED = True
    try:
        from sentence_transformers import CrossEncoder

        _NLI_MODEL = CrossEncoder(DEFAULT_MODEL_NAME)
        return _NLI_MODEL
    except Exception:
        return None


def score_nli(premise: str, hypothesis: str) -> dict[str, Any]:
    """
    Evaluate natural language inference (NLI) between an evidence premise
    and a candidate claim hypothesis.

    Returns:
    {
        "entailment": float | None,
        "contradiction": float | None,
        "neutral": float | None,
        "model": str,
        "available": bool,
    }
    """
    if not premise or not hypothesis:
        return {
            "entailment": 0.0,
            "contradiction": 0.0,
            "neutral": 1.0,
            "model": "none",
            "available": False,
        }

    model = _get_nli_model()
    if model is None:
        return {
            "entailment": None,
            "contradiction": None,
            "neutral": None,
            "model": "unavailable",
            "available": False,
        }

    try:
        raw_output = model.predict([(premise, hypothesis)])
        logits = raw_output[0] if hasattr(raw_output, "__len__") else [0.0, 0.0, 0.0]

        # Extract mapping from model configuration (e.g. {0: 'contradiction', 1: 'entailment', 2: 'neutral'})
        config = getattr(model, "config", None)
        id2label = getattr(config, "id2label", None) if config else None

        probs = _softmax(list(logits))

        if id2label and isinstance(id2label, dict):
            label_map = {id2label[i].lower(): probs[i] for i in range(min(len(probs), len(id2label)))}
            p_contra = label_map.get("contradiction", probs[0] if len(probs) > 0 else 0.0)
            p_entail = label_map.get("entailment", probs[1] if len(probs) > 1 else 0.0)
            p_neutral = label_map.get("neutral", probs[2] if len(probs) > 2 else 0.0)
        else:
            # Default standard MNLI order: 0=contradiction, 1=entailment, 2=neutral
            p_contra = probs[0] if len(probs) > 0 else 0.0
            p_entail = probs[1] if len(probs) > 1 else 0.0
            p_neutral = probs[2] if len(probs) > 2 else 0.0

        return {
            "entailment": p_entail,
            "contradiction": p_contra,
            "neutral": p_neutral,
            "model": DEFAULT_MODEL_NAME,
            "available": True,
        }
    except Exception:
        return {
            "entailment": None,
            "contradiction": None,
            "neutral": None,
            "model": "error",
            "available": False,
        }
