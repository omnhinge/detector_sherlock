import re
from typing import Any, Optional


_SPACY_NLP: Any = None


def _load_spacy():
    """
    Load spaCy English model once and cache it.
    Returns None if spaCy or the model is unavailable.
    """
    global _SPACY_NLP
    if _SPACY_NLP is not None:
        return _SPACY_NLP
    try:
        import spacy
        _SPACY_NLP = spacy.load("en_core_web_sm")
        return _SPACY_NLP
    except Exception:
        return None


class StructuredClaim:
    """
    Represents an atomic claim decomposed into Subject-Predicate-Object (SPO) components.
    """
    def __init__(
        self,
        text: str,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        object: Optional[str] = None,
        negated: bool = False,
        is_complete: bool = False,
        parser_available: bool = True,
    ):
        self.text = text
        self.subject = subject
        self.predicate = predicate
        self.object = object
        self.negated = negated
        self.is_complete = is_complete
        self.parser_available = parser_available

    def __repr__(self) -> str:
        return (
            f"StructuredClaim(text={self.text!r}, subject={self.subject!r}, "
            f"predicate={self.predicate!r}, object={self.object!r}, "
            f"negated={self.negated}, is_complete={self.is_complete})"
        )


def _split_with_spacy(sentence: str) -> list[str]:
    """
    Use spaCy dependency parsing to split compound sentences with coordinated
    verb predicates into independent atomic propositions.
    If the second clause shares the first clause's subject, it inherits that subject.
    """
    nlp = _load_spacy()
    if nlp is None:
        return [sentence]

    doc = nlp(sentence)
    tokens = list(doc)

    # Find coordinated verb predicates:
    # A token with dep_ == 'conj', pos_ == 'VERB', whose head is also a VERB.
    conj_verb = None
    for token in tokens:
        if token.dep_ == "conj" and token.pos_ == "VERB" and token.head.pos_ == "VERB":
            conj_verb = token
            break

    # If no conj verb, check for adverbial participial clause (e.g. "...March 2023, delivering over 1,200 units...")
    is_advcl = False
    if conj_verb is None:
        for token in tokens:
            if token.dep_ == "advcl" and token.pos_ == "VERB" and token.head.pos_ == "VERB":
                token_left = token.left_edge if hasattr(token, "left_edge") else (list(token.subtree)[0] if hasattr(token, "subtree") else token)
                # Check if preceded by comma
                if token.i > 0 and (tokens[token.i - 1].text == "," or (token_left.i > 0 and tokens[token_left.i - 1].text == ",")):
                    conj_verb = token
                    is_advcl = True
                    break

    if conj_verb is None:
        return [sentence]

    head_verb = conj_verb.head

    # Find subject of head verb
    head_subj = None
    for child in head_verb.children:
        if child.dep_ in ("nsubj", "nsubjpass"):
            head_subj = child
            break

    # Find subject of conj verb if explicitly present
    conj_subj = None
    for child in conj_verb.children:
        if child.dep_ in ("nsubj", "nsubjpass"):
            conj_subj = child
            break

    # Find coordinating conjunction token ('and', 'but', etc.)
    cc_token = None
    for token in tokens:
        if token.dep_ == "cc" and token.head in (head_verb, conj_verb):
            cc_token = token
            break

    # Determine split token: prefer cc_token if present, else leftmost edge of conj_verb clause
    conj_left = conj_verb.left_edge if hasattr(conj_verb, "left_edge") else (list(conj_verb.subtree)[0] if hasattr(conj_verb, "subtree") else conj_verb)
    conj_left_idx = conj_left.i
    split_token_idx = cc_token.i if cc_token is not None else conj_left_idx
    if split_token_idx > 0 and tokens[split_token_idx - 1].dep_ == "punct":
        split_token_idx -= 1

    # Clause 1: tokens from start of sentence up to the split point
    clause1_tokens = [t for t in tokens if t.i < split_token_idx]
    clause1_text = " ".join(t.text for t in clause1_tokens).strip()
    clause1_text = re.sub(r"[,;:\s]+$", "", clause1_text)

    # Trailing punctuation of the original sentence (e.g. '.')
    trailing_punct = ""
    if tokens and tokens[-1].dep_ == "punct":
        trailing_punct = tokens[-1].text

    # Clause 2:
    if conj_subj is not None:
        # Explicit subject already part of conj_verb's subtree
        conj_tokens = [t for t in conj_verb.subtree if t.dep_ != "punct"]
        clause2_body = " ".join(t.text for t in sorted(conj_tokens, key=lambda x: x.i))
    else:
        # Inherits subject from head_verb
        subj_text = " ".join(t.text for t in sorted(head_subj.subtree, key=lambda x: x.i)) if head_subj else ""
        conj_tokens = [t for t in conj_verb.subtree if t.dep_ != "punct"]
        conj_body = " ".join(t.text for t in sorted(conj_tokens, key=lambda x: x.i))
        conj_body = re.sub(r"^(while|whereas|although|and|but|so)\s+", "", conj_body.strip(), flags=re.I)
        clause2_body = f"{subj_text} {conj_body}" if subj_text else conj_body

    # Clean leading subordinate conjunctions and resolve possessive pronouns
    clause2_body = re.sub(r"^(while|whereas|although|and|but|so)\s+", "", clause2_body.strip(), flags=re.I)
    if head_subj and head_subj.text:
        clause2_body = re.sub(r"^(its|their)\s+", f"{head_subj.text}'s ", clause2_body, flags=re.I)
    clause2_text = f"{clause2_body}{trailing_punct}"

    return [clause1_text, clause2_text]


def _extract_structured_claims(text: str) -> list[StructuredClaim]:
    """
    Decompose text into structured claims with Subject-Predicate-Object annotations.
    Falls back gracefully if spaCy is not available.
    """
    if not text or not text.strip():
        return []

    # Split text by sentence-level punctuation
    raw_sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+|\n+', text.strip()) if s.strip()]

    nlp = _load_spacy()
    if nlp is None:
        return [
            StructuredClaim(
                text=sentence,
                subject=None,
                predicate=None,
                object=None,
                negated=False,
                is_complete=False,
                parser_available=False,
            )
            for sentence in raw_sentences
        ]

    structured_results: list[StructuredClaim] = []

    for raw_sentence in raw_sentences:
        atomic_clauses = _split_with_spacy(raw_sentence)
        for clause in atomic_clauses:
            doc = nlp(clause)
            tokens = list(doc)

            # Detect negation
            negated = any(t.dep_ == "neg" or t.text.lower() in ("not", "never", "n't") for t in tokens)

            # Find root verb
            root = None
            for t in tokens:
                if t.dep_ == "ROOT":
                    root = t
                    break
            if root is None:
                for t in tokens:
                    if t.pos_ in ("VERB", "AUX"):
                        root = t
                        break

            # Find subject
            subject = None
            subj_token = None
            if root is not None:
                for child in root.children:
                    if child.dep_ in ("nsubj", "nsubjpass"):
                        subj_token = child
                        break
            if subj_token is None:
                for t in tokens:
                    if t.dep_ in ("nsubj", "nsubjpass"):
                        subj_token = t
                        break

            if subj_token is not None:
                subj_tokens = [t for t in subj_token.subtree if t.dep_ != "punct"]
                subject = " ".join(t.text for t in sorted(subj_tokens, key=lambda x: x.i)).strip()

            # Find predicate
            predicate = root.text if root is not None else None

            # Find object
            object_val = None
            obj_token = None
            if root is not None:
                for child in root.children:
                    if child.dep_ in ("dobj", "pobj", "attr"):
                        obj_token = child
                        break
            if obj_token is None:
                for t in tokens:
                    if t.dep_ in ("dobj", "pobj", "attr"):
                        obj_token = t
                        break

            if obj_token is not None:
                obj_tokens = [t for t in obj_token.subtree if t.dep_ != "punct"]
                object_val = " ".join(t.text for t in sorted(obj_tokens, key=lambda x: x.i)).strip()

            is_complete = bool(subject and predicate and object_val)

            structured_results.append(
                StructuredClaim(
                    text=clause,
                    subject=subject,
                    predicate=predicate,
                    object=object_val,
                    negated=negated,
                    is_complete=is_complete,
                    parser_available=True,
                )
            )

    return structured_results


def extract_claims(text: str) -> list[str]:
    """
    Split an AI-generated answer into candidate claims.
    Uses spaCy dependency parsing when available to decompose compound statements,
    falling back to regex sentence splitting.
    """
    if not text or not text.strip():
        return []

    # Split text at sentence-ending punctuation or line breaks.
    raw_sentences = re.split(r'(?<=[.!?])\s+|\n+', text.strip())

    claims = []

    for sentence in raw_sentences:
        clean_sentence = sentence.strip()
        if not clean_sentence:
            continue

        # Try splitting compound clauses
        sub_clauses = _split_with_spacy(clean_sentence)

        for clause in sub_clauses:
            claim = clause.strip()
            # Remove empty fragments and very short pieces.
            if len(claim) < 15:
                continue
            claims.append(claim)

    return claims
