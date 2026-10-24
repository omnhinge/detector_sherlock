from app import claims


class FakeToken:
    def __init__(self, text, dep, pos, index, start):
        self.text = text
        self.dep_ = dep
        self.pos_ = pos
        self.i = index
        self.idx = start
        self.head = self
        self.children = []

    @property
    def subtree(self):
        tokens = [self]
        for child in self.children:
            tokens.extend(child.subtree)
        return sorted(tokens, key=lambda token: token.i)


class FakeNlp:
    def __init__(self, tokens):
        self.tokens = tokens

    def __call__(self, text):
        tokens = []
        position = 0
        for index, (word, dep, pos, head_index) in enumerate(self.tokens):
            start = text.index(word, position)
            token = FakeToken(word, dep, pos, index, start)
            tokens.append(token)
            position = start + len(word)

        for token, (_, _, _, head_index) in zip(tokens, self.tokens):
            token.head = tokens[head_index]
            if token.head is not token:
                token.head.children.append(token)

        return tokens


class FakeParser:
    def __init__(self, token_maps):
        self.token_maps = token_maps

    def __call__(self, text):
        return FakeNlp(self.token_maps[text])(text)


def _use_fake_parse(monkeypatch, tokens):
    monkeypatch.setattr(claims, "_load_spacy", lambda: FakeNlp(tokens))


def _use_fake_parses(monkeypatch, token_maps):
    monkeypatch.setattr(
        claims, "_load_spacy", lambda: FakeParser(token_maps)
    )


def test_split_coordinated_predicates_inherits_shared_subject(monkeypatch):
    _use_fake_parse(
        monkeypatch,
        [
            ("Aspirin", "nsubj", "NOUN", 1),
            ("reduces", "ROOT", "VERB", 1),
            ("fever", "dobj", "NOUN", 1),
            ("and", "cc", "CCONJ", 1),
            ("prevents", "conj", "VERB", 1),
            ("heart", "compound", "NOUN", 6),
            ("attacks", "dobj", "NOUN", 4),
            (".", "punct", "PUNCT", 1),
        ],
    )

    assert claims._split_with_spacy(
        "Aspirin reduces fever and prevents heart attacks."
    ) == [
        "Aspirin reduces fever",
        "Aspirin prevents heart attacks.",
    ]


def test_split_preserves_explicit_subject_and_contrast(monkeypatch):
    _use_fake_parse(
        monkeypatch,
        [
            ("Aspirin", "nsubj", "NOUN", 1),
            ("reduces", "ROOT", "VERB", 1),
            ("fever", "dobj", "NOUN", 1),
            (",", "punct", "PUNCT", 1),
            ("but", "cc", "CCONJ", 1),
            ("ibuprofen", "nsubj", "NOUN", 6),
            ("causes", "conj", "VERB", 1),
            ("nausea", "dobj", "NOUN", 6),
            (".", "punct", "PUNCT", 1),
        ],
    )

    assert claims._split_with_spacy(
        "Aspirin reduces fever, but ibuprofen causes nausea."
    ) == [
        "Aspirin reduces fever",
        "ibuprofen causes nausea.",
    ]


def test_coordinated_adjectives_are_not_split(monkeypatch):
    _use_fake_parse(
        monkeypatch,
        [
            ("The", "det", "DET", 2),
            ("drug", "nsubj", "NOUN", 3),
            ("is", "cop", "AUX", 3),
            ("safe", "ROOT", "ADJ", 3),
            ("and", "cc", "CCONJ", 3),
            ("effective", "conj", "ADJ", 3),
            (".", "punct", "PUNCT", 3),
        ],
    )

    sentence = "The drug is safe and effective."
    assert claims._split_with_spacy(sentence) == [sentence]


def test_coordinated_noun_objects_are_not_split(monkeypatch):
    _use_fake_parse(
        monkeypatch,
        [
            ("Aspirin", "nsubj", "NOUN", 1),
            ("reduces", "ROOT", "VERB", 1),
            ("fever", "dobj", "NOUN", 1),
            ("and", "cc", "CCONJ", 1),
            ("pain", "conj", "NOUN", 2),
            (".", "punct", "PUNCT", 1),
        ],
    )

    sentence = "Aspirin reduces fever and pain."
    assert claims._split_with_spacy(sentence) == [sentence]


def test_extract_claims_splits_compound_sentence_and_inherits_subject(monkeypatch):
    _use_fake_parse(
        monkeypatch,
        [
            ("Aspirin", "nsubj", "NOUN", 1),
            ("reduces", "ROOT", "VERB", 1),
            ("fever", "dobj", "NOUN", 1),
            ("and", "cc", "CCONJ", 1),
            ("prevents", "conj", "VERB", 1),
            ("heart", "compound", "NOUN", 6),
            ("attacks", "dobj", "NOUN", 4),
            (".", "punct", "PUNCT", 1),
        ],
    )

    assert claims.extract_claims(
        "Aspirin reduces fever and prevents heart attacks."
    ) == [
        "Aspirin reduces fever",
        "Aspirin prevents heart attacks.",
    ]


def test_extract_claims_preserves_sentence_fallback_without_spacy(monkeypatch):
    monkeypatch.setattr(claims, "_load_spacy", lambda: None)
    sentence = "Aspirin reduces fever and prevents heart attacks."

    assert claims.extract_claims(sentence) == [sentence]


def test_structured_claim_extracts_subject_predicate_and_object(monkeypatch):
    _use_fake_parse(
        monkeypatch,
        [
            ("Aspirin", "nsubj", "NOUN", 1),
            ("reduces", "ROOT", "VERB", 1),
            ("fever", "dobj", "NOUN", 1),
            (".", "punct", "PUNCT", 1),
        ],
    )

    result = claims._extract_structured_claims("Aspirin reduces fever.")

    assert len(result) == 1
    assert result[0].subject == "Aspirin"
    assert result[0].predicate == "reduces"
    assert result[0].object == "fever"
    assert result[0].is_complete
    assert not result[0].negated
    assert result[0].parser_available


def test_structured_claim_marks_missing_object_incomplete(monkeypatch):
    _use_fake_parse(
        monkeypatch,
        [
            ("The", "det", "DET", 1),
            ("treatment", "nsubj", "NOUN", 2),
            ("helped", "ROOT", "VERB", 2),
            (".", "punct", "PUNCT", 2),
        ],
    )

    result = claims._extract_structured_claims("The treatment helped.")

    assert result[0].subject == "The treatment"
    assert result[0].predicate == "helped"
    assert result[0].object is None
    assert not result[0].is_complete


def test_structured_claim_records_negation_without_losing_it(monkeypatch):
    _use_fake_parse(
        monkeypatch,
        [
            ("Aspirin", "nsubj", "NOUN", 3),
            ("does", "aux", "AUX", 3),
            ("not", "neg", "PART", 3),
            ("reduce", "ROOT", "VERB", 3),
            ("fever", "dobj", "NOUN", 3),
            (".", "punct", "PUNCT", 3),
        ],
    )

    result = claims._extract_structured_claims(
        "Aspirin does not reduce fever."
    )

    assert result[0].subject == "Aspirin"
    assert result[0].predicate == "reduce"
    assert result[0].object == "fever"
    assert result[0].negated
    assert result[0].is_complete


def test_structured_claims_keep_coordinated_shared_subjects(monkeypatch):
    _use_fake_parses(
        monkeypatch,
        {
            "Aspirin reduces fever and prevents heart attacks.": [
                ("Aspirin", "nsubj", "NOUN", 1),
                ("reduces", "ROOT", "VERB", 1),
                ("fever", "dobj", "NOUN", 1),
                ("and", "cc", "CCONJ", 1),
                ("prevents", "conj", "VERB", 1),
                ("heart", "compound", "NOUN", 6),
                ("attacks", "dobj", "NOUN", 4),
                (".", "punct", "PUNCT", 1),
            ],
            "Aspirin reduces fever": [
                ("Aspirin", "nsubj", "NOUN", 1),
                ("reduces", "ROOT", "VERB", 1),
                ("fever", "dobj", "NOUN", 1),
            ],
            "Aspirin prevents heart attacks.": [
                ("Aspirin", "nsubj", "NOUN", 1),
                ("prevents", "ROOT", "VERB", 1),
                ("heart", "compound", "NOUN", 3),
                ("attacks", "dobj", "NOUN", 1),
                (".", "punct", "PUNCT", 1),
            ],
        },
    )

    result = claims._extract_structured_claims(
        "Aspirin reduces fever and prevents heart attacks."
    )

    assert [(claim.subject, claim.predicate, claim.object) for claim in result] == [
        ("Aspirin", "reduces", "fever"),
        ("Aspirin", "prevents", "heart attacks"),
    ]


def test_structured_claims_preserve_sentence_boundaries(monkeypatch):
    _use_fake_parses(
        monkeypatch,
        {
            "Aspirin reduces fever.": [
                ("Aspirin", "nsubj", "NOUN", 1),
                ("reduces", "ROOT", "VERB", 1),
                ("fever", "dobj", "NOUN", 1),
                (".", "punct", "PUNCT", 1),
            ],
            "Ibuprofen treats pain.": [
                ("Ibuprofen", "nsubj", "NOUN", 1),
                ("treats", "ROOT", "VERB", 1),
                ("pain", "dobj", "NOUN", 1),
                (".", "punct", "PUNCT", 1),
            ],
        },
    )

    result = claims._extract_structured_claims(
        "Aspirin reduces fever. Ibuprofen treats pain."
    )

    assert [claim.text for claim in result] == [
        "Aspirin reduces fever.",
        "Ibuprofen treats pain.",
    ]
    assert [claim.subject for claim in result] == ["Aspirin", "Ibuprofen"]


def test_structured_claim_fallback_marks_spo_unavailable(monkeypatch):
    monkeypatch.setattr(claims, "_load_spacy", lambda: None)

    result = claims._extract_structured_claims(
        "Aspirin reduces fever. Ibuprofen treats pain."
    )

    assert [claim.text for claim in result] == [
        "Aspirin reduces fever.",
        "Ibuprofen treats pain.",
    ]
    assert all(not claim.parser_available for claim in result)
    assert all(not claim.is_complete for claim in result)
    assert all(claim.subject is None for claim in result)
