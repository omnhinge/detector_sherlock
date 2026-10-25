from app.claims import extract_claims


def test_extract_claims_empty():
    assert extract_claims("") == []
    assert extract_claims("   ") == []


def test_extract_claims_filters_short_fragments():
    text = "Short. This is a sufficiently long factual claim that should be extracted."
    claims = extract_claims(text)
    assert len(claims) == 1
    assert claims[0] == "This is a sufficiently long factual claim that should be extracted."


def test_extract_claims_multiple():
    text = "Python was created by Guido van Rossum. It was first released in 1991."
    claims = extract_claims(text)
    assert len(claims) == 2
    assert "Python was created by Guido van Rossum." in claims
    assert "It was first released in 1991." in claims


def test_extract_claims_adverbial_clause_split():
    text = "India had a GDP exceeding 3 trillion USD in 2023, while its annual inflation rate exceeded 85 % in the same year."
    claims = extract_claims(text)
    assert len(claims) == 2
    assert "India had a GDP exceeding 3 trillion USD in 2023" in claims[0]
    assert "annual inflation rate exceeded 85 % in the same year." in claims[1]
    assert not claims[0].endswith("while its annual inflation rate")
