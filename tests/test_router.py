from app.router import detect_domain, get_sources_for_domain


def test_detect_domain_biomedical():
    claim = "Metformin is used to treat patients with type 2 diabetes."
    assert detect_domain(claim) == "biomedical"
    sources = get_sources_for_domain("biomedical")
    assert "PubMed" in sources
    assert "Crossref" in sources
    assert "OpenAlex" in sources


def test_detect_domain_computing():
    claim = "Python software compiler was implemented using an efficient algorithm."
    assert detect_domain(claim) == "computing"
    sources = get_sources_for_domain("computing")
    assert "PubMed" not in sources
    assert "Crossref" in sources
    assert "OpenAlex" in sources


def test_detect_domain_space():
    claim = "The telescope observed a distant galaxy near Mars."
    assert detect_domain(claim) == "space"
    sources = get_sources_for_domain("space")
    assert "PubMed" not in sources


def test_detect_domain_general():
    claim = "The historical event occurred in the early nineteenth century."
    assert detect_domain(claim) == "general"
    sources = get_sources_for_domain("general")
    assert "PubMed" not in sources
